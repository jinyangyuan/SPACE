import torch
import torch.nn as nn
from .arch import arch
from .fg import SpaceFg
from .bg import SpaceBg


class Space(nn.Module):
    
    def __init__(self, config):
        nn.Module.__init__(self)

        self.seg_overlap = config['seg_overlap']
        
        self.fg_module = SpaceFg()
        self.bg_module = SpaceBg(config)
        
    def forward(self, data, global_step, require_results=True):
        """
        Inference.
        
        :param x: (B, 3, H, W)
        :param global_step: global training step
        :return:
            loss: a scalor. Note it will be better to return (B,)
            log: a dictionary for visualization
        """

        image, segment, overlap = self.convert_data(data)
        x = image
        
        # Background extraction
        # (B, 3, H, W), (B, 3, H, W), (B,)
        bg_likelihood, bg, kl_bg, log_bg = self.bg_module(x, global_step)
        
        # Foreground extraction
        fg_likelihood, fg, alpha_map, kl_fg, loss_boundary, log_fg = self.fg_module(x, global_step)

        # Fix alpha trick
        if global_step and global_step < arch.fix_alpha_steps:
            alpha_map = torch.full_like(alpha_map, arch.fix_alpha_value)
            
        # Compute final mixture likelhood
        # (B, 3, H, W)
        fg_likelihood = (fg_likelihood + (alpha_map + 1e-5).log())
        bg_likelihood = (bg_likelihood + (1 - alpha_map + 1e-5).log())
        # (B, 2, 3, H, W)
        log_like = torch.stack((fg_likelihood, bg_likelihood), dim=1)
        # (B, 3, H, W)
        log_like = torch.logsumexp(log_like, dim=1)
        # (B,)
        log_like = log_like.flatten(start_dim=1).sum(1)

        # Take mean as reconstruction
        y = alpha_map * fg + (1.0 - alpha_map) * bg
        
        # Elbo
        elbo = log_like - kl_bg - kl_fg
        
        # Mean over batch
        loss = (-elbo + loss_boundary).mean()
        
        log = {
            'imgs': x,
            'y': y,
            # (B,)
            'mse': ((y-x)**2).flatten(start_dim=1).sum(dim=1),
            'log_like': log_like
        }
        log.update(log_fg)
        log.update(log_bg)
        apc_all = torch.cat([log['apc'], bg[:, None]], dim=1)
        shp = log['shp']
        shp_all = torch.cat([shp, torch.ones([shp.shape[0], 1, *shp.shape[2:]], device=shp.device)], dim=1)
        pres = log['pres']
        pres_all = torch.cat([pres, torch.ones([pres.shape[0], 1], device=pres.device)], dim=1)
        results = {'image': x, 'recon': y, 'apc': apc_all, 'shp': shp_all, 'pres': pres_all}
        results.update({key: log[key] for key in ['order', 'mask']})
        metrics = self.compute_metrics(image, segment, overlap, results)
        if require_results:
            for key, val in results.items():
                if key not in ['order']:
                    results[key] = (val.clamp(0, 1) * 255).to(torch.uint8)
        else:
            results = {}
        return results, metrics, loss

    @staticmethod
    def convert_data(data):
        data = {key: val.cuda(non_blocking=True) for key, val in data.items()}
        image = data['image'].float() / 255
        segment_base = data['segment'][:, None, None].long()
        scatter_shape = [segment_base.shape[0], segment_base.max() + 1, *segment_base.shape[2:]]
        segment = torch.zeros(scatter_shape, device=segment_base.device).scatter_(1, segment_base, 1)
        overlap = torch.gt(data['overlap'][:, None, None], 1).float()
        return image, segment, overlap

    @staticmethod
    def compute_ari(mask_true, mask_pred):
        def comb2(x):
            x = x * (x - 1)
            if x.ndim > 1:
                x = x.sum([*range(1, x.ndim)])
            return x
        num_pixels = mask_true.sum([*range(1, mask_true.ndim)])
        mask_true = mask_true.reshape(
            [mask_true.shape[0], mask_true.shape[1], 1, mask_true.shape[-2] * mask_true.shape[-1]])
        mask_pred = mask_pred.reshape(
            [mask_pred.shape[0], 1, mask_pred.shape[1], mask_pred.shape[-2] * mask_pred.shape[-1]])
        mat = (mask_true * mask_pred).sum(-1)
        sum_row = mat.sum(1)
        sum_col = mat.sum(2)
        comb_mat = comb2(mat)
        comb_row = comb2(sum_row)
        comb_col = comb2(sum_col)
        comb_num = comb2(num_pixels)
        comb_prod = (comb_row * comb_col) / comb_num
        comb_mean = 0.5 * (comb_row + comb_col)
        diff = comb_mean - comb_prod
        score = (comb_mat - comb_prod) / diff
        invalid = ((comb_num == 0) + (diff == 0)) > 0
        score = torch.where(invalid, torch.ones_like(score), score)
        return score

    def compute_metrics(self, image, segment, overlap, results):
        segment_obj = segment[:, :-1]
        # ARI
        segment_obj_sel = segment_obj if self.seg_overlap else segment_obj * (1 - overlap)
        mask = results['mask']
        mask_oh_all = torch.argmax(mask, dim=1, keepdim=True)
        mask_oh_all = torch.zeros_like(mask).scatter_(1, mask_oh_all, 1)
        mask_oh_obj = torch.argmax(mask[:, :-1], dim=1, keepdim=True)
        mask_oh_obj = torch.zeros_like(mask[:, :-1]).scatter_(1, mask_oh_obj, 1)
        ari_all = self.compute_ari(segment_obj_sel, mask_oh_all)
        ari_obj = self.compute_ari(segment_obj_sel, mask_oh_obj)
        # MSE
        sq_diff = (results['recon'] - image).square()
        mse = sq_diff.mean([*range(1, sq_diff.ndim)])
        # Count
        count_true = segment_obj.reshape(*segment_obj.shape[:-3], -1).max(-1).values.sum(1)
        count_pred = results['pres'][:, :-1].sum(1)
        count_acc = torch.eq(count_true, count_pred).to(dtype=torch.float)
        metrics = {'ari_all': ari_all, 'ari_obj': ari_obj, 'mse': mse, 'count': count_acc}
        return metrics
