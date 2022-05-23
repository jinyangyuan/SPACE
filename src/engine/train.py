from model import get_model
from solver import get_optimizers
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_


def compute_overview(config, results, dpi=100):
    def convert_image(image):
        image = np.moveaxis(image, 0, 2)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        return image
    def plot_image(ax, image, xlabel=None, ylabel=None, color=None):
        plot = ax.imshow(image, interpolation='bilinear')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(xlabel, color='k' if color is None else color, fontfamily='monospace') if xlabel else None
        ax.set_ylabel(ylabel, color='k' if color is None else color, fontfamily='monospace') if ylabel else None
        ax.xaxis.set_label_position('top')
        return plot
    def get_overview(fig_idx):
        image = results_sel['image'][fig_idx]
        recon = results_sel['recon'][fig_idx]
        apc = results_sel['apc'][fig_idx]
        shp = results_sel['shp'][fig_idx]
        pres = results_sel['pres'][fig_idx]
        rows, cols = apc.shape[0] + 1, 2
        fig, axes = plt.subplots(rows, cols, figsize=(cols + 0.2, rows), dpi=dpi)
        plot_image(axes[0, 0], convert_image(image), ylabel='scene')
        plot_image(axes[0, 1], convert_image(recon))
        for idx in range(apc.shape[0]):
            ylabel = 'obj_{}'.format(idx) if idx < apc.shape[0] - 1 else 'back'
            color = [1.0, 0.5, 0.0] if pres[idx] >= 128 else [0.0, 0.5, 1.0]
            plot_image(axes[idx + 1, 0], convert_image(apc[idx]), ylabel=ylabel, color=color)
            plot_image(axes[idx + 1, 1], convert_image(shp[idx]))
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        out = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, -1)
        plt.close(fig)
        return out
    summ_image_count = min(config['summ_image_count'], config['batch_size'])
    results_sel = {key: val[:summ_image_count].data.cpu().numpy() for key, val in results.items()}
    overview_list = [get_overview(idx) for idx in range(summ_image_count)]
    overview = np.concatenate(overview_list, axis=1)
    overview = np.moveaxis(overview, 2, 0)
    return overview


def train(cfg, data_loaders, config):
    def data_loader_train():
        while True:
            for x in data_loaders['train']:
                yield x
    data_loader_valid = data_loaders['valid']
    model = get_model(cfg, config)
    model = nn.DataParallel(model.cuda())
    optimizer_fg, optimizer_bg = get_optimizers(cfg, model)
    path_ckpt = os.path.join(config['folder_out'], config['file_ckpt'])
    path_model = os.path.join(config['folder_out'], config['file_model'])
    if config['resume']:
        checkpoint = torch.load(path_ckpt)
        step = checkpoint['step']
        best_step = checkpoint['best_step']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_fg.load_state_dict(checkpoint['optimizer_fg_state_dict'])
        optimizer_bg.load_state_dict(checkpoint['optimizer_bg_state_dict'])
        print('Resume training from step {}'.format(step))
    else:
        step = 0
        best_step = -1
        best_loss = float('inf')
        if config['path_pretrain'] is not None:
            model.load_state_dict(torch.load(config['path_pretrain']))
        print('Start training')
    print()
    with SummaryWriter(log_dir=config['folder_log'], purge_step=step + 1) as writer:
        for data_train in data_loader_train():
            step += 1
            if step > config['num_steps']:
                break
            model.train(True)
            with torch.set_grad_enabled(True):
                _, metrics, loss = model(data_train, step - 1, require_results=False)
            metrics = {key: val.mean() for key, val in metrics.items()}
            for key, val in metrics.items():
                writer.add_scalar('train/metric_{}'.format(key), val, global_step=step)
            loss = loss.mean()
            optimizer_fg.zero_grad()
            optimizer_bg.zero_grad()
            loss.backward()
            if cfg.train.clip_norm:
                clip_grad_norm_(model.parameters(), cfg.train.clip_norm)
            optimizer_fg.step()
            optimizer_bg.step()
            if step % config['ckpt_intvl'] == 0:
                with torch.set_grad_enabled(False):
                    results, _, _ = model(data_train, step - 1)
                overview = compute_overview(config, results)
                writer.add_image('train', overview, global_step=step)
                model.train(False)
                sum_metrics, sum_loss = {}, 0
                num_data = 0
                for idx_batch, data_valid in enumerate(data_loader_valid):
                    batch_size = data_valid['image'].shape[0]
                    with torch.set_grad_enabled(False):
                        results, metrics, loss = model(
                            data_valid, config['num_steps'] - 1, require_results=idx_batch == 0)
                    if idx_batch == 0:
                        overview = compute_overview(config, results)
                        writer.add_image('valid', overview, global_step=step)
                    for key, val in metrics.items():
                        if key in sum_metrics:
                            sum_metrics[key] += val.sum().item()
                        else:
                            sum_metrics[key] = val.sum().item()
                    sum_loss += loss.sum().item()
                    num_data += batch_size
                mean_metrics = {key: val / num_data for key, val in sum_metrics.items()}
                mean_loss = sum_loss / num_data
                for key, val in mean_metrics.items():
                    writer.add_scalar('valid/metric_{}'.format(key), val, global_step=step)
                writer.flush()
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    best_step = step
                    torch.save(model.state_dict(), path_model)
                save_dict = {
                    'step': step,
                    'best_step': best_step,
                    'best_loss': best_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_fg_state_dict': optimizer_fg.state_dict(),
                    'optimizer_bg_state_dict': optimizer_bg.state_dict(),
                }
                torch.save(save_dict, path_ckpt)
                print('Step: {}/{}'.format(step, config['num_steps']))
                print((' ' * 4).join([
                    'ARI_A: {:.3f}'.format(mean_metrics['ari_all']),
                    'ARI_O: {:.3f}'.format(mean_metrics['ari_obj']),
                    'MSE: {:.2e}'.format(mean_metrics['mse']),
                    'Count: {:.3f}'.format(mean_metrics['count']),
                ]))
                print('Best Step: {}'.format(best_step))
                print()
    return
