from model import get_model
import h5py
import numpy as np
import os
import torch
from torch import nn


def eval(cfg, data_loaders, config):
    def get_path_detail():
        return os.path.join(config['folder_out'], '{}.h5'.format(phase))
    for phase in data_loaders:
        path_detail = get_path_detail()
        if os.path.exists(path_detail):
            raise FileExistsError(path_detail)
    path_model = os.path.join(config['folder_out'], config['file_model'])
    model = get_model(cfg, config)
    model = nn.DataParallel(model.cuda())
    model.load_state_dict(torch.load(path_model))
    model.train(False)
    for phase in data_loaders:
        path_detail = get_path_detail()
        results_all = {}
        for data in data_loaders[phase]:
            results = {}
            for idx_run in range(config['num_tests']):
                with torch.set_grad_enabled(False):
                    sub_results, _, _ = model(data, config['num_steps'])
                for key, val in sub_results.items():
                    if key in ['image']:
                        continue
                    val = val.data.cpu().numpy()
                    if key not in  ['pres', 'order']:
                        val = np.moveaxis(val, -3, -1)
                    if key in results:
                        results[key].append(val)
                    else:
                        results[key] = [val]
            for key, val in results.items():
                val = np.stack(val)
                if key in results_all:
                    results_all[key].append(val)
                else:
                    results_all[key] = [val]
        results_all = {key: np.concatenate(val, axis=1) for key, val in results_all.items()}
        with h5py.File(path_detail, 'w') as f:
            for key, val in results_all.items():
                f.create_dataset(key, data=val, compression='gzip')
    return
