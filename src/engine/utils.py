import argparse
import datetime
import h5py
import numpy as np
import os
import torch
import torch.utils.data as utils_data
import yaml
from argparse import ArgumentParser
from config import cfg


class Dataset(utils_data.Dataset):

    def __init__(self, data):
        super(Dataset, self).__init__()
        self.images = torch.tensor(data['image'])
        self.segments = torch.tensor(data['segment'])
        self.overlaps = torch.tensor(data['overlap'])
        if self.images.ndim == 5 and self.segments.ndim == 4 and self.overlaps.ndim == 4:
            self.images = self.images.reshape(-1, *self.images.shape[2:])
            self.segments = self.segments.reshape(-1, *self.segments.shape[2:])
            self.overlaps = self.overlaps.reshape(-1, *self.overlaps.shape[2:])
        assert self.images.ndim == 4
        assert self.segments.ndim == 3
        assert self.overlaps.ndim == 3

    def __getitem__(self, idx):
        image = self.images[idx]
        segment = self.segments[idx]
        overlap = self.overlaps[idx]
        data = {'image': image, 'segment': segment, 'overlap': overlap}
        return data

    def __len__(self):
        return self.images.shape[0]


def get_data_loaders(config):
    image_shape = None
    datasets = {}
    with h5py.File(config['path_data'], 'r', libver='latest', swmr=True) as f:
        phase_list = [*f.keys()]
        if not config['train']:
            phase_list = [val for val in phase_list if val not in ['train', 'valid']]
        index_sel = slice(config['batch_size']) if config['debug'] else ()
        for phase in phase_list:
            data = {key: f[phase][key][index_sel] for key in f[phase] if key not in ['layers', 'masks']}
            data['image'] = np.moveaxis(data['image'], -1, -3)
            if image_shape is None:
                image_shape = data['image'].shape[-3:]
            else:
                assert image_shape == data['image'].shape[-3:]
            datasets[phase] = Dataset(data)
    if 'train' in datasets and 'valid' not in datasets:
        datasets['valid'] = datasets['train']
    data_loaders = {
        key:
            utils_data.DataLoader(
                val,
                batch_size=config['batch_size'],
                num_workers=1,
                shuffle=(key == 'train'),
                drop_last=(key == 'train'),
                pin_memory=True,
            )
        for key, val in datasets.items()
    }
    return data_loaders, image_shape


def get_config():
    parser = ArgumentParser()
    parser.add_argument('--path_config')
    parser.add_argument('--path_data')
    parser.add_argument('--path_pretrain')
    parser.add_argument('--folder_log')
    parser.add_argument('--folder_out')
    parser.add_argument('--timestamp')
    parser.add_argument('--num_tests', type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--use_timestamp', action='store_true')
    parser.add_argument('--file_ckpt', default='ckpt.pth')
    parser.add_argument('--file_model', default='model.pth')
    parser.add_argument(
        '--task',
        type=str,
        default='train',
        metavar='TASK',
        help='What to do. See engine'
    )
    parser.add_argument(
        '--config-file',
        type=str,
        default='',
        metavar='FILE',
        help='Path to config file'
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command line',
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    # Use config file name as the default experiment name
    if cfg.exp_name == '':
        if args.config_file:
            cfg.exp_name = os.path.splitext(os.path.basename(args.config_file))[0]
        else:
            raise ValueError('exp_name cannot be empty without specifying a config file')
    with open(args.path_config) as f:
        config_extra = yaml.safe_load(f)
    for key, val in args.__dict__.items():
        if key not in config_extra or val is not None:
            config_extra[key] = val
    if config_extra['debug']:
        config_extra['ckpt_intvl'] = 1
    if config_extra['resume']:
        config_extra['train'] = True
    if config_extra['timestamp'] is None:
        config_extra['timestamp'] = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    if config_extra['use_timestamp']:
        for key in ['folder_log', 'folder_out']:
            config_extra[key] = os.path.join(config_extra[key], config_extra['timestamp'])
    if config_extra['train'] and not config_extra['resume']:
        for key in ['folder_log', 'folder_out']:
            if os.path.exists(config_extra[key]):
                raise FileExistsError(config_extra[key])
            os.makedirs(config_extra[key])
        with open(os.path.join(config_extra['folder_out'], 'config.yaml'), 'w') as f:
            yaml.safe_dump(config_extra, f)
    data_loaders, image_shape = get_data_loaders(config_extra)
    config_extra['image_shape'] = image_shape
    return cfg, args.task, data_loaders, config_extra


