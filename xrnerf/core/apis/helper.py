import argparse
import importlib
import os
import warnings
from functools import partial, reduce

import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import (DistSamplerSeedHook, EMAHook, IterBasedRunner,
                         OptimizerHook, build_optimizer, get_dist_info)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from xrnerf.datasets import DistributedSampler, build_dataset

__all__ = ['parse_args', 'build_dataloader', 'get_optimizer', 'register_hooks', \
            'get_runner', 'update_config']


def parse_args():
    """parse args."""
    parser = argparse.ArgumentParser(description='train a nerf')
    parser.add_argument('--config',
                        help='train config file path',
                        default='configs/nerfs/nerf_base01.py')
    parser.add_argument('--dataname',
                        help='data name in dataset',
                        default='ficus')
    parser.add_argument('--test_only',
                        help='set to influence on testset once',
                        action='store_true')
    parser.add_argument(
        '--render_only',
        help='set to influence on testset once for visualization',
        action='store_true')
    parser.add_argument('--load_from', help='reset load_from', default='')
    args = parser.parse_args()
    return args


def replace_dataname(dataname, cfg):
    """Recursively replace all '#DATANAME#' to dataname, dataname is specified
    in the input args."""
    if isinstance(cfg, str):
        cfg = cfg.replace('#DATANAME#', dataname)
    elif isinstance(cfg, Config) or isinstance(cfg, dict):
        for k in cfg:
            cfg[k] = replace_dataname(dataname, cfg[k])
    return cfg


def kilo_replace(dataname, cfg):
    """Recursively replace in the input args."""
    resolution = cfg.resolution_table[dataname]
    # print("resolution:", resolution)
    if cfg.phase == 'pretrain':
        cfg.build_occupancy_tree_config.update({'resolution': resolution})
    elif cfg.phase == 'distill':
        cfg.fix_resolution = resolution
        cfg.total_num_networks = reduce(lambda x, y: x * y, resolution)
        cfg.data['train'].cfg.update({'fixed_resolution': resolution})
        cfg.data['val'].cfg.update({'fixed_resolution': resolution})
    else:
        cfg.model['mlp'].update({'resolution': resolution})
    return cfg


def update_config(dataname, cfg):
    """update_config."""
    cfg = replace_dataname(dataname, cfg)
    if cfg.method == 'kilo_nerf':
        cfg = kilo_replace(dataname, cfg)
    return cfg


def update_loadfrom(load_from, cfg):
    """update_loadfrom."""
    if len(load_from) > 0:
        cfg.load_from = os.path.join(cfg.work_dir, load_from)
    return cfg


def build_dataloader(cfg, mode='train'):
    """build_dataloader."""
    num_gpus = cfg.num_gpus
    dataset = build_dataset(cfg.data[mode])
    if num_gpus > 0:  # ddp多卡模式
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(dataset,
                                     world_size,
                                     rank,
                                     shuffle=(mode == 'train'))
    else:  # 单卡模式
        sampler = RandomSampler(
            dataset) if mode == 'train' else SequentialSampler(dataset)

    loader_cfg = cfg.data['{}_loader'.format(mode)]
    num_workers = loader_cfg['num_workers']
    bs_per_gpu = loader_cfg['batch_size']  # 分到每个gpu的bs数
    bs_all_gpus = bs_per_gpu * num_gpus  # 总的bs数

    data_loader = DataLoader(dataset,
                             batch_size=bs_all_gpus,
                             sampler=sampler,
                             num_workers=num_workers,
                             collate_fn=partial(collate,
                                                samples_per_gpu=bs_per_gpu),
                             shuffle=False)

    return data_loader, dataset


def get_optimizer(model, cfg):
    """get_optimizer."""
    if cfg.method == 'animatable_nerf':
        params = model.get_params()
        optimizer = torch.optim.Adam(params=params, lr=cfg.optimizer.lr)
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
    return optimizer


def register_hooks(hook_cfgs, **variables):
    """auto register hooks."""
    def get_variates(hook_cfg):
        variates = {}
        if 'variables' in hook_cfg:
            for k, v_name in hook_cfg['variables'].items():
                variates[k] = variables[v_name]
        return variates

    runner = variables['runner']
    hook_module = importlib.import_module('xrnerf.core.hooks')
    for hook_cfg in hook_cfgs:
        HookClass = getattr(hook_module, hook_cfg['type'])
        runner.register_hook(
            HookClass(**hook_cfg['params'], **get_variates(hook_cfg)))
    return runner


def get_runner(runner_cfg):
    """get_runner."""
    runner_module = importlib.import_module('xrnerf.core.runner')
    RunnerClass = getattr(runner_module, runner_cfg['type'])
    return RunnerClass
