
import torch
import warnings
import argparse
from functools import partial
from torch.utils.data import DataLoader, RandomSampler
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import (DistSamplerSeedHook, IterBasedRunner, OptimizerHook,
                         build_optimizer, get_dist_info)
from xrnerf.datasets import build_dataset, DistributedSampler


__all__ = ['parse_args', 'build_dataloader', 'get_optimizer']


def parse_args():
    parser = argparse.ArgumentParser(description='train a nerf')
    parser.add_argument('--config', help='train config file path', default='configs/nerfs/nerf_base01.py')
    parser.add_argument('--test_only', help='set to influence on testset once', action='store_true')
    args = parser.parse_args()
    return args

def build_dataloader(cfg, mode='train'):

    num_gpus = cfg.num_gpus
    dataset = build_dataset(cfg.data[mode])
    if num_gpus>0: # ddp多卡模式 
        rank, world_size = get_dist_info()
        sampler = DistributedSampler(dataset, world_size, rank, shuffle=True)
    else: # 单卡模式
        sampler = RandomSampler(dataset)

    loader_cfg = cfg.data['{}_loader'.format(mode)]
    num_workers = loader_cfg['num_workers']
    bs_per_gpu =  loader_cfg['batch_size'] # 分到每个gpu的bs数
    bs_all_gpus = bs_per_gpu*num_gpus # 总的bs数

    data_loader = DataLoader(dataset, batch_size=bs_all_gpus, 
        sampler=sampler, num_workers=num_workers, 
        collate_fn=partial(collate, samples_per_gpu=bs_per_gpu),
        shuffle=False)

    return data_loader, dataset

def get_optimizer(model, cfg):
    if cfg.method in ['hash_nerf']:
        # not finished
        embedding_params, grad_vars = model.get_params() 
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                            ], lr=cfg.lr_rate, betas=(0.9, 0.99))
    else:
        # default optimizer setting for [nerf, mip_nerf, kilo_nerf, ...]
        # torch.optim.Adam(params=model.parameters(), lr=cfg.lr_rate, betas=(0.9, 0.999))
        optimizer = build_optimizer(model, cfg.optimizer)
    return optimizer