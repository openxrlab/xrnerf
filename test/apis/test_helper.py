import os
import shutil
import sys

import numpy as np
import torch

sys.path.append('/home/zhengchengyao/Document/Nerf/git/xrnerf')
from mmcv import Config, ConfigDict
from mmcv.runner import EpochBasedRunner

from xrnerf.core.apis.helper import *
from xrnerf.models.builder import build_network
from xrnerf.utils import get_root_logger


def get_nerf_network():

    model_cfg = dict(
        type='NerfNetwork',
        cfg=dict(
            phase='train',  # 'train' or 'test'
            N_importance=128,  # number of additional fine samples per ray
            is_perturb=False,
            chunk=256,  # mainly work for val
            bs_data='rays_o',
        ),
        mlp=dict(  # coarse model
            type='NerfMLP',
            skips=[4],
            netdepth=8,  # layers in network
            netwidth=256,  # channels per layer
            netchunk=1024 *
            32,  # number of pts sent through network in parallel;
            output_ch=5,  # 5 if cfg.N_importance>0 else 4
            use_viewdirs=True,
            embedder=dict(
                type='BaseEmbedder',
                i_embed=0,
                multires=10,
                multires_dirs=4,
            ),
        ),
        mlp_fine=dict(  # fine model
            type='NerfMLP',
            skips=[4],
            netdepth=8,  # layers in fine network
            netwidth=256,  # channels per layer in fine network
            netchunk=256,
            output_ch=5,
            use_viewdirs=True,
            embedder=dict(
                type='BaseEmbedder',
                i_embed=0,
                multires=10,
                multires_dirs=4,
            ),
        ),
        render=dict(  # render model
            type='NerfRender',
            white_bkgd=True,
            raw_noise_std=0,
        ),
    )
    model_cfg = ConfigDict(model_cfg)
    model = build_network(model_cfg)
    return model


def test_helper():

    cfg = {
        'method': 'nerf',
        'work_dir': 'workspace/#DATANAME#',
        'data_cfg': {
            'datadir': 'workspace/#DATANAME#'
        }
    }
    update_config('lego', ConfigDict(cfg))

    Runner = get_runner(dict(type='NerfTestRunner'))
    runner = Runner(get_nerf_network(),
                    logger=get_root_logger(log_level='INFO'))

    hooks = [
        dict(type='ValidateHook', params=dict(save_folder='val_results/')),
        dict(type='SaveSpiralHook',
             params=dict(save_folder='spiral_results/')),
        dict(type='PassDatasetHook',
             params=dict(),
             variables=dict(dataset='trainset')),
        dict(type='PassIterHook', params=dict()),
        dict(type='OccupationHook', params=dict()),
        dict(type='ModifyBatchsizeHook', params=dict()),
        dict(type='PassSamplerIterHook', params=dict()),
    ]

    variables = {'runner': runner, 'trainset': None}
    register_hooks(hooks, **variables)
