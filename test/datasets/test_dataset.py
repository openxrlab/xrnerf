import os
import shutil
import sys

import numpy as np
import torch
# sys.path.append('/home/zhengchengyao/Document/Nerf/git/xrnerf')
from mmcv import Config, ConfigDict

from xrnerf.datasets import build_dataset


def test_scene_dataset():

    K = np.array([[555.5555156, 0., 200.], [0., 555.5555156, 200.],
                  [0., 0., 1.]])
    pipeline = [
        dict(type='Sample'),
        dict(type='DeleteUseless', keys=['images', 'poses', 'i_data', 'idx']),
        dict(type='ToTensor', enable=True, keys=['pose', 'target_s']),
        dict(type='GetRays', enable=True, H=400, W=400, K=K),
        dict(type='SelectRays',
             enable=True,
             sel_n=256,
             precrop_iters=500,
             precrop_frac=0.5,
             H=400,
             W=400,
             K=K),
        dict(type='GetViewdirs', enable=True),
        dict(type='GetBounds', enable=True, near=2, far=6),
        dict(type='GetZvals', lindisp=False, N_samples=64),
        dict(type='PerturbZvals', enable=True),
        dict(type='GetPts', enable=True),
        dict(type='DeleteUseless', enable=True, keys=['pose', 'iter_n']),
    ]

    data_cfg = dict(dataset_type='blender',
                    datadir='test/datasets/data/nerf_synthetic/lego',
                    half_res=True,
                    testskip=1,
                    white_bkgd=False,
                    is_batching=False,
                    mode='train')
    train = dict(
        type='SceneBaseDataset',
        cfg=data_cfg,
        pipeline=pipeline,
    )
    dataset_cfg = ConfigDict(train)

    dataset = build_dataset(dataset_cfg)
    dataset.get_info()
    dataset.__getitem__(0)
    len(dataset)


def test_mip_dataset():

    ray_keys = [
        'rays_o', 'rays_d', 'viewdirs', 'radii', 'lossmult', 'near', 'far'
    ]
    pipeline = [
        dict(type='MipMultiScaleSample',
             keys=['target_s'] + ray_keys,
             N_rand=1024),
        dict(type='GetZvals',
             enable=True,
             lindisp=False,
             N_samples=128 + 1,
             randomized=True),
        dict(type='ToTensor', keys=['target_s'] + ray_keys),
    ]

    data_cfg = dict(dataset_type='multiscale',
                    datadir='test/datasets/data/multiscale/lego',
                    white_bkgd=False,
                    mode='train')
    train = dict(type='MipMultiScaleDataset', cfg=data_cfg, pipeline=pipeline)
    dataset_cfg = ConfigDict(train)

    dataset = build_dataset(dataset_cfg)
    dataset.__getitem__(0)
    len(dataset)


def test_hash_dataset():

    pipeline = [
        dict(type='HashBatchSample', N_rand=1024),
        dict(type='RandomBGColor'),
        dict(type='DeleteUseless', keys=['rays_rgb', 'iter_n', 'idx']),
    ]

    data_cfg = dict(
        dataset_type='blender',
        N_rand_per_sampler=1024,
        datadir='test/datasets/data/nerf_synthetic/lego',
        half_res=False,
        testskip=1,
        white_bkgd=False,
        load_alpha=True,
        is_batching=True,
        mode='train',
        val_n=1,
    )

    train = dict(
        type='HashNerfDataset',
        cfg=data_cfg,
        pipeline=pipeline,
    )
    dataset_cfg = ConfigDict(train)

    dataset = build_dataset(dataset_cfg)
    dataset.get_info()
    dataset.get_alldata()
    dataset.__getitem__(0)
    len(dataset)
