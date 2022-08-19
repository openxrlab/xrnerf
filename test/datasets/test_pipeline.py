import os
import shutil

import numpy as np
# import pytest
import torch

from xrnerf.datasets.pipelines import Compose


def test_nerf_no_batching():
    K = np.array([[555.5555156, 0., 200.], [0., 555.5555156, 200.],
                  [0., 0., 1.]])

    no_batching_pipeline = [
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
    n_imgs = 20
    data = {
        'poses': np.random.rand(n_imgs, 4, 4),
        'images': np.random.rand(n_imgs, 400, 400, 3),
        'i_data': np.array(range(0, n_imgs)),
        'idx': 0,
        'iter_n': 0
    }
    pipeline = Compose(no_batching_pipeline)
    data = pipeline(data)

    assert isinstance(data['pts'], torch.Tensor)
    assert data['pts'].shape[0] == 256
    assert data['pts'].shape[1] == 64
    assert data['pts'].shape[2] == 3
    assert isinstance(data['z_vals'], torch.Tensor)
    assert data['z_vals'].shape[0] == 256
    assert data['z_vals'].shape[1] == 64


def test_nerf_batching():
    K = np.array([[555.5555156, 0., 200.], [0., 555.5555156, 200.],
                  [0., 0., 1.]])

    batching_pipeline = [
        dict(type='BatchSample', N_rand=256),
        dict(type='DeleteUseless', keys=['rays_rgb', 'idx']),
        dict(type='ToTensor', keys=['rays_o', 'rays_d', 'target_s']),
        dict(type='GetViewdirs', enable=True),
        dict(type='GetBounds', enable=True, near=2, far=6),
        dict(type='GetZvals', lindisp=False, N_samples=64),
        dict(type='PerturbZvals', enable=True),
        dict(type='GetPts', enable=True),
        dict(type='DeleteUseless', enable=True, keys=['iter_n']),
    ]
    data = {
        'rays_rgb': np.random.rand(3238704, 3, 3),
        'idx': 0,
    }
    pipeline = Compose(batching_pipeline)
    data = pipeline(data)

    assert isinstance(data['pts'], torch.Tensor)
    assert data['pts'].shape[0] == 256
    assert data['pts'].shape[1] == 64
    assert data['pts'].shape[2] == 3
    assert isinstance(data['z_vals'], torch.Tensor)
    assert data['z_vals'].shape[0] == 256
    assert data['z_vals'].shape[1] == 64
