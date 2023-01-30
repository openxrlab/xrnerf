import os
import shutil
import sys

import pytest

try:
    import torch
    import numpy as np
    from mmcv import Config, ConfigDict

    from xrnerf.datasets import build_dataset
    from xrnerf.models.builder import build_network
except:
    pass

# sys.path.append('/home/zhengchengyao/Document/Nerf/git/xrnerf')


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def get_train_dataset():

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

    return dataset


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_nerf_network():

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
            netchunk=1024 * 32,
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
    model.cuda()

    dataset = get_train_dataset()
    data = dataset.__getitem__(0)
    for k in data:
        data[k] = data[k].cuda().unsqueeze(0)
    ret = model.train_step(data, None)

    # dataset = get_val_dataset()
    # dataset.hwf = [20, 20, 1111]
    # data = dataset.__getitem__(0)
    # data['spiral_poses'] = data['spiral_poses'][:1]
    # data['images'] = data['images'][:,:20,:20,:]
    # for k in data:
    #     data[k] = torch.tensor(data[k]).cuda().unsqueeze(0)
    #     print(k, data[k].shape)
    # # exit(0)
    # model.val_pipeline = dataset.pipeline
    # with torch.no_grad():
    #     ret = model.val_step(data, None)


# test_nerf_network2()
