_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

method = 'bungeenerf'  # [nerf, kilo_nerf, mip_nerf, bungeenerf]

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 200000
lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=500, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=5,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 500), ('val', 1)]

# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='valset')),
    dict(type='ValidateHook',
         params=dict(save_folder='visualizations/validation')),
    dict(type='SaveSpiralHook',
         params=dict(save_folder='visualizations/spiral')),
    dict(type='PassIterHook', params=dict()),  # 将当前iter数告诉dataset
    dict(type='OccupationHook',
         params=dict()),  # no need for open-source vision
]

test_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='testset')),
    dict(type='TestHook', params=dict()),
]

# runner
train_runner = dict(type='BungeeNerfTrainRunner')
test_runner = dict(type='BungeeNerfTestRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
stage = 0  # current stage for training
work_dir = './work_dirs/bungeenerf/#DATANAME#/stage_%d/' % stage
timestamp = datetime.now().strftime('%d-%b-%H-%M')

# shared params by model and data and ...
dataset_type = 'mutiscale_google'
no_batching = True  # only take random rays from 1 image at a time

white_bkgd = False  # set to render synthetic data on a white bkgd (always use for dvoxels)
is_perturb = False  # set to 0. for no jitter, 1. for jitter
use_viewdirs = True  # use full 5D input instead of 3D
N_rand_per_sampler = 1024 * 2  # how many N_rand in get_item() function
lindisp = False  # sampling linearly in disparity rather than depth
N_samples = 65  # number of coarse samples per ray

# resume_from = os.path.join(work_dir, 'latest.pth')
load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='BungeeNerfNetwork',
    cfg=dict(
        phase='train',  # 'train' or 'test'
        ray_shape='cone',  # The shape of cast rays ('cone' or 'cylinder').
        resample_padding=0.01,  # Dirichlet/alpha "padding" on the histogram.
        N_importance=65,  # number of additional fine samples per ray
        is_perturb=is_perturb,
        chunk=1024 * 32,  # mainly work for val
        bs_data=
        'rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
    ),
    mlp=dict(  # coarse model
        type='BungeeNerfMLP',
        cur_stage=stage,  # resblock nums
        netwidth=256,  # channels per layer
        netchunk=1024 * 64,  # number of pts sent through network in parallel;
        embedder=dict(
            type='BungeeEmbedder',
            i_embed=0,  # set 0 for default positional encoding, -1 for none
            multires=
            10,  # log2 of max freq for positional encoding (3D location)
            multires_dirs=
            4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
        ),
    ),
    render=dict(  # render model
        type='BungeeNerfRender',
        white_bkgd=
        white_bkgd,  # set to render synthetic data on a white bkgd (always use for dvoxels)
        raw_noise_std=
        0,  # std dev of noise added to regularize sigma_a output, 1e0 recommended
    ),
)

basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir='data/multiscale_google/#DATANAME#',
    white_bkgd=white_bkgd,
    factor=3,
    N_rand_per_sampler=N_rand_per_sampler,
    mode='train',
    cur_stage=stage,
    holdout=16,
    is_batching=True,  # True for blender, False for llff
)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
testdata_cfg = basedata_cfg.copy()

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))
testdata_cfg.update(dict(mode='test', testskip=0))

train_pipeline = [
    dict(
        type='BungeeBatchSample',
        enable=True,
        N_rand=N_rand_per_sampler,
    ),
    dict(type='DeleteUseless', keys=['rays_rgb', 'idx']),
    dict(
        type='ToTensor',
        enable=True,
        keys=['rays_o', 'rays_d', 'target_s', 'scale_code'],
    ),
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(type='BungeeGetBounds', enable=True),
    dict(type='BungeeGetZvals',
         enable=True,
         lindisp=lindisp,
         N_samples=N_samples),  # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=is_perturb),
    dict(type='DeleteUseless', enable=True,
         keys=['pose', 'iter_n']),  # 删除pose 其实求完ray就不再需要了
]

test_pipeline = [
    dict(
        type='ToTensor',
        enable=True,
        keys=['pose'],
    ),
    dict(
        type='GetRays',
        include_radius=True,
        enable=True,
    ),
    dict(type='FlattenRays', include_radius=True,
         enable=True),  # 原来是(H, W, ..) 变成(H*W, ...) 记录下原来的尺寸
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(type='BungeeGetBounds', enable=True),
    dict(type='BungeeGetZvals',
         enable=True,
         lindisp=lindisp,
         N_samples=N_samples),  # 同上train_pipeline
    dict(type='PerturbZvals', enable=False),  # 测试集不扰动
    dict(type='DeleteUseless', enable=True,
         keys=['pose']),  # 删除pose 其实求完ray就不再需要了
]

data = dict(
    train_loader=dict(batch_size=1, num_workers=4),
    train=dict(
        type='BungeeDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='BungeeDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='BungeeDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline,  # same pipeline as validation
    ),
)
