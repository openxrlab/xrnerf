_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

method = 'kilo_nerf'  # [nerf, kilo_nerf, mip_nerf]
model_type = 'single_network'  #[single_network, multi_network]
phase = 'pretrain'  # [pretrain, distill, finetune]

resolution_table = dict(
    Character=[128, 256, 128],
    Fountain=[224, 256, 224],
    Jade=[256, 224, 256],
    Statues=[192, 224, 256],
)

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 600000
# max_iters = 100000 # Character only needs 100000 iterations, other scenes need  600000 iterations
lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=50000, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=10000,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 50000), ('val', 1)]

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
    dict(type='CalElapsedTimeHook', params=dict()),
    dict(type='BuildOccupancyTreeHook',
         params=dict(),
         variables=dict(cfg='cfg'))
]

# runner
train_runner = dict(type='KiloNerfTrainRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './work_dirs/kilonerfs/BlendedMVS_#DATANAME#_base01/pretrain'
timestamp = datetime.now().strftime('%d-%b-%H-%M')

# shared params by model and data and ...
dataset_type = 'nsvf'
datadir = 'data/nsvf/BlendedMVS/#DATANAME#'
no_batching = True  # only take random rays from 1 image at a time
no_ndc = True  # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

white_bkgd = True  # set to render synthetic data on a white bkgd (Fountain and Jade have black background, set white_bkgd=False)
is_perturb = True  # set to 0. for no jitter, 1. for jitter
use_viewdirs = True  # use full 5D input instead of 3D
N_rand_per_sampler = 1024  # how many N_rand in get_item() function
lindisp = False  # sampling linearly in disparity rather than depth
N_samples = 384  # number of coarse samples per ray

# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

build_occupancy_tree_config = dict(
    subsample_resolution=[3, 3, 3],
    threshold=10,
    voxel_batch_size=16384,
    work_dir=
    './work_dirs/kilonerfs/BlendedMVS_#DATANAME#_base01/pretrain_occupancy')

model = dict(
    type='NerfNetwork',
    cfg=dict(
        phase='train',  # 'train' or 'test'
        N_importance=0,  # number of additional fine samples per ray
        is_perturb=is_perturb,
        chunk=16384,  # chunk_size, mainly work for val
        bs_data=
        'rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
    ),
    mlp=dict(  # coarse model
        type='NerfMLP',
        skips=[4],
        netdepth=8,  # layers in network
        netwidth=256,  # channels per layer
        netchunk=1024 * 64,  # number of pts sent through network in parallel;
        output_ch=4,  # 5 if cfg.N_importance>0 else 4
        use_viewdirs=use_viewdirs,
        embedder=dict(
            type='BaseEmbedder',
            i_embed=0,  # set 0 for default positional encoding, -1 for none
            multires=
            10,  # log2 of max freq for positional encoding (3D location)
            multires_dirs=
            4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
        ),
    ),
    mlp_fine=None,
    render=dict(  # render model
        type='NerfRender',
        white_bkgd=
        white_bkgd,  # set to render synthetic data on a white bkgd (always use for dvoxels)
        raw_noise_std=
        0,  # std dev of noise added to regularize sigma_a output, 1e0 recommended
    ),
)

basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir=datadir,
    half_res=False,  # load nsvf synthetic data at 800x800
    testskip=
    8,  # will load 1/N images from test/val sets, useful for large datasets like deepvoxels
    white_bkgd=white_bkgd,
    is_batching=False,
    render_test=True,
    mode='train',
)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))

train_pipeline = [
    dict(type='Sample'),
    dict(type='DeleteUseless', keys=['images', 'poses', 'i_data', 'idx']),
    dict(
        type='ToTensor',
        enable=True,
        keys=['pose', 'target_s'],
    ),
    dict(
        type='GetRays',
        enable=True,
    ),  # 与batching型dataset不同的是, 需要从pose生成rays
    dict(type='SelectRays',
         enable=True,
         sel_n=N_rand_per_sampler,
         precrop_iters=10000,
         precrop_frac=0.5),  # 抽取N个射线
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(
        type='ToNDC',
        enable=(not no_ndc),
    ),
    dict(type='GetBounds', enable=True),
    dict(type='GetZvals', enable=True, lindisp=lindisp,
         N_samples=N_samples),  # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=is_perturb),
    dict(type='GetPts', enable=True),
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
        enable=True,
    ),
    dict(type='FlattenRays',
         enable=True),  # 原来是(H, W, ..) 变成(H*W, ...) 记录下原来的尺寸
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(
        type='ToNDC',
        enable=(not no_ndc),
    ),
    dict(type='GetBounds', enable=True),
    dict(type='GetZvals', enable=True, lindisp=lindisp,
         N_samples=N_samples),  # 同上train_pipeline
    dict(type='PerturbZvals', enable=False),  # 测试集不扰动
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True,
         keys=['pose']),  # 删除pose 其实求完ray就不再需要了
]

data = dict(
    train_loader=dict(batch_size=1, num_workers=4),
    train=dict(
        type='SceneBaseDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='SceneBaseDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
)
