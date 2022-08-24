_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

method = 'kilo_nerf'  # [nerf, kilo_nerf, mip_nerf]
model_type = 'multi_network'  #[single_network, multi_network]
phase = 'finetune'  # [pretrain, distill, finetune]

resolution_table = dict(Chair=[208, 208, 256],
                        Drums=[256, 208, 192],
                        Ficus=[128, 176, 256],
                        Hotdog=[256, 256, 96],
                        Lego=[144, 256, 160],
                        Materials=[256, 224, 80],
                        Mic=[256, 256, 240],
                        Ship=[256, 256, 144])

# optimizer
optimizer = dict(type='Adam', lr=0.001, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 1000000
lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=50000, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=10000,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', max_iters), ('val', 1)]

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
    dict(type='CalElapsedTimeHook', params=dict()),
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
train_runner = dict(type='KiloNerfTrainRunner')
test_runner = dict(type='KiloNerfTestRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './work_dirs/kilonerfs/Synthetic_NeRF_#DATANAME#_base01/finetune'
timestamp = datetime.now().strftime('%d-%b-%H-%M')

# shared params by model and data and ...
dataset_type = 'nsvf'
datadir = 'data/nsvf/Synthetic_NeRF/#DATANAME#'
no_batching = True  # only take random rays from 1 image at a time
no_ndc = True  # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

white_bkgd = True  # set to render synthetic data on a white bkgd (always use for dvoxels)
is_perturb = True  # set to 0. for no jitter, 1. for jitter
use_viewdirs = True  # use full 5D input instead of 3D
N_rand_per_sampler = 8192  # how many N_rand in get_item() function
lindisp = False  # sampling linearly in disparity rather than depth
N_samples = 384  # number of coarse samples per ray

# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

occupancy_checkpoint = './work_dirs/kilonerfs/Synthetic_NeRF_#DATANAME#_base01/pretrain_occupancy/occupancy.pth'
distilled_config = './configs/kilonerfs/kilonerf_distill_Synthetic_NeRF_base01.py'
distilled_checkpoint = './work_dirs/kilonerfs/Synthetic_NeRF_#DATANAME#_base01/distill/checkpoint.pth'

model = dict(
    type='KiloNerfNetwork',
    cfg=dict(
        phase='train',  # 'train' or 'test'
        N_importance=0,  # number of additional fine samples per ray
        is_perturb=is_perturb,
        chunk=40000,  # chunk_size, mainly work for val
        l2_regularization_lambda=1.0e-06,
        bs_data=
        'rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
    ),
    mlp=dict(  # multi_network model
        type='KiloNerfMLP',
        distilled_config=distilled_config,
        distilled_checkpoint=distilled_checkpoint,
        occupancy_checkpoint=occupancy_checkpoint,
        embedder=dict(
            type='KiloNerfFourierEmbedder',
            num_networks=1,  # num_networks, teacher nerf network only have 1
            input_ch=3,
            multires=
            10,  # num_frequencies, log2 of max freq for positional encoding (3D location)
            multires_dirs=
            4,  # num_frequencies_direction, this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
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
testdata_cfg = basedata_cfg.copy()

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))
testdata_cfg.update(dict(mode='test', testskip=1))

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
         precrop_iters=0,
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
        type='KilonerfGetRays',
        enable=True,
        expand_origin=True,
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
        type='KiloNerfDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='KiloNerfDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='KiloNerfDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline,  # same pipeline as validation
    ),
)
