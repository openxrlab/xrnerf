_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

method = 'neuralbody'

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

lr_rate = 5e-4
max_iters = 2000000
evalute_config = dict()
lr_config = dict(policy='step', step=500*1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=10000, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=10000,  by_epoch=False, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 10000), ('val', 1)]

# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='valset')),
    dict(type='ValidateHook',
         params=dict(save_folder='visualizations/validation')),
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
train_runner = dict(type='NerfTrainRunner')
test_runner = dict(type='NerfTestRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus>1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './work_dirs/neuralbody/zjumocap_386/'  # noqa
timestamp = datetime.now().strftime("%d-%b-%H-%M")

# shared params by model and data and ...
dataset_type = "blender"
no_batching = True # only take random rays from 1 image at a time
no_ndc = True # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

white_bkgd = False # set to render synthetic data on a white bkgd (always use for dvoxels)
is_perturb = True # set to 0. for no jitter, 1. for jitter
use_viewdirs = True # use full 5D input instead of 3D
N_rand_per_sampler = 1024*1     # how many N_rand in get_item() function
lindisp = False # sampling linearly in disparity rather than depth
N_samples = 64 # number of coarse samples per ray

# resume_from = os.path.join(work_dir, 'latest.pth')
load_from = os.path.join(work_dir, 'latest.pth')

num_train_frame = 300
model = dict(
    type='NeuralBodyNetwork',
    cfg=dict(
        raw_noise_std=0, # std dev of noise added to regularize sigma_a output, 1e0 recommended
        white_bkgd=white_bkgd, # set to render synthetic data on a white bkgd (always use for dvoxels)
        use_viewdirs=use_viewdirs,
        is_perturb=is_perturb,
        chunk=1024*4, # mainly work for val
        smpl_embedder=dict(
            type='SmplEmbedder',
            voxel_size=[0.005, 0.005, 0.005],
            ),
        num_train_frame=num_train_frame,
        nerf_mlp=dict(
            type='NB_NeRFMLP',
            num_frame=num_train_frame,
            embedder=dict(
                type='BaseEmbedder',
                i_embed=0, # set 0 for default positional encoding, -1 for none
                multires=10, # log2 of max freq for positional encoding (3D location)
                multires_dirs=4, # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
                )
            ),
        bs_data='rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
        ),

    render=dict( # render model
        type='NerfRender',
        ),
    )


img_path_to_smpl_idx = lambda x: int(os.path.basename(x)[:-4])
img_path_to_frame_idx = lambda x: int(os.path.basename(x)[:-4])


frame_interval = 1
val_frame_interval = 30
basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir="data/zju_mocap/CoreView_386",
    smpl_vertices_dir="new_vertices",
    smpl_params_dir="new_params",
    ratio=0.5, # reduce the image resolution by ratio
    unit=1000.,
    training_view=[0, 6, 12, 18],
    test_view=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20, 21, 22],
    num_train_frame = num_train_frame,
    training_frame=[0, num_train_frame * frame_interval], # [begin_frame, end_frame]
    frame_interval=frame_interval,
    val_frame_interval=val_frame_interval,
    white_bkgd=white_bkgd,
    mode='train',

    img_path_to_smpl_idx=img_path_to_smpl_idx,
    img_path_to_frame_idx=img_path_to_frame_idx,
    )


traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
traindata_cfg.update( dict() )
valdata_cfg.update( dict(mode='val') )


train_pipeline = [
    dict(type='LoadImageAndCamera', enable=True,), # 读取图片和相机参数
    dict(type='LoadSmplParam', enable=True,), # 读取SMPL参数
    dict(type='NBGetRays', enable=True,), # 与batching型dataset不同的是, 需要从pose生成rays
    dict(type='NBSelectRays', enable=True, sel_n=N_rand_per_sampler), # 抽取N个射线
    dict(type='ToTensor', enable=True, keys=['rays_o', 'rays_d', 'target_s', 'near', 'far'],),
    dict(type='GetZvals', enable=True, lindisp=lindisp, N_samples=N_samples), # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=is_perturb),
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True, keys=['iter_n', 'cams', 'cam_inds', 'ims', 'cfg', 'data_root', 'idx', 'img_path', 'num_cams']),
]


test_pipeline = [
    dict(type='LoadImageAndCamera', enable=True,), # 读取图片和相机参数
    dict(type='LoadSmplParam', enable=True,), # 读取SMPL参数
    dict(type='NBGetRays', enable=True,),
    dict(type='NBSelectRays', enable=True, sel_all=True), # 抽取N个射线
    dict(type='ToTensor', enable=True, keys=['rays_o', 'rays_d', 'target_s', 'near', 'far', 'mask_at_box'],),
    dict(type='GetZvals', enable=True, lindisp=lindisp, N_samples=N_samples), # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=False),
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True, keys=['iter_n', 'cams', 'cam_inds', 'ims', 'cfg', 'data_root', 'idx', 'img_path', 'num_cams']),
]


data = dict(
    train_loader=dict(batch_size=1, num_workers=0),
    train=dict(
        type='NeuralBodyDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
        ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='NeuralBodyDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
        ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='NeuralBodyDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
        ),
    )
