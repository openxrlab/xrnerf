_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime


method = 'nerf' # [nerf, kilo_nerf, mip_nerf]

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

max_iters = 200000
evalute_config = dict()
lr_config = dict(policy='step', step=500*1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=10000, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=10000,  by_epoch=False, hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 10000), ('val', 1)]

# runtime settings
num_gpus = 1
distributed = (num_gpus>1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './work_dirs/nerfsv3/nerf_ship_base01/'  # noqa
timestamp = datetime.now().strftime("%d-%b-%H-%M")

# shared params by model and data and ...
dataset_type = "blender"
no_batching = True # only take random rays from 1 image at a time
no_ndc = True # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

white_bkgd = True # set to render synthetic data on a white bkgd (always use for dvoxels)
is_perturb = True # set to 0. for no jitter, 1. for jitter
use_viewdirs = True # use full 5D input instead of 3D
N_rand_per_sampler = 1024*4     # how many N_rand in get_item() function
lindisp = False # sampling linearly in disparity rather than depth
N_samples = 64 # number of coarse samples per ray

# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='NerfNetwork',
    cfg=dict(
        phase='train', # 'train' or 'test'
        N_importance=128, # number of additional fine samples per ray
        is_perturb=is_perturb,
        chunk=1024*32, # mainly work for val
        ),
    mlp=dict( # coarse model
        type='NerfMLP',
        skips=[4],
        netdepth=8, # layers in network
        netwidth=256, # channels per layer
        netchunk=1024*32, # number of pts sent through network in parallel;
        output_ch=5, # 5 if cfg.N_importance>0 else 4
        use_viewdirs=use_viewdirs, 

        embedder=dict(
            type='BaseEmbedder',
            i_embed=0, # set 0 for default positional encoding, -1 for none
            multires=10, # log2 of max freq for positional encoding (3D location)
            multires_dirs=4, # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
            ),
        ),
    mlp_fine=dict( # fine model
        type='NerfMLP',
        skips=[4],
        netdepth=8, # layers in fine network
        netwidth=256, # channels per layer in fine network
        netchunk=1024*32,
        output_ch=5, # 5 if cfg.N_importance>0 else 4
        use_viewdirs=use_viewdirs,  # same as above

        embedder=dict(
            type='BaseEmbedder',
            i_embed=0, # set 0 for default positional encoding, -1 for none
            multires=10, # log2 of max freq for positional encoding (3D location)
            multires_dirs=4, # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
            ), 
        ),
    render=dict( # render model
        type='NerfRender',
        white_bkgd=white_bkgd, # set to render synthetic data on a white bkgd (always use for dvoxels)
        raw_noise_std=0, # std dev of noise added to regularize sigma_a output, 1e0 recommended
        
        ),    
    )

basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir="data/nerf_synthetic/ship",
    half_res=True, # load blender synthetic data at 400x400 instead of 800x800 
    testskip=8, # will load 1/N images from test/val sets, useful for large datasets like deepvoxels
    white_bkgd=white_bkgd,
    mode='train',
    )


traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
testdata_cfg = basedata_cfg.copy()

traindata_cfg.update( dict() )
valdata_cfg.update( dict(mode='val') )
testdata_cfg.update( dict(mode='test', testskip=0) )


train_pipeline = [
    dict(type='GetRays', enable=True,), # 与batching型dataset不同的是, 需要从pose生成rays
    dict(type='SelectRays', enable=True, sel_n=N_rand_per_sampler, precrop_iters=500, precrop_frac=0.5), # 抽取N个射线
    dict(type='GetViewdirs', enable=use_viewdirs,),
    dict(type='ToNDC', enable=(not no_ndc),),
    dict(type='GetBounds', enable=True),
    dict(type='GetZvals', enable=True, lindisp=lindisp, N_samples=N_samples), # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=is_perturb),
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True, keys=['pose', 'iter_n']), # 删除pose 其实求完ray就不再需要了
]


test_pipeline = [
    dict(type='GetRays', enable=True,), 
    dict(type='FlattenRays', enable=True), # 原来是(H, W, ..) 变成(H*W, ...) 记录下原来的尺寸
    dict(type='GetViewdirs', enable=use_viewdirs,),
    dict(type='ToNDC', enable=(not no_ndc),),
    dict(type='GetBounds', enable=True),
    dict(type='GetZvals', enable=True, lindisp=lindisp, N_samples=N_samples), # 同上train_pipeline
    dict(type='PerturbZvals', enable=False), # 测试集不扰动
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless', enable=True, keys=['pose']), # 删除pose 其实求完ray就不再需要了
]


data = dict(
    train_loader=dict(batch_size=1, num_workers=4),
    train=dict(
        type='NerfNoBatchingDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
        ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='NerfNoBatchingDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
        ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='NerfNoBatchingDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline, # same pipeline as validation
        ),      
    )
