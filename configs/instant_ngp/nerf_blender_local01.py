_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

# [nerf, kilo_nerf, mip_nerf]
method = 'nerf'

# optimizer
optimizer = dict(type='Adam',
                 lr=1e-2,
                 betas=(0.9, 0.99),
                 eps=1e-15,
                 weight_decay=1e-6)
optimizer_config = dict(grad_clip=None)

max_iters = 50000
lr_config = dict(policy='step', step=10000, gamma=0.2, by_epoch=False)
checkpoint_config = dict(interval=10000, by_epoch=False)
custom_hooks = [dict(type='EMAHook', momentum=0.05)]
log_level = 'INFO'
log_config = dict(interval=500,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 500), ('val', 1)]

# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='valset')),
    dict(type='ValidateHook',
         params=dict(save_folder='visualizations/validation')),
    dict(type='OccupationHook', params=dict()),
    dict(type='PassIterHook', params=dict()),
    dict(type='PassDatasetHook',
         params=dict(),
         variables=dict(dataset='trainset')),
    dict(type='ModifyBatchsizeHook', params=dict()),
    dict(type='PassSamplerIterHook', params=dict()),
]

test_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='testset')),
    dict(type='PassDatasetHook',
         params=dict(),
         variables=dict(dataset='testset')),
    dict(type='HashSaveSpiralHook',
         params=dict(save_folder='visualizations/spirals', ),
         variables=dict(cfg='cfg')),
]

# runner
train_runner = dict(type='NerfTrainRunner')
test_runner = dict(type='NerfTestRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)
work_dir = './work_dirs/instant_ngp/nerf_#DATANAME#_base01/'
timestamp = datetime.now().strftime('%d-%b-%H-%M')

dataset_type = 'blender'
no_batching = True  # only take random rays from 1 image at a time
no_ndc = True  # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

white_bkgd = False  # set to render synthetic data on a white bkgd (always use for dvoxels)
load_alpha = True
use_viewdirs = True  # use full 5D input instead of 3D
N_rand_per_sampler = 4096  # how many N_rand in get_item() function
# lindisp = False  # sampling linearly in disparity rather than depth

# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='HashNerfNetwork',
    cfg=dict(
        phase='train',  # 'train' or 'test'
        chunk=4096,  # mainly work for val
        bs_data='rays_o',
    ),
    mlp=dict(  # coarse model
        type='HashNerfMLP',
        bound=1,
        embedder_pos=dict(n_input_dims=3,
                          encoding_config=dict(
                              otype='HashGrid',
                              n_levels=16,
                              n_features_per_level=2,
                              log2_hashmap_size=19,
                              base_resolution=16,
                              interpolation='Linear',
                          )),
        embedder_dir=dict(n_input_dims=3,
                          encoding_config=dict(
                              otype='SphericalHarmonics',
                              degree=4,
                          )),
        density_net=dict(n_input_dims=32,
                         n_output_dims=16,
                         network_config=dict(
                             otype='FullyFusedMLP',
                             activation='ReLU',
                             output_activation='None',
                             n_neurons=64,
                             num_layers=1,
                         )),
        color_net=dict(
            # n_input_dims=32, # embedder_dir's out + density_net's out
            n_output_dims=3,
            network_config=dict(
                otype='FullyFusedMLP',
                activation='ReLU',
                output_activation='None',
                n_neurons=64,
                num_layers=2,
            )),
    ),
    sampler=dict(
        type='NGPGridSampler',
        update_grid_freq=16,
        update_block_size=5000000,
        n_rays_per_batch=N_rand_per_sampler,
        cone_angle_constant=0.00390625,
        near_distance=0.2,
        target_batch_size=1 << 18,
        rgb_activation=2,
        density_activation=3,
    ),
    render=dict(
        type='HashNerfRender',
        bg_color=[0, 0, 0],
    ),
)

basedata_cfg = dict(
    dataset_type=dataset_type,
    N_rand_per_sampler=N_rand_per_sampler,
    datadir='data/nerf_synthetic/#DATANAME#',
    half_res=False,  # load blender synthetic data at 400x400 or 800x800
    testskip=1,
    white_bkgd=white_bkgd,
    load_alpha=load_alpha,
    is_batching=True,  # True for hashnerf
    mode='train',
    val_n=10,
)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
testdata_cfg = basedata_cfg.copy()

traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val', ))
testdata_cfg.update(dict(mode='test', testskip=100))

train_pipeline = [
    dict(type='HashBatchSample', N_rand=N_rand_per_sampler),
    dict(type='RandomBGColor'),
    dict(type='DeleteUseless', keys=['rays_rgb', 'iter_n', 'idx']),
]

test_pipeline = [
    dict(
        type='HashGetRays',
        enable=True,
    ),
    dict(type='FlattenRays', enable=True),
    dict(
        type='HashSetImgids',
        enable=True,
    ),
    # dict(
    #     type='RandomBGColor',
    #     enable=True,
    # ),
    dict(type='DeleteUseless', enable=True, keys=['pose', 'idx']),
]

data = dict(
    # num_workers>0 lead to low psnr ?
    train_loader=dict(batch_size=1, num_workers=0),
    train=dict(
        type='HashNerfDataset',
        cfg=traindata_cfg,
        pipeline=train_pipeline,
    ),
    val_loader=dict(batch_size=1, num_workers=0),
    val=dict(
        type='HashNerfDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ),
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='HashNerfDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline,  # same pipeline as validation
    ),
)
