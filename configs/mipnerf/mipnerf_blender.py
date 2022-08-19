import os
from datetime import datetime

method = 'mip_nerf'  # [nerf, kilo_nerf, mip_nerf]
use_multiscale = False

# optimizer
optimizer = dict(type='Adam', lr=5e-4)
optimizer_config = dict(grad_clip=None)
max_iters = 1000000
lr_config = dict(
    policy='Mip',
    lr_init=5e-4,
    lr_final=5e-6,
    max_steps=max_iters,
    lr_delay_steps=2500,
    lr_delay_mult=0.01,
    by_epoch=False,
)
checkpoint_config = dict(interval=100000, by_epoch=False)
optimizer_config = dict(grad_clip=None)
log_level = 'INFO'
log_config = dict(interval=10000,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
workflow = [('train', 100000), ('val', 1)]

# hooks
# 'params' are numeric type value, 'variables' are variables in local environment
train_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='valset')),
    dict(type='ValidateHook', params=dict(save_folder='val_results/')),
    dict(type='SaveSpiralHook', params=dict(save_folder='spiral_results/')),
    dict(type='PassIterHook', params=dict()),  # 将当前iter数告诉dataset
    dict(type='OccupationHook',
         params=dict()),  # no need for open-source vision
]

test_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='testset')),
    dict(type='TestHook',
         params=dict(ndown=1,
                     dump_json=True,
                     save_img=True,
                     save_folder='test_results/'),
         variables=dict()),
]

# runner
train_runner = dict(type='NerfTrainRunner')
test_runner = dict(type='NerfTestRunner')

# runtime settings
num_gpus = 1
distributed = (num_gpus > 1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = f'/mnt/lustre/ganshikang/Projects/xrnerf/single_results/#DATANAME#'

timestamp = datetime.now().strftime('%d-%b-%H-%M')

# shared params by model and data and ...
dataset_type = 'blender'
no_batching = True  # only take random rays from 1 image at a time
no_ndc = True  # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

# set to render synthetic data on a white bkgd (always use for dvoxels)
white_bkgd = True
use_viewdirs = True  # use full 5D input instead of 3D
N_rand_per_sampler = 1024  # how many N_rand in get_item() function
lindisp = False  # sampling linearly in disparity rather than depth
num_samples = 128  # number of samples per ray

# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='MipNerfNetwork',
    cfg=dict(
        num_levels=2,  # The number of sampling levels.
        # If True, sample linearly in disparity, not in depth.
        ray_shape='cone',  # The shape of cast rays ('cone' or 'cylinder').
        resample_padding=0.01,  # Dirichlet/alpha "padding" on the histogram.
        use_multiscale=use_multiscale,  # If True, use multiscale.
        coarse_loss_mult=0.1,  # How much to downweight the coarse loss(es).
        chunk=800,  # mainly work for val
        bs_data='rays_o'),
    mlp=dict(  # coarse model
        type='NerfMLP',
        skips=[4],
        netdepth=8,  # layers in network
        netwidth=256,  # channels per layer
        netchunk=1024 * 32,  # number of pts sent through network in parallel;
        use_viewdirs=use_viewdirs,
        embedder=dict(
            type='MipNerfEmbedder',
            # Min degree of positional encoding for 3D points.
            min_deg_point=0,
            # Max degree of positional encoding for 3D points.
            max_deg_point=16,
            min_deg_view=0,  # Min degree of positional encoding for viewdirs.
            max_deg_view=4,  # Max degree of positional encoding for viewdirs.
            use_viewdirs=use_viewdirs,
            append_identity=True),
    ),
    render=dict(  # render model
        type='MipNerfRender',
        # set to render synthetic data on a white bkgd (always use for dvoxels)
        white_bkgd=white_bkgd,
        raw_noise_std=0,  # Standard deviation of noise added to raw density.
        density_bias=-1.,  # The shift added to raw densities pre-activation.
        rgb_padding=0.001,  # Padding added to the RGB outputs.
        density_activation='softplus',  # density activation
    ),
)

basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir=f'data/multiscale/#DATANAME#',
    half_res=False,  # load blender synthetic data at 400x400 instead of 800x800
    testskip=16,
    white_bkgd=white_bkgd,
    is_batching=False,
    mode='train',
)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
testdata_cfg = basedata_cfg.copy()
traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))
testdata_cfg.update(dict(mode='test', testskip=0))

train_pipeline = [
    dict(type='Sample'),
    dict(type='DeleteUseless', keys=['images', 'poses', 'i_data', 'idx']),
    dict(
        type='ToTensor',
        enable=True,
        keys=['pose', 'target_s'],
    ),
    dict(type='GetRays', enable=True,
         include_radius=True),  # 与batching型dataset不同的是, 需要从pose生成rays
    dict(type='SelectRays',
         enable=True,
         sel_n=N_rand_per_sampler,
         include_radius=True),  # 抽取N个射线
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(
        type='ToNDC',
        enable=(not no_ndc),
    ),
    dict(type='GetBounds', enable=True, near_new=2., far_new=6.),
    dict(type='GetZvals',
         enable=True,
         lindisp=lindisp,
         N_samples=num_samples + 1,
         randomized=True),
    dict(type='DeleteUseless', enable=True,
         keys=['pose', 'iter_n']),  # 删除pose 其实求完ray就不再需要了
]

test_pipeline = [
    dict(
        type='ToTensor',
        enable=True,
        keys=['pose'],
    ),
    dict(type='GetRays', enable=True, include_radius=True),
    dict(type='FlattenRays', enable=True,
         include_radius=True),  # 原来是(H, W, ..) 变成(H*W, ...) 记录下原来的尺寸
    dict(
        type='GetViewdirs',
        enable=use_viewdirs,
    ),
    dict(
        type='ToNDC',
        enable=(not no_ndc),
    ),
    dict(type='GetBounds', enable=True, near_new=2., far_new=6.),
    dict(type='GetZvals',
         enable=True,
         lindisp=lindisp,
         N_samples=num_samples + 1,
         randomized=False),
    dict(type='DeleteUseless', enable=True,
         keys=['pose']),  # 删除pose 其实求完ray就不再需要了
]

data = dict(
    train_loader=dict(batch_size=1, num_workers=0),
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
    test_loader=dict(batch_size=1, num_workers=0),
    test=dict(
        type='SceneBaseDataset',
        cfg=testdata_cfg,
        pipeline=test_pipeline,  # same pipeline as validation
    ),
)
