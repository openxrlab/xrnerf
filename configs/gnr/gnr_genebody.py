_base_ = [
    # '../_base_/models/nerf.py',
    # '../_base_/schedules/adam_20w_iter.py',
    # '../_base_/default_runtime.py'
]

import os
from datetime import datetime

method = 'gnr'

# optimizer
optimizer = dict(type='Adam', lr=5e-4, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)

lr_rate = 5e-4
max_iters = 2000000
evalute_config = dict()
lr_config = dict(policy='step', step=500 * 1000, gamma=0.1, by_epoch=False)
checkpoint_config = dict(interval=10000, by_epoch=False)
log_level = 'INFO'
log_config = dict(interval=1,
                  by_epoch=False,
                  hooks=[dict(type='TextLoggerHook')])
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
distributed = (num_gpus > 1)  # 是否多卡，mmcv对dp多卡支持不好，故而要么单卡要么ddp多卡
work_dir = './work_dirs/gnr/'  # noqa
timestamp = datetime.now().strftime('%d-%b-%H-%M')

# shared params by model and data and ...
dataset_type = 'blender'
no_batching = True  # only take random rays from 1 image at a time
no_ndc = True  # 源代码中'if args.dataset_type != 'llff' or args.no_ndc:' 就设置no_ndc

white_bkgd = False  # set to render synthetic data on a white bkgd (always use for dvoxels)
is_perturb = True  # set to 0. for no jitter, 1. for jitter
use_viewdirs = False  # use full 5D input instead of 3D
N_rand_per_sampler = 1024 * 1  # how many N_rand in get_item() function
lindisp = False  # sampling linearly in disparity rather than depth
N_samples = 256  # number of coarse samples per ray
use_feat_sr = False
# resume_from = os.path.join(work_dir, 'latest.pth')
# load_from = os.path.join(work_dir, 'latest.pth')

model = dict(
    type='GnrNetwork',
    cfg=dict(
        raw_noise_std=
        0,  # std dev of noise added to regularize sigma_a output, 1e0 recommended
        white_bkgd=
        white_bkgd,  # set to render synthetic data on a white bkgd (always use for dvoxels)
        use_viewdirs=use_viewdirs,
        projection_mode='perspective',
        is_perturb=is_perturb,
        use_feat_sr=False,
        use_smpl_sdf=True,
        use_t_pose=True,
        use_smpl_depth=True,
        use_attention=True,
        ddp=False,
        chunk=524288,  # mainly work for val
        num_views=4,
        image_filter=dict(type='HGFilter',
                          opt=dict(norm='group',
                                   num_stack=4,
                                   num_hourglass=2,
                                   skip_hourglass=True,
                                   hg_down='ave_pool',
                                   hourglass_dim=256)),
        sr_filter=dict(type='SRFilters', order=2, out_ch=256),
        nerf=dict(type='GNRMLP',
                  opt=dict(
                      input_ch_feat=64 if use_feat_sr else 256,
                      smpl_type='smplx',
                      use_smpl_sdf=True,
                      use_t_pose=True,
                      use_nml=True,
                      use_attention=True,
                      weighted_pool=True,
                      use_sh=True,
                      use_viewdirs=True,
                      use_occlusion=True,
                      use_smpl_depth=True,
                      use_occlusion_net=True,
                      angle_diff=False,
                      use_bn=False,
                      skips=[2, 4, 6],
                      num_views=4,
                  )),
        bs_data=
        'rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
        nerf_renderer=dict(  # render model
            type='GnrRenderer',
            opt=dict(model=None,
                     N_samples=256,
                     ddp=False,
                     train_encoder=False,
                     projection_mode='perspective',
                     loadSize=512,
                     num_views=4,
                     N_rand=1024,
                     N_grid=512,
                     use_nml=True,
                     use_attention=True,
                     debug=False,
                     use_vgg=False,
                     use_smpl_sdf=True,
                     use_t_pose=True,
                     use_smpl_depth=True,
                     regularization=False,
                     angle_diff=False,
                     use_occlusion=True,
                     use_occlusion_net=True,
                     use_vh_free=False,
                     use_white_bkgd=False,
                     chunk=524288,
                     N_rand_infer=4096,
                     use_vh=True,
                     laplacian=5,
                     vh_overhead=1),
        ),
        train_encoder=False))

basedata_cfg = dict(dataset_type=dataset_type,
                    dataroot='path/to/GeneBodyDataset',
                    eval_skip=1,
                    train_skip=1,
                    loadSize=512,
                    num_views=4,
                    use_smpl_sdf=True,
                    use_t_pose=True,
                    smpl_type='smplx',
                    t_pose_path='path/to/smpl_t_pose',
                    use_smpl_depth=True,
                    use_white_bkgd=False,
                    random_multiview=False)

traindata_cfg = basedata_cfg.copy()
valdata_cfg = basedata_cfg.copy()
traindata_cfg.update(dict())
valdata_cfg.update(dict(mode='val'))

data = dict(
    train_loader=dict(batch_size=1, num_workers=6),
    train=dict(type='GeneBodyDataset',
               opt=traindata_cfg,
               phase='train',
               pipeline=[]),
    val_loader=dict(batch_size=1, num_workers=6),
    val=dict(type='GeneBodyDataset', opt=valdata_cfg, phase='val',
             pipeline=[]),
    test_loader=dict(batch_size=1, num_workers=6),
    test=dict(type='GeneBodyDataset',
              opt=valdata_cfg,
              phase='test',
              pipeline=[]),
)
