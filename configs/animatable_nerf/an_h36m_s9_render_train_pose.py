_base_ = ['an_h36m_s9_train_pose.py']
from configs.animatable_nerf.an_h36m_s9_train_pose import *

test_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='testset')),
    dict(type='NBSaveSpiralHook', params=dict()),
]

ratio = 1.
basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir='data/h36m/S9/Posing',
    smpl_vertices_dir='new_vertices',
    smpl_params_dir='new_params',
    ratio=ratio,  # reduce the image resolution by ratio
    unit=1000.,
    training_view=[0, 1, 2],
    test_view=[3],
    num_train_pose=num_train_pose,
    training_frame=[0, num_train_pose * frame_interval
                    ],  # [begin_frame, end_frame]
    frame_interval=frame_interval,
    val_frame_interval=val_frame_interval,
    white_bkgd=white_bkgd,
    mode='train',
    phase=phase,
    img_path_to_smpl_idx=img_path_to_smpl_idx,
    img_path_to_frame_idx=img_path_to_frame_idx,
)

frame_idx_to_smpl_idx = lambda x: x
frame_idx_to_latent_idx = lambda x: x
valdata_cfg = basedata_cfg.copy()
valdata_cfg.update(
    dict(mode='render',
         num_render_views=50,
         frame_idx=0,
         frame_idx_to_smpl_idx=frame_idx_to_smpl_idx,
         frame_idx_to_latent_idx=frame_idx_to_latent_idx,
         render_H=int(1000 * ratio),
         render_W=int(1000 * ratio),
         ratio=ratio))

test_pipeline = [
    dict(
        type='LoadCamAndSmplParam',
        enable=True,
    ),  # 读取相机和Smpl参数
    dict(
        type='CalculateSkelTransf',
        enable=True,
    ),  # 计算骨架变换矩阵
    dict(
        type='AninerfIdxConversion',
        enable=True,
    ),  # 变换latent index
    dict(
        type='NBGetRays',
        enable=True,
    ),
    dict(type='NBSelectRays', enable=True, sel_all=True,
         sel_rgb=False),  # 抽取N个射线
    dict(
        type='ToTensor',
        enable=True,
        keys=['rays_o', 'rays_d', 'near', 'far', 'mask_at_box'],
    ),
    dict(type='GetZvals', enable=True, lindisp=lindisp,
         N_samples=N_samples),  # N_samples: number of coarse samples per ray
    dict(type='PerturbZvals', enable=False),
    dict(type='GetPts', enable=True),
    dict(type='DeleteUseless',
         enable=True,
         keys=[
             'iter_n', 'cams', 'cam_inds', 'ims', 'cfg', 'data_root', 'idx',
             'img_path', 'num_cams', 'parents', 'joints', 'spiral_poses', 'K'
         ]),
]

data.update(
    dict(test=dict(
        type='AniNeRFDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ), ))
