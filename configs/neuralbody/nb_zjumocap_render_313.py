_base_ = ['nb_zjumocap_313.py']
from configs.neuralbody.nb_zjumocap_313 import *

test_hooks = [
    dict(type='SetValPipelineHook',
         params=dict(),
         variables=dict(valset='testset')),
    dict(type='NBSaveSpiralHook', params=dict()),
]

ratio = 0.5
basedata_cfg = dict(
    dataset_type=dataset_type,
    datadir='data/zju_mocap/CoreView_313',
    smpl_vertices_dir='new_vertices',
    smpl_params_dir='new_params',
    ratio=ratio,  # reduce the image resolution by ratio
    unit=1000.,
    training_view=[0, 6, 12, 18],
    test_view=[1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 19, 20],
    num_train_frame=num_train_frame,
    training_frame=[0, num_train_frame * frame_interval
                    ],  # [begin_frame, end_frame]
    frame_interval=frame_interval,
    val_frame_interval=val_frame_interval,
    white_bkgd=white_bkgd,
    mode='train',
    img_path_to_smpl_idx=img_path_to_smpl_idx,
    img_path_to_frame_idx=img_path_to_frame_idx,
)

frame_idx_to_smpl_idx = lambda x: x + 1
frame_idx_to_latent_idx = lambda x: x
valdata_cfg = basedata_cfg.copy()
valdata_cfg.update(
    dict(mode='render',
         num_render_views=50,
         frame_idx=0,
         frame_idx_to_smpl_idx=frame_idx_to_smpl_idx,
         frame_idx_to_latent_idx=frame_idx_to_latent_idx,
         render_H=int(1024 * ratio),
         render_W=int(1024 * ratio),
         ratio=ratio))

test_pipeline = [
    dict(
        type='LoadCamAndSmplParam',
        enable=True,
    ),  # 读取相机和Smpl参数
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
             'iter_n', 'cams', 'cam_inds', 'cfg', 'data_root', 'idx',
             'spiral_poses', 'K'
         ]),
]

data.update(
    dict(test=dict(
        type='NeuralBodyDataset',
        cfg=valdata_cfg,
        pipeline=test_pipeline,
    ), ))
