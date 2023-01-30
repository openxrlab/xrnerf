import os
import shutil
import sys

import pytest

try:
    import torch
    from mmcv import Config, ConfigDict
    sys.path.extend(['.', '..'])
    from xrnerf.models.builder import build_network
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_gnr_network():

    num_views = 4
    img = torch.rand((1, num_views + 1, 3, 512, 512)).cuda()
    mask = torch.rand((1, num_views, 1, 512, 512)).cuda()
    persps = torch.rand((1, num_views + 1, 11)).cuda()
    calib = torch.rand((1, num_views + 1, 4, 4)).cuda()
    bbox = torch.tensor([[45, 467, 100, 412]]).float().cuda()
    render_gt = torch.tensor([]).cuda()
    smpl_depth = torch.rand((1, num_views, 512, 512)).cuda()
    spatial_freq = torch.tensor([229.]).float().cuda()
    center = torch.rand((1, 3)).cuda()
    smpl_rot = torch.rand((1, 3, 3)).cuda()
    smpl_verts = torch.rand((1, 10475, 3)).float().cuda()
    smpl_faces = torch.rand((1, 20908, 3)).int().cuda()
    smpl_betas = torch.rand((1, 10)).cuda()
    smpl_t_verts = torch.rand((1, 10475, 3)).float().cuda()
    smpl_t_faces = torch.rand((1, 20908, 3)).int().cuda()
    idx = torch.tensor([0]).float().cuda()

    data = {'img': img, 'mask':mask, 'persps':persps, 'calib':calib, 'bbox':bbox, 'render_gt': render_gt, 'smpl_depth':smpl_depth, \
            'spatial_freq':spatial_freq, 'center':center, 'smpl_rot':smpl_rot, 'smpl_verts':smpl_verts, \
            'smpl_faces':smpl_faces, 'smpl_betas':smpl_betas, 'smpl_t_verts':smpl_t_verts, 'smpl_t_faces':smpl_t_faces, 'idx':idx}

    white_bkgd = False  # set to render synthetic data on a white bkgd (always use for dvoxels)
    is_perturb = True  # set to 0. for no jitter, 1. for jitter
    use_viewdirs = False  # use full 5D input instead of 3D
    use_feat_sr = False
    N_rand = 1024
    model_cfg = dict(
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
            num_views=num_views,
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
                         N_rand=N_rand,
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
    model_cfg = ConfigDict(model_cfg)
    model = build_network(model_cfg).cuda()

    ret = model.train_step(data, None)
    assert isinstance(ret['loss'], torch.Tensor)
    assert ret['num_samples'] == N_rand


if __name__ == '__main__':
    test_gnr_network()
