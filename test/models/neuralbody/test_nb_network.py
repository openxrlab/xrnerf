import os
import shutil
import sys

import pytest

try:
    import torch
    sys.path.extend(['.', '..'])
    import numpy as np
    from mmcv import Config, ConfigDict

    from xrnerf.models.builder import build_network
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_base_network():
    white_bkgd = False  # set to render synthetic data on a white bkgd (always use for dvoxels)
    is_perturb = True  # set to 0. for no jitter, 1. for jitter
    use_viewdirs = True  # use full 5D input instead of 3D
    N_rand_per_sampler = 1024 * 1  # how many N_rand in get_item() function
    lindisp = False  # sampling linearly in disparity rather than depth
    N_samples = 64  # number of coarse samples per ray
    num_train_frame = 60

    model_cfg = dict(
        type='NeuralBodyNetwork',
        cfg=dict(
            raw_noise_std=
            0,  # std dev of noise added to regularize sigma_a output, 1e0 recommended
            white_bkgd=
            white_bkgd,  # set to render synthetic data on a white bkgd (always use for dvoxels)
            use_viewdirs=use_viewdirs,
            is_perturb=is_perturb,
            chunk=1024 * 4,  # mainly work for val
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
                    i_embed=
                    0,  # set 0 for default positional encoding, -1 for none
                    multires=
                    10,  # log2 of max freq for positional encoding (3D location)
                    multires_dirs=
                    4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
                )),
            bs_data=
            'rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
        ),
        render=dict(  # render model
            type='NerfRender', ),
    )

    model_cfg = ConfigDict(model_cfg)

    n_rays = 128
    N_samples_per_ray = 64
    # n_pts = n_rays*N_samples_per_ray
    target_s = torch.rand((n_rays, 3)).cuda()
    pts = torch.rand((n_rays, N_samples_per_ray, 3)).cuda()
    viewdirs = torch.rand((n_rays, 3)).cuda()
    z_vals = torch.rand((n_rays, N_samples_per_ray)).cuda()

    smpl_verts = torch.rand((6890, 3)).cuda()
    smpl_T = torch.rand((1, 3)).cuda()
    smpl_R = torch.rand((3, 3)).cuda()
    latent_idx = torch.tensor([0]).cuda()

    data = {
        'target_s': target_s,
        'pts': pts,
        'rays_d': viewdirs,
        'z_vals': z_vals,
        'smpl_verts': smpl_verts,
        'smpl_T': smpl_T,
        'smpl_R': smpl_R,
        'latent_idx': latent_idx
    }
    model = build_network(model_cfg).cuda()

    data = {k: data[k].unsqueeze(0) for k in data.keys()}
    ret = model.train_step(data, None)

    # ret = model(data)
    # ret = model.train_step(data, None)

    assert isinstance(ret['loss'], torch.Tensor)
    # assert isinstance(ret['rgb'], torch.Tensor)
    # assert ret['rgb'].shape[0] == n_rays
    # assert ret['rgb'].shape[1] == 3


if __name__ == '__main__':
    test_base_network()
