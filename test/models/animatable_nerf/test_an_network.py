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
    print('please install env')


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_base_network():
    phase = 'train_pose'
    num_train_pose = 250
    num_novel_pose = 87

    model_cfg = dict(
        type='AniNeRFNetwork',
        cfg=dict(
            chunk=1024 * 4,  # mainly work for val
            phase=phase,
            tpose_human=dict(
                type='TPoseHuman',
                density_mlp=dict(
                    type='AN_DensityMLP',
                    embedder=dict(
                        type='BaseEmbedder',
                        i_embed=
                        0,  # set 0 for default positional encoding, -1 for none
                        multires=
                        6,  # log2 of max freq for positional encoding (3D location)
                        multires_dirs=
                        4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
                    )),
                color_mlp=dict(
                    type='AN_ColorMLP',
                    num_train_pose=num_train_pose,
                    embedder=dict(
                        type='BaseEmbedder',
                        i_embed=
                        0,  # set 0 for default positional encoding, -1 for none
                        multires=
                        6,  # log2 of max freq for positional encoding (3D location)
                        multires_dirs=
                        4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
                    )),
            ),
            deform_field=dict(
                type='DeformField',
                smpl_threshold=0.05,
                phase=phase,
                bw_mlp=dict(
                    type='AN_BlendWeightMLP',
                    num_pose=num_train_pose,
                    embedder=dict(
                        type='BaseEmbedder',
                        i_embed=
                        0,  # set 0 for default positional encoding, -1 for none
                        multires=
                        10,  # log2 of max freq for positional encoding (3D location)
                        multires_dirs=
                        4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
                    )),
                novel_pose_bw_mlp=dict(
                    type='AN_BlendWeightMLP',
                    num_pose=num_novel_pose,
                    embedder=dict(
                        type='BaseEmbedder',
                        i_embed=
                        0,  # set 0 for default positional encoding, -1 for none
                        multires=
                        10,  # log2 of max freq for positional encoding (3D location)
                        multires_dirs=
                        4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
                    )),
            ),
            bs_data=
            'rays_o',  # the data's shape indicates the real batch-size, this's also the num of rays
        ),
        render=dict(  # render model
            type='NerfRender', ),
    )

    model_cfg = ConfigDict(model_cfg)
    model = build_network(model_cfg)

    n_rays = 128
    N_samples_per_ray = 64
    target_s = torch.rand((n_rays, 3))
    pts = torch.rand((n_rays, N_samples_per_ray, 3))
    viewdirs = torch.rand((n_rays, 3))
    z_vals = torch.rand((n_rays, N_samples_per_ray))
    latent_idx = torch.tensor([0])

    A = torch.rand([24, 4, 4])
    big_A = torch.rand([24, 4, 4])
    canonical_smpl_verts = torch.rand([6890, 3])
    smpl_verts = torch.rand((6890, 3))
    smpl_T = torch.rand((1, 3))
    smpl_R = torch.rand((3, 3))
    smpl_bw = torch.rand([6890, 24])

    data = {
        'target_s': target_s,
        'pts': pts,
        'rays_d': viewdirs,
        'z_vals': z_vals,
        'latent_idx': latent_idx,
        'bw_latent_idx': latent_idx,
        'color_latent_idx': latent_idx,
        'A': A,
        'big_A': big_A,
        'canonical_smpl_verts': canonical_smpl_verts,
        'smpl_verts': smpl_verts,
        'smpl_T': smpl_T,
        'smpl_R': smpl_R,
        'smpl_bw': smpl_bw
    }

    # ret = model(data)
    data = {k: data[k].unsqueeze(0) for k in data.keys()}
    ret = model.train_step(data, None)

    assert isinstance(ret['loss'], torch.Tensor)
    # assert isinstance(ret['rgb'], torch.Tensor)
    # assert ret['rgb'].shape[0] == n_rays
    # assert ret['rgb'].shape[1] == 3


if __name__ == '__main__':
    test_base_network()
