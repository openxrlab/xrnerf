import os
import shutil
import sys

import pytest

try:
    import torch
    sys.path.extend(['.', '..'])
    import numpy as np

    from xrnerf.models.builder import build_mlp
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_base_mlp():

    n_rays = 128
    N_samples_per_ray = 64
    pts = torch.rand((n_rays, N_samples_per_ray, 3))
    viewdirs = torch.rand((n_rays, 3))
    latent_idx = torch.tensor([0])

    A = torch.rand([24, 4, 4])
    big_A = torch.rand([24, 4, 4])
    canonical_smpl_verts = torch.rand([6890, 3])
    smpl_verts = torch.rand((6890, 3))
    smpl_T = torch.rand((1, 3))
    smpl_R = torch.rand((3, 3))
    smpl_bw = torch.rand([6890, 24])

    data = {
        'pts': pts,
        'rays_d': viewdirs,
        'latent_idx': latent_idx,
        'bw_latent_idx': latent_idx,
        'A': A,
        'big_A': big_A,
        'canonical_smpl_verts': canonical_smpl_verts,
        'smpl_verts': smpl_verts,
        'smpl_T': smpl_T,
        'smpl_R': smpl_R,
        'smpl_bw': smpl_bw
    }

    phase = 'train'
    num_train_pose = 250
    num_novel_pose = 87
    deform_field_cfg = dict(
        type='DeformField',
        smpl_threshold=0.05,
        phase=phase,
        bw_mlp=dict(
            type='AN_BlendWeightMLP',
            num_pose=num_train_pose,
            embedder=dict(
                type='BaseEmbedder',
                i_embed=0,  # set 0 for default positional encoding, -1 for none
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
                i_embed=0,  # set 0 for default positional encoding, -1 for none
                multires=
                10,  # log2 of max freq for positional encoding (3D location)
                multires_dirs=
                4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
            )),
    )

    nerf_mlp = build_mlp(deform_field_cfg)

    datas = nerf_mlp(data)

    assert isinstance(datas['tpose'], torch.Tensor)
    assert datas['pbw'].shape[1] == 24


if __name__ == '__main__':
    test_base_mlp()
