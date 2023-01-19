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

    n_pts = 4096
    tpose = torch.rand((n_pts, 3))
    tpose_dirs = torch.rand((n_pts, 3))
    pbw = torch.rand((1, 24, n_pts))
    tbw = torch.rand((1, 24, n_pts))
    latent_idx = torch.tensor([0])

    data = {
        'color_latent_idx': latent_idx,
        'tpose': tpose,
        'tpose_dirs': tpose_dirs,
        'pbw': pbw,
        'tbw': tbw
    }

    phase = 'train'
    num_train_pose = 250
    num_novel_pose = 87
    tpose_human_cfg = dict(
        type='TPoseHuman',
        density_mlp=dict(
            type='AN_DensityMLP',
            embedder=dict(
                type='BaseEmbedder',
                i_embed=0,  # set 0 for default positional encoding, -1 for none
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
                i_embed=0,  # set 0 for default positional encoding, -1 for none
                multires=
                6,  # log2 of max freq for positional encoding (3D location)
                multires_dirs=
                4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
            )),
    )

    nerf_mlp = build_mlp(tpose_human_cfg)

    raw = nerf_mlp(data, data)

    assert isinstance(raw, torch.Tensor)
    assert raw.shape[0] == n_pts


if __name__ == '__main__':
    test_base_mlp()
