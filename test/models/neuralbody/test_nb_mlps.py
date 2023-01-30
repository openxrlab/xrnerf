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
    # n_pts = n_rays*N_samples_per_ray
    pts = torch.rand((n_rays, N_samples_per_ray, 3))
    viewdirs = torch.rand((n_rays, 3))
    latent_idx = torch.tensor([0])
    data = {'pts': pts, 'rays_d': viewdirs, 'latent_idx': latent_idx}

    nerf_mlp_cfg = dict(
        type='NB_NeRFMLP',
        num_frame=60,
        embedder=dict(
            type='BaseEmbedder',
            i_embed=0,  # set 0 for default positional encoding, -1 for none
            multires=
            10,  # log2 of max freq for positional encoding (3D location)
            multires_dirs=
            4,  # this is 'multires_views' in origin codes, log2 of max freq for positional encoding (2D direction)
        ))

    nerf_mlp = build_mlp(nerf_mlp_cfg)

    xyzc_features = torch.rand((1, 352, n_rays * N_samples_per_ray))
    datas = nerf_mlp(xyzc_features, data)

    assert isinstance(data['raw'], torch.Tensor)
    assert data['raw'].shape[0] == n_rays
    assert data['raw'].shape[1] == N_samples_per_ray
    assert data['raw'].shape[2] == 4


if __name__ == '__main__':
    test_base_mlp()
