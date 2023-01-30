import os
import shutil

import pytest

try:
    import torch

    from xrnerf.models.builder import build_mlp
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_nerf_mlp():

    n_rays = 128
    N_samples_per_ray = 64
    # n_pts = n_rays*N_samples_per_ray
    pts = torch.rand((n_rays, N_samples_per_ray, 3))
    viewdirs = torch.rand((n_rays, 3))
    data = {'pts': pts, 'viewdirs': viewdirs}
    mlp_cfg = dict(
        type='NerfMLP',
        skips=[4],
        netdepth=8,  # layers in network
        netwidth=256,  # channels per layer
        netchunk=1024 * 1,  # number of pts sent through network in parallel;
        output_ch=4,
        use_viewdirs=True,
        embedder=dict(
            type='BaseEmbedder',
            i_embed=0,
            multires=10,
            multires_dirs=4,
        ),
    )
    mlp = build_mlp(mlp_cfg)

    data = mlp(data)

    assert isinstance(data['raw'], torch.Tensor)
    assert data['raw'].shape[0] == n_rays
    assert data['raw'].shape[1] == N_samples_per_ray
    assert data['raw'].shape[2] == 4
