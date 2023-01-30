import os
import shutil

import pytest

try:
    import torch

    from xrnerf.models.builder import build_render
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_nerf_render():

    n_rays = 128
    N_samples_per_ray = 64
    raw = torch.rand((n_rays, N_samples_per_ray, 4))
    z_vals = torch.rand((n_rays, N_samples_per_ray))
    rays_d = torch.rand((n_rays, 3))
    data = {'raw': raw, 'z_vals': z_vals, 'rays_d': rays_d}
    render_cfg = dict(  # render model
        type='NerfRender',
        white_bkgd=True,
        raw_noise_std=0,
    )
    render = build_render(render_cfg)

    data, ret = render(data)

    assert isinstance(ret['rgb'], torch.Tensor)
    assert ret['rgb'].shape[0] == n_rays
    assert ret['rgb'].shape[1] == 3
