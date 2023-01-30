import os
import shutil
import sys

import pytest

try:
    import torch
    sys.path.extend(['.', '..'])
    import numpy as np

    from xrnerf.models.builder import build_render
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_base_render():

    n_rays = 128
    N_samples_per_ray = 64
    raw = torch.rand((n_rays, N_samples_per_ray, 4))
    z_vals = torch.rand((n_rays, N_samples_per_ray))
    rays_d = torch.rand((n_rays, 3))
    data = {'raw': raw, 'z_vals': z_vals, 'rays_d': rays_d}
    render_cfg = dict(  # render model
        type='NerfRender',
        white_bkgd=False,
        raw_noise_std=0,
    )
    render = build_render(render_cfg)

    data, ret = render(data)

    assert isinstance(ret['rgb'], torch.Tensor)
    assert ret['rgb'].shape[0] == n_rays
    assert ret['rgb'].shape[1] == 3


if __name__ == '__main__':
    test_base_render()
