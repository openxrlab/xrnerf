import os
import shutil
import sys

import pytest

try:
    import torch
    sys.path.extend(['.', '..'])
    import numpy as np

    from xrnerf.models.builder import build_embedder
except:
    pass
# @pytest.fixture(scope='module', autouse=True)
# def fixture():
#     if os.path.exists(output_dir):
#         shutil.rmtree(output_dir)
#     os.makedirs(output_dir, exist_ok=False)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_base_embedder():

    smpl_verts = torch.rand((6890, 3)).cuda()
    smpl_T = torch.rand((1, 3)).cuda()
    smpl_R = torch.rand((3, 3)).cuda()

    n_rays = 128
    N_samples_per_ray = 64
    n_pts = n_rays * N_samples_per_ray
    pts = torch.rand((n_rays, N_samples_per_ray, 3)).cuda()

    data = {
        'pts': pts,
        'smpl_verts': smpl_verts,
        'smpl_T': smpl_T,
        'smpl_R': smpl_R
    }

    smpl_embedder_cfg = dict(
        type='SmplEmbedder',
        voxel_size=[0.005, 0.005, 0.005],
    )
    embedder = build_embedder(smpl_embedder_cfg).cuda()

    xyzc_features = embedder(data)

    assert isinstance(xyzc_features, torch.Tensor)
    assert xyzc_features.shape[1] == 352
    assert xyzc_features.shape[2] == n_pts


if __name__ == '__main__':
    test_base_embedder()
