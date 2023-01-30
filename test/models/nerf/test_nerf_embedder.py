import os
import shutil

import pytest

try:
    import torch

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
def test_nerf_embedder():

    n_rays = 128
    N_samples_per_ray = 64
    n_pts = n_rays * N_samples_per_ray
    pts = torch.rand((n_rays, N_samples_per_ray, 3))
    viewdirs = torch.rand((n_rays, 3))
    embedder_cfg = dict(
        type='BaseEmbedder',
        i_embed=0,  # set 0 for default positional encoding, -1 for none
        multires=10,
        multires_dirs=4,
    )
    embedder = build_embedder(embedder_cfg)
    embed_ch, embed_ch_dirs = embedder.get_embed_ch()
    data = {'pts': pts, 'viewdirs': viewdirs}

    data = embedder(data)

    assert isinstance(data['embedded'], torch.Tensor)
    assert data['embedded'].shape[0] == n_pts
    assert data['embedded'].shape[1] == (embed_ch + embed_ch_dirs)
