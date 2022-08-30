import sys
import pytest
try:
    import torch
    import numpy as np
    from mmcv import ConfigDict
    sys.path.extend(['.', '..'])
    from distutils.command.build import build
    from xrnerf.models.builder import build_embedder
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), 
    reason='No GPU device has been found.')
def test_gnr_embedder():

    n_rays = 128
    N_samples_per_ray = 64
    n_pts = n_rays * N_samples_per_ray
    pts = torch.rand((n_pts, 3))
    att_dirs = torch.rand((n_rays, 3))
    num_views = 4
    load_size = 512
    image_data = torch.rand((num_views, 3, load_size, load_size))

    image_filter_cfg=dict(
        type='HGFilter',
        opt=dict(
            norm='group',
            num_stack=4,
            num_hourglass=2,
            skip_hourglass=True,
            hg_down='ave_pool',
            hourglass_dim=256
        )
    )

    sr_filter_cfg=dict(
        type='SRFilters',
        order=2,
        out_ch=256
    )

    pose_embeder_cfg=dict(
        type='PositionalEncoding',
        d=3,
        num_freqs=10,
        min_freq=0.1/256,
        max_freq=10/256
    )

    att_embeder_cfg=dict(
        type='SphericalHarmonics',
        d=3

    )
    image_filter_cfg = ConfigDict(image_filter_cfg)
    image_filter = build_embedder(image_filter_cfg)
    sr_filter = build_embedder(sr_filter_cfg)

    pose_embeder = build_embedder(pose_embeder_cfg)
    att_embeder = build_embedder(att_embeder_cfg)


    img_feat = image_filter(image_data)
    sr_feat = sr_filter(img_feat, image_data)
    pose_embed = pose_embeder.embed(pts)
    att_embed = att_embeder.embed(att_dirs)

    assert isinstance(img_feat, torch.Tensor)
    assert isinstance(sr_feat, torch.Tensor)
    assert isinstance(pose_embed, torch.Tensor)
    assert isinstance(att_embed, torch.Tensor)

    assert img_feat.shape[0] == num_views
    assert sr_feat.shape[0] == num_views
    assert pose_embed.shape[0] == n_pts
    assert att_embed.shape[0] == n_rays

if __name__ == '__main__':
    test_gnr_embedder()
