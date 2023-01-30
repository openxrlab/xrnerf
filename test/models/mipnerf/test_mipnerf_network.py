import os
import shutil
import sys

import pytest

try:
    import torch
    from mmcv import Config, ConfigDict

    from xrnerf.datasets.pipelines import Compose
    from xrnerf.models.builder import build_network
except:
    print('please install env')


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_nerf_network():

    ########################## get data ##########################
    ray_keys = [
        'rays_o', 'rays_d', 'viewdirs', 'radii', 'lossmult', 'near', 'far'
    ]
    mip_pipeline = [
        dict(type='MipMultiScaleSample',
             keys=['target_s'] + ray_keys,
             N_rand=256),
        dict(type='GetZvals',
             enable=True,
             lindisp=False,
             N_samples=128 + 1,
             randomized=True),
        dict(type='ToTensor', keys=['target_s'] + ray_keys),
    ]

    n_imgs = 5
    n_rays = 1700000
    data = {
        'target_s': torch.rand(n_rays, 3),
        'rays_o': torch.rand(n_rays, 3),
        'rays_d': torch.rand(n_rays, 3),
        'viewdirs': torch.rand(n_rays, 3),
        'radii': torch.rand(n_rays, 1),
        'lossmult': torch.rand(n_rays, 1),
        'near': torch.ones(n_rays, 1),
        'far': torch.ones(n_rays, 1),
    }
    pipeline = Compose(mip_pipeline)
    data = pipeline(data)
    for k in data:
        data[k] = data[k].cuda().unsqueeze(0)

    ########################## get data ##########################

    model_cfg = dict(
        type='MipNerfNetwork',
        cfg=dict(
            num_levels=2,  # The number of sampling levels.
            ray_shape='cone',  # The shape of cast rays ('cone' or 'cylinder').
            resample_padding=0.01,  # Dirichlet/alpha "padding" on the histogram.
            use_multiscale=True,  # If True, use multiscale.
            coarse_loss_mult=0.1,  # How much to downweight the coarse loss(es).
            chunk=800,  # mainly work for val
            bs_data='rays_o'),
        mlp=dict(  # coarse model
            type='NerfMLP',
            skips=[4],
            netdepth=8,  # layers in network
            netwidth=256,  # channels per layer
            netchunk=1024 *
            32,  # number of pts sent through network in parallel;
            use_viewdirs=True,
            embedder=dict(
                type='MipNerfEmbedder',
                min_deg_point=0,
                max_deg_point=16,
                min_deg_view=
                0,  # Min degree of positional encoding for viewdirs.
                max_deg_view=
                4,  # Max degree of positional encoding for viewdirs.
                use_viewdirs=True,
                append_identity=True),
        ),
        render=dict(  # render model
            type='MipNerfRender',
            white_bkgd=False,
            raw_noise_std=0,  # Standard deviation of noise added to raw density.
            density_bias=-1.,  # The shift added to raw densities pre-activation.
            rgb_padding=0.001,  # Padding added to the RGB outputs.
            density_activation='softplus',  # density activation
        ),
    )

    model_cfg = ConfigDict(model_cfg)
    model = build_network(model_cfg)
    model.cuda()

    ret = model.train_step(data, None)

    assert isinstance(ret['loss'], torch.Tensor)


# test_nerf_network()
