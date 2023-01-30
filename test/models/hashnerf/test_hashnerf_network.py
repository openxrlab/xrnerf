import os
import shutil
import sys

import pytest

try:
    import torch
    import numpy as np
    from mmcv import Config, ConfigDict

    # sys.path.append('/home/zhengchengyao/Document/Nerf/git/xrnerf')
    from xrnerf.models.builder import build_network
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_hasnerf_network():

    model_cfg = dict(
        type='HashNerfNetwork',
        cfg=dict(
            phase='train',  # 'train' or 'test'
            chunk=2048,  # mainly work for val
            bs_data='rays_o',
        ),
        mlp=dict(  # coarse model
            type='HashNerfMLP',
            bound=1,
            embedder_pos=dict(n_input_dims=3,
                              encoding_config=dict(
                                  otype='HashGrid',
                                  n_levels=16,
                                  n_features_per_level=2,
                                  log2_hashmap_size=19,
                                  base_resolution=16,
                                  interpolation='Linear',
                              )),
            embedder_dir=dict(n_input_dims=3,
                              encoding_config=dict(
                                  otype='SphericalHarmonics',
                                  degree=4,
                              )),
            density_net=dict(n_input_dims=32,
                             n_output_dims=16,
                             network_config=dict(
                                 otype='FullyFusedMLP',
                                 activation='ReLU',
                                 output_activation='None',
                                 n_neurons=64,
                                 num_layers=1,
                             )),
            color_net=dict(
                # n_input_dims=32, # embedder_dir's out + density_net's out
                n_output_dims=3,
                network_config=dict(
                    otype='FullyFusedMLP',
                    activation='ReLU',
                    output_activation='None',
                    n_neurons=64,
                    num_layers=2,
                )),
        ),
        sampler=dict(
            type='NGPGridSampler',
            update_grid_freq=16,
            update_block_size=5000000,
            n_rays_per_batch=2048,
            cone_angle_constant=0.00390625,
            near_distance=0.2,
            target_batch_size=1 << 18,
            rgb_activation=2,
            density_activation=3,
        ),
        render=dict(
            type='HashNerfRender',
            bg_color=[0, 0, 0],
        ),
    )

    n_imgs = 10
    alldata = {
        'aabb_scale': 1,
        'aabb_range': (0, 1),
        'images': np.random.rand(n_imgs, 800, 800, 4),
        'poses': np.random.rand(n_imgs, 4, 3),
        'focal': np.ones((n_imgs, 2), dtype=float) * 1110,
        'metadata': np.random.rand(n_imgs, 11),
    }
    K = np.array([[1111, 0., 400.], [0., 1111, 400.], [0., 0., 1.]])
    datainfo = {
        'H': 800,
        'W': 800,
        'focal': 1111,
        'K': K,
        'hwf': [800, 800, 1111],
        'near': 2.0,
        'far': 6.0
    }
    model = build_network(ConfigDict(model_cfg))
    model.sampler.set_data(alldata, datainfo)
    model.cuda()

    data = {
        'rays_o': torch.rand((2048, 3)).to(torch.float32),
        'rays_d': torch.rand((2048, 3)).to(torch.float32),
        'target_s': torch.rand((2048, 3)).to(torch.float32),
        'alpha': torch.rand((2048, 1)).to(torch.float32),
        'img_ids': torch.zeros((2048, 1)).to(torch.int32),
        'bg_color': torch.rand((2048, 3)).to(torch.float32),
    }
    for k in data:
        data[k] = data[k].cuda().unsqueeze(0)

    ret = model.train_step(data, None)
    assert isinstance(ret['loss'], torch.Tensor)


# test_hasnerf_network()
