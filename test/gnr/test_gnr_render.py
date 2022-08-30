import os
import shutil
import sys
from matplotlib.transforms import Bbox
import numpy as np
import pytest
import torch
import tornado
sys.path.extend(['.', '..'])
from xrnerf.models.builder import build_render, build_mlp
from mmcv import ConfigDict


@pytest.mark.skipif(not torch.cuda.is_available(), 
    reason='No GPU device has been found.')
def test_gnr_render():

    num_views = 4
    feats = torch.rand((num_views, 256, 128, 128))
    images = torch.rand((num_views+1, 3, 512, 512))
    masks = torch.rand((num_views, 1, 512, 512))
    calibs = torch.rand(num_views+1, 4, 4)
    bbox = [45, 467, 100, 412]
    mesh_param = {'center':torch.rand((3)), 'spatia_freq':229.}
    smpl = {'rot':torch.rand((1, 3, 3)), 'verts':torch.rand((10475, 3)), 'faces':torch.rand((20908, 3)), 't_verts':torch.rand((10475, 3)), 't_faces':torch.rand((20908, 3)), 'depth':torch.torch.rand((num_views, 1, 512, 512))}
    persps = torch.rand((num_views+1, 11))
    data = {'feats': feats, 'images': images, 'masks':masks, 'calibs':calibs, 'bbox':bbox, 'mesh_param': mesh_param, 'smpl':smpl, 'persps':persps}

    input_ch_feat = 256
    skips=[2, 4, 6]
    mlp_cfg = dict(
        type='GNRMLP',
        opt=dict(
        input_ch_feat=input_ch_feat,
        smpl_type='smplx',
        use_smpl_sdf=True,
        use_t_pose=True,
        use_nml=True,
        use_attention=True,
        weighted_pool=True,
        use_sh=True,
        use_viewdirs=True,
        use_occlusion=True,
        use_smpl_depth=True,
        use_occlusion_net=True,
        angle_diff=False,
        use_bn=False,
        skips=skips,
        num_views=num_views
        )
    )
    mlp_cfg = ConfigDict(mlp_cfg)
    mlp = build_mlp(mlp_cfg)

    N_rand = 1024
    render_cfg=dict(  # render model
        type='GnrRenderer',
        opt=dict(
            model=mlp,
            N_samples=256,
            ddp=False,
            train_encoder=False,
            projection_mode='perspective',
            loadSize=512,
            num_views=4,
            N_rand=N_rand,
            N_grid=512,
            use_nml=True,
            use_attention=True,
            debug=False,
            use_vgg=False,
            use_smpl_sdf=True,
            use_t_pose=True,
            use_smpl_depth=True,
            regularization=False,
            angle_diff=False,
            use_occlusion=True,
            use_occlusion_net=True,
            use_vh_free=False,
            use_white_bkgd=False,
            chunk=524288,
            N_rand_infer=4096,
            use_vh=True,
            laplacian=5,
            vh_overhead=1
            ),
    )
    render_cfg = ConfigDict(render_cfg)
    render = build_render(render_cfg)
    data = render.render(**data)

    assert isinstance(data['loss'], torch.Tensor)
    assert data['num_samples'] == N_rand

if __name__ == '__main__':
    test_gnr_render()


    