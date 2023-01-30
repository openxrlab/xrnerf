import os
import shutil
import sys

import pytest

try:
    import torch
    from mmcv import ConfigDict
    sys.path.extend(['.', '..'])
    from xrnerf.models.builder import build_mlp
except:
    pass


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason='No GPU device has been found.')
def test_gnr_mlp():

    chunk = 1024
    input_ch_pos_enc = 3
    input_ch_smpl = 7
    input_ch_feat = 256
    dim_x = input_ch_pos_enc + input_ch_smpl + input_ch_feat + 3
    num_views = 4
    x = torch.rand(chunk, num_views, dim_x).cuda()
    viewdirs = torch.rand((chunk, num_views + 1, 3)).cuda()
    smpl_vis = torch.rand((chunk, num_views)).cuda()

    data = {
        'x': x,
        'attdirs': viewdirs,
        'alpha_only': False,
        'smpl_vis': smpl_vis
    }

    skips = [2, 4, 6]
    mlp_cfg = dict(type='GNRMLP',
                   opt=dict(input_ch_feat=input_ch_feat,
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
                            num_views=num_views))
    mlp_cfg = ConfigDict(mlp_cfg)
    mlp = build_mlp(mlp_cfg).cuda()

    data = mlp(**data)

    assert isinstance(data, torch.Tensor)
    assert data.shape[0] == chunk
    assert data.shape[1] == 13


if __name__ == '__main__':
    test_gnr_mlp()
