# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import builder
from ..builder import MLPS
from .base import BaseMLP

try:
    import tinycudann as tcnn
except Exception as e:
    print('please install tcnn for instant ngp')


def get_per_level_scale(bound):
    per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))
    # b = np.exp(np.log(2048*scale/N_min)/(L-1))
    return per_level_scale


@MLPS.register_module()
class HashNerfMLP(BaseMLP):
    def __init__(self,
                 bound=1,
                 embedder_pos=None,
                 embedder_dir=None,
                 density_net=None,
                 color_net=None,
                 **kwarg):
        super().__init__()

        embedder_pos['encoding_config'][
            'per_level_scale'] = get_per_level_scale(1)
        self.embedder_pos = tcnn.Encoding(**embedder_pos)
        self.embedder_dir = tcnn.Encoding(**embedder_dir)

        density_net['n_input_dims'] = self.embedder_pos.n_output_dims
        self.density_net = tcnn.Network(**density_net)
        # color_net['n_input_dims'] = self.embedder_dir.n_output_dims + \
        #                             density_net['n_output_dims']
        color_net['n_input_dims'] = self.embedder_dir.n_output_dims + \
            density_net['n_output_dims'] - 1
        self.color_net = tcnn.Network(**color_net)

    def forward(self, data):
        unflatten_shape = data['pts'].shape[:-1]
        outputs_flat = self.run_mlp(data)
        data['raw'] = torch.reshape(
            outputs_flat,
            list(unflatten_shape) + [outputs_flat.shape[-1]])
        return data

    def run_mlp(self, data):

        # embedder
        pts_flat = torch.reshape(data['pts'],
                                 [-1, data['pts'].shape[-1]]).detach()
        pts_embedded = self.embedder_pos(pts_flat)

        viewdirs = data['viewdirs']
        if len(data['pts'].shape) > len(viewdirs.shape):
            viewdirs = viewdirs[:, None].expand(data['pts'].shape)
        else:
            viewdirs = viewdirs  # 如果chunk为None， inputs也是2维，不需要expand
        viewdirs_flat = torch.reshape(viewdirs,
                                      [-1, viewdirs.shape[-1]]).detach()
        viewdirs_embedded = self.embedder_dir(viewdirs_flat)

        # mlp
        density_out = self.density_net(pts_embedded)
        color_output = self.color_net(
            torch.cat([density_out[..., 1:], viewdirs_embedded], dim=-1))

        outputs = torch.cat([color_output, density_out[..., :1]], -1)
        outputs = outputs.to(torch.float32).contiguous()

        return outputs

    def run_mlp2(self, data):

        # embedder
        pts_flat = torch.reshape(data['pts'],
                                 [-1, data['pts'].shape[-1]]).detach()
        pts_embedded = self.embedder_pos(pts_flat)

        viewdirs = data['viewdirs']
        if len(data['pts'].shape) > len(viewdirs.shape):
            viewdirs = viewdirs[:, None].expand(data['pts'].shape)
        else:
            viewdirs = viewdirs  # 如果chunk为None， inputs也是2维，不需要expand
        viewdirs_flat = torch.reshape(viewdirs,
                                      [-1, viewdirs.shape[-1]]).detach()
        viewdirs_embedded = self.embedder_dir(viewdirs_flat)

        # mlp
        density_out = self.density_net(pts_embedded)
        color_output = self.color_net(
            torch.cat([density_out, viewdirs_embedded], dim=-1))

        outputs = torch.cat([color_output, density_out[..., :1]], -1)
        outputs = outputs.to(torch.float32).contiguous()

        return outputs

    def run_density(self, pts_flat):
        pts_embedded = self.embedder_pos(pts_flat)
        density_out = self.density_net(pts_embedded)
        density = density_out[:, :1].to(torch.float32)
        return density
