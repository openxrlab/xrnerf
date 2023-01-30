# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .. import builder
from ..builder import MLPS
from .base import BaseMLP


class BungeeNerfBaseBlock(nn.Module):
    def __init__(self, netwidth=256, input_ch=3, input_ch_views=3):
        super(BungeeNerfBaseBlock, self).__init__()
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, netwidth)] +
            [nn.Linear(netwidth, netwidth) for _ in range(3)])
        self.views_linear = nn.Linear(input_ch_views + netwidth, netwidth // 2)
        self.feature_linear = nn.Linear(netwidth, netwidth)
        self.alpha_linear = nn.Linear(netwidth, 1)
        self.rgb_linear = nn.Linear(netwidth // 2, 3)

    def forward(self, input_pts, input_views):
        h = input_pts.float()
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = torch.cat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = F.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h


class BungeeNerfResBlock(nn.Module):
    def __init__(self, netwidth=256, input_ch=3, input_ch_views=3):
        super(BungeeNerfResBlock, self).__init__()
        self.pts_linears = nn.ModuleList([
            nn.Linear(input_ch + netwidth, netwidth),
            nn.Linear(netwidth, netwidth)
        ])
        self.views_linear = nn.Linear(input_ch_views + netwidth, netwidth // 2)
        self.feature_linear = nn.Linear(netwidth, netwidth)
        self.alpha_linear = nn.Linear(netwidth, 1)
        self.rgb_linear = nn.Linear(netwidth // 2, 3)

    def forward(self, input_pts, input_views, h):
        h = torch.cat([input_pts, h], -1)
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
        alpha = self.alpha_linear(h)
        feature0 = self.feature_linear(h)
        h0 = torch.cat([feature0, input_views], -1)
        h0 = self.views_linear(h0)
        h0 = F.relu(h0)
        rgb = self.rgb_linear(h0)
        return rgb, alpha, h


@MLPS.register_module()
class BungeeNerfMLP(BaseMLP):
    def __init__(self,
                 cur_stage=0,
                 netwidth=256,
                 netchunk=1024 * 32,
                 embedder=None,
                 **kwarg):
        super().__init__()  # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        self.chunk = netchunk
        self.embedder = builder.build_embedder(embedder)
        self.num_resblocks = cur_stage
        self.init_mlp(netwidth)

    def init_mlp(self, netwidth):
        W = netwidth
        self.input_ch, self.input_ch_dirs = self.embedder.get_embed_ch()
        self.baseblock = BungeeNerfBaseBlock(netwidth=W,
                                             input_ch=self.input_ch,
                                             input_ch_views=self.input_ch_dirs)
        self.resblocks = nn.ModuleList([
            BungeeNerfResBlock(netwidth=W,
                               input_ch=self.input_ch,
                               input_ch_views=self.input_ch_dirs)
            for _ in range(self.num_resblocks)
        ])
        return

    def forward(self, data):

        data = self.embedder(data)
        data['embedded'] = data['embedded'].float()
        outputs_flat = self.batchify_run_mlp(data['embedded'])
        data['raw'] = torch.reshape(
            outputs_flat,
            list(data['unflatten_shape']) + list(outputs_flat.shape[1:]))
        del data['unflatten_shape']
        return data

    def batchify_run_mlp(self, x):
        if self.chunk is None:
            return self.run_mlp(x)
        else:
            outputs = torch.cat([
                self.run_mlp(x[i:i + self.chunk])
                for i in range(0, x.shape[0], self.chunk)
            ], 0)
            return outputs

    def run_mlp(self, x):
        input_pts, input_views = torch.split(
            x, [self.input_ch, self.input_ch_dirs], dim=-1)
        alphas = []
        rgbs = []
        base_rgb, base_alpha, h = self.baseblock(input_pts, input_views)
        alphas.append(base_alpha)
        rgbs.append(base_rgb)
        for i in range(self.num_resblocks):
            res_rgb, res_alpha, h = self.resblocks[i](input_pts, input_views,
                                                      h)
            alphas.append(res_alpha)
            rgbs.append(res_rgb)

        outputs = torch.cat([torch.stack(rgbs, 1), torch.stack(alphas, 1)], -1)

        return outputs
