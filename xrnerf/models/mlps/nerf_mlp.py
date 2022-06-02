# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .. import builder
from ..builder import MLPS
from .base import BaseMLP


@MLPS.register_module()
class NerfMLP(BaseMLP):
    def __init__(self, skips=[4], netdepth=8, netwidth=256, output_ch=4, use_viewdirs=True, 
                netchunk=1024*32, embedder=None, **kwarg):
        super().__init__() # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        self.skips = skips
        self.chunk = netchunk
        self.use_viewdirs = use_viewdirs
        self.embedder = builder.build_embedder(embedder)
        self.init_mlp(output_ch, netdepth, netwidth)

    def init_mlp(self, output_ch, netdepth, netwidth):
        D, W = netdepth, netwidth
        self.input_ch, self.input_ch_dirs = self.embedder.get_embed_ch()

        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.input_ch, W) for i in range(D-1)])
        
        if self.use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(self.input_ch_dirs + W, W//2)])
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3) # need to fit the output shape of self.views_linears
        else:
            self.output_linear = nn.Linear(W, output_ch)
        return 

    def forward(self, data):
        data = self.embedder(data)
        outputs_flat = self.batchify_run_mlp(data['embedded'])
        data['raw'] = torch.reshape(outputs_flat, list(data['pts'].shape[:-1]) + [outputs_flat.shape[-1]])
        return data

    def batchify_run_mlp(self, x):
        if self.chunk is None:
            return self.run_mlp(x)
        else:
            outputs = torch.cat([self.run_mlp(x[i:i+self.chunk]) for i in range(0, x.shape[0], self.chunk)], 0)
            return outputs

    def run_mlp(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_dirs], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

