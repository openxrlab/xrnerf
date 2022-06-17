import math
from turtle import forward

import numpy as np
import torch
from torch import nn

from ..builder import EMBEDDERS
from .base import BaseEmbedder


@EMBEDDERS.register_module()
class MipNerfEmbedder(BaseEmbedder):
    def __init__(self,
                 min_deg_point,
                 max_deg_point,
                 min_deg_view,
                 max_deg_view,
                 input_ch=3,
                 use_viewdirs=False,
                 diag=True,
                 append_identity=True):
        """Encode `x` with sinusoids scaled by 2^[min_deg:max_deg-1]."""
        super().__init__()  # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        self.min_deg = min_deg_point
        self.max_deg = max_deg_point
        self.min_deg_view = min_deg_view
        self.max_deg_view = max_deg_view
        self.use_viewdirs = use_viewdirs
        self.diag = diag
        self.append_identity = append_identity
        self.input_ch = input_ch

    @staticmethod
    def expected_sin(x, x_var):
        """Estimates mean and variance of sin(z), z ~ N(x, var)."""
        y = torch.exp(-0.5 * x_var) * torch.sin(x)
        y_var = torch.maximum(
            torch.tensor(0).cuda(),
            0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2)
        return y, y_var

    def integrated_pos_enc(self, x_coord):
        if self.diag:
            x, x_cov_diag = x_coord
            scales = torch.tensor(
                [2**i for i in range(self.min_deg, self.max_deg)]).cuda()
            shape = list(x.shape[:-1]) + [-1]
            y = torch.reshape(x[..., None, :] * scales[:, None], shape)
            y_var = torch.reshape(
                x_cov_diag[..., None, :] * scales[:, None]**2, shape)
        else:
            x, x_cov = x_coord
            num_dims = x.shape[-1]
            basis = torch.cat([
                2**i * torch.eye(num_dims)
                for i in range(self.min_deg, self.max_deg)
            ], 1).cuda()
            y = torch.matmul(x, basis)
            y_var = torch.sum((torch.matmul(x_cov, basis)) * basis, -2)

        return self.expected_sin(
            torch.cat([y, y + 0.5 * torch.tensor(math.pi).cuda()], -1),
            torch.cat([y_var] * 2, -1))[0]

    def pos_enc(self, x):
        """The positional encoding used by the original NeRF paper."""
        scales = torch.tensor([
            2**i for i in range(self.min_deg_view, self.max_deg_view)
        ]).cuda()
        xb = torch.reshape((x[..., None, :] * scales[:, None]),
                           list(x.shape[:-1]) + [-1])
        four_feat = torch.sin(
            torch.cat([xb, xb + 0.5 * torch.tensor(math.pi).cuda()], dim=-1))
        if self.append_identity:
            return torch.cat([x] + [four_feat], dim=-1)
        else:
            return four_feat

    def get_embed_ch(self):
        d = self.input_ch
        ch_ipe = 2 * d * (self.max_deg - self.min_deg)
        ch_pe = 2 * d * (self.max_deg_view - self.min_deg_view)
        if self.append_identity:
            ch_pe += d
        return ch_ipe, ch_pe

    def forward(self, data):
        samples_enc = self.integrated_pos_enc(data['samples'])
        viewdirs_enc = self.pos_enc(data['viewdirs'])
        num_samples = samples_enc.shape[1]
        data['unflatten_shape'] = samples_enc.shape[:-1]
        samples_enc = torch.reshape(samples_enc, (-1, samples_enc.shape[-1]))
        viewdirs_enc = torch.reshape(
            torch.tile(viewdirs_enc[:, None, :], (1, num_samples, 1)),
            (-1, viewdirs_enc.shape[-1]))
        data['embedded'] = torch.cat([samples_enc, viewdirs_enc], -1)
        return data
