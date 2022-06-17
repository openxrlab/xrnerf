# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import RENDERS
from .nerf_render import NerfRender


@RENDERS.register_module()
class MipNerfRender(NerfRender):
    def get_disp_map(self, weights, z_vals):
        z_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])

        depth_map = (weights * z_mids).sum(axis=-1)

        disp_map = torch.max(
            torch.min(
                torch.nan_to_num(depth_map / weights.sum(axis=-1),
                                 torch.tensor(float('inf')).to(z_vals.device)),
                z_vals[:, -1]), z_vals[:, 0])

        return disp_map

    def get_weights(self, density_delta):
        alpha = 1 - torch.exp(-density_delta)
        weights = alpha * torch.exp(-torch.cat([
            torch.zeros_like(density_delta[..., :1]),
            torch.cumsum(density_delta[..., :-1], dim=-1)
        ],
                                               dim=-1))
        return weights
