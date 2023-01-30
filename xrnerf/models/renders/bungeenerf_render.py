# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import RENDERS
from .base import BaseRender


@RENDERS.register_module()
class BungeeNerfRender(BaseRender):
    def __init__(self,
                 stage=0,
                 white_bkgd=False,
                 raw_noise_std=0,
                 rgb_padding=0,
                 density_bias=-1,
                 density_activation='softplus',
                 **kwarg):
        super().__init__()  # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std
        self.rgb_padding = rgb_padding
        self.density_bias = density_bias
        self.stage = stage

        if density_activation == 'softplus':  # Density activation.
            self.density_activation = F.softplus
        elif density_activation == 'relu':
            self.density_activation = F.relu
        else:
            raise NotImplementedError

    def get_disp_map(self, weights, z_vals):
        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                                  depth_map / torch.sum(weights, -1))
        return disp_map

    def get_weights(self, density_delta):
        alpha = 1 - torch.exp(density_delta)
        weights = alpha * torch.cumprod(
            torch.cat([
                torch.ones(
                    (alpha.shape[0], 1)).to(alpha.device), 1. - alpha + 1e-10
            ], -1), -1)[:, :-1]
        return weights

    def forward(self, data, is_test=False):
        """Transforms model's predictions to semantically meaningful values.

        Args:
            data: inputs
            is_test: is_test
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            disp_map: [num_rays]. Disparity map. Inverse of depth map.
            acc_map: [num_rays]. Sum of weights along each ray.
            weights: [num_rays, num_samples]. Weights assigned to each sampled color.
            depth_map: [num_rays]. Estimated distance to object.
            ret: return values
        """
        raw = data['raw']
        z_vals = data['z_vals']
        # z_vals: [N_rays, N_samples] for nerf or [N_rays, N_samples+1] for mip
        viewdirs = data['viewdirs']
        raw_noise_std = 0 if is_test else self.raw_noise_std
        device = raw.device
        z_vals = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        if dists.shape[1] != raw.shape[1]:  # if z_val: [N_rays, N_samples]
            dists = torch.cat([
                dists,
                torch.Tensor([1e10]).to(device).expand(dists[..., :1].shape)
            ], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(viewdirs[..., None, :], dim=-1)

        acc_rgb = torch.sum(raw[..., :self.stage + 1, :3], dim=2)

        rgb = (1 + 2 * self.rgb_padding) / (
            1 + torch.exp(-acc_rgb)) - self.rgb_padding

        acc_alpha = torch.sum(raw[..., :self.stage + 1, 3], dim=2)

        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(acc_alpha.shape) * raw_noise_std
            noise = noise.to(device)

        density_delta = -self.density_activation(acc_alpha + noise +
                                                 self.density_bias) * dists

        weights = self.get_weights(density_delta)

        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        disp_map = self.get_disp_map(weights, z_vals)
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        ret = {'rgb': rgb_map, 'disp': disp_map, 'acc': acc_map}
        data['weights'] = weights  # 放在data里面，给sample函数用

        return data, ret
