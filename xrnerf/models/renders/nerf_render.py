# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import RENDERS
from .base import BaseRender


@RENDERS.register_module()
class NerfRender(BaseRender):
    def __init__(self, white_bkgd=False, raw_noise_std=0, **kwarg):
        super().__init__() # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        self.white_bkgd = white_bkgd
        self.raw_noise_std = raw_noise_std

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
        rays_d = data['rays_d']
        raw_noise_std = 0 if is_test else self.raw_noise_std
                
        device = raw.device
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        noise = 0.
        if raw_noise_std > 0.:
            noise = torch.randn(raw[...,3].shape) * raw_noise_std
            noise = noise.to(device)

        alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        ret = {'rgb':rgb_map, 'disp':disp_map, 'acc':acc_map}
        data['weights'] = weights # 放在data里面，给sample函数用
        # data['depth'] = depth_map 
        
        return data, ret

