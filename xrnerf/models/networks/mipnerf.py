# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.runner import get_dist_info, load_checkpoint
from torch import nn

from .. import builder
from ..builder import NETWORKS
from .nerf import NerfNetwork
from .utils import (merge_ret, mse2psnr, resample_along_rays,
                    sample_along_rays, unfold_batching)


@NETWORKS.register_module()
class MipNerfNetwork(NerfNetwork):
    def __init__(self, cfg, mlp=None, render=None):

        super().__init__(cfg, mlp=mlp, render=render)
        self.num_levels = cfg.num_levels
        self.resample_padding = cfg.resample_padding
        self.ray_shape = cfg.ray_shape
        self.use_multiscale = cfg.use_multiscale
        self.coarse_loss_mult = cfg.coarse_loss_mult

    def forward(self, data, is_test):
        randomized = not is_test
        ret = {}
        for i_level in range(self.num_levels):
            if i_level == 0:
                data = sample_along_rays(data, self.ray_shape)
            else:
                data = resample_along_rays(data, randomized, self.ray_shape,
                                           self.resample_padding)

            data, temp_ret = self.render(self.mlp(data), is_test)
            if not ret:
                ret = temp_ret
            else:
                ret = merge_ret(ret, temp_ret)
        return ret

    def train_step(self, data, optimizer, **kwargs):

        for k in data:
            data[k] = unfold_batching(data[k])

        ret = self.forward(data, is_test=False)

        if 'lossmult' in data:
            mask = torch.broadcast_to(data['lossmult'], ret['rgb'].shape)
        else:
            mask = torch.ones_like(ret['rgb']).to(ret['rgb'].device)

        loss_fine = (mask *
                     (ret['rgb'] - data['target_s'])**2).sum() / mask.sum()
        loss_coarse = (
            mask *
            (ret['coarse_rgb'] - data['target_s'])**2).sum() / mask.sum()
        loss = loss_fine + self.coarse_loss_mult * loss_coarse
        psnr = mse2psnr(loss_fine)

        log_vars = {
            'loss': loss.item(),
            'loss_fine': loss_fine.item(),
            'loss_coarse': loss_coarse.item(),
            'psnr': psnr.item()
        }

        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': ret['rgb'].shape[0]
        }
        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        if not self.use_multiscale:
            return super().val_step(data, **kwargs)
        else:
            rank, world_size = get_dist_info()
            if rank == 0:
                rgb, image, disp, idx = self.evaluate_once(data, **kwargs)
                if self.phase == 'test':
                    outputs = {
                        'rgb': rgb,
                        'gt_img': image,
                        'disp': disp,
                        'idx': idx
                    }
                else:
                    outputs = {
                        'rgbs': [rgb],
                        'gt_imgs': [image],
                        'disps': [disp]
                    }
            else:
                outputs = {}
            return outputs

    def evaluate_once(self, data, **kwargs):

        H, W = data['image'].shape[1:3]
        idx = data['idx'].item()
        del data['idx']
        for key in data.keys():
            data[key] = data[key].squeeze(0).reshape(H * W,
                                                     -1).to(torch.float32)

        ret = self.batchify_forward(data, is_test=True)
        rgb = ret['rgb'].reshape((H, W, -1)).cpu().numpy()
        disp = ret['disp'].reshape((H, W, -1)).cpu().numpy()

        image = data['image'].reshape((H, W, -1)).cpu().numpy()

        outputs = {'rgb': rgb, 'gt_img': image, 'disp': disp, 'idx': idx}

        return rgb, image, disp, idx
