# Copyright (c) OpenMMLab. All rights reserved.
import time

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch import nn
from tqdm import tqdm

from .. import builder
from ..builder import NETWORKS
from .nerf import NerfNetwork
from .utils import *


@NETWORKS.register_module()
class HashNerfNetwork(NerfNetwork):
    def __init__(self, cfg, sampler=None, mlp=None, render=None):
        super().__init__(cfg)
        self.sampler = builder.build_sampler(sampler)
        self.mlp = builder.build_mlp(mlp)
        self.render = builder.build_render(render)

    def forward(self, data, is_test=False):

        data = self.sampler.sample(data, self.mlp, is_test)
        data = self.mlp(data)
        data, ret = self.render(data, self.sampler, is_test)

        return ret

    def train_step(self, data, optimizer, **kwargs):
        for k in data:
            data[k] = unfold_batching(data[k])
        ret = self.forward(data, is_test=False)

        bs = ret['rgb'].shape[0]
        alpha = data['alpha'].detach()
        huber_loss = HuberLoss(ret['rgb'], data['target_s'], 0.1, 'sum')
        mse_loss = img2mse(ret['rgb'] * alpha, data['target_s'] * alpha)

        psnr = mse2psnr(mse_loss)
        # loss = mse_loss * bs * 40
        loss = huber_loss * 5

        log_vars = {'loss': loss.item(), 'psnr': psnr.item()}
        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': bs,
        }
        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        if self.phase == 'test':
            return self.test_step(data, **kwargs)

        rank, world_size = get_dist_info()
        if rank == 0:
            for k in data:
                data[k] = unfold_batching(data[k])
                print(k)
            poses = data['poses']
            images = data['images']

            rgbs, disps, gt_imgs = [], [], []
            elapsed_time_list = []
            for i in tqdm(range(poses.shape[0])):
                start = time.time()
                data = self.val_pipeline({'pose': poses[i], 'idx': i})
                ret = self.batchify_forward(data, is_test=True)
                end = time.time()
                elapsed_time = end - start
                rgb = recover_shape(ret['rgb'], data['src_shape'])
                alpha = images[i].cpu().numpy()[:, :, 3:]
                rgb = rgb.cpu().numpy()
                gt_img = images[i].cpu().numpy()[:, :, :3]

                rgb = rgb * alpha
                gt_img = gt_img * alpha

                rgbs.append(rgb)
                gt_imgs.append(gt_img)
                elapsed_time_list.append(elapsed_time)
            outputs = {
                'rgbs': rgbs,
                'disps': disps,
                'gt_imgs': gt_imgs,
                'elapsed_time': elapsed_time_list
            }
        else:
            outputs = {}
        return outputs

    def test_step(self, data, **kwargs):
        """process spiral poses."""
        rank, world_size = get_dist_info()
        if rank == 0:
            for k in data:
                data[k] = unfold_batching(data[k])
            # for k in data:
            #     print(k, data[k].shape, data[k].device, data[k].dtype)
            idx = data['idx'].item()

            ret = self.batchify_forward(data, is_test=True)
            rgb = recover_shape(ret['rgb'], data['src_shape']).cpu().numpy()
            alpha = recover_shape(ret['alpha'],
                                  data['src_shape']).cpu().numpy()
            outputs = {'spiral_rgb': rgb, 'spiral_alpha': alpha, 'idx': idx}
        else:
            outputs = {}
        return outputs
