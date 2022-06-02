# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2022-04-20 17:05:14
# @Last Modified by:   zcy
# @Last Modified time: 2022-06-02 12:24:18

import os
import torch 
import imageio
import numpy as np
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook 
from skimage.metrics import structural_similarity as ssim


img2mse = lambda x, y : np.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * np.log(x) / np.log(np.array([10.]))
def calculate_ssim(im1, im2, data_range=255, multichannel=True):
    if multichannel:
        full_ssim = ssim(im1, im2, val_range=data_range, multichannel=True, full=True)[1]
        out_ssim = full_ssim.mean()
    else:
        full_ssim = ssim(im1, im2, val_range=data_range, multichannel=False, full=True)[1]
        out_ssim = full_ssim.mean()

    return out_ssim

@HOOKS.register_module() 
class CalTestMetricsHook(Hook): 
    """
        In test phase, calculate metrics over all testset
    """ 
    def __init__(self, cfg=None): 
        self.cfg = cfg
 
    def before_val_epoch(self, runner):
        self.rgbs = []
        self.gt_imgs = []

    def after_val_iter(self, runner):
        rank, _ = get_dist_info()
        if rank==0:
            cur_iter = runner.iter
            rgb = runner.outputs['rgb']
            gt_img = runner.outputs['gt_img']
            self.rgbs.append(rgb)
            self.gt_imgs.append(gt_img)
            
    def after_val_epoch(self, runner):
        rank, _ = get_dist_info()
        if rank==0:        
            mse_list, psnr_list, ssim_list = [], [], []
            for i, rgb in enumerate(self.rgbs):
                gt_img = self.gt_imgs[i]
                if isinstance(gt_img, torch.Tensor):
                    gt_img = gt_img.cpu().numpy()

                mse = img2mse(rgb, gt_img)
                psnr = mse2psnr(mse)
                ssim = calculate_ssim(rgb, gt_img, data_range=gt_img.max() - gt_img.min(), multichannel=True)
                mse_list.append(mse.item())
                psnr_list.append(psnr.item())
                ssim_list.append(ssim)

            average_mse = sum(mse_list) / len(mse_list)
            average_psnr = sum(psnr_list) / len(psnr_list)
            average_ssim = sum(ssim_list) / len(ssim_list)

            metrics = "In test phase on whole testset, mse is {:.5f}, psnr is {:.5f}, ssim is {:.5f}".format(average_mse, average_psnr, average_ssim)
            runner.logger.info(metrics)
            ''' 
                in mmcv's EpochBasedRunner, only 'after_train_epoch' epoch will be updated
                but in our test phase, we only want to run ('val', 1), 
                so we need to update runner_epoch additionally
            '''
            runner._epoch += 1 
