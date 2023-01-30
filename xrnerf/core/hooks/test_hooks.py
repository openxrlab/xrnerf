import json
import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook

from .utils import calculate_ssim, img2mse, mse2psnr, to8b


@HOOKS.register_module()
class TestHook(Hook):
    """In test phase, calculate metrics over all testset.

    ndown: multiscales for mipnerf, set to 0 for others
    """
    def __init__(self,
                 ndown=1,
                 save_img=False,
                 dump_json=False,
                 save_folder='test'):
        self.ndown = ndown
        self.dump_json = dump_json
        self.save_img = save_img
        self.save_folder = save_folder

    def before_val_epoch(self, runner):
        """init list."""
        self.psnr = {}
        self.ssim = {}
        self.mse = {}
        for i in range(self.ndown):
            self.psnr[i] = []
            self.mse[i] = []
            self.ssim[i] = []

    def after_val_iter(self, runner):
        """after_val_iter."""
        rank, _ = get_dist_info()
        if rank == 0:
            cur_iter = runner.iter
            rgb = runner.outputs['rgb']
            gt_img = runner.outputs['gt_img']
            idx = runner.outputs['idx']

            if self.save_img:  # save image
                testset_dir = os.path.join(runner.work_dir, self.save_folder)
                os.makedirs(testset_dir, exist_ok=True)
                filename = os.path.join(testset_dir, '{:03d}.png'.format(idx))
                imageio.imwrite(filename, to8b(rgb))

            # cal metrics
            mse = img2mse(rgb, gt_img)
            psnr = mse2psnr(mse)
            ssim = calculate_ssim(rgb,
                                  gt_img,
                                  data_range=gt_img.max() - gt_img.min(),
                                  multichannel=True)

            scale = idx % self.ndown  # for 'self.ndown==1', scale is 0
            self.psnr[scale].append(float(psnr))
            self.mse[scale].append(float(mse))
            self.ssim[scale].append(float(ssim))

    def after_val_epoch(self, runner):
        """after_val_epoch."""
        rank, _ = get_dist_info()
        if rank == 0:
            metrics = 'In test phase on whole testset, \n  '
            for scale in range(self.ndown):
                average_mse = sum(self.mse[scale]) / len(self.mse[scale])
                average_psnr = sum(self.psnr[scale]) / len(self.psnr[scale])
                average_ssim = sum(self.ssim[scale]) / len(self.ssim[scale])
                metrics += f' for scale {scale}, mse is {average_mse}, psnr is {average_psnr}, ssim is {average_ssim}. \n'
            runner.logger.info(metrics)

            if self.dump_json:
                filename = os.path.join(runner.work_dir, self.save_folder,
                                        'test_results.json')
                with open(filename, 'w') as f:
                    json.dump(
                        {
                            'results': metrics,
                            'psnrs': self.psnr,
                            'ssims': self.ssim
                        }, f)
            '''
                in mmcv's EpochBasedRunner, only 'after_train_epoch' epoch will be updated
                but in our test phase, we only want to run ('val', 1),
                so we need to update runner_epoch additionally
            '''
            runner._epoch += 1
