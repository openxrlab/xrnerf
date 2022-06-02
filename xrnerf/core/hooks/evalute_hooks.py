# -*- coding: utf-8 -*-
# @Author: zcy
# @Date:   2022-04-20 17:05:14
# @Last Modified by:   zcy
# @Last Modified time: 2022-05-07 16:19:58

import os
import torch 
import imageio
import numpy as np
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook 
from skimage.metrics import structural_similarity as ssim


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

@HOOKS.register_module() 
class SaveTestHook(Hook): 
    """
    save testset's render results with test poses
    在每次val_step()之后调用 用于保存test数据集的渲染图片
    这些图片是有groundtruth的 可用来计算指标
    """ 
    def __init__(self, cfg=None): 
        self.cfg = cfg 
 
    def after_val_iter(self, runner):
        rank, _ = get_dist_info()
        if rank==0:          
            # print(runner.iter, runner._inner_iter)
            cur_iter = runner.iter
            rgbs = runner.outputs['rgbs']
            disps = runner.outputs['disps']
            gt_imgs = runner.outputs['gt_imgs']

            testset_dir = os.path.join(runner.work_dir, 'visualizations/testset/{}'.format(cur_iter))
            os.makedirs(testset_dir, exist_ok=True)

            for i, rgb in enumerate(rgbs):
                filename = os.path.join(testset_dir, '{:03d}.png'.format(i))
                final_img, gt_img = rgb, gt_imgs[i]
                # print(type(gt_img), isinstance(gt_img, torch.Tensor))
                # if isinstance(rgb, torch.Tensor):
                #     rgb = rgb.cpu().numpy()
                # if isinstance(gt_img, torch.Tensor):
                #     gt_img = gt_img.cpu().numpy()
                # print(rgbs[i].shape, disps[i].shape, gt_imgs[i].shape, flush=True)
                # print(torch.is_tensor(gt_img), )
                # print(type(gt_img), type(final_img))
                final_img = np.hstack((final_img, gt_img))
                imageio.imwrite(filename, to8b(final_img))


@HOOKS.register_module() 
class SaveSpiralHook(Hook): 
    """
    save testset's render results with spiral poses
    在每次val_step()之后调用 用于保存test数据集的环型pose渲染图片
    这些图片是没有groundtruth的 以视频方式保存
    """ 
    def __init__(self, cfg=None): 
        self.cfg = cfg 
 
    def after_val_iter(self, runner):
        rank, _ = get_dist_info()
        if rank==0:        
            cur_iter = runner.iter
            spiral_rgbs = np.stack(runner.outputs['spiral_rgbs'], 0)
            spiral_disps = np.stack(runner.outputs['spiral_disps'], 0)        

            spiral_dir = os.path.join(runner.work_dir, 'visualizations/spiral')
            os.makedirs(spiral_dir, exist_ok=True)

            imageio.mimwrite(os.path.join(spiral_dir, '{}_rgb.mp4'.format(cur_iter)), \
                to8b(spiral_rgbs), fps=30, quality=8)
            imageio.mimwrite(os.path.join(spiral_dir, '{}_disp.mp4'.format(cur_iter)), \
                to8b(spiral_disps / np.max(spiral_disps)), fps=30, quality=8)


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
class CalMetricsHook(Hook): 
    """
    在测试集上计算ssim psnr指标
    """ 
    def __init__(self, cfg=None): 
        self.cfg = cfg 
 
    def after_val_iter(self, runner):
        rank, _ = get_dist_info()
        if rank==0:
            cur_iter = runner.iter
            rgbs = runner.outputs['rgbs']
            disps = runner.outputs['disps']
            gt_imgs = runner.outputs['gt_imgs']
            if len(rgbs)==0: return
            if rgbs[0].shape!=rgbs[0].shape: return
            mse_list, psnr_list, ssim_list = [], [], []
            for i, rgb in enumerate(rgbs):
                gt_img = gt_imgs[i]
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

            # metrics = {'test_mse':average_mse, 'test_psnr':average_psnr, 'test_ssim':average_ssim}
            # runner.log_buffer.update(metrics) # 不合适，没法做到每次val_step后输出当前值，他会跟之前的求一个滑动平均
            
            metrics = "On testset, mse is {:.5f}, psnr is {:.5f}, ssim is {:.5f}".format(average_mse, average_psnr, average_ssim)
            runner.logger.info(metrics)
            # exit(0)
            