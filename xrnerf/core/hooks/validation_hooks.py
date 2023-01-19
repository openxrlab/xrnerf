import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook

from .utils import calculate_ssim, img2mse, mse2psnr, to8b


@HOOKS.register_module()
class SetValPipelineHook(Hook):
    """pass val dataset's pipeline to network."""
    def __init__(self, valset=None):
        self.val_pipeline = valset.pipeline

    def before_run(self, runner):
        """only run once."""
        runner.model.module.set_val_pipeline(self.val_pipeline)
        del self.val_pipeline


@HOOKS.register_module()
class SaveSpiralHook(Hook):
    """save testset's render results with spiral poses 在每次val_step()之后调用
    用于保存test数据集的环型pose渲染图片 这些图片是没有groundtruth的 以视频方式保存."""
    def __init__(self, save_folder='validation'):
        self.save_folder = save_folder

    def after_val_iter(self, runner):
        """SaveSpiralHook."""
        rank, _ = get_dist_info()
        if rank == 0:
            cur_iter = runner.iter
            spiral_rgbs = np.stack(runner.outputs['spiral_rgbs'], 0)
            spiral_disps = np.stack(runner.outputs['spiral_disps'], 0)

            spiral_dir = os.path.join(runner.work_dir, self.save_folder)
            os.makedirs(spiral_dir, exist_ok=True)

            imageio.mimwrite(os.path.join(spiral_dir,
                                          '{}_rgb.mp4'.format(cur_iter)),
                             to8b(spiral_rgbs),
                             fps=30,
                             quality=8)
            imageio.mimwrite(os.path.join(spiral_dir,
                                          '{}_disp.mp4'.format(cur_iter)),
                             to8b(spiral_disps / np.max(spiral_disps)),
                             fps=30,
                             quality=8)


@HOOKS.register_module()
class NBSaveSpiralHook(Hook):
    """save testset's render results with spiral poses 在每次val_step()之后调用
    用于保存test数据集的环型pose渲染图片 这些图片是没有groundtruth的 以视频方式保存."""
    def __init__(self, save_folder='validation'):
        self.save_folder = save_folder
        self.rgbs = []
        self.disps = []

    def after_val_iter(self, runner):
        """NBSaveSpiralHook."""
        rank, _ = get_dist_info()
        if rank == 0:
            cur_iter = runner.iter
            self.rgbs.append(runner.outputs['rgbs'][0])
            self.disps.append(runner.outputs['disps'][0])

    def after_val_epoch(self, runner):
        """NBSaveSpiralHook."""
        spiral_dir = os.path.join(runner.work_dir, self.save_folder)
        os.makedirs(spiral_dir, exist_ok=True)

        spiral_rgbs = np.array(self.rgbs)
        spiral_disps = np.array(self.disps)

        imageio.mimwrite(os.path.join(spiral_dir, 'rgb.mp4'),
                         to8b(spiral_rgbs),
                         fps=30,
                         quality=8)
        imageio.mimwrite(os.path.join(spiral_dir, 'disp.mp4'),
                         to8b(spiral_disps / np.max(spiral_disps)),
                         fps=30,
                         quality=8)
        '''
            in mmcv's EpochBasedRunner, only 'after_train_epoch' epoch will be updated
            but in our test phase, we only want to run ('val', 1),
            so we need to update runner_epoch additionally
        '''
        runner._epoch += 1


@HOOKS.register_module()
class ValidateHook(Hook):
    """在测试集上计算ssim psnr指标 保存图片."""
    def __init__(self, save_folder='validation'):
        self.save_folder = save_folder

    def after_val_iter(self, runner):
        """ValidateHook."""
        rank, _ = get_dist_info()
        if rank == 0:
            cur_iter = runner.iter
            rgbs = runner.outputs['rgbs']
            gt_imgs = runner.outputs['gt_imgs']
            if len(rgbs) == 0:
                return
            if rgbs[0].shape != gt_imgs[0].shape:
                return

            ########### calculate metrics ###########
            mse_list, psnr_list, ssim_list = [], [], []
            for i, rgb in enumerate(rgbs):
                gt_img = gt_imgs[i]
                if isinstance(gt_img, torch.Tensor):
                    gt_img = gt_img.cpu().numpy()

                mse = img2mse(rgb, gt_img)
                psnr = mse2psnr(mse)
                ssim = calculate_ssim(rgb,
                                      gt_img,
                                      data_range=gt_img.max() - gt_img.min(),
                                      multichannel=True)
                mse_list.append(mse.item())
                psnr_list.append(psnr.item())
                ssim_list.append(ssim)

            average_mse = sum(mse_list) / len(mse_list)
            average_psnr = sum(psnr_list) / len(psnr_list)
            average_ssim = sum(ssim_list) / len(ssim_list)
            ########### calculate metrics ###########

            ########### save test images ###########
            testset_dir = os.path.join(runner.work_dir, self.save_folder,
                                       str(cur_iter))
            os.makedirs(testset_dir, exist_ok=True)
            for i, rgb in enumerate(rgbs):
                filename = os.path.join(testset_dir, '{:03d}.png'.format(i))
                final_img, gt_img = rgb, gt_imgs[i]
                final_img = np.hstack((final_img, gt_img))
                imageio.imwrite(filename, to8b(final_img))
            ########### save test images ###########

            # metrics = {'test_mse':average_mse, 'test_psnr':average_psnr, 'test_ssim':average_ssim}
            # runner.log_buffer.update(metrics) # 不合适，没法做到每次val_step后输出当前值，他会跟之前的求一个滑动平均

            metrics = 'On testset, mse is {:.5f}, psnr is {:.5f}, ssim is {:.5f}'.format(
                average_mse, average_psnr, average_ssim)
            runner.logger.info(metrics)


@HOOKS.register_module()
class CalElapsedTimeHook(Hook):
    """calculate average elapsed_time in val step."""
    def __init__(self, cfg=None):
        self.cfg = cfg

    def after_val_iter(self, runner):
        """after_val_iter."""
        rank, _ = get_dist_info()
        if rank == 0:
            if 'elapsed_time' in runner.outputs:
                elapsed_time_list = runner.outputs['elapsed_time']
            else:
                elapsed_time_list = []
            if len(elapsed_time_list) == 0: return

            #calculate average elapsed time
            average_elapsed_time = 1000 * sum(elapsed_time_list) / len(
                elapsed_time_list)

            metrics = 'On testset, elapsed_time is {:7.2f} ms'.format(
                average_elapsed_time)
            runner.logger.info(metrics)
            # exit(0)
