import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook

from .utils import to8b


@HOOKS.register_module()
class PassDatasetHook(Hook):
    """pass data in dataset to network's sampler, work for instant-ngp."""
    def __init__(self, dataset=None):
        self.dataset = dataset

    def before_run(self, runner):  # only run once
        alldata = self.dataset.get_alldata()
        datainfo = self.dataset.get_info()
        runner.model.module.sampler.set_data(alldata, datainfo)
        del self.dataset


@HOOKS.register_module()
class PassSamplerIterHook(Hook):
    """PassSamplerIterHook."""
    def before_train_iter(self, runner):
        runner.model.module.sampler.set_iter(runner.iter)


@HOOKS.register_module()
class ModifyBatchsizeHook(Hook):
    """change n_rays, work for instant-ngp."""
    def __init__(self):
        self.bs = 0

    def after_train_iter(self, runner):
        bs = runner.model.module.sampler.n_rays_per_batch
        if bs != self.bs:
            self.bs = bs
            runner.data_loader.iter_loader._dataset.set_batchsize(self.bs)


@HOOKS.register_module()
class HashSaveSpiralHook(Hook):
    """NGP save."""
    def __init__(self, save_folder='validation', cfg=None):
        self.save_folder = save_folder
        self.prefix = cfg.load_from.split('/')[-1].split('.')[-2].replace(
            'iter_', '')
        self.prefix = 'latest' if len(self.prefix) == 0 else self.prefix

    def before_val_epoch(self, runner):
        """init list."""
        self.spiral_data = []

    def after_val_iter(self, runner):
        """append image."""
        rank, _ = get_dist_info()
        if rank == 0:
            idx = runner.outputs['idx']
            spiral_rgb = runner.outputs['spiral_rgb']
            spiral_alpha = runner.outputs['spiral_alpha']
            self.spiral_data.append([idx, spiral_rgb, spiral_alpha])
            print(idx, spiral_rgb.shape, flush=True)

    def after_val_epoch(self, runner):
        """write images."""
        rank, _ = get_dist_info()
        if rank == 0:
            spiral_dir = os.path.join(runner.work_dir, self.save_folder)
            os.makedirs(spiral_dir, exist_ok=True)

            self.spiral_data = sorted(self.spiral_data, key=lambda x: x[0])
            self.spiral_data = self.apply_mask(self.spiral_data)

            spiral_rgbs = [x[1] for x in self.spiral_data]
            spiral_rgbs = np.stack(spiral_rgbs, 0)
            spiral_path = os.path.join(spiral_dir,
                                       '{}_rgb.mp4'.format(self.prefix))
            imageio.mimwrite(spiral_path, to8b(spiral_rgbs), fps=25, quality=8)

            spiral_alphas = [x[2] for x in self.spiral_data]
            spiral_alphas = np.stack(spiral_alphas, 0)
            spiral_path = os.path.join(spiral_dir,
                                       '{}_alpha.mp4'.format(self.prefix))
            imageio.mimwrite(spiral_path,
                             to8b(spiral_alphas),
                             fps=25,
                             quality=8)

            runner._epoch += 1

    def apply_mask(self, spiral_data):
        """apply_mask."""
        for i in range(len(spiral_data)):
            alpha = spiral_data[i][2]
            rgb = spiral_data[i][1]
            mask = (alpha >= 0.99).astype(np.float)
            rgb = rgb * mask + 1 * (1 - mask)
            spiral_data[i][1] = rgb
        return spiral_data
