# # Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from .builder import DATASETS
from .load_data import load_data, load_rays_bungee
from .scene_dataset import SceneBaseDataset


@DATASETS.register_module()
class BungeeDataset(SceneBaseDataset):
    def __init__(self, cfg, pipeline):
        self.cur_stage = cfg.cur_stage
        super().__init__(cfg, pipeline)

    def _init_load(self):  # load dataset when init
        self.images, self.poses, self.render_poses, self.hwf, self.K, self.scene_scaling_factor, self.scene_origin, self.scale_split, self.i_train, self.i_val, self.i_test, self.n_images = load_data(
            self.cfg)

        if self.is_batching and self.mode == 'train':
            # for batching dataset, rays must be computed when init()
            self.N_rand = self.cfg.N_rand_per_sampler
            self.rays_rgb, self.radii, self.scale_codes = load_rays_bungee(
                self.hwf[0], self.hwf[1], self.hwf[2], self.poses, self.images,
                self.i_train, self.n_images, self.scale_split, self.cur_stage)

    def _fetch_train_data(self, idx):
        if self.is_batching:  # for batching dataset, rays are randomly selected from all images
            data = {
                'rays_rgb': self.rays_rgb,
                'radii': self.radii,
                'scale_code': self.scale_codes,
                'idx': idx
            }
        else:  # for batching dataset, rays are selected from one images
            data = {
                'poses': self.poses,
                'images': self.images,
                'n_images': self.n_images,
                'i_data': self.i_train,
                'idx': idx
            }
        data['iter_n'] = self.iter_n
        return data

    def _fetch_val_data(self, idx):  # for val mode, fetch all data in one time
        data = {
            'spiral_poses': self.render_poses,
            'poses': self.poses[self.i_test],
            'images': self.images[self.i_test],
        }
        return data

    def _fetch_test_data(
            self, idx):  # different from val: test return one image once
        data = {
            'pose': self.poses[self.i_test][idx],
            'image': self.images[self.i_test][idx],
            'idx': idx
        }
        return data

    def get_info(self):
        res = {
            'H': self.hwf[0],
            'W': self.hwf[1],
            'focal': self.hwf[2],
            'K': self.K,
            'render_poses': self.render_poses,
            'hwf': self.hwf,
            'cur_stage': self.cur_stage,
            'scene_origin': self.scene_origin,
            'scene_scaling_factor': self.scene_scaling_factor,
            'scale_split': self.scale_split,
        }
        return res
