# Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from .base import BaseDataset
from .builder import DATASETS
from .load_data import load_data, load_rays


@DATASETS.register_module()
class SceneBaseDataset(BaseDataset):
    def __init__(self, cfg, pipeline):
        super().__init__()
        self.iter_n = 0
        self.cfg = cfg
        self.mode = cfg.mode
        self.is_batching = cfg.is_batching
        self._init_load()
        self._init_pipeline(pipeline)

    def _init_load(self):  # load dataset when init
        self.images, self.poses, self.render_poses, self.hwf, self.K, self.near, \
            self.far, self.i_train, self.i_val, self.i_test = load_data(self.cfg)
        if self.is_batching and self.mode == 'train':
            # for batching dataset, rays must be computed when init()
            self.N_rand = self.cfg.N_rand_per_sampler
            self.rays_rgb = load_rays(self.hwf[0], self.hwf[1], self.K,
                                      self.poses, self.images, self.i_train)

    def get_info(self):
        res = {
            'H': self.hwf[0],
            'W': self.hwf[1],
            'focal': self.hwf[2],
            'K': self.K,
            'render_poses': self.render_poses,
            'hwf': self.hwf,
            'near': self.near,
            'far': self.far
        }
        return res

    def _fetch_train_data(self, idx):
        if self.is_batching:  # for batching dataset, rays are randomly selected from all images
            data = {'rays_rgb': self.rays_rgb, 'idx': idx}
        else:  # for batching dataset, rays are selected from one images
            data = {
                'poses': self.poses,
                'images': self.images,
                'i_data': self.i_train,
                'idx': idx
            }
        data['iter_n'] = self.iter_n
        return data

    def _fetch_val_data(self, idx):  # for val mode, fetch all data in one time
        data = {'spiral_poses':self.render_poses, 'poses':self.poses[self.i_test], \
                'images':self.images[self.i_test]}
        return data

    def _fetch_test_data(
            self, idx):  # different from val: test return one image once
        data = {'pose':self.poses[self.i_test][idx], 'image':self.images[self.i_test][idx], \
                'idx':idx}
        return data

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self._fetch_train_data(idx)
            data = self.pipeline(data)
            return data
        elif self.mode == 'val':  # for some complex reasons，pipeline have to be moved to network.val_step() in val phase
            return self._fetch_val_data(idx)
        elif self.mode == 'test':  # for some complex reasons，pipeline have to be moved to network.val_step() in test phase
            data = self._fetch_test_data(idx)
            return data

    def __len__(self):
        if self.mode == 'train':
            if self.is_batching:
                return self.rays_rgb.shape[0] // self.N_rand
            else:
                return self.i_train.shape[0]
        elif self.mode == 'val':
            return 1
        elif self.mode == 'test':
            return self.i_test.shape[0]
