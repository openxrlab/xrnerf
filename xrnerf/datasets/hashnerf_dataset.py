# Copyright (c) OpenMMLab. All rights reserved.

import sys

import numpy as np
import torch

from .builder import DATASETS
from .load_data import load_data, load_rays_hash
from .scene_dataset import SceneBaseDataset
from .utils import poses_nerf2ngp


@DATASETS.register_module()
class HashNerfDataset(SceneBaseDataset):
    def __init__(self, cfg, pipeline):
        if 'val_n' in cfg: self.val_n = cfg.val_n
        super().__init__(cfg, pipeline)

    def check_img(self):
        if self.images.shape[3] == 3:
            print('image has no alpha channel, set to 1')
            alpha = np.ones(list(self.images.shape[:3]) + [1])
            self.images = np.concatenate([self.images, alpha], 3)

    def _init_load(self):
        self.N_rand = self.cfg.N_rand_per_sampler
        assert self.is_batching is True, 'HashNerfDataset only support batching mode'
        self.images, self.poses, self.render_poses, self.hwf, self.K, self.near, \
            self.far, self.i_train, self.i_val, self.i_test = load_data(self.cfg)
        self.check_img()

        i_index = np.concatenate((self.i_val, self.i_train))

        self.images, self.poses = self.images[i_index], self.poses[i_index]
        correct_pose = [1, -1, -1]
        offset = [0.5, 0.5, 0.5]
        scale = 0.33

        self.poses = poses_nerf2ngp(self.poses, correct_pose, scale, offset)
        if self.mode == 'train':
            self.rays_rgb = load_rays_hash(self.hwf[0], self.hwf[1], self.K,
                                           self.poses, self.images)
            np.random.shuffle(self.rays_rgb)  # slow

        elif self.mode == 'test':
            # self.render_poses = self.render_poses[:2] # tmp
            self.n_render = self.render_poses.shape[0]
            self.images = np.zeros(
                (self.n_render, self.hwf[0], self.hwf[1], 4))
            self.render_poses = poses_nerf2ngp(self.render_poses.numpy(),
                                               correct_pose, scale, offset)
            self.rays_rgb = load_rays_hash(self.hwf[0], self.hwf[1], self.K,
                                           self.render_poses, self.images)

    def get_alldata(self):
        aabb_scale = 1
        aabb_center = (0.5, 0.5)
        aabb_range = (aabb_center[0] - aabb_scale / 2,
                      aabb_center[1] + aabb_scale / 2)
        n_img = self.images.shape[0]
        focal = np.ones((n_img, 2), dtype=float) * self.hwf[2]
        metadata = [0, 0, 0, 0, 0.5, 0.5, self.hwf[2], self.hwf[2], 0, 0, 0]
        metadata = np.expand_dims(metadata, 0).repeat(n_img, axis=0)
        poses = self.render_poses if self.mode == 'test' else self.poses
        res = {
            'aabb_scale': aabb_scale,
            'aabb_range': aabb_range,
            'images': self.images,
            'poses': poses,
            'focal': focal,
            'metadata': metadata,
        }
        return res

    def get_info(self):
        res = {
            'H': self.hwf[0],
            'W': self.hwf[1],
            'focal': self.hwf[2],
            'K': self.K,
            # 'render_poses': self.render_poses,
            'hwf': self.hwf,
            'near': self.near,
            'far': self.far
        }
        return res

    def set_batchsize(self, bs):
        self.N_rand = bs  # ModifyBatchsizeHook

    def _fetch_train_data(self, idx):
        # N_rays may changes, during training
        data = {'rays_rgb': self.rays_rgb, 'idx': idx, 'N_rand': self.N_rand}
        data['iter_n'] = self.iter_n
        return data

    def _fetch_val_data(self, idx):
        # ngp paper use all images to train and val
        data = {'poses':self.poses[:self.val_n], \
                'images':self.images[:self.val_n]}
        return data

    def _fetch_test_data(self, idx):
        n_pixel = self.hwf[0] * self.hwf[1]
        start_i, end_i = idx * n_pixel, (idx + 1) * n_pixel
        data = {
            'pose': self.render_poses[idx],
            'rays_o': self.rays_rgb[start_i:end_i, :3],
            'rays_d': self.rays_rgb[start_i:end_i, 3:6],
            'img_ids': np.ones((n_pixel, 1)) * idx,
            'src_shape': np.array([self.hwf[0], self.hwf[1], 3]),
            'idx': idx
        }
        return data

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self._fetch_train_data(idx)
            data = self.pipeline(data)
            return data
        elif self.mode == 'val':
            return self._fetch_val_data(idx)
        elif self.mode == 'test':
            data = self._fetch_test_data(idx)
            return data

    def __len__(self):
        if self.mode == 'train':
            # *4 to make sure all index can be fetched
            return self.rays_rgb.shape[0] // self.cfg.N_rand_per_sampler * 4
        elif self.mode == 'val':
            return 1
        elif self.mode == 'test':
            return self.n_render
