# Copyright (c) OpenMMLab. All rights reserved.

import torch
import numpy as np
from .nerf_dataset import NerfDataset
from .builder import DATASETS


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


@DATASETS.register_module()
class NerfBatchingDataset(NerfDataset):
    ''' 
        BatchingDataset for llff datatype,
        each batch, select rays over all images
        in __init__() function, we must concat all images
    '''
    def __init__(self, cfg, pipeline):
        super().__init__(cfg, pipeline)
        self.N_rand = cfg.N_rand_per_sampler
        print('get rays')
        rays = np.stack([get_rays_np(self.hwf[0], self.hwf[1], self.K, p) for p in self.poses[:,:3,:4]], 0) # [N, ro+rd, H, W, 3]
        self.rays_rgb = np.concatenate([rays, self.images[:,None]], 1) # [N, ro+rd+rgb, H, W, 3]
        self.rays_rgb = np.transpose(self.rays_rgb, [0,2,3,1,4]) # [N, H, W, ro+rd+rgb, 3]
        self.rays_rgb = np.stack([self.rays_rgb[i] for i in self.i_train], 0) # train self.images only
        self.rays_rgb = np.reshape(self.rays_rgb, [-1,3,3]) # [(N-1)*H*W, ro+rd+rgb, 3]
        self.rays_rgb = self.rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(self.rays_rgb)
        print('done')

        # Move training data to GPU
        self.poses = torch.Tensor(self.poses)
        self.render_poses = torch.Tensor(self.render_poses)
        self.images = torch.Tensor(self.images)
        self.rays_rgb = torch.Tensor(self.rays_rgb)

    def fetch_train_data(self, idx):
        start_i = self.N_rand*idx
        batch = self.rays_rgb[start_i:start_i+self.N_rand] # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        # batch_rays, target_s = batch[:2], batch[2]
        rays_o, rays_d, target_s = batch[0], batch[1], batch[2]
        # batch_rays (N_rand_per_sampler, 3)
        # target_s   (2, N_rand_per_sampler, 3)
        # bs>1
        data = {'rays_o':rays_o, 'rays_d':rays_d, 'target_s':target_s}
        return data

    def __len__(self):
        if self.mode=='train':
            return self.rays_rgb.shape[0]//self.N_rand
        elif self.mode=='val':
            return 1
        elif self.mode=='test':
            return self.i_test.shape[0]