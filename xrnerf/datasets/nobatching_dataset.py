# Copyright (c) OpenMMLab. All rights reserved.

import torch
import numpy as np
from .nerf_dataset import NerfDataset
from .builder import DATASETS


@DATASETS.register_module()
class NerfNoBatchingDataset(NerfDataset):
    ''' 
        NoBatchingDataset for blender datatype,
        each batch, select rays over one images
        in __init__() function, we don't concat all images
    '''
    def __init__(self, cfg, pipeline):
        super().__init__(cfg, pipeline)
        # self.N_rand = cfg.N_rand_per_sampler
        self.poses = torch.Tensor(self.poses)
        self.render_poses = torch.Tensor(self.render_poses)
        self.images = torch.Tensor(self.images)

    def fetch_train_data(self, idx):
        # 此时选择一张图，从该图里面随机选择N_rand个射线
        img_i = self.i_train[idx] # idx是self.i_train中的下标
        target_s = self.images[img_i]
        pose = self.poses[img_i, :3, :4]
        data = {'pose':pose, 'target_s':target_s}
        return data

    def __len__(self):
        if self.mode=='train':
            return self.i_train.shape[0]
        elif self.mode=='val':
            return 1
        elif self.mode=='test':
            return self.i_test.shape[0]