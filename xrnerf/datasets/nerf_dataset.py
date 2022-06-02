# Copyright (c) OpenMMLab. All rights reserved.

import torch
import numpy as np
from .base import BaseDataset
from .pipelines import Compose
from .load_data import load_data


class NerfDataset(BaseDataset):

    def __init__(self, cfg, pipeline):
        super().__init__()
        self.mode = cfg.mode
        self.images, self.poses, self.render_poses, self.hwf, self.K, self.near, \
            self.far, self.i_train, self.i_val, self.i_test = load_data(cfg)
        datainfo = self.get_info() # 只有载入数据了，才能知道这些参数值
        for p, _ in enumerate(pipeline):
            for d in datainfo:
                pipeline[p][d] = datainfo[d]
        self.pipeline = Compose(pipeline)
        self.iter_n = 0 

    def get_info(self):
        res = {'H':self.hwf[0], 'W':self.hwf[1], 'focal':self.hwf[2], 'K':self.K,
                'render_poses':self.render_poses, 'hwf':self.hwf, 'near':self.near,
                'far':self.far}
        return res

    def set_iter(self, iter_n):
        # print('iter_n', iter_n, flush=True)
        self.iter_n = iter_n # 在hook里，会传进来

    def fetch_train_data(self, idx):
        raise NotImplementedError 

    def fetch_val_data(self):
        # for val mode, fetch all data in one time
        data = {'spiral_poses':self.render_poses, 'poses':self.poses[self.i_test], \
                'images':self.images[self.i_test]}
        return data

    def fetch_test_data(self, idx):
        data = {'pose':self.poses[self.i_test][idx], 'image':self.images[self.i_test][idx]}
        return data

    def __getitem__(self, idx):
        if self.mode=='train':
            data = self.fetch_train_data(idx)
            data['iter_n'] = self.iter_n
            data = self.pipeline(data)
            return data
        elif self.mode=='val':
            # for some complex reasons，pipeline have to be moved to network.val_step() in val phase
            return self.fetch_val_data()
        elif self.mode=='test':
            # for some complex reasons，pipeline have to be moved to network.val_step() in test phase
            data = self.fetch_test_data(idx)
            return data

    def __len__(self):
        raise NotImplementedError 
