# # Copyright (c) OpenMMLab. All rights reserved.

import numpy as np
import torch

from xrnerf.utils.data_helper import get_global_domain_min_and_max

from .builder import DATASETS
from .scene_dataset import SceneBaseDataset


@DATASETS.register_module()
class KiloNerfDataset(SceneBaseDataset):
    def __init__(self, cfg, pipeline):
        super().__init__(cfg, pipeline)
        self.global_domain_min, self.global_domain_max = get_global_domain_min_and_max(
            cfg, torch.device('cpu'))

    def _fetch_train_data(self, idx):
        if self.is_batching:  # for batching dataset, rays are randomly selected from all images
            data = {'rays_rgb': self.rays_rgb, 'idx': idx}
        else:  # for no_batching dataset, rays are selected from one images
            data = {
                'poses': self.poses,
                'images': self.images,
                'i_data': self.i_train,
                'idx': idx,
                'global_domain_min': self.global_domain_min,
                'global_domain_max': self.global_domain_max,
            }
        data['iter_n'] = self.iter_n
        return data

    def _fetch_val_data(self, idx):  # for val mode, fetch all data in one time
        data = {'spiral_poses':self.render_poses, 'poses':self.poses[self.i_test], \
                'images':self.images[self.i_test], 'global_domain_min':self.global_domain_min, \
                'global_domain_max':self.global_domain_max}
        return data

    def _fetch_test_data(
        self, idx
    ):  # the difference between test and val is: test return one image once
        data = {'pose':self.poses[self.i_test][idx], 'image':self.images[self.i_test][idx], \
                'idx':idx, 'global_domain_min':self.global_domain_min, 'global_domain_max':self.global_domain_max}
        return data
