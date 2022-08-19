import os
import shutil
import sys

import numpy as np
import torch
# sys.path.append('/home/zhengchengyao/Document/Nerf/git/xrnerf')
from mmcv import Config, ConfigDict

from xrnerf.datasets.load_data import load_data


def test_load():

    data_cfg = dict(dataset_type='blender',
                    datadir='test/datasets/data/nerf_synthetic/lego',
                    half_res=True,
                    testskip=1,
                    white_bkgd=False,
                    is_batching=False,
                    mode='train')
    data_cfg = ConfigDict(data_cfg)
    images, poses, render_poses, hwf, K, near, far, i_train, \
            i_val, i_test = load_data(data_cfg)

    data_cfg = dict(dataset_type='llff',
                    datadir='test/datasets/data/nerf_llff_data/fern',
                    half_res=False,
                    testskip=1,
                    N_rand_per_sampler=256,
                    llffhold=1,
                    no_ndc=True,
                    white_bkgd=False,
                    spherify=False,
                    shape='greek',
                    factor=8,
                    is_batching=True,
                    mode='train')
    data_cfg = ConfigDict(data_cfg)
    images, poses, render_poses, hwf, K, near, far, i_train, \
            i_val, i_test = load_data(data_cfg)


# test_load()
