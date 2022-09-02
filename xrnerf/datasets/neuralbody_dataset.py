# Copyright (c) OpenMMLab. All rights reserved.

import os

import cv2
import imageio
import numpy as np
import torch

from .base import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from .utils import gen_spiral_path


@DATASETS.register_module()
class NeuralBodyDataset(BaseDataset):
    """NoBatchingDataset for blender datatype, each batch, select rays over one
    images in __init__() function, we don't concat all images."""
    def __init__(self, cfg, pipeline):
        super().__init__()

        self.is_train = cfg.mode == 'train'

        self.data_root = cfg.datadir
        self.ratio = cfg.ratio
        self.white_bkgd = cfg.white_bkgd
        self.smpl_vertices_dir = cfg.smpl_vertices_dir
        self.smpl_params_dir = cfg.smpl_params_dir
        self.img_path_to_smpl_idx = cfg.img_path_to_smpl_idx
        self.img_path_to_frame_idx = cfg.img_path_to_frame_idx
        self.cfg = cfg
        self.iter_n = 0

        self._init_load()
        self.pipeline = Compose(pipeline)

    def _init_load(self):  # load dataset when init
        cfg = self.cfg

        # load data
        ann_file = os.path.join(cfg.datadir, 'annots.npy')
        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        # generate camera poses for rendering
        self.spiral_poses = gen_spiral_path(self.cams['R'], self.cams['T'],
                                            cfg.get('num_render_views', 50))

        num_cams = len(self.cams['K'])
        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if cfg.mode == 'train' else test_view
        if len(view) == 0:
            view = [0]

        begin_frame, end_frame = cfg.training_frame
        frame_interval = cfg.frame_interval
        if cfg.get('phase', 'train_pose') == 'novel_pose':
            begin_frame, end_frame = cfg.novel_pose_frame
        if cfg.mode != 'train':
            frame_interval = cfg.get('val_frame_interval', 1)
        self.ims = np.array([
            np.array(ims_data['ims'])[view] for ims_data in annots['ims']
            [begin_frame:end_frame][::frame_interval]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view] for ims_data in annots['ims']
            [begin_frame:end_frame][::frame_interval]
        ]).ravel()
        self.num_cams = len(view)

    def _fetch_train_data(self, idx):
        if self.cfg.mode == 'render':
            datas = {
                'data_root': self.data_root,
                'idx': idx,
                'K': np.array(self.cams['K'][0]),
                'spiral_poses': self.spiral_poses,
                'cfg': self.cfg
            }
        else:
            datas = {
                'data_root': self.data_root,
                'idx': idx,
                'cams': self.cams,
                'cam_inds': self.cam_inds,
                'ims': self.ims,
                'cfg': self.cfg,
                'num_cams': self.num_cams
            }
        return datas

    def __getitem__(self, idx):
        if self.cfg.mode == 'val' or self.cfg.mode == 'test':
            idx = 0
        datas = self._fetch_train_data(idx)
        datas['iter_n'] = self.iter_n
        datas = self.pipeline(datas)
        return datas

    def __len__(self):
        if self.cfg.mode == 'render':
            return len(self.spiral_poses)
        else:
            return len(self.ims)
