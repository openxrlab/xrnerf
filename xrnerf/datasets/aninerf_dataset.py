# Copyright (c) OpenMMLab. All rights reserved.

import os

import cv2
import imageio
import numpy as np
import torch

from .base import BaseDataset
from .builder import DATASETS
from .neuralbody_dataset import NeuralBodyDataset
from .pipelines import Compose
from .utils import get_rigid_transformation


@DATASETS.register_module()
class AniNeRFDataset(NeuralBodyDataset):
    """NoBatchingDataset for blender datatype, each batch, select rays over one
    images in __init__() function, we don't concat all images."""
    def __init__(self, cfg, pipeline):
        super().__init__(cfg, pipeline)

        self.is_train = cfg.mode == 'train'
        self.cfg = cfg
        self._init_load()

    def _init_load(self):
        super()._init_load()

        # load joints, parents, blend weights, big poses
        self.lbs_root = os.path.join(self.data_root, 'lbs')
        self.joints = np.load(os.path.join(self.lbs_root,
                                           'joints.npy')).astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        self.weights = np.load(os.path.join(self.lbs_root,
                                            'weights.npy')).astype(np.float32)
        self.canonical_smpl_verts = np.load(
            os.path.join(self.lbs_root,
                         'bigpose_vertices.npy')).astype(np.float32)
        self.big_A = self.load_bigpose()

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)

        template_pose_path = os.path.join(self.lbs_root, 'template_pose.npy')
        if os.path.exists(template_pose_path):
            big_poses = np.load(template_pose_path)

        big_poses = big_poses.reshape(-1, 3)
        big_A = get_rigid_transformation(big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A

    def _fetch_train_data(self, idx):
        datas = super()._fetch_train_data(idx)
        datas.update({
            'big_A': self.big_A,
            'canonical_smpl_verts': self.canonical_smpl_verts,
            'smpl_bw': self.weights,
            'joints': self.joints,
            'parents': self.parents
        })
        return datas
