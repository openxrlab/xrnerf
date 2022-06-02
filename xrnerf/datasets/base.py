# Copyright (c) OpenMMLab. All rights reserved.
import copy
import mmcv
import torch
import warnings
import numpy as np
import os.path as osp
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self):
        super().__init__()

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)
