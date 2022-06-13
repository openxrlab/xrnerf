# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod

import mmcv
import numpy as np
import torch
from torch.utils.data import Dataset

from .pipelines import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self):
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def _init_pipeline(self, pipeline):
        datainfo = self.get_info(
        )  # those params are not clear until data are loaded
        for p, _ in enumerate(pipeline):
            for d in datainfo:
                pipeline[p][d] = datainfo[d]
        self.pipeline = Compose(pipeline)

    def _fetch_train_data(self, idx):
        raise NotImplementedError

    def _fetch_val_data(self, idx):
        raise NotImplementedError

    def _fetch_test_data(self, idx):
        raise NotImplementedError

    def set_iter(self, iter_n):
        self.iter_n = iter_n  # see PassIterHook
