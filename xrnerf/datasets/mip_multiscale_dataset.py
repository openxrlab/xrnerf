import json
import os
import time
from turtle import pd

import numpy as np
import torch
from PIL import Image

from .builder import DATASETS
from .load_data import load_data, load_rays_multiscale
from .pipelines import Compose
from .scene_dataset import SceneBaseDataset
from .utils import flatten


@DATASETS.register_module()
class MipMultiScaleDataset(SceneBaseDataset):
    def _init_load(self):
        self.meta, self.images, self.n_examples = load_data(self.cfg)
        self.rays = load_rays_multiscale(self.meta, self.n_examples)
        if self.mode == 'train':
            self.images = flatten(self.images)
            for key in self.rays.keys():
                self.rays[key] = flatten(self.rays[key])

    def _init_pipeline(self, pipeline):
        self.pipeline = Compose(pipeline)

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self._fetch_train_data(idx)
            data = self.pipeline(data)
        else:
            data = self._fetch_test_data(idx)
            data = self.pipeline(data)
        return data

    def __len__(self):
        return self.n_examples

    def _fetch_train_data(self, idx):
        data = {'target_s': self.images}
        for key in self.rays.keys():
            data[key] = self.rays[key]
        return data

    def _fetch_test_data(self, idx):
        """get one test example."""
        datas = {'image': self.images[idx], 'idx': idx}
        for key in self.rays.keys():
            datas[key] = self.rays[key][idx]
        return datas
