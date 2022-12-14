# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from .. import builder


class BaseNerfNetwork(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers. All recognizers should subclass it. All
    subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.
    Args:
        cfg (dict): backbone config
        mlp (dict | None): mlp config
        render (dict | None): render config
    """

    # def __init__(self, cfg, mlp=None, render=None):
    #     super().__init__()
    #     # record the source of the backbone
    #     # self.embedder = builder.build_embedder(mlp)
    #     # self.mlp = builder.build_mlp(mlp)
    #     # self.render = builder.build_render(render)
    def __init__(self, **kwarg):
        super().__init__()

    @abstractmethod
    def train_step(self, data, optimizer, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def val_step(self, data, **kwargs):
        raise NotImplementedError
