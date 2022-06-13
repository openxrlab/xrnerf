# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from ..builder import MLPS


@MLPS.register_module()
class BaseMLP(nn.Module, metaclass=ABCMeta):
    def __init__(self, **kwarg):
        super().__init__()  # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError
