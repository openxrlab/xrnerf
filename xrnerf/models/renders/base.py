# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
from torch import nn

from .. import builder


class BaseRender(nn.Module, metaclass=ABCMeta):
    """Base class for recognizers. All recognizers should subclass it. All
    subclass should overwrite:

    - Methods:``forward_train``, supporting to forward when training.
    - Methods:``forward_test``, supporting to forward when testing.
    Args:
        cfg (dict): backbone config
        mlp (dict | None): mlp config
        render (dict | None): render config
    """
    def __init__(self, **kwargs):
        super().__init__()
        pass

    @abstractmethod
    def forward(self):
        pass
