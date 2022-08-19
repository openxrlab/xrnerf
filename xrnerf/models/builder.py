# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
MLPS = MODELS
RENDERS = MODELS
EMBEDDERS = MODELS
NETWORKS = MODELS
SAMPLERS = MODELS


def build_mlp(cfg):
    """Build backbone."""
    return MLPS.build(cfg)


def build_render(cfg):
    """Build head."""
    return RENDERS.build(cfg)


def build_embedder(cfg):
    """Build backbone."""
    return EMBEDDERS.build(cfg)


def build_network(cfg):
    # print(cfg.keys())
    return NETWORKS.build(cfg)


def build_sampler(cfg):
    return SAMPLERS.build(cfg)


# def build_optimizer(grad_vars, args):
#     # Create optimizer
#     optimizer = torch.optim.Adam(params=grad_vars, lr=args.lr_rate, betas=(0.9, 0.999))
