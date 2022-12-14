# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import get_dist_info, load_checkpoint
from torch import nn
from tqdm import tqdm

from .. import builder
from ..builder import NETWORKS
from .nerf import NerfNetwork
from .utils import *


@NETWORKS.register_module()
class NeuralBodyNetwork(NerfNetwork):
    def __init__(self, cfg, embedder=None, render=None):
        nn.Module.__init__(self)

        self.chunk = cfg.chunk
        self.bs_data = cfg.bs_data
        self.phase = cfg.get('phase', 'train')
        self.idx = 0

        self.smpl_conv = builder.build_embedder(cfg.smpl_embedder)
        self.nerf_mlp = builder.build_mlp(cfg.nerf_mlp)

        self.render = builder.build_render(render)

    def forward(self, datas, is_test=False):
        # keep the batch norm staying in the training mode
        self.train()
        # extract features from structured latent codes
        xyzc_features = self.smpl_conv(datas)
        # predict colors and densities
        datas = self.nerf_mlp(xyzc_features, datas)
        datas, ret = self.render(datas, is_test)
        return ret

    def val_step(self, datas, *args, **kwargs):
        rank, world_size = get_dist_info()
        if rank == 0:
            for k in datas:
                datas[k] = unfold_batching(datas[k])

            ret = self.batchify_forward(
                datas, is_test=True)  # 测试时 raw_noise_std=False
            rgb = nb_recover_shape(ret['rgb'], datas['src_shape'],
                                   datas['mask_at_box']).cpu().numpy()
            disp = nb_recover_shape(ret['disp'], datas['src_shape'],
                                    datas['mask_at_box']).cpu().numpy()
            rgbs = [rgb]
            disps = [disp]

            outputs = {
                'rgbs': rgbs,
                'disps': disps,
                'rgb': rgb,
                'idx': self.idx
            }

            if self.phase != 'render':
                image = nb_recover_shape(datas['target_s'], datas['src_shape'],
                                         datas['mask_at_box']).cpu().numpy()
                outputs.update({'gt_imgs': [image], 'gt_img': image})

            self.idx = self.idx + 1
        else:
            outputs = {}
        return outputs
