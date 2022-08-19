# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import get_dist_info, load_checkpoint
from torch import nn
from tqdm import tqdm

from .. import builder
from ..builder import NETWORKS
from .neuralbody import NeuralBodyNetwork
from .utils import *


@NETWORKS.register_module()
class AniNeRFNetwork(NeuralBodyNetwork):
    def __init__(self, cfg, render=None):
        nn.Module.__init__(self)

        self.cfg = cfg
        self.chunk = cfg.chunk
        self.bs_data = cfg.bs_data
        self.phase = cfg.phase
        self.idx = 0

        self.tpose_human = builder.build_mlp(cfg.tpose_human)
        self.deform_field = builder.build_mlp(cfg.deform_field)

        self.render = builder.build_render(render)

    def get_params(self):
        if self.cfg.phase == 'train_pose':
            params = []
            params += list(self.tpose_human.parameters())
            params += list(self.deform_field.bw_mlp.parameters())
            for param in self.deform_field.novel_pose_bw_mlp.parameters():
                param.requires_grad = False
        else:
            for param in self.tpose_human.parameters():
                param.requires_grad = False
            for param in self.deform_field.bw_mlp.parameters():
                param.requires_grad = False
            params = list(self.deform_field.novel_pose_bw_mlp.parameters())
        return params

    def forward(self, datas, is_test=False):
        deform_ret = self.deform_field(datas)

        # predict the color and density
        raw = self.tpose_human(deform_ret, datas)

        datas, tpose_ret = self.tpose_human.filter_and_format_prediction(
            raw, deform_ret, datas)

        datas, ret = self.render(datas, is_test)
        ret['pbw'] = tpose_ret['pbw']
        ret['tbw'] = tpose_ret['tbw']

        return ret

    def train_pose_stage(self, datas):
        ret = self.forward(datas, is_test=False)

        img_loss = img2mse(ret['rgb'], datas['target_s'])
        psnr = mse2psnr(img_loss)
        loss = img_loss

        bw_loss = F.smooth_l1_loss(ret['pbw'], ret['tbw'])
        loss = loss + bw_loss

        log_vars = {'loss': loss.item(), 'psnr': psnr.item()}
        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': ret['rgb'].shape[0]
        }

        return outputs

    def train_step(self, datas, optimizer, **kwargs):
        for k in datas:
            datas[k] = unfold_batching(datas[k])

        if self.cfg.phase == 'train_pose':
            outputs = self.train_pose_stage(datas)
        else:
            outputs = NovelPoseTraining.calculate_loss(self, datas)

        return outputs
