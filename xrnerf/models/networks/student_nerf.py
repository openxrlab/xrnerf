# Copyright (c) OpenMMLab. All rights reserved.
from re import S

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import get_dist_info
from torch import nn

from .. import builder
from ..builder import NETWORKS, build_network
from .base import BaseNerfNetwork
from .utils import unfold_batching, transform_examples


@NETWORKS.register_module()
class StudentNerfNetwork(BaseNerfNetwork):
    """StudentNerfNetwork learns from a pretrained nerf model, and has a mlp
    structure which is a multi_network."""
    def __init__(self,
                 cfg,
                 pretrained_kwargs=None,
                 multi_network=None,
                 render=None):
        super().__init__()

        if 'outputs' in cfg: self.outputs = cfg.outputs
        if 'query_batch_size' in cfg:
            self.query_batch_size = cfg.query_batch_size
        self.test_batch_size = cfg.get('test_batch_size', 0)

        if pretrained_kwargs is not None:
            pretrain_cfg = Config.fromfile(pretrained_kwargs.config)
            pretrained_nerf = build_network(pretrain_cfg.model)
            checkpoint = torch.load(pretrained_kwargs.checkpoint)
            pretrained_nerf.load_state_dict(checkpoint['state_dict'])
            self.teacher_nerf = pretrained_nerf.mlp

        if multi_network is not None:
            self.multi_network = builder.build_mlp(multi_network)
        if render is not None:
            self.render = builder.build_render(render)

    def get_params(self):
        grad_vars = list(self.multi_network.parameters())
        return grad_vars

    def teacher_forward(self, data):
        """use teacher nerf to get target."""
        data, ret = self.render(self.teacher_nerf(data))
        return ret

    def teacher_batchify_forward(self, data):
        num_networks, num_examples_per_network, num_channels = data[
            'batch_examples'].shape
        batch_examples = data['batch_examples'].reshape(-1, num_channels)
        N = len(batch_examples)
        if self.query_batch_size is not None:
            self.query_batch_size = N

        with torch.no_grad():
            for i in range(0, N, self.query_batch_size):
                # change data type to feed pretrained_nerf model
                datas = {'pts': batch_examples[i:i+self.query_batch_size,:3], \
                         'viewdirs': batch_examples[i:i+self.query_batch_size,3:6]}
                batch_examples[i:i + self.query_batch_size,
                               6:] = self.teacher_forward(datas)

        data['batch_examples'] = batch_examples.view(num_networks,
                                                     num_examples_per_network,
                                                     -1)
        return data

    def forward(self, data):
        data, ret = self.render(self.multi_network(data))
        return ret

    def batchify_forward(self, data):
        """forward in smaller batch_size to avoid OOM."""
        num_networks = data['target_s'].size(0)
        N = data['target_s'].size(1)

        if self.test_batch_size == 0:
            self.test_batch_size = N
        if self.outputs == 'color_and_density':
            num_output_channels = 4

        out = torch.empty(num_networks, N,
                          num_output_channels).to(data['target_s'])
        for i in range(0, N, self.test_batch_size):
            # prepare data
            data_chunk = {}
            for k in data:
                if data[k].shape[1] == N:
                    data_chunk[k] = data[k][:, i:i + self.test_batch_size]
                else:
                    data_chunk[k] = data[k]
            # run
            ret = self.forward(data_chunk)
            out[:, i:i + self.test_batch_size] = ret
        return out

    def train_step(self, data, optimizer, **kwargs):
        for k in data:
            data[k] = unfold_batching(data[k])
        data = self.teacher_batchify_forward(data)
        data = transform_examples(data)
        out = self.forward(data)

        loss = nn.functional.mse_loss(out, data['target_s'], reduction='none')
        loss = loss.mean(dim=2).mean(dim=1).sum()

        log_vars = {
            'sum_loss': loss.item(),
            'avg_loss': loss.item() / out.size(0)
        }
        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': out.size(1)
        }

        return outputs

    def val_step(self, data, **kwargs):
        rank, world_size = get_dist_info()
        if rank == 0:
            for k in data:
                data[k] = unfold_batching(data[k])
            data = self.teacher_batchify_forward(data)
            data = transform_examples(data)
            num_networks = data['target_s'].size(0)
            domain_mins = data['domain_mins']
            domain_maxs = data['domain_maxs']

            error_log = [
                '{} {}\n'.format(domain_mins[network_index].cpu().tolist(),
                                 domain_maxs[network_index].cpu().tolist())
                for network_index in range(num_networks)
            ]

            out = self.batchify_forward(data)
            outputs = {'out':out,  'target_s': data['target_s'], \
                       'test_points':data['test_points'], 'error_log':error_log}
        else:
            outputs = {}
        return outputs
