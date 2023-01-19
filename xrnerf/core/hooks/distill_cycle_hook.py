from functools import partial

import torch
import torch.nn as nn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import init_dist
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.iter_based_runner import IterLoader

from xrnerf.core.apis.helper import *
from xrnerf.core.hooks import *
from xrnerf.models.builder import build_network


@HOOKS.register_module()
class DistllCycleHook(Hook):
    """
    change dataloader and model by updating cfg info in distill phase
    Args:
        cfg (dict): The config dict of distill
    """
    def __init__(self, cfg=None):
        assert cfg, f'cfg not input in {self.__name__}'
        self.cfg = cfg

    def before_run(self, runner):
        """DistllCycleHook."""
        if self.cfg.total_num_networks % self.cfg.max_num_networks == 0:
            runner._max_iters = (
                self.cfg.total_num_networks //
                self.cfg.max_num_networks) * self.cfg.max_iters
        else:
            runner._max_iters = (
                self.cfg.total_num_networks // self.cfg.max_num_networks +
                1) * self.cfg.max_iters
        print('max_iters:', runner._max_iters)

    def after_val_iter(self, runner):
        """DistllCycleHook."""
        if (runner.iter % self.cfg.max_iters
                == 0) and runner.iter < runner._max_iters:
            print('current iter:', runner.iter)
            index = runner.iter // self.cfg.max_iters
            self._update_train_distill_cyle(runner, index)

    def _update_train_distill_cyle(self, runner, index):
        """DistllCycleHook."""
        self.cfg.data['train'].cfg.update({'batch_index': index})
        train_loader, trainset = build_dataloader(self.cfg, mode='train')

        self.cfg.data['val'].cfg.update({'batch_index': index})
        val_loader, valset = build_dataloader(self.cfg, mode='val')

        # update data_loaders
        print('reload dataloader...')
        # runner.data_loaders = [train_loader, val_loader]
        runner.iter_loaders = [
            IterLoader(train_loader),
            IterLoader(val_loader)
        ]

        datas = trainset.get_info()
        self.cfg.update({'num_networks': len(datas['node_batch'])})
        self.cfg.model.multi_network.update(
            {'num_networks': len(datas['node_batch'])})
        self.cfg.model.multi_network.embedder.update(
            {'num_networks': len(datas['node_batch'])})

        nerf_net = build_network(self.cfg.model)

        # update optimizer
        if datas['processing_saturated_nodes'] == True:
            self.cfg.optimizer.update({'lr': 0.0001})
        # print(self.cfg.optimizer)
        optimizer = get_optimizer(nerf_net, self.cfg)
        print('reload the optimizer ...')
        runner.optimizer = optimizer

        if self.cfg.distributed:
            print('init_dist...', flush=True)
            init_dist('slurm', **self.cfg.get('dist_param', {}))
            find_unused_parameters = self.cfg.get('find_unused_parameters',
                                                  False)
            nerf_net = MMDistributedDataParallel(
                nerf_net.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
        else:
            nerf_net = MMDataParallel(nerf_net.cuda(), device_ids=[0])
        # update model
        print('rebuild model...')
        runner.model = nerf_net

        # update the SaveDistillResultsHook using new model and data
        for index, hook in enumerate(runner.hooks):
            if isinstance(hook, SaveDistillResultsHook):
                new_hook = SaveDistillResultsHook(self.cfg, trainset)
                runner.hooks[index] = new_hook
