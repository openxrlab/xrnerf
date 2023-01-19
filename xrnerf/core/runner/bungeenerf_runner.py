import time
import warnings

import mmcv
import torch
from mmcv.runner import EpochBasedRunner, IterBasedRunner
from mmcv.runner.utils import get_host_info


class BungeeNerfTrainRunner(IterBasedRunner):
    def train(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._epoch = data_loader.epoch
        data_batch = next(data_loader)
        self.data_batch = data_batch
        scale_code = data_batch['scale_code']
        for stage in range(int(torch.max(scale_code) + 1)):
            kwargs['stage'] = stage
            self.call_hook('before_train_iter')
            outputs = self.model.train_step(data_batch, self.optimizer,
                                            **kwargs)
            if not isinstance(outputs, dict):
                raise TypeError('model.train_step() must return a dict')
            if 'log_vars' in outputs:
                if outputs['log_vars']['loss'] == 0.:
                    continue
                self.log_buffer.update(outputs['log_vars'],
                                       outputs['num_samples'])
                self.log_buffer.output['stage'] = stage
            self.outputs = outputs
            self.call_hook('after_train_iter')
        del self.data_batch
        self._inner_iter += 1
        self._iter += 1


class BungeeNerfTestRunner(EpochBasedRunner):
    """BungeeNerfTestRunner."""
    pass
