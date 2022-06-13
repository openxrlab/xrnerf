# @Author: zcy
# @Date:   2022-04-20 17:05:14
# @Last Modified by:   zcy
# @Last Modified time: 2022-06-08 10:28:13

import os

import imageio
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class PassIterHook(Hook):
    """思路来自于-- https://discuss.pytorch.org/t/pass-extra-arguments-to-
    getitem/100926/3
    https://github.com/ptrblck/pytorch_misc/blob/master/shared_array.py#L57
    通过这个hook，把iter传给train的dataset."""
    def __init__(self, cfg=None):
        self.cfg = cfg

    def after_train_iter(self, runner):
        # print(runner.iter, flush=True)
        runner.data_loader.iter_loader._dataset.set_iter(runner.iter)
        return


@HOOKS.register_module()
class OccupationHook(Hook):
    """GPU source occupation hook GPU cards are fucking hard to queue recently
    Don't blame on me, I need only one card."""
    def __init__(self, cfg=None):
        self.first_run = True

    def func(self, runner):
        flag_folder = os.path.join(runner.work_dir, 'delete_me_to_stop')
        if self.first_run:
            os.makedirs(flag_folder, exist_ok=True)
            self.first_run = False
        else:
            # deldete that folder if you want to stop
            if not os.path.exists(flag_folder):
                print('Stop now!!!', flush=True)
                exit(0)

    def after_train_iter(self, runner):
        self.func(runner)

    def after_val_iter(self, runner):
        self.func(runner)
