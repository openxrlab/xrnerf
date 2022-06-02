
import torch
import warnings
from mmcv.runner import init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import (EpochBasedRunner, get_dist_info)

from xrnerf.utils import *
from xrnerf.core.hooks import *
from xrnerf.models.builder import build_network
from .helper import build_dataloader, get_optimizer


def test_nerf(cfg):
    """test model entry function.
    Args:
        cfg (dict): The config dict for test, the same config as train.
        the difference between test and val is: 
                    in test phase, use 'EpochBasedRunner' to influence all testset, in one iter
                    in val phase, use 'IterBasedRunner' to influence 1/N testset, in one epoch (several iters)
    """
    cfg.workflow = [('val', 1)] # only run val_step one epoch

    if cfg.method=='kilo_nerf':
        return test_kilonerf(cfg)

    test_loader, testset = build_dataloader(cfg, mode='test')
    dataloaders = [test_loader]
    
    nerf_net = build_network(cfg.model)
    nerf_net.set_val_pipeline(testset.pipeline)

    if cfg.distributed:
        print("init_dist...", flush=True)
        init_dist('slurm', **cfg.get('dist_param', {}))
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        nerf_net = MMDistributedDataParallel(
            nerf_net.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        nerf_net = MMDataParallel(nerf_net.cuda(), device_ids=[0])
    
    print("EpochBasedRunner...", flush=True)
    logger = get_root_logger(log_level=cfg.log_level)
    timestamp = cfg.get('timestamp', None)
    runner = EpochBasedRunner(nerf_net, 
                    work_dir=cfg.work_dir,
                    logger=logger,
                    meta=None)
    runner.timestamp = timestamp

    runner.register_hook(CalTestMetricsHook(cfg.evalute_config))
    
    runner.load_checkpoint(cfg.load_from) # for test phase, we must load checkpoint

    print("start test...", flush=True)
    runner.run(data_loaders=dataloaders, workflow=cfg.workflow, max_epochs=1)

def test_kilonerf(cfg):
    pass
