import warnings

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import EpochBasedRunner, get_dist_info, init_dist

from xrnerf.models.builder import build_network
from xrnerf.utils import get_root_logger

from .helper import build_dataloader, get_optimizer, get_runner, register_hooks


def test_nerf(cfg):
    """test model entry function.

    Args:
        cfg (dict): The config dict for test, the same config as train.
        the difference between test and val is:
                    in test phase, use 'EpochBasedRunner' to influence all testset, in one iter
                    in val phase, use 'IterBasedRunner' to influence 1/N testset, in one epoch (several iters)
    """
    cfg.workflow = [('val', 1)]  # only run val_step one epoch

    test_loader, testset = build_dataloader(cfg, mode='test')
    dataloaders = [test_loader]

    network = build_network(cfg.model)
    # network.set_val_pipeline(testset.pipeline)

    if cfg.distributed:
        print('init_dist...', flush=True)
        init_dist('slurm', **cfg.get('dist_param', {}))
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        network = MMDistributedDataParallel(
            network.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        network = MMDataParallel(network.cuda(), device_ids=[0])

    Runner = get_runner(cfg.test_runner)
    runner = Runner(network,
                    work_dir=cfg.work_dir,
                    logger=get_root_logger(log_level=cfg.log_level),
                    meta=None)
    runner.timestamp = cfg.get('timestamp', None)
    register_hooks(cfg.test_hooks, **locals())

    runner.load_checkpoint(
        cfg.load_from)  # for test phase, we must load checkpoint

    print('start test...', flush=True)
    runner.run(data_loaders=dataloaders, workflow=cfg.workflow, max_epochs=1)
