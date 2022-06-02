
import os
import torch
import warnings
from mmcv.runner import init_dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from mmcv.runner import (IterBasedRunner, get_dist_info)

from xrnerf.utils import *
from xrnerf.core.hooks import *
from xrnerf.models.builder import build_network
from .helper import build_dataloader, get_optimizer


def train_nerf(cfg):
    """Train model entry function.
    Args:
        cfg (dict): The config dict for training.
    """
    if cfg.method=='kilo_nerf':
        return train_kilonerf(cfg)

    train_loader, trainset = build_dataloader(cfg, mode='train')
    val_loader, valset = build_dataloader(cfg, mode='val')
    dataloaders = [train_loader, val_loader]
    
    nerf_net = build_network(cfg.model)
    nerf_net.set_val_pipeline(valset.pipeline)

    optimizer = get_optimizer(nerf_net, cfg)

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
    
    print("IterBasedRunner...", flush=True)
    logger = get_root_logger(log_level=cfg.log_level)
    timestamp = cfg.get('timestamp', None)
    runner = IterBasedRunner(nerf_net,
                    optimizer=optimizer,
                    work_dir=cfg.work_dir,
                    logger=logger,
                    meta=None)
    runner.timestamp = timestamp

    # register hooks
    print("register hooks...", flush=True)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    runner.register_hook(SaveTestHook(cfg.evalute_config))
    runner.register_hook(SaveSpiralHook(cfg.evalute_config))
    runner.register_hook(CalMetricsHook(cfg.evalute_config))
    runner.register_hook(PassIterHook()) # 将当前iter数告诉dataset
    # runner.register_hook(OccupationHook())  # no need for open-source vision
    
    # resume_from是载入ckpt和runner的训练信息，load_checkpoint只载入ckpt
    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None) and os.path.exists(cfg.load_from):
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()

    print("start train...", flush=True)
    runner.run(dataloaders, cfg.workflow, cfg.max_iters, **runner_kwargs)


def train_kilonerf(cfg):
    """Train model entry function. Some special codes for kilo-nerf 
    Args:
        cfg (dict): The config dict for training.
    """
    from xrnerf.core.runner import DistillCycleRunner

    set_random_seed(cfg.rng_seed)

    train_loader, trainset = build_dataloader(cfg, mode='train')
    val_loader, valset = build_dataloader(cfg, mode='val')
    dataloaders = [train_loader, val_loader]
    
    nerf_net = build_network(cfg.model)
    nerf_net.set_val_pipeline(valset.pipeline)

    optimizer = get_optimizer(nerf_net, cfg)

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
    
    print("IterBasedRunner...", flush=True)
    logger = get_root_logger(log_level=cfg.log_level)
    timestamp = cfg.get('timestamp', None)

    Runner = DistillCycleRunner if cfg.phase=='distill' else IterBasedRunner
    runner = Runner(nerf_net,
                    optimizer=optimizer,
                    work_dir=cfg.work_dir,
                    logger=logger,
                    meta=None)
    runner.timestamp = timestamp

    if cfg.checkpoint_config is not None:
        # save  torch_rng_state, torch_cuda_rng_state and numpy_rng_state in checkpoints as meta data
        torch_rng_state = torch.get_rng_state()
        torch_cuda_rng_state = torch.cuda.get_rng_state()
        numpy_rng_state = np.random.get_state()
        cfg.checkpoint_config.meta = dict(
            torch_rng_state=torch_rng_state,
            torch_cuda_rng_state=torch_cuda_rng_state,
            numpy_rng_state=numpy_rng_state)
        if cfg.get('rng_seed_fix', None):
            runner.logger.info('Saving rng state. torch: {}, torch cuda: {}, numpy: {}'.format(torch_rng_state.sum(), torch_cuda_rng_state.sum(), numpy_rng_state[1].sum()))

    # register hooks
    print("register hooks...", flush=True)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)
    runner.register_hook(OccupationHook())  # no need for open-source vision

    if cfg.model_type!='single_network':
        nerf_net.set_train_pipeline(trainset.pipeline)

    if cfg.phase=='distill':
        print("local distill phase ...")
        datas = trainset.get_datas()
        runner.register_hook(SaveDistillResultsHook(nerf_net, cfg, datas))
        distill_cycler = DistllCycleHook(cfg) 
        runner.register_hook(distill_cycler)
    else:
        runner.register_hook(SaveTestHook(cfg.evalute_config))
        runner.register_hook(SaveSpiralHook(cfg.evalute_config))
        runner.register_hook(CalMetricsHook(cfg.evalute_config))
        runner.register_hook(PassIterHook()) # 将当前iter数告诉dataset
        
    build_occupancy_tree_config = cfg.get('build_occupancy_tree_config', None)
    if build_occupancy_tree_config is not None and cfg.model_type=='single_network':
        build_occupancy_tree_hook = BuildOccupancyTreeHook(cfg)
        runner.register_hook(build_occupancy_tree_hook)

    # resume_from是载入ckpt和runner的训练信息，load_checkpoint只载入ckpt
    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        runner.load_checkpoint(cfg.load_from)
    runner_kwargs = dict()

    print("start train...", flush=True)
    runner.run(dataloaders, cfg.workflow, cfg.max_iters, **runner_kwargs)

