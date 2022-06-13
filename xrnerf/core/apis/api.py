from mmcv import Config

from .helper import update_config
from .test import test_nerf
from .train import train_nerf

__all__ = ['run_nerf']


def run_nerf(args):
    dataname = args.dataname
    cfg = Config.fromfile(args.config)
    cfg = update_config(dataname, cfg)
    if args.test_only:
        cfg['model']['cfg']['phase'] = 'test'
        test_nerf(cfg)
    else:
        train_nerf(cfg)
