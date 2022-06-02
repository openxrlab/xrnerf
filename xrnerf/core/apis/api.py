
from mmcv import Config
from .train import train_nerf
from .test import test_nerf


__all__ = ['run_nerf']


def run_nerf(args):
    cfg = Config.fromfile(args.config)
    if args.test_only:
        cfg['model']['cfg']['phase'] = 'test'
        test_nerf(cfg)
    else:
        train_nerf(cfg)

