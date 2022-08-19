from mmcv import Config

from .helper import update_config, update_loadfrom
from .test import test_nerf
from .train import train_nerf

__all__ = ['run_nerf']


def run_nerf(args):
    cfg = Config.fromfile(args.config)
    cfg = update_config(dataname, cfg)
    cfg = update_loadfrom(args.load_from, cfg)
    if args.test_only or args.render_only:
        cfg['model']['cfg']['phase'] = 'test' if args.test_only else 'render'
        test_nerf(cfg)
    else:
        train_nerf(cfg)
