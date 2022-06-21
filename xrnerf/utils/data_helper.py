import os

import numpy as np
import torch


class Node:
    """Node is used in distill phase."""
    def __init__(self):
        pass


def calculate_volume(domain_min, domain_max):
    """
    calculate volume by domain_min and domain_max
    Args:
        domain_min: min value of domain
        domain_max: max value of domain
    """
    return (domain_max[0] - domain_min[0]) * (
        domain_max[1] - domain_min[1]) * (domain_max[2] - domain_min[2])


def load_matrix(path):
    """
    load matrix from txt file path
    Args:
        path: txt file path
    """
    return np.array([[float(w) for w in line.strip().split()]
                     for line in open(path)],
                    dtype=np.float32)


def get_global_domain_min_and_max(cfg, device=None):
    """
    get global_domain_min and global_domain_max
    Args:
        cfg (dict): the config dict of dataset
        device: cpu or cuda
    """
    if 'global_domain_min' in cfg and 'global_domain_max' in cfg:
        global_domain_min = cfg['global_domain_min']
        global_domain_max = cfg['global_domain_max']
    elif 'datadir' in cfg and cfg.dataset_type == 'nsvf':
        bbox_path = os.path.join(cfg.datadir, 'bbox.txt')
        bounding_box = load_matrix(bbox_path)[0, :-1]
        global_domain_min = bounding_box[:3]
        global_domain_max = bounding_box[3:]
    result = global_domain_min, global_domain_max
    if device:
        result = [
            torch.tensor(x, dtype=torch.float, device=device) for x in result
        ]
    return result
