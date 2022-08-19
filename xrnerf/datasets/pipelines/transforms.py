import mmcv
import numpy as np
import torch

from ..builder import PIPELINES
from ..utils import get_rigid_transformation


@PIPELINES.register_module()
class ToNDC:
    """use normalized device coordinates
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        self.H = kwargs['H']
        self.W = kwargs['W']
        self.K = kwargs['K']

    def __call__(self, results):
        """use normalized device coordinates
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            results['rays_o'], results['rays_d'] = self.ndc_rays(self.H, self.W, self.K[0][0], \
                1., results['rays_o'], results['rays_d'])
        return results

    def ndc_rays(self, H, W, focal, near, rays_o, rays_d):
        # Shift ray origins to near plane
        t = -(near + rays_o[..., 2]) / rays_d[..., 2]
        rays_o = rays_o + t[..., None] * rays_d
        # Projection
        o0 = -1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
        o1 = -1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
        o2 = 1. + 2. * near / rays_o[..., 2]
        d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] -
                                         rays_o[..., 0] / rays_o[..., 2])
        d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] -
                                         rays_o[..., 1] / rays_o[..., 2])
        d2 = -2. * near / rays_o[..., 2]
        rays_o = torch.stack([o0, o1, o2], -1)
        rays_d = torch.stack([d0, d1, d2], -1)
        return rays_o, rays_d

    def __repr__(self):
        return '{}:use normalized device coordinates'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class FlattenRays:
    """change rays from (H, W, ..) to (H*W, ...)
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, include_radius=False, **kwargs):
        self.enable = enable
        self.include_radius = include_radius

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            # 测试模式下，rays_d和rays_o本来是(h,w,..)的，需要变成(h*w,...)网络才能处理
            # [..., 3] 记录一下，最后reshape test的rays
            src_shape = results['rays_d'].shape
            results['rays_o'] = torch.reshape(results['rays_o'],
                                              [-1, 3]).float()
            results['rays_d'] = torch.reshape(results['rays_d'],
                                              [-1, 3]).float()
            if self.include_radius:
                results['radii'] = torch.reshape(results['radii'],
                                                 [-1, 1]).float()
            results['src_shape'] = torch.tensor(src_shape)
        return results

    def __repr__(self):
        return '{}:change rays from (H, W, ..) to (H*W, ...)'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class CalculateSkelTransf:
    """Calculate skeletal transformation
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            smpl_pose = results['smpl_pose']
            joints = results['joints']
            parents = results['parents']
            # calculate the skeleton transformation
            smpl_pose = smpl_pose.reshape(-1, 3)
            A = get_rigid_transformation(smpl_pose, joints, parents)
            results['A'] = A
        return results

    def __repr__(self):
        return '{}:calculate the skeletal transformation'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class AninerfIdxConversion:
    """Convert latent index to indices of blend weight and color
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            results['bw_latent_idx'] = results['latent_idx'].copy()
            results['color_latent_idx'] = results['latent_idx'].copy()
            if results['cfg'].phase == 'novel_pose':
                results['color_latent_idx'][:] = 0
        return results

    def __repr__(self):
        return '{}:convert the aninerf index'.format(self.__class__.__name__)
