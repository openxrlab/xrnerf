
import torch
import numpy as np
from ..builder import PIPELINES



@PIPELINES.register_module()
class GetRays:
    """get rays from pose
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, include_radius=False, **kwargs):
        self.enable = enable
        self.kwargs = kwargs
        self.include_radius = include_radius

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            pose = results['pose']
            c2w = pose[:3,:4]
            device = pose.device
            H, W, K = self.kwargs['H'], self.kwargs['W'], self.kwargs['K']

            i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1).to(device)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3,-1].expand(rays_d.shape)

            results['rays_d'] = rays_d
            results['rays_o'] = rays_o

            if self.include_radius:
                # for mip nerf support
                dx = torch.sqrt(torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :])**2, -1))
                dx = torch.cat([dx, dx[-2:-1, :]], 0)
                results['radii'] = dx[..., None] * 2 / torch.sqrt(torch.tensor(12)).to(device)

        return results

    def __repr__(self):
        return "{}:get rays from pose and camera's params".format(self.__class__.__name__)


@PIPELINES.register_module()
class GetViewdirs:
    """get viewdirs from rays_d
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            # print(results.keys())
            viewdirs = results['rays_d'].clone()
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1,3]).float()
            results['viewdirs'] = viewdirs
        return results

    def __repr__(self):
        return "{}:get viewdirs from rays' direction".format(self.__class__.__name__)


@PIPELINES.register_module()
class GetBounds:
    """get near and far
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        # kwargs来自于dataset读取完毕后，记录的datainfo信息
        self.near = kwargs['near']
        self.far = kwargs['far']

    def __call__(self, results):
        """get bound(near and far)
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            results['near'] = self.near*torch.ones_like(results['rays_d'][...,:1])
            results['far'] = self.far*torch.ones_like(results['rays_d'][...,:1])
        return results

    def __repr__(self):
        return "{}:get bounds(near and far)".format(self.__class__.__name__)


@PIPELINES.register_module()
class GetZvals:
    """get intervals between samples
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, lindisp=False, N_samples=64, **kwargs):
        self.enable = enable
        self.lindisp = lindisp
        self.N_samples = N_samples

    def __call__(self, results):
        """get intervals between samples
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            device = results['rays_o'].device
            N_rays = results['rays_o'].shape[0]
            t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
            if not self.lindisp:
                z_vals = results['near']*(1.-t_vals)+results['far']*(t_vals)
            else:
                z_vals = 1./(1./results['near']*(1.-t_vals)+1./results['far']*(t_vals))
            results['z_vals'] = z_vals.expand([N_rays, self.N_samples])
        return results

    def __repr__(self):
        return "{}:get intervals between samples".format(self.__class__.__name__)


@PIPELINES.register_module()
class GetPts:
    """get pts
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            results['pts'] = results['rays_o'][...,None,:]+results['rays_d'][...,None,:]*results['z_vals'][...,:,None]
        return results

    def __repr__(self):
        return "{}:get viewdirs from rays' direction".format(self.__class__.__name__)


@PIPELINES.register_module()
class DeleteUseless:
    """delete useless params 
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, keys=[], **kwargs):
        self.enable = enable
        self.keys = keys

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            for k in self.keys:
                if k in results:
                    del results[k]
        return results

    def __repr__(self):
        return "{}:delete useless params".format(self.__class__.__name__)
