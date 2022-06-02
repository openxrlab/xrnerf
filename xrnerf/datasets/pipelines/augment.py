
import time
import torch
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class SelectRays:
    """random select rays when training
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, sel_n=1024, precrop_iters=0, precrop_frac=0.5, include_radius=False, **kwargs):
        self.enable = enable
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.kwargs = kwargs
        self.sel_n = sel_n
        self.include_radius = include_radius

    def __call__(self, results):
        """random select rays when training, support precrop
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            H, W, K = self.kwargs['H'], self.kwargs['W'], self.kwargs['K']
            if self.precrop_iters!=0 and results['iter_n']<self.precrop_iters:
                # print(results['iter_n'], "precrop now!", flush=True)
                # 在blender数据集的train时，靠前的iter，只取中间部分训练
                dH = int(H//2*self.precrop_frac)
                dW = int(W//2*self.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H//2 - dH, H//2 + dH - 1, 2*dH), 
                        torch.linspace(W//2 - dW, W//2 + dW - 1, 2*dW)
                    ), -1)
            else:
                coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1,2])  # (H * W, 2)
            
            np.random.seed(int(time.time())) # fix a bug, for detials please ref to https://github.com/pytorch/pytorch/issues/5059
            
            select_inds = np.random.choice(coords.shape[0], size=[self.sel_n], replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            results['rays_o'] = results['rays_o'][select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            results['rays_d'] = results['rays_d'][select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            results['target_s'] = results['target_s'][select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            if self.include_radius: 
                results['radii'] = results['radii'][select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
        return results

    def __repr__(self):
        return "{}:random select rays when training".format(self.__class__.__name__)


@PIPELINES.register_module()
class PerturbZvals:
    """apply perturb to zvals
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get intervals between samples
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            z_vals = results['z_vals']
            # get intervals between samples
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(z_vals.device)
            results['z_vals'] = lower + (upper - lower) * t_rand
        return results

    def __repr__(self):
        return "{}:apply perturb to zvals".format(self.__class__.__name__)

