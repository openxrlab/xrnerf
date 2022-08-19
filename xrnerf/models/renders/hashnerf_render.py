# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function

from ..builder import RENDERS
from .base import BaseRender

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


@RENDERS.register_module()
class HashNerfRender(BaseRender):
    def __init__(self, bg_color=None, **kwarg):
        super().__init__()
        self.bg_color = torch.tensor(bg_color).to(dtype=torch.float32)

    def forward(self, data, sampler, is_test=False):
        """Transforms model's predictions to semantically meaningful values.

        Args:
            data: inputs
            sampler: ngp sampler
            is_test: is_test
        Returns:
            rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
            ret: return values
        """
        network_output = data['raw']

        coords = sampler.coords
        aabb_range = sampler.aabb_range
        rays_numsteps = sampler.rays_numsteps
        density_grid_mean = sampler.density_grid_mean

        rgb_activation = int(sampler.rgb_activation)
        density_activation = int(sampler.density_activation)

        if is_test:
            rgb_output, alpha_output = calc_rgb_nobp(
                network_output, coords, rays_numsteps, self.bg_color,
                rgb_activation, density_activation, aabb_range)
            ret = {'rgb': rgb_output, 'alpha': alpha_output}
        else:
            bg_color = data['bg_color'].detach()
            rays_numsteps_compacted = sampler.rays_numsteps_compacted
            rgb_output = calc_rgb_bp(network_output, coords, rays_numsteps,
                                     rays_numsteps_compacted, bg_color,
                                     density_grid_mean, rgb_activation,
                                     density_activation, aabb_range)
            ret = {'rgb': rgb_output}

        return data, ret


class _calc_rgb_bp(Function):
    @staticmethod
    def forward(ctx, network_output, coords_in, rays_numsteps,
                rays_numsteps_compacted, training_background_color,
                density_grid_mean, rgb_activation, density_activation,
                aabb_range):

        coords_in = coords_in.detach()
        rays_numsteps = rays_numsteps.detach()
        rays_numsteps_compacted = rays_numsteps_compacted.detach()
        density_grid_mean = density_grid_mean.detach()

        n_rays_per_batch = rays_numsteps.shape[0]
        rgb_output = torch.zeros((n_rays_per_batch, 3),
                                 dtype=torch.float32).to(network_output.device)

        assert network_output.is_contiguous(), 'tensor must be contiguous!!!'
        assert coords_in.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_numsteps.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_numsteps_compacted.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert training_background_color.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert rgb_output.is_contiguous(), 'tensor must be contiguous!!!'

        assert network_output.dtype in [
            torch.float32,
        ], 'data type error!!!'
        assert coords_in.dtype in [
            torch.float32,
        ], 'data type error!!!'
        assert rays_numsteps.dtype in [torch.int32], 'data type error!!!'
        assert rays_numsteps_compacted.dtype in [torch.int32
                                                 ], 'data type error!!!'
        assert training_background_color.dtype in [torch.float32
                                                   ], 'data type error!!!'
        assert rgb_output.dtype in [torch.float32], 'data type error!!!'

        raymarch_cuda.calc_rgb_forward_api(network_output, coords_in,
                                           rays_numsteps,
                                           rays_numsteps_compacted,
                                           training_background_color,
                                           int(rgb_activation),
                                           int(density_activation),
                                           float(aabb_range[0]),
                                           float(aabb_range[1]), rgb_output)

        ctx.save_for_backward(network_output, rays_numsteps_compacted,
                              coords_in, rgb_output, density_grid_mean)
        ctx.extro = [rgb_activation, density_activation, aabb_range]

        return rgb_output

    @staticmethod
    def backward(ctx, grad_rgb_output):

        network_output, rays_numsteps_compacted, coords_in, rgb_output, \
            density_grid_mean = ctx.saved_tensors
        rgb_activation, density_activation, aabb_range = ctx.extro

        num_elements = network_output.shape[0]
        grad_network_output = torch.zeros((num_elements, 4),
                                          dtype=torch.float32)
        grad_network_output = grad_network_output.to(network_output.device)

        assert network_output.is_contiguous(), 'tensor must be contiguous!!!'
        assert coords_in.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_numsteps_compacted.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert rgb_output.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_grid_mean.is_contiguous(
        ), 'tensor must be contiguous!!!'

        raymarch_cuda.calc_rgb_backward_api(network_output,
                                            rays_numsteps_compacted, coords_in,
                                            grad_rgb_output, rgb_output,
                                            density_grid_mean,
                                            int(rgb_activation),
                                            int(density_activation),
                                            float(aabb_range[0]),
                                            float(aabb_range[1]),
                                            grad_network_output)

        return grad_network_output, None, None, None, None, None, None, None, None


calc_rgb_bp = _calc_rgb_bp.apply


class _calc_rgb_nobp(Function):
    @staticmethod
    def forward(ctx, network_output, coords_in, rays_numsteps, bg_color_cpu,
                rgb_activation, density_activation, aabb_range):
        '''
            bg_color_cpu: shape is (3,)
        '''

        coords_in = coords_in.detach()
        rays_numsteps = rays_numsteps.detach()

        n_rays_per_batch = rays_numsteps.shape[0]
        rgb_output = torch.zeros((n_rays_per_batch, 3),
                                 dtype=torch.float32).to(network_output.device)
        alpha_output = torch.zeros(
            (n_rays_per_batch, 1),
            dtype=torch.float32).to(network_output.device)

        raymarch_cuda.calc_rgb_influence_api(network_output, coords_in,
                                             rays_numsteps, bg_color_cpu.cpu(),
                                             rgb_activation,
                                             density_activation, aabb_range[0],
                                             aabb_range[1], rgb_output,
                                             alpha_output)

        return rgb_output.detach(), alpha_output.detach()


calc_rgb_nobp = _calc_rgb_nobp.apply
