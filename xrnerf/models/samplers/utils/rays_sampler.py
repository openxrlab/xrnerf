import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _rays_sampler(Function):
    @staticmethod
    def forward(ctx, rays_o, rays_d, imgs_id, density_grid_bitfield, metadata,
                xforms, aabb_range, near_distance, cone_angle_constant,
                num_coords_elements, device):

        n_rays_per_batch = rays_o.shape[0]
        coords_out = torch.zeros((num_coords_elements, 7),
                                 dtype=torch.float32).to(device)
        rays_index = torch.zeros((n_rays_per_batch, 1),
                                 dtype=torch.int32).to(device)
        rays_numsteps = torch.zeros((n_rays_per_batch, 2),
                                    dtype=torch.int32).to(device)
        ray_numstep_counter = torch.zeros((2, ), dtype=torch.int32).to(device)

        assert rays_o.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_d.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_grid_bitfield.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert metadata.is_contiguous(), 'tensor must be contiguous!!!'
        assert xforms.is_contiguous(), 'tensor must be contiguous!!!'
        assert coords_out.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_index.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_numsteps.is_contiguous(), 'tensor must be contiguous!!!'
        assert ray_numstep_counter.is_contiguous(
        ), 'tensor must be contiguous!!!'

        assert rays_o.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert rays_d.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert imgs_id.dtype in [
            torch.int32,
        ], 'tensor dtype error!'
        assert density_grid_bitfield.dtype in [
            torch.uint8,
        ], 'tensor dtype error!'
        assert metadata.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert xforms.dtype in [
            torch.float32,
        ], 'tensor dtype error!'

        raymarch_cuda.rays_sampler_api(rays_o, rays_d, density_grid_bitfield,
                                       metadata, imgs_id, xforms,
                                       float(aabb_range[0]),
                                       float(aabb_range[1]),
                                       float(near_distance),
                                       float(cone_angle_constant), coords_out,
                                       rays_index, rays_numsteps,
                                       ray_numstep_counter)

        coords_out = coords_out.detach()
        rays_index = rays_index.detach()
        rays_numsteps = rays_numsteps.detach()
        ray_numstep_counter = ray_numstep_counter.detach()
        samples = ray_numstep_counter[1].item()
        coords_out = coords_out[:samples]

        return coords_out, rays_index, rays_numsteps, ray_numstep_counter


rays_sampler = _rays_sampler.apply
