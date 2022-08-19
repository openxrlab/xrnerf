import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _compacted_coords(Function):
    @staticmethod
    def forward(ctx, network_output, coords_in, rays_numsteps,
                compacted_elements, aabb_range, rgb_activation,
                density_activation, device):

        coords_out = torch.zeros((compacted_elements, 7),
                                 dtype=torch.float32).to(device)
        rays_numsteps_compacted = torch.zeros_like(
            rays_numsteps, dtype=torch.int32).to(device)
        compacted_rays_counter = torch.zeros((1, ),
                                             dtype=torch.int32).to(device)
        compacted_numstep_counter = torch.zeros((1, ),
                                                dtype=torch.int32).to(device)

        # activation 0:None 1:relu 2:sigmoid 3:exp
        # rgb_activation = int(2)
        # density_activation = int(3)
        bg_color_cpu = torch.tensor([1, 1, 1], dtype=torch.float32).cpu()

        assert network_output.is_contiguous(), 'tensor must be contiguous!!!'
        assert coords_in.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_numsteps.is_contiguous(), 'tensor must be contiguous!!!'
        assert bg_color_cpu.is_contiguous(), 'tensor must be contiguous!!!'
        assert coords_out.is_contiguous(), 'tensor must be contiguous!!!'
        assert rays_numsteps_compacted.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert compacted_rays_counter.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert compacted_numstep_counter.is_contiguous(
        ), 'tensor must be contiguous!!!'

        assert network_output.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert coords_in.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert rays_numsteps.dtype in [
            torch.int32,
        ], 'tensor dtype error!'

        raymarch_cuda.compacted_coord_api(
            network_output, coords_in, rays_numsteps, bg_color_cpu,
            rgb_activation, density_activation, aabb_range[0], aabb_range[1],
            coords_out, rays_numsteps_compacted, compacted_rays_counter,
            compacted_numstep_counter)

        return coords_out, rays_numsteps_compacted, compacted_numstep_counter


compacted_coords = _compacted_coords.apply
