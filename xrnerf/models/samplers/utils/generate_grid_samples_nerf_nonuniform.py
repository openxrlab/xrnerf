import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _generate_grid_samples_nerf_nonuniform(Function):
    @staticmethod
    def forward(ctx, density_grid, n_elements, density_grid_ema_step,
                max_cascade, thresh, aabb_range, device):

        positions_uniform = torch.empty((n_elements, 3), dtype=torch.float32)
        indices_uniform = torch.empty((n_elements, ), dtype=torch.int32)
        positions_uniform = positions_uniform.to(device)
        indices_uniform = indices_uniform.to(device)

        assert positions_uniform.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert indices_uniform.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_grid.is_contiguous(), 'tensor must be contiguous!!!'

        # assert density_grid_indices.dtype in [torch.int32,], "tensor dtype error!"
        # assert density_grid_tmp.dtype in [torch.float32,], "tensor dtype error!"

        raymarch_cuda.generate_grid_samples_nerf_nonuniform_api(
            density_grid, int(density_grid_ema_step), int(n_elements),
            int(max_cascade), float(thresh), float(aabb_range[0]),
            float(aabb_range[1]), positions_uniform, indices_uniform)

        return positions_uniform, indices_uniform


generate_grid_samples_nerf_nonuniform = _generate_grid_samples_nerf_nonuniform.apply
