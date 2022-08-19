import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _splat_grid_samples_nerf_max_nearest_neighbor(Function):
    @staticmethod
    def forward(ctx, density_out, density_grid_indices, padded_output_width,
                n_density_grid_samples, density_grid_tmp, device):

        density_grid_tmp = density_grid_tmp.to(device)
        density_grid_tmp.zero_()

        assert density_grid_tmp.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_out.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_grid_indices.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert density_out.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert density_grid_indices.dtype in [
            torch.int32,
        ], 'tensor dtype error!'
        assert density_grid_tmp.dtype in [
            torch.float32,
        ], 'tensor dtype error!'

        raymarch_cuda.splat_grid_samples_nerf_max_nearest_neighbor_api(
            density_out, density_grid_indices, int(padded_output_width),
            int(n_density_grid_samples), density_grid_tmp)

        return density_grid_tmp


splat_grid_samples_nerf_max_nearest_neighbor = _splat_grid_samples_nerf_max_nearest_neighbor.apply
