import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _ema_grid_samples_nerf(Function):
    @staticmethod
    def forward(ctx, density_grid_tmp, density_grid, n_elements, decay):

        assert density_grid_tmp.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_grid.is_contiguous(), 'tensor must be contiguous!!!'

        assert density_grid_tmp.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert density_grid.dtype in [
            torch.float32,
        ], 'tensor dtype error!'

        raymarch_cuda.ema_grid_samples_nerf_api(density_grid_tmp,
                                                int(n_elements), float(decay),
                                                density_grid)

        return density_grid


ema_grid_samples_nerf = _ema_grid_samples_nerf.apply
