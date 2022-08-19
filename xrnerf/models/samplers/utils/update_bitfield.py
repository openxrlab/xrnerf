import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _update_bitfield(Function):
    @staticmethod
    def forward(ctx, density_grid, density_grid_mean, density_grid_bitfield,
                device):

        density_grid_mean = density_grid_mean.to(device)
        density_grid_bitfield = density_grid_bitfield.to(device)
        density_grid_mean.zero_()

        assert density_grid_mean.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert density_grid_bitfield.is_contiguous(
        ), 'tensor must be contiguous!!!'
        assert density_grid.is_contiguous(), 'tensor must be contiguous!!!'

        assert density_grid.dtype in [torch.float32], 'tensor dtype error '
        assert density_grid_bitfield.dtype in [torch.uint8
                                               ], 'tensor dtype error '
        assert density_grid_mean.dtype in [torch.float32
                                           ], 'tensor dtype error '

        raymarch_cuda.update_bitfield_api(density_grid, density_grid_mean, \
            density_grid_bitfield)

        return density_grid_bitfield, density_grid_mean


update_bitfield = _update_bitfield.apply
