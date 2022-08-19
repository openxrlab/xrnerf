import os

import numpy as np
import torch
from torch.autograd import Function

try:
    import raymarch_cuda
except Exception as e:
    print('please build extensions/ngp_raymarch for NGPGridSampler')


class _mark_untrained_density_grid(Function):
    @staticmethod
    def forward(ctx, focal_lengths, transforms, n_elements, n_img, resolutions,
                device):

        density_grid = torch.empty((n_elements, ), dtype=torch.float32)
        density_grid = density_grid.to(device)

        assert focal_lengths.is_contiguous(), 'tensor must be contiguous!!!'
        assert transforms.is_contiguous(), 'tensor must be contiguous!!!'
        assert density_grid.is_contiguous(), 'tensor must be contiguous!!!'
        assert focal_lengths.dtype in [
            torch.float32,
        ], 'tensor dtype error!'
        assert transforms.dtype in [
            torch.float32,
        ], 'tensor dtype error!'

        raymarch_cuda.mark_untrained_density_grid_api(focal_lengths, transforms, \
            int(n_elements), int(n_img), int(resolutions[0]), int(resolutions[1]), \
            density_grid)

        return density_grid


mark_untrained_density_grid = _mark_untrained_density_grid.apply
