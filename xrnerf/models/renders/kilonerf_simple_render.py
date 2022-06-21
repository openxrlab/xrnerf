# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from torch import nn

from ..builder import RENDERS
from .nerf_render import NerfRender


@RENDERS.register_module()
class KiloNerfSimpleRender(NerfRender):
    def __init__(self, 
                 white_bkgd=False, 
                 raw_noise_std=0, 
                 rgb_padding=0, 
                 density_bias=0, 
                 density_activation='relu', 
                 convert_density_to_alpha=True, 
                 alpha_distance=0, 
                 **kwarg):
        super().__init__(white_bkgd=white_bkgd, 
                         raw_noise_std=raw_noise_std, 
                         rgb_padding=rgb_padding, 
                         density_bias=density_bias, 
                         density_activation=density_activation, 
                         **kwarg)
        self.convert_density_to_alpha = convert_density_to_alpha
        self.alpha_distance = alpha_distance

    def process_density(self, raw_output):
        if self.convert_density_to_alpha:
            if len(raw_output.shape) == 2:
                return (1. - torch.exp(-F.relu(raw_output[:, 3]) *
                                       self.alpha_distance)).unsqueeze(1)
            else:
                return (
                    1. - torch.exp(-F.leaky_relu(raw_output[:, :, 3]) *
                                   self.alpha_distance)
                ).unsqueeze(
                    2
                )  # Convert to alpha with typical distance encountered during training
        else:
            if len(raw_output.shape) == 2:
                return F.relu(raw_output[:, 3]).unsqueeze(1)
            else:
                return F.leaky_relu(raw_output[:, :, 3]).unsqueeze(
                    2)  # Only apply ReLU to density output

    def forward(self, data):
        """Transforms model's predictions to semantically meaningful values.

        Args:
            data: inputs
        Returns:
            ret: return values
        """
        raw = data['raw']

        if len(raw.shape) == 2:
            rgb = F.sigmoid(raw[:, 0:3])
            density = self.process_density(raw)
            ret = torch.cat((rgb, density), dim=1)
        else:
            rgb = F.sigmoid(raw[:, :, 0:3])
            density = self.process_density(raw)
            ret = torch.cat((rgb, density), dim=2)
        return data, ret
