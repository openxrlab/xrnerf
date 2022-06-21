# Copyright (c) OpenMMLab. All rights reserved.
import kilonerf_cuda
import torch
from torch import nn

from ..builder import EMBEDDERS


class MultiNetworkFourierEmbedding(nn.Module):
    def __init__(
        self,
        num_networks,
        num_input_channels,
        num_frequencies,
    ):
        super(MultiNetworkFourierEmbedding, self).__init__()

        max_frequency = num_frequencies - 1
        self.frequency_bands = 2.**torch.linspace(0.,
                                                  max_frequency,
                                                  steps=num_frequencies)
        self.num_frequencies = num_frequencies
        self.num_output_channels = (2 * num_frequencies +
                                    1) * num_input_channels
        self.num_networks = num_networks

    def forward(self,
                x,
                implementation='pytorch',
                num_blocks=46,
                num_threads=512):
        # x: num_networks x batch_size x num_input_channels
        batch_size, num_input_channels = x.size(1), x.size(2)
        if implementation == 'pytorch':
            x = x.unsqueeze(3).expand(
                self.num_networks, batch_size, num_input_channels,
                2 * self.num_frequencies + 1).contiguous()
            x[:, :, :, 1:1 + self.num_frequencies] = x[:, :, :, 0].unsqueeze(
                3) * self.frequency_bands.unsqueeze(0).unsqueeze(0).unsqueeze(
                    0).to(x)
            x[:, :, :,
              1 + self.num_frequencies:] = x[:, :, :,
                                             1:1 + self.num_frequencies]
            x[:, :, :, 1:1 + self.num_frequencies] = torch.cos(
                x[:, :, :, 1:1 + self.num_frequencies])
            x[:, :, :, 1 + self.num_frequencies:] = torch.sin(
                x[:, :, :, 1 + self.num_frequencies:])
        else:

            self.frequency_bands = self.frequency_bands.to(x)
            x = kilonerf_cuda.compute_fourier_features(x.contiguous().view(-1),
                                                       self.frequency_bands,
                                                       num_blocks, num_threads,
                                                       implementation)
        return x.view(self.num_networks, batch_size, -1)


@EMBEDDERS.register_module()
class KiloNerfFourierEmbedder(nn.Module):
    """KiloNerfFourierEmbedder is used to build multi_network."""
    def __init__(self,
                 num_networks,
                 multires=10,
                 multires_dirs=4,
                 input_ch=3,
                 **kwargs):
        super().__init__()
        # params of create_embedding_fn change，so MultiNetworkFourierEmbedder couldn't extend base class
        self.embed_fns, self.embed_ch = self.create_embedding_fn(
            num_networks, multires, input_ch=input_ch)
        self.embed_fns_dirs, self.embed_ch_dirs = self.create_embedding_fn(
            num_networks, multires_dirs, input_ch=input_ch)

    def create_embedding_fn(self, num_networks, multires, input_ch=3):
        fourier_embedding = MultiNetworkFourierEmbedding(
            num_networks, input_ch, multires)
        return fourier_embedding, fourier_embedding.num_output_channels

    def get_embed_ch(self):
        return self.embed_ch, self.embed_ch_dirs

    def forward(self, data, fourier_embedding_implementation='pytorch'):
        if fourier_embedding_implementation == 'pytorch':
            batch_positions = self.embed_fns(data['batch_positions'])
            batch_directions = self.embed_fns_dirs(data['batch_directions'])
            embedded = torch.cat((batch_positions, batch_directions), dim=2)
            data['embedded'] = embedded

        # for fast training in finetune phase
        elif fourier_embedding_implementation == 'custom_kernel_v2':
            embedded_points = self.embed_fns(
                data['points_reordered'].unsqueeze(0),
                implementation=fourier_embedding_implementation).squeeze(0)
            embedded_dirs = self.embed_fns_dirs(
                data['directions_reordered'].unsqueeze(0),
                implementation=fourier_embedding_implementation).squeeze(0)
            embedded = [embedded_points, embedded_dirs]
            del embedded_points
            del embedded_dirs
            data['embedded'] = embedded
        return data
