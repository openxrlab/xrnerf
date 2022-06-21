# Copyright (c) OpenMMLab. All rights reserved.
import itertools

import kilonerf_cuda
import torch
import torch.nn.functional as F
from mmcv import Config
from torch import nn

from .multi_modules import MultiNetwork

from .. import builder
from ..builder import MLPS
from .base import BaseMLP

kilonerf_cuda.init_stream_pool(16)
kilonerf_cuda.init_magma()


@MLPS.register_module()
class KiloNerfMultiNetwork(BaseMLP):
    """KiloNerfMultiNetwork build the multi_netowrk for local distill."""
    def __init__(self,
                 num_networks,
                 alpha_rgb_initalization,
                 bias_initialization_method,
                 direction_layer_size,
                 hidden_layer_size,
                 late_feed_direction,
                 network_rng_seed,
                 nonlinearity_initalization,
                 num_hidden_layers,
                 num_output_channels,
                 refeed_position_index,
                 use_same_initialization_for_all_networks,
                 weight_initialization_method,
                 embedder=None,
                 embedder_dir=None):
        super().__init__()
        self.embedder = builder.build_embedder(embedder)

        self.init_multi_network(num_networks, alpha_rgb_initalization,
                                bias_initialization_method,
                                direction_layer_size, hidden_layer_size,
                                late_feed_direction, network_rng_seed,
                                nonlinearity_initalization, num_hidden_layers,
                                num_output_channels, refeed_position_index,
                                use_same_initialization_for_all_networks,
                                weight_initialization_method)

    def init_multi_network(self, num_networks, alpha_rgb_initalization,
                           bias_initialization_method, direction_layer_size,
                           hidden_layer_size, late_feed_direction,
                           network_rng_seed, nonlinearity_initalization,
                           num_hidden_layers, num_output_channels,
                           refeed_position_index,
                           use_same_initialization_for_all_networks,
                           weight_initialization_method):

        position_num_input_channels, direction_num_input_channels = self.embedder.get_embed_ch(
        )

        self.multi_network = MultiNetwork(
            num_networks,
            position_num_input_channels,
            direction_num_input_channels,
            num_output_channels,
            hidden_layer_size,
            num_hidden_layers,
            refeed_position_index,
            late_feed_direction,
            direction_layer_size,
            nonlinearity='relu',
            nonlinearity_initalization=nonlinearity_initalization,
            use_single_net=False,
            linear_implementation='bmm',
            use_same_initialization_for_all_networks=
            use_same_initialization_for_all_networks,
            network_rng_seed=network_rng_seed,
            weight_initialization_method=weight_initialization_method,
            bias_initialization_method=bias_initialization_method,
            alpha_rgb_initalization=alpha_rgb_initalization,
            use_hard_parameter_sharing_for_color=False,
            view_dependent_dropout_probability=-1,
            use_view_independent_color=False)
        return

    def forward(self, data):
        data = self.embedder(data)
        raw_output = self.multi_network(data['embedded'])
        data['raw'] = raw_output
        return data

    def get_single_network(self, network_index):
        single_network = self.multi_network.extract_single_network(
            network_index)
        return single_network
