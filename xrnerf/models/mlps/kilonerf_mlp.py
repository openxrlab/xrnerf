# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import time

import kilonerf_cuda
import torch
import torch.nn.functional as F
from mmcv import Config
from torch import nn

from .multi_modules import MultiNetwork, extract_linears
from xrnerf.models.networks.utils.transforms import reorder_points_and_dirs

from .. import builder
from ..builder import MLPS
from .base import BaseMLP

kilonerf_cuda.init_stream_pool(16)
kilonerf_cuda.init_magma()


@MLPS.register_module()
class KiloNerfMLP(BaseMLP):
    """KiloNerfMLP uses the distilled_checkpoint to build the multi_network."""
    def __init__(self,
                 resolution=None,
                 distilled_config=None,
                 occupancy_checkpoint=None,
                 distilled_checkpoint=None,
                 embedder=None):
        super().__init__()
        self.resolution = resolution
        self.distilled_config = Config.fromfile(distilled_config)

        self.embedder = builder.build_embedder(embedder)
        self.occupancy_grid = self.load_occupancy_grid(occupancy_checkpoint)
        self.init_mlp(distilled_checkpoint)

    def init_mlp(self, distilled_checkpoint):
        # Checkpoint loading
        cp = torch.load(distilled_checkpoint)

        root_nodes = cp['root_nodes']
        # Merging individual networks into multi network for efficient inference
        single_networks = []
        domain_mins, domain_maxs = [], []
        nodes_to_process = root_nodes.copy()
        for node in nodes_to_process:
            if hasattr(node, 'network'):
                node.network_index = len(single_networks)
                single_networks.append(node.network)
                domain_mins.append(node.domain_min)
                domain_maxs.append(node.domain_max)
            else:
                nodes_to_process.append(node.leq_child)
                nodes_to_process.append(node.gt_child)

        self.domain_mins = torch.tensor(domain_mins)
        self.domain_maxs = torch.tensor(domain_maxs)
        linear_implementation = 'multimatmul_differentiable'
        num_networks = len(single_networks)
        p = single_networks[0]
        try:
            use_hard_parameter_sharing_for_color = p.use_hard_parameter_sharing_for_color
        except AttributeError:
            use_hard_parameter_sharing_for_color = False

        try:
            use_view_independent_color = p.use_view_independent_color
        except AttributeError:
            use_view_independent_color = False

        # The initialization parameters do not need to be passed, because weights are overwritten anyhow
        self.multi_network = MultiNetwork(
            num_networks,
            p.num_position_channels,
            p.num_direction_channels,
            p.num_output_channels,
            p.hidden_layer_size,
            p.num_hidden_layers,
            p.refeed_position_index,
            p.late_feed_direction,
            p.direction_layer_size,
            p.nonlinearity,
            linear_implementation=linear_implementation,
            use_hard_parameter_sharing_for_color=
            use_hard_parameter_sharing_for_color,
            use_view_independent_color=use_view_independent_color)

        multi_linears, multi_shared_linears = extract_linears(
            self.multi_network)
        linears_per_network = [
            extract_linears(network) for network in single_networks
        ]
        num_linear_layers = len(multi_linears)
        num_linear_layers_shared = len(multi_shared_linears)
        transpose_weight = linear_implementation.startswith('multimatmul')
        with torch.no_grad():
            for layer_index in range(num_linear_layers):
                for network_index in range(self.multi_network.num_networks):
                    new_weight = linears_per_network[network_index][0][
                        layer_index].weight.data[0]
                    new_bias = linears_per_network[network_index][0][
                        layer_index].bias.data[0]
                    # new multimatmul implementation requires transposed weights: in_features x out_features
                    if transpose_weight:
                        new_weight = new_weight.t()
                        #new_bias = new_bias.t()
                    multi_linears[layer_index].weight.data[
                        network_index] = new_weight
                    multi_linears[layer_index].bias.data[
                        network_index] = new_bias

            for layer_index in range(num_linear_layers_shared):
                new_weight = linears_per_network[0][1][layer_index].weight.data
                new_bias = linears_per_network[0][1][layer_index].bias.data
                multi_shared_linears[layer_index].weight.data = new_weight
                multi_shared_linears[layer_index].bias.data = new_bias
        self.multi_network.activation = nn.ReLU(
            inplace=True
        )  # TODO: make sure that other activation functions are also inplace
        return

    def get_view_dependent_parameters(self):
        return self.multi_network.view_dependent_parameters

    def load_occupancy_grid(self, occupancy_checkpoint):
        return torch.load(occupancy_checkpoint).reshape(-1)

    def forward(self, data):
        num_rays = data['pts'].size(0)
        num_samples = data['pts'].size(1)

        self.domain_mins = self.domain_mins.to(data['pts'].device)
        self.domain_maxs = self.domain_maxs.to(data['pts'].device)

        num_networks = self.multi_network.num_networks
        fixed_res = [x // 16 for x in self.resolution]

        reorder_data = reorder_points_and_dirs(data, fixed_res,
                                               self.resolution,
                                               self.occupancy_grid,
                                               num_networks)

        num_points_to_process = reorder_data['points_reordered'].size(
            0) if reorder_data['points_reordered'].ndim > 0 else 0
        # print("#points to process:", num_points_to_process, flush=True)
        if num_points_to_process == 0:
            data['raw'] = torch.zeros(num_rays,
                                      num_samples,
                                      4,
                                      dtype=torch.float,
                                      device=data['pts'].device)
        else:
            # Convert global to local coordinates
            if not ('use_global_coordinates' in self.distilled_config
                    and self.distilled_config.use_global_coordinates):
                kilonerf_cuda.global_to_local(
                    reorder_data['points_reordered'], self.domain_mins,
                    self.domain_maxs, reorder_data['batch_size_per_network'],
                    1, 64)

            reorder_data = self.embedder(
                reorder_data,
                fourier_embedding_implementation='custom_kernel_v2')
            raw_outputs = self.multi_network(
                reorder_data['embedded'],
                reorder_data['batch_size_per_network'],
                random_directions=None)

            # Naive reordering is extremely fast even without any explicit measures to guarantee coherence => DeRF authors were telling lies
            raw_outputs_backordered = torch.empty_like(raw_outputs)
            raw_outputs_backordered[
                reorder_data['reorder_indices']] = raw_outputs
            #raw_outputs_backordered = kilonerf_cuda.scatter_int32_float4(reorder_indices, raw_outputs)
            raw_outputs_full = torch.zeros(
                num_rays * num_samples,
                4,
                dtype=torch.float,
                device=raw_outputs_backordered.device)
            raw_outputs_full[
                reorder_data['active_samples_mask']] = raw_outputs_backordered
            data['raw'] = raw_outputs_full.view(num_rays, num_samples, -1)
        return data
