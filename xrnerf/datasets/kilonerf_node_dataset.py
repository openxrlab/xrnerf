# Copyright (c) OpenMMLab. All rights reserved.

import itertools
from collections import deque

import numpy as np
import torch

from xrnerf.utils.data_helper import (Node, calculate_volume,
                                      get_global_domain_min_and_max)

from .builder import DATASETS
from .pipelines import Compose
from .scene_dataset import SceneBaseDataset


@DATASETS.register_module()
class KiloNerfNodeDataset(SceneBaseDataset):
    """KiloNerfNodeDataset for node data in distill phase, which uses the
    pretrained nerf model to predict examples."""
    def __init__(self, cfg, pipeline):
        super().__init__(cfg, pipeline)
        self._init_examples()

    def _init_load(self):
        batch_index = self.cfg.batch_index
        if batch_index == 0:
            print('batch_index:', batch_index)
            #init the cp
            self.cp = {}
            self.cp['fitted_volume'] = 0
            self.cp['num_networks_fitted'] = 0
            self.cp['phase'] = 'discovery'

            self.global_domain_min, self.global_domain_max = get_global_domain_min_and_max(
                self.cfg)
            if not 'fixed_resolution' in self.cfg:
                root_node = Node()
                root_node.domain_min = self.global_domain_min
                root_node.domain_max = self.global_domain_max
                self.cp['root_nodes'] = [root_node]
            else:
                self.cp['root_nodes'] = self.get_nodes_fixed_resolution(
                    self.cfg.fixed_resolution, self.global_domain_min,
                    self.global_domain_max)

            self.cp['nodes_to_process'] = deque(self.cp['root_nodes'])
            self.cp['saturated_nodes_to_process'] = deque([])
            self.cp['total_volume'] = calculate_volume(self.global_domain_min,
                                                       self.global_domain_max)

        else:
            print('batch_index:', batch_index)
            #load from the previous checkpoint
            checkpoint_filename = self.cfg.work_dir + '/checkpoint.pth'
            self.cp = torch.load(checkpoint_filename)
            print('load from the previous checkpoint!')

        if self.cp['nodes_to_process']:
            self.processing_saturated_nodes = False
            self.node_batch = [
                self.cp['nodes_to_process'].popleft() for _ in range(
                    min(self.cfg.max_num_networks,
                        len(self.cp['nodes_to_process'])))
            ]
        else:
            self.processing_saturated_nodes = True
            self.node_batch = [
                self.cp['saturated_nodes_to_process'].popleft() for _ in range(
                    min(self.cfg.max_num_networks,
                        len(self.cp['saturated_nodes_to_process'])))
            ]

    def _init_examples(self):
        num_networks = len(self.node_batch)
        self.all_examples = torch.empty(num_networks *
                                        self.cfg.num_examples_per_network,
                                        10)  # x,y,z,dir_x,dir_y,dir_z,r,g,b,a
        start = 0
        for network_index in range(num_networks):
            start = network_index * self.cfg.num_examples_per_network
            end = (network_index + 1) * self.cfg.num_examples_per_network
            self.all_examples[start:end, 0:3] = torch.tensor(
                self.get_random_points_inside_domain(
                    self.cfg.num_examples_per_network,
                    self.node_batch[network_index].domain_min,
                    self.node_batch[network_index].domain_max),
                dtype=torch.float)
            self.all_examples[start:end, 3:6] = torch.tensor(
                self.get_random_directions(self.cfg.num_examples_per_network),
                dtype=torch.float)
        self.all_examples = self.all_examples.view(
            num_networks, self.cfg.num_examples_per_network, -1)
        print('{} examples shape: {}'.format(self.cfg.mode,
                                             self.all_examples.shape))

        self.domain_mins = [
            self.node_batch[network_index].domain_min
            for network_index in range(num_networks)
        ]
        self.domain_maxs = [
            self.node_batch[network_index].domain_max
            for network_index in range(num_networks)
        ]

    def _init_pipeline(self, pipeline):
        self.pipeline = Compose(pipeline)

    def get_nodes_fixed_resolution(self, fixed_resolution, global_domain_min,
                                   global_domain_max):
        """
        get nodes according to fixed_resolution
        Args:
            fixed_resolution: a list, fix resolution of xyz axis
            global_domain_min: global min value of domain
            global_domain_max: global min value of domain
        Return:
            nodes: Nodes
        """
        fixed_resolution = np.array(fixed_resolution)
        global_domain_min = np.array(global_domain_min)
        global_domain_max = np.array(global_domain_max)
        voxel_size = (global_domain_max - global_domain_min) / fixed_resolution
        nodes = []
        for voxel_indices in itertools.product(*[
                range(axis_resolution) for axis_resolution in fixed_resolution
        ]):
            node = Node()
            node.domain_min = (global_domain_min +
                               voxel_indices * voxel_size).tolist()
            node.domain_max = (
                global_domain_min +
                (voxel_indices + np.array(1)) * voxel_size).tolist()
            nodes.append(node)
        return nodes

    def get_random_points_inside_domain(self, num_points, domain_min,
                                        domain_max):
        """
        generate random point btw domain_min and domain_max
        Args:
            num_points: number of points
            domain_min: min value of domain
            domain_max: max value of domain
        Return:
            points: points in x,y,z
        """
        x = np.random.uniform(domain_min[0],
                              domain_max[0],
                              size=(num_points, ))
        y = np.random.uniform(domain_min[1],
                              domain_max[1],
                              size=(num_points, ))
        z = np.random.uniform(domain_min[2],
                              domain_max[2],
                              size=(num_points, ))
        return np.column_stack((x, y, z))

    def get_random_directions(self, num_samples):
        """
        generate random directions
        Args:
            num_samples: number of samples
        Return:
            directions: random_directions
        """
        random_directions = np.random.randn(num_samples, 3)
        random_directions /= np.linalg.norm(random_directions,
                                            axis=1).reshape(-1, 1)
        return random_directions

    def get_info(self):
        res = {'cp':self.cp, 'processing_saturated_nodes':self.processing_saturated_nodes, \
               'node_batch':self.node_batch}
        return res

    def _fetch_train_data(self, idx):
        # indices = np.random.choice(self.cfg.num_examples_per_network, size=(self.cfg.train_batch_size,))
        # train_batch = self.all_examples[:, indices]
        # data = {'batch_examples':train_batch, 'domain_mins':self.domain_mins, 'domain_maxs':self.domain_maxs}
        data = {'all_examples':self.all_examples, \
                'domain_mins':self.domain_mins, 'domain_maxs':self.domain_maxs}
        return data

    def _fetch_val_data(self, idx):
        # for val mode, fetch all data in one time
        data = {'batch_examples':self.all_examples, \
                'domain_mins':self.domain_mins, 'domain_maxs':self.domain_maxs}
        # data = {'node_batch':self.node_batch}
        return data

    def _fetch_test_data(self, idx):
        #for test mode, fetch all data in one time
        data = {'batch_examples':self.all_examples, \
                'domain_mins':self.domain_mins, 'domain_maxs':self.domain_maxs}
        return data

    def __getitem__(self, idx):
        if self.mode == 'train':
            data = self._fetch_train_data(idx)
            data = self.pipeline(data)
            return data
        else:
            data = self._fetch_val_data(idx)
            data = self.pipeline(data)
            return data

    def __len__(self):
        if self.mode == 'train':
            return self.all_examples.shape[1] // self.cfg.train_batch_size
        else:
            return 1
