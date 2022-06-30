# @Author: fr
# @Date:   2022-05-04 17:05:14
# @Last Modified by:   fr
# @Last Modified time: 2022-05-12 20:46:46

import os

import imageio
try:
    import kilonerf_cuda
except:
    print('Please install kilonerf_cuda for training KiloNeRF')
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from torch import nn

from xrnerf.utils.data_helper import get_global_domain_min_and_max


@HOOKS.register_module()
class BuildOccupancyTreeHook(Hook):
    """
    use the pretrained nerf model to build occupancy tree,
    save occupancy grid which will be used in finetune stage
    Args:
        cfg (dict): The config dict of pretraining
        occupancy_config (dict): The config dict for building occupancy tree
    """
    def __init__(self, cfg=None):
        assert cfg, f'cfg not input in {self.__name__}'
        self.cfg = cfg
        self.occupancy_config = cfg.build_occupancy_tree_config

    def after_run(self, runner):
        rank, _ = get_dist_info()
        if rank == 0:
            pretrained_nerf = runner.model.module.mlp

            global_domain_min, global_domain_max = get_global_domain_min_and_max(
                self.cfg, torch.device('cpu'))
            global_domain_size = global_domain_max - global_domain_min
            occupancy_res = self.occupancy_config.resolution
            total_num_voxels = occupancy_res[0] * occupancy_res[
                1] * occupancy_res[2]
            occupancy_resolution = torch.tensor(occupancy_res,
                                                dtype=torch.long,
                                                device=torch.device('cpu'))
            occupancy_voxel_size = global_domain_size / occupancy_resolution
            first_voxel_min = global_domain_min
            first_voxel_max = first_voxel_min + occupancy_voxel_size

            first_voxel_samples = []
            for dim in range(3):
                first_voxel_samples.append(
                    torch.linspace(
                        first_voxel_min[dim], first_voxel_max[dim],
                        self.occupancy_config.subsample_resolution[dim]))
            first_voxel_samples = torch.stack(
                torch.meshgrid(*first_voxel_samples), dim=3).view(-1, 3)

            ranges = []
            for dim in range(3):
                ranges.append(torch.arange(0, occupancy_res[dim]))
            index_grid = torch.stack(torch.meshgrid(*ranges), dim=3)
            index_grid = (index_grid * occupancy_voxel_size).unsqueeze(3)

            points = first_voxel_samples.unsqueeze(0).unsqueeze(0).unsqueeze(
                0).expand(occupancy_res + list(first_voxel_samples.shape))
            points = points + index_grid
            points = points.view(total_num_voxels, -1, 3)
            num_samples_per_voxel = points.size(1)

            mock_directions = torch.empty(min(
                self.occupancy_config.voxel_batch_size, total_num_voxels),
                                          3,
                                          device=torch.device('cuda'))

            # We query in a fixed grid at a higher resolution than the occupancy grid resolution to detect fine structures.
            all_densities = torch.empty(total_num_voxels,
                                        num_samples_per_voxel)
            end = 0
            while end < total_num_voxels:
                runner.logger.info('sampling network: {}/{} ({:.4f}%)'.format(
                    end, total_num_voxels, 100 * end / total_num_voxels))
                start = end
                end = min(start + self.occupancy_config.voxel_batch_size,
                          total_num_voxels)
                actual_batch_size = end - start
                points_subset = points[start:end].to(
                    mock_directions).contiguous(
                    )  # voxel_batch_size x num_samples_per_voxel x 3
                mock_directions_subset = mock_directions[:actual_batch_size]
                density_dim = 3
                with torch.no_grad():
                    mock_directions_subset = mock_directions_subset.unsqueeze(
                        1).expand(points_subset.size())
                    # points_and_dirs = torch.cat([points_subset.reshape(-1, 3), mock_directions_subset.reshape(-1, 3)], dim=-1)
                    # change data type to feed pretrained_nerf model
                    points_and_dirs = {
                        'pts': points_subset.reshape(-1, 3),
                        'viewdirs': mock_directions_subset.reshape(-1, 3)
                    }
                    ret = pretrained_nerf(points_and_dirs)
                    result = ret['raw'][:, density_dim].view(
                        actual_batch_size, -1)
                    all_densities[start:end] = result.cpu()

            occupancy_grid = all_densities.to(
                mock_directions) > self.occupancy_config.threshold

            occupancy_grid = occupancy_grid.view(
                self.occupancy_config.resolution + [-1])
            occupancy_grid = occupancy_grid.any(
                dim=3
            )  # checks if any point in the voxel is above the threshold

            runner.logger.info(
                '{} out of {} voxels are occupied. {:.2f}%'.format(
                    occupancy_grid.sum().item(), occupancy_grid.numel(), 100 *
                    occupancy_grid.sum().item() / occupancy_grid.numel()))
            os.makedirs(self.occupancy_config.work_dir, exist_ok=True)
            occupancy_filename = self.occupancy_config.work_dir + '/occupancy.pth'
            torch.save(occupancy_grid, occupancy_filename)
            runner.logger.info(
                'Saved occupancy grid to {}'.format(occupancy_filename))
