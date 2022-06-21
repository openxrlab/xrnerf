# @Author: fr
# @Date:   2022-05-12 17:05:14
# @Last Modified by:   fr
# @Last Modified time: 2022-05-17 13:07:52

import os
from collections import deque

import imageio
import kilonerf_cuda
import numpy as np
import torch
from mmcv.runner import get_dist_info
from mmcv.runner.hooks import HOOKS, Hook
from torch import nn

from xrnerf.utils.data_helper import Node, calculate_volume


def get_equal_error_split_threshold(test_points, errors, split_axis):
    """calculate the split threshold by equal error
    Args:
        test_points: x,y,z value of test examples
        errors: different types of errors
        split_axis: axis to split
    Return:
        split_threshold: split threshold has equal error
    """
    test_points = test_points.numpy()
    errors = errors.numpy()
    half_error_sum = np.sum(errors) / np.array(2.)
    points_sort = np.argsort(test_points[:, split_axis])
    split_threshold = test_points[points_sort][np.nonzero(
        np.cumsum(
            np.cumsum(errors[points_sort]) > half_error_sum) == 1)][0,
                                                                    split_axis]
    return split_threshold


def calculate_error_metrics(out, test_targets, cfg):
    """
    calculate mse/mae/mape/quantile_se of each network on testset
    Args:
        out: predict value of test
        test_targets: target value of test
        cfg (dict): the config dict of distill
    Return:
        errors_per_point: errors of per_point
        errors_per_network: errors of per_network
        errors_per_network_color: errors of per_network color
        errors_per_network_density: errors of per_network denesity
        saturation: detect whether get trapped in an all 0 or 1 state
    """
    # For a small fraction of networks/regions the RGB sigmoids get trapped in an all 0 or 1 state
    # We detect when this happens in order to retrain these networks with a smaller learning rate
    tolerance = 0.001
    close_to_zero = (torch.abs(out[:, :, :3] - torch.zeros_like(out[:, :, :3]))
                     < tolerance).all(dim=1)
    gt_close_to_zero = (torch.abs(test_targets[:, :, :3] -
                                  torch.zeros_like(test_targets[:, :, :3])) <
                        tolerance).all(dim=1)
    saturation_zero = torch.logical_and(
        close_to_zero, torch.logical_not(gt_close_to_zero)).any(dim=1)

    close_to_one = (torch.abs(out[:, :, :3] - torch.ones_like(out[:, :, :3])) <
                    tolerance).all(dim=1)
    gt_close_to_one = (torch.abs(test_targets[:, :, :3] -
                                 torch.ones_like(test_targets[:, :, :3])) <
                       tolerance).all(dim=1)
    saturation_one = torch.logical_and(
        close_to_one, torch.logical_not(gt_close_to_one)).any(dim=1)

    saturation = torch.logical_or(saturation_zero, saturation_one)

    errors, errors_per_point, errors_per_network, errors_per_network_color, errors_per_network_density = {}, {}, {}, {}, {}
    errors['mse'] = nn.functional.mse_loss(out, test_targets, reduction='none')
    errors['mae'] = torch.abs(out - test_targets)
    mape_epsilon = 0.1
    errors['mape'] = errors['mae'] / (torch.abs(test_targets) + mape_epsilon)

    for metric in ['mse', 'mape', 'mae']:
        errors_per_point[metric] = errors[metric].mean(dim=2)
        errors_per_network[metric] = errors_per_point[metric].mean(dim=1).cpu()
        if cfg.outputs == 'density':
            errors_per_network_density[metric] = errors_per_network[metric]
        if cfg.outputs == 'color_and_density':
            errors_per_network_color[metric] = errors[metric][:, :, :3].mean(
                dim=2).mean(dim=1).cpu()
            errors_per_network_density[metric] = errors[metric][:, :, 3].mean(
                dim=1).cpu()
        errors_per_point[metric] = errors_per_point[metric].cpu()

    def calcululate_quantile(se_per_point):
        num_test_samples = errors['mse'].size(1)
        quantile_index = int(num_test_samples * cfg.quantile_se)
        sorted_se_per_point = torch.sort(se_per_point, dim=1)[0]
        return sorted_se_per_point[:, quantile_index].cpu()

    errors_per_point[
        'quantile_se'] = None  # not really defined and this value should never be used
    errors_per_network['quantile_se'] = calcululate_quantile(
        errors['mse'].mean(dim=2))
    errors_per_network_color['quantile_se'] = calcululate_quantile(
        errors['mse'][:, :, :3].mean(dim=2))
    errors_per_network_density['quantile_se'] = calcululate_quantile(
        errors['mse'][:, :, 3])
    return errors_per_point, errors_per_network, errors_per_network_color, errors_per_network_density, saturation


def log_error_stats(initial_nodes, phase, cfg, filename):
    """
    traverse all root_nodes，log the best results of mse/mae/mape/quantile_se
    Args:
        initial_nodes: root nodes in checkpoint
        phase: discovery
        cfg (dict): the config dict of distill
        filename: log filename
    """
    domain_mins = []
    domain_maxs = []
    volumes = []
    best_errors = {}
    if cfg.outputs == 'color_and_density':
        best_errors_color = {}
        best_errors_density = {}
    for metric in ['mse', 'mae', 'mape', 'quantile_se']:
        best_errors[metric] = []
        if cfg.outputs == 'color_and_density':
            best_errors_color[metric] = []
            best_errors_density[metric] = []

    nodes_to_visit = deque(initial_nodes)
    while nodes_to_visit:
        node = nodes_to_visit.popleft()
        if hasattr(node, 'leq_child'):
            nodes_to_visit.append(node.leq_child)
            nodes_to_visit.append(node.gt_child)
        if (phase == 'discovery' and hasattr(node, 'discovery_best_error')
            ) or (phase == 'final' and hasattr(node, 'final_best_error')):
            domain_mins.append(node.domain_min)
            domain_maxs.append(node.domain_max)
            volumes.append(calculate_volume(node.domain_min, node.domain_max))
            for metric in ['mse', 'mae', 'mape', 'quantile_se']:
                if phase == 'discovery':
                    best_errors[metric].append(
                        node.discovery_best_error[metric])
                    if cfg.outputs == 'color_and_density':
                        best_errors_color[metric].append(
                            node.discovery_best_error_color[metric])
                        best_errors_density[metric].append(
                            node.discovery_best_error_density[metric])
                if phase == 'final':
                    best_errors[metric].append(node.final_best_error[metric])
                    if cfg.outputs == 'color_and_density':
                        best_errors_color[metric].append(
                            node.final_best_error_color[metric])
                        best_errors_density[metric].append(
                            node.final_best_error_density[metric])

    def write_log(prefix, domain_mins, domain_maxs, volumes, best_errors,
                  filename):
        best_errors = torch.tensor(best_errors)
        weighted_mean_error = (volumes * best_errors).sum() / volumes.sum()
        max_error_index = torch.argmax(best_errors)
        with open(filename, 'a') as log_file:
            log_file.write(
                '\t{} | weighted mean: {:.5f}, mean: {:.5f}, max: {} {} {:.5f}'
                .format(
                    prefix, weighted_mean_error.item(),
                    best_errors.mean().item(), domain_mins[max_error_index],
                    domain_maxs[max_error_index], best_errors[max_error_index])
                + '\n')

    if len(best_errors['mse']) > 0:
        volumes = torch.tensor(volumes)
        for metric in ['mse', 'mae', 'mape', 'quantile_se']:
            with open(filename, 'a') as log_file:
                log_file.write('[' + metric + ']')
            write_log('total', domain_mins, domain_maxs, volumes,
                      best_errors[metric], filename)
            if cfg.outputs == 'color_and_density':
                write_log('color', domain_mins, domain_maxs, volumes,
                          best_errors_color[metric], filename)
                write_log('density', domain_mins, domain_maxs, volumes,
                          best_errors_density[metric], filename)


@HOOKS.register_module()
class SaveDistillResultsHook(Hook):
    """
    postprocess the node batch according to the val results,
    and save distill results to checkpoint
    Args:
        cfg (dict): the config dict of distill
        trainset: train dataset
    """
    def __init__(self, cfg=None, trainset=None):
        assert cfg, f'cfg not input in {self.__name__}'
        assert trainset, f'cfg not input in {self.__name__}'
        self.cfg = cfg
        self.trainset = trainset

    def before_train_iter(self, runner):
        #init best_error before train step, and then update in the val step
        if (runner.iter % self.cfg.max_iters == 0):
            num_networks = self.cfg.num_networks
            self.best_errors_per_network, self.best_errors_per_network_color, self.best_errors_per_network_density = {}, {}, {}
            for metric in ['mse', 'mae', 'mape', 'quantile_se']:
                self.best_errors_per_network[metric] = float(
                    'inf') * torch.ones(num_networks)
                self.best_errors_per_network_color[metric] = float(
                    'inf') * torch.ones(num_networks)
                self.best_errors_per_network_density[metric] = float(
                    'inf') * torch.ones(num_networks)

    def after_val_iter(self, runner):
        rank, _ = get_dist_info()
        if rank == 0:
            cur_iter = runner.iter
            out = runner.outputs['out']
            target_s = runner.outputs['target_s']
            self.error_log = runner.outputs['error_log']

            errors_per_point, errors_per_network, errors_per_network_color, errors_per_network_density, saturation=\
                calculate_error_metrics(out, target_s, self.cfg)

            #compare and save the best validation results
            for metric in ['mse', 'mae', 'mape', 'quantile_se']:
                self.best_errors_per_network[metric] = torch.min(
                    errors_per_network[metric],
                    self.best_errors_per_network[metric])
                if self.cfg.outputs == 'color_and_density':
                    self.best_errors_per_network_color[metric] = torch.min(
                        errors_per_network_color[metric],
                        self.best_errors_per_network_color[metric])
                    self.best_errors_per_network_density[metric] = torch.min(
                        errors_per_network_density[metric],
                        self.best_errors_per_network_density[metric])

            num_networks = len(self.error_log)
            for network_index in range(num_networks):
                self.error_log[
                    network_index] += 'network_index:{}, it: {} | '.format(
                        network_index, cur_iter)
                for metric in ['mse', 'mae', 'mape', 'quantile_se']:
                    self.error_log[
                        network_index] += metric + ': {:.5f} '.format(
                            errors_per_network[metric][network_index].item())
                    self.error_log[
                        network_index] += '(d: {:.5f}, c: {:.5f}) '.format(
                            errors_per_network_density[metric]
                            [network_index].item(),
                            errors_per_network_color[metric]
                            [network_index].item())
                if saturation[network_index]:
                    self.error_log[network_index] += ' [saturation detected]'
                self.error_log[network_index] += '\n'

            if cur_iter % self.cfg.max_iters == 0:
                test_points = runner.outputs['test_points']
                checkpoint_filename = runner.work_dir + '/checkpoint.pth'

                datas = self.trainset.get_info()
                cp = datas['cp']
                processing_saturated_nodes = datas[
                    'processing_saturated_nodes']
                node_batch = datas['node_batch']

                num_networks = len(node_batch)
                num_networks_below_threshold = 0
                for network_index in range(num_networks):
                    split_further = not ('stop_after_one_iteration'
                                         in self.cfg)
                    if 'test_error_metric_color' in self.cfg:  # use different metric for density and color
                        split_further = split_further and (self.best_errors_per_network_color[self.cfg.test_error_metric_color][network_index] > self.cfg.max_error_color or\
                            self.best_errors_per_network_density[self.cfg.test_error_metric_density][network_index] > self.cfg.max_error_density)
                    else:  # use same metric for density and color
                        split_further = split_further and self.best_errors_per_network[
                            self.cfg.test_error_metric][
                                network_index] > self.cfg.max_error
                    if 'termination_volume' in self.cfg:
                        fitted_volume_ratio = cp['fitted_volume'] / cp[
                            'total_volume']
                        split_further = split_further and fitted_volume_ratio < self.cfg.termination_volume

                    #if nodes split further，the number of nodes_to_process will increase
                    if split_further:
                        if 'saturation_detection' in self.cfg and saturation[
                                network_index] and not processing_saturated_nodes:
                            cp['saturated_nodes_to_process'].append(
                                node_batch[network_index])
                        else:
                            if self.cfg.tree_type == 'kdtree_random':
                                split_axis = np.random.randint(low=0, high=3)
                            elif self.cfg.tree_type == 'kdtree_longest' or self.cfg.tree_type == 'kdtree_equal_error_split':
                                split_axis = np.argmax(
                                    np.array(
                                        node_batch[network_index].domain_max) -
                                    np.array(
                                        node_batch[network_index].domain_min))
                            node_batch[network_index].split_axis = split_axis

                            if self.cfg.tree_type == 'kdtree_equal_error_split':
                                node_batch[
                                    network_index].split_threshold = get_equal_error_split_threshold(
                                        test_points[network_index],
                                        errors_per_point[
                                            self.cfg.equal_split_metric]
                                        [network_index],
                                        node_batch[network_index].split_axis)

                            if self.cfg.tree_type == 'kdtree_random' or self.cfg.tree_type == 'kdtree_longest':
                                domain_min_coord = node_batch[
                                    network_index].domain_min[
                                        node_batch[network_index].split_axis]
                                domain_max_coord = node_batch[
                                    network_index].domain_max[
                                        node_batch[network_index].split_axis]
                                node_batch[
                                    network_index].split_threshold = domain_min_coord + (
                                        domain_max_coord -
                                        domain_min_coord) / 2

                            node_batch[network_index].leq_child = Node()
                            node_batch[network_index].gt_child = Node()

                            node_batch[
                                network_index].leq_child.domain_min = node_batch[
                                    network_index].domain_min.copy()
                            node_batch[
                                network_index].leq_child.domain_max = node_batch[
                                    network_index].domain_max.copy()
                            node_batch[network_index].leq_child.domain_max[
                                node_batch[network_index].
                                split_axis] = node_batch[
                                    network_index].split_threshold

                            node_batch[
                                network_index].gt_child.domain_min = node_batch[
                                    network_index].domain_min.copy()
                            node_batch[
                                network_index].gt_child.domain_max = node_batch[
                                    network_index].domain_max.copy()
                            node_batch[network_index].gt_child.domain_min[
                                node_batch[network_index].
                                split_axis] = node_batch[
                                    network_index].split_threshold

                            if processing_saturated_nodes:
                                cp['saturated_nodes_to_process'].append(
                                    node_batch[network_index].leq_child)
                                cp['saturated_nodes_to_process'].append(
                                    node_batch[network_index].gt_child)
                            else:
                                cp['nodes_to_process'].append(
                                    node_batch[network_index].leq_child)
                                cp['nodes_to_process'].append(
                                    node_batch[network_index].gt_child)
                    else:
                        num_networks_below_threshold += 1
                        cp['fitted_volume'] += calculate_volume(
                            node_batch[network_index].domain_min,
                            node_batch[network_index].domain_max)
                        node_batch[network_index].discovery_best_error = {}
                        node_batch[
                            network_index].discovery_best_error_color = {}
                        node_batch[
                            network_index].discovery_best_error_density = {}
                        for metric in ['mse', 'mae', 'mape', 'quantile_se']:
                            node_batch[network_index].discovery_best_error[
                                metric] = self.best_errors_per_network[metric][
                                    network_index]
                            node_batch[network_index].discovery_best_error_color[
                                metric] = self.best_errors_per_network_color[
                                    metric][network_index]
                            node_batch[network_index].discovery_best_error_density[
                                metric] = self.best_errors_per_network_density[
                                    metric][network_index]
                        node_batch[
                            network_index].network = runner.model.module.multi_network.get_single_network(
                                network_index)
                    #del node_batch[network_index].examples
                cp['num_networks_fitted'] += num_networks_below_threshold

                runner.logger.info('detected saturated networks: {}'.format(
                    saturation.sum().item()))
                runner.logger.info(
                    'num networks below threshold: {}/{}'.format(
                        num_networks_below_threshold, num_networks))
                runner.logger.info(
                    'fitted volume: {}/{} ({}%), num networks fitted: {}'.
                    format(cp['fitted_volume'], cp['total_volume'],
                           100 * cp['fitted_volume'] / cp['total_volume'],
                           cp['num_networks_fitted']))
                runner.logger.info('number of nodes_to_process: {}'.format(
                    len(cp['nodes_to_process'])))

                # save metrics and error status of each network into log.txt
                with open(os.path.join(runner.work_dir, 'log.txt'),
                          'a') as log_file:
                    log_file.write('\n'.join(self.error_log))
                log_error_stats(cp['root_nodes'], 'discovery', self.cfg,
                                os.path.join(runner.work_dir, 'log.txt'))

                torch.save(cp, checkpoint_filename)
                runner.logger.info('Saved to {}'.format(checkpoint_filename))
                # check all nodes have been processed
                all_nodes_processed = len(cp['nodes_to_process']) == 0 and len(
                    cp['saturated_nodes_to_process']) == 0
                if (not all_nodes_processed) and cur_iter == runner._max_iters:
                    runner._max_iters += self.cfg.max_iters
            else:
                pass
