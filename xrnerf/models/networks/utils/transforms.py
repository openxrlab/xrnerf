import numpy as np
import torch


def recover_shape(data, to_shape):
    # 对于测试数据，回复到(H, W, ...)的格式
    to_shape = list(to_shape[:-1]) + list(data.shape[1:])
    data = torch.reshape(data, to_shape)
    return data


def nb_recover_shape(data, to_shape, mask):
    num_data = torch.cumprod(to_shape[:-1], -1)[-1].item()
    to_shape = list(to_shape[:-1]) + list(data.shape[1:])
    if len(data.shape) > 1:
        full_data = torch.zeros([num_data, data.shape[-1]]).to(data)
    else:
        full_data = torch.zeros([num_data]).to(data)
    full_data[mask] = data
    full_data = full_data.view(to_shape)
    return full_data


def merge_ret(ret, fine_ret):
    ret['coarse_rgb'] = ret['rgb']
    ret['coarse_disp'] = ret['disp']
    ret['coarse_acc'] = ret['acc']

    ret['rgb'] = fine_ret['rgb']
    ret['disp'] = fine_ret['disp']
    ret['acc'] = fine_ret['acc']
    return ret


def convert_to_local_coords_multi(points, domain_mins, domain_maxs):
    converted_points = torch.empty_like(points)
    for i in [0, 1, 2]:
        # values between -1 and 1
        converted_points[:, :,
                         i] = 2 * (points[:, :, i] -
                                   domain_mins[:, i].unsqueeze(1)) / (
                                       domain_maxs[:, i].unsqueeze(1) -
                                       domain_mins[:, i].unsqueeze(1)) - 1
    return converted_points


def transform_examples(data):
    batch_positions = data['batch_examples'][:, :, 0:3]
    data['batch_positions'] = convert_to_local_coords_multi(
        batch_positions, data['domain_mins'], data['domain_maxs'])
    data['batch_directions'] = data['batch_examples'][:, :, 3:6]
    data['target_s'] = data['batch_examples'][:, :, 6:10]
    data['test_points'] = data['batch_examples'][:, :, :3]
    return data


def reorder_points_and_dirs(data, fixed_res, res, occupancy_grid,
                            num_networks):
    """
    reorder point and directions
    Args:
        fixed_res: fixed resolution of distill
        res: occupancy resolution
        occupancy_grid: occupancy grid
        num_networks: number of networks
    Return:
        data: reordered results
    """
    device = data['pts'].device
    points_flat = data['pts'].view(-1, 3)

    #get point indices
    fixed_resolution = torch.tensor(fixed_res, dtype=torch.long, device=device)
    network_strides = torch.tensor(
        [fixed_res[2] * fixed_res[1], fixed_res[2], 1],
        dtype=torch.long,
        device=device)  # assumes row major ordering
    global_domain_size = data['global_domain_max'] - data['global_domain_min']
    voxel_size = global_domain_size / fixed_resolution
    point_indices_3d = ((points_flat - data['global_domain_min']) /
                        voxel_size).to(network_strides)
    point_indices = (point_indices_3d * network_strides).sum(dim=1)

    # get point in occupied space
    # define a mapping to filter empty regions: 0 -> -1, 1 -> 1, 2 -> 2, 3 -> -1, 4 -> -1
    if occupancy_grid is not None:
        occupancy_resolution = torch.tensor(res,
                                            dtype=torch.long,
                                            device=device)
        strides = torch.tensor([res[2] * res[1], res[2], 1],
                               dtype=torch.long,
                               device=device)  # assumes row major ordering
        voxel_size = global_domain_size / occupancy_resolution
        occupancy_indices = ((points_flat - data['global_domain_min']) /
                             voxel_size).to(torch.long)
        torch.max(torch.tensor([0, 0, 0], device=device),
                  occupancy_indices,
                  out=occupancy_indices)
        torch.min(occupancy_resolution - 1,
                  occupancy_indices,
                  out=occupancy_indices)
        occupancy_indices = (occupancy_indices * strides).sum(dim=1)
        point_in_occupied_space = occupancy_grid[occupancy_indices]
        del occupancy_indices

    # Filtering points outside global domain
    epsilon = 0.001
    active_samples_mask = torch.logical_and(
        (points_flat > data['global_domain_min'] + epsilon).all(dim=1),
        (points_flat < data['global_domain_max'] - epsilon).all(dim=1))
    if occupancy_grid is not None:
        active_samples_mask = torch.logical_and(active_samples_mask,
                                                point_in_occupied_space)
        del point_in_occupied_space
    proper_index = torch.logical_and(
        point_indices >= 0, point_indices < num_networks
    )  # probably this is not needed if we check for points_flat <= global_domain_max
    active_samples_mask = torch.nonzero(torch.logical_and(
        active_samples_mask, proper_index),
                                        as_tuple=False).squeeze()
    data['active_samples_mask'] = active_samples_mask
    del proper_index

    filtered_point_indices = point_indices[active_samples_mask]
    del point_indices

    # Sort according to network
    filtered_point_indices, reorder_indices = torch.sort(
        filtered_point_indices)
    data['reorder_indices'] = reorder_indices

    # make sure that also batch sizes are given for networks which are queried 0 points
    contained_nets, batch_size_per_network_incomplete = torch.unique_consecutive(
        filtered_point_indices, return_counts=True)
    del filtered_point_indices
    batch_size_per_network = torch.zeros(num_networks,
                                         device=device,
                                         dtype=torch.long)
    batch_size_per_network[contained_nets] = batch_size_per_network_incomplete
    data['batch_size_per_network'] = batch_size_per_network.cpu()

    # Reordering
    directions_flat = data['viewdirs'].unsqueeze(1).expand(
        data['pts'].size()).reshape(-1, 3)
    points_reordered = points_flat[active_samples_mask]
    directions_reordered = directions_flat[active_samples_mask]
    del points_flat, directions_flat
    # reorder so that points handled by the same network are packed together in the list of points
    data['points_reordered'] = points_reordered[reorder_indices]
    data['directions_reordered'] = directions_reordered[reorder_indices]
    return data
