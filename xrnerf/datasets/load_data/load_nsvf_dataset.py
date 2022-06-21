import os

import imageio
import numpy as np


def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.], [0., fy, cy, 0],
                               [0., 0, 1, 0], [0, 0, 0, 1]])

    return full_intrinsic


def load_matrix(path):
    """
    load matrix from txt file path
    Args:
        path: txt file path
    """
    return np.array([[float(w) for w in line.strip().split()]
                     for line in open(path)],
                    dtype=np.float32)


def get_distance_to_closest_point_in_box(point, domain_min, domain_max):
    """
    get the closest point by domain_min and domain_max
    Args:
        point: point in the box
        domain_min: min value of domain
        domain_max: max value of domain
    """
    closest_point = np.array([0., 0., 0.])
    for dim in range(3):
        if point[dim] < domain_min[dim]:
            closest_point[dim] = domain_min[dim]
        elif domain_max[dim] < point[dim]:
            closest_point[dim] = domain_max[dim]
        else:  # in between domain_min and domain_max
            closest_point[dim] = point[dim]
    return np.linalg.norm(point - closest_point)


def get_distance_to_furthest_point_in_box(point, domain_min, domain_max):
    """
    get the furthest point by domain_min and domain_max
    Args:
        point: point in the box
        domain_min: min value of domain
        domain_max: max value of domain
    """
    furthest_point = np.array([0., 0., 0.])
    for dim in range(3):
        mid = (domain_min[dim] + domain_max[dim]) / 2
        if point[dim] > mid:
            furthest_point[dim] = domain_min[dim]
        else:
            furthest_point[dim] = domain_max[dim]
    return np.linalg.norm(point - furthest_point)


def parse_intrinsics(intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return fx, fy, cx, cy


class CameraIntrinsics:
    def __init__(self, H, W, fx, fy, cx, cy):
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


# TODO: Correct value of world2camera?
def parse_extrinsics(extrinsics, world2camera=True):
    """this function is only for numpy for now."""
    if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
        extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
    if extrinsics.shape[0] == 1 and extrinsics.shape[1] == 16:
        extrinsics = extrinsics.reshape(4, 4)
    if world2camera:
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
    return extrinsics


def load_nsvf_dataset(path, testskip, test_traj_path=None):
    rgb_base_path = os.path.join(path, 'rgb')
    pose_base_path = os.path.join(path, 'pose')
    intrinsics_path = os.path.join(path, 'intrinsics.txt')
    bbox_path = os.path.join(path, 'bbox.txt')

    rgbs, poses, poses_unfiltered = [], [], []
    index = 0
    val_index, test_index = 0, 0
    i_split = [[], [], []]
    for rgb_filename in sorted(os.listdir(rgb_base_path)):
        split_prefix = int(
            rgb_filename.split('_')[0])  # 0 = train, 1 = val, 2 = test
        # reduce test set size by only using every testskip-th image

        rgb_filename_without_extension = rgb_filename.split('.')[0]

        pose_filename = rgb_filename_without_extension + '.txt'
        pose_path = os.path.join(pose_base_path, pose_filename)
        pose = load_matrix(pose_path)
        pose = parse_extrinsics(pose, world2camera=False)
        pose[:3, 1:
             3] = -pose[:3, 1:
                        3]  # TODO: why do we need to do this for NSVF style poses? probably they are assuming different coordinate system
        poses_unfiltered.append(pose[None, :])

        if split_prefix == 0 or (split_prefix == 1 and val_index % testskip
                                 == 0) or (split_prefix == 2
                                           and test_index % testskip == 0):
            i_split[split_prefix].append(index)
            rgb_path = os.path.join(rgb_base_path, rgb_filename)
            rgb = imageio.imread(rgb_path)
            rgb = (np.array(rgb) / 255.).astype(np.float32)
            rgbs.append(rgb[None, :])
            poses.append(pose[None, :])
            index += 1
        if split_prefix == 1:
            val_index += 1
        if split_prefix == 2:
            test_index += 1

    rgbs = np.concatenate(rgbs, 0)
    poses = np.concatenate(poses, 0)
    poses_unfiltered = np.concatenate(poses_unfiltered, 0)
    i_split = [np.array(x) for x in i_split]
    H, W = rgbs.shape[1], rgbs.shape[2]
    fx, fy, cx, cy = parse_intrinsics(load_intrinsics(intrinsics_path))
    intrinsics = CameraIntrinsics(H, W, fx, fy, cx, cy)

    near_and_far_path = os.path.join(path, 'near_and_far.txt')
    if os.path.isfile(near_and_far_path):
        near, far = load_matrix(near_and_far_path)[0]
    else:
        # Calculate 'near' and 'far' values based on domain and camera positions
        bounding_box = load_matrix(bbox_path)[0, :-1]
        global_domain_min = bounding_box[:3]
        global_domain_max = bounding_box[3:]
        camera_positions = poses_unfiltered[:, :3, -1]
        near, far = float('inf'), 0.
        for camera_position in camera_positions:
            near = min(
                near,
                get_distance_to_closest_point_in_box(camera_position,
                                                     global_domain_min,
                                                     global_domain_max))
            far = max(
                far,
                get_distance_to_furthest_point_in_box(camera_position,
                                                      global_domain_min,
                                                      global_domain_max))

    background_color = None
    background_color_path = os.path.join(path, 'background_color.txt')
    if os.path.isfile(background_color_path):
        background_color = load_matrix(background_color_path)[0]

    render_poses = np.array([])
    if test_traj_path is None:
        test_traj_path = os.path.join(path, 'test_traj.txt')
    if os.path.isfile(test_traj_path):
        render_poses = []
        test_traj = load_matrix(test_traj_path)
        test_traj = test_traj.reshape((-1, 4, 4))
        for pose in test_traj:
            pose = parse_extrinsics(pose, world2camera=False)
            pose[:3, 1:
                 3] = -pose[:3, 1:
                            3]  # TODO: why do we need to do this for NSVF style poses? probably they are assuming different coordinate system
            render_poses.append(pose[None, :])
        render_poses = np.concatenate(render_poses, 0)

    return rgbs, poses, intrinsics, near, far, background_color, render_poses, i_split
