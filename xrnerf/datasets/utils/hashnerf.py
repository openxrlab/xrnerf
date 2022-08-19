import numpy as np


def matrix_nerf2ngp(matrix, correct_pose, scale, offset):
    matrix[:, 0] *= correct_pose[0]
    matrix[:, 1] *= correct_pose[1]
    matrix[:, 2] *= correct_pose[2]
    matrix[:, 3] = matrix[:, 3] * scale + offset
    # cycle
    matrix = matrix[[1, 2, 0]]
    return matrix


def poses_nerf2ngp(poses, correct_pose, scale, offset):

    ngp_poses = []
    for i in range(poses.shape[0]):
        ngp_poses.append(
            matrix_nerf2ngp(poses[i, :-1, :], correct_pose, scale, offset))
    ngp_poses = np.array(ngp_poses).astype(np.float32)
    ngp_poses = ngp_poses.transpose(0, 2, 1)

    return ngp_poses
