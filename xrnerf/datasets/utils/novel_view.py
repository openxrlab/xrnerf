import numpy as np


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec0_avg = up
    vec1 = normalize(np.cross(vec2, vec0_avg))
    vec0 = normalize(np.cross(vec1, vec2))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt


def gen_spiral_path(R, T, num_render_views, center=None):
    lower_row = np.array([[0., 0., 0., 1.]])

    # transfer RT to camera_to_world matrix
    RT = np.concatenate([np.array(R), np.array(T) / 1000.], axis=2)
    RT = np.concatenate([RT, np.tile(lower_row[None], (len(RT), 1, 1))],
                        axis=1)
    RT[:] = np.linalg.inv(RT[:])

    RT = np.concatenate(
        [RT[:, :, 1:2], RT[:, :, 0:1], -RT[:, :, 2:3], RT[:, :, 3:4]], 2)

    up = normalize(RT[:, :3, 0].sum(0))  # average up vector
    z = normalize(RT[0, :3, 2])
    vec1 = normalize(np.cross(z, up))
    vec2 = normalize(np.cross(up, vec1))
    z_off = 0

    if center is None:
        center = RT[:, :3, 3].mean(0)
        z_off = 1.3

    c2w = np.stack([up, vec1, vec2, center], 1)

    # get radii for spiral path
    tt = ptstocam(RT[:, :3, 3], c2w).T
    rads = np.percentile(np.abs(tt), 80, -1)
    rads = rads * 1.3
    rads = np.array(list(rads) + [1.])

    render_w2c = []
    for theta in np.linspace(0., 2 * np.pi, num_render_views + 1)[:-1]:
        # camera position
        cam_pos = np.array([0, np.sin(theta), np.cos(theta), 1] * rads)
        cam_pos_world = np.dot(c2w[:3, :4], cam_pos)
        # z axis
        z = normalize(cam_pos_world -
                      np.dot(c2w[:3, :4], np.array([z_off, 0, 0, 1.])))
        # vector -> 3x4 matrix (camera_to_world)
        mat = viewmatrix(z, up, cam_pos_world)

        mat = np.concatenate(
            [mat[:, 1:2], mat[:, 0:1], -mat[:, 2:3], mat[:, 3:4]], 1)
        mat = np.concatenate([mat, lower_row], 0)
        mat = np.linalg.inv(mat)
        render_w2c.append(mat)

    return render_w2c
