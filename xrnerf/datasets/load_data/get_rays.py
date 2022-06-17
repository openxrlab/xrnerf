import numpy as np


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)],
        -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3],
        -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def load_rays(H, W, K, poses, images, i_data):
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]],
                    0)  # [N, ro+rd, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None]],
                              1)  # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.transpose(rays_rgb,
                            [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.stack([rays_rgb[i] for i in i_data],
                        0)  # train self.images only
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    return rays_rgb


def load_rays_multiscale(meta, n_examples):
    """Generating rays for all images."""
    pix2cam = meta['pix2cam']
    cam2world = meta['cam2world']
    width = meta['width']
    height = meta['height']

    def res2grid(w, h):
        return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(w, dtype=np.float32) + .5,  # X-Axis (columns)
            np.arange(h, dtype=np.float32) + .5,  # Y-Axis (rows)
            indexing='xy')

    xy = [res2grid(w, h) for w, h in zip(width, height)]
    pixel_dirs = [np.stack([x, y, np.ones_like(x)], axis=-1) for x, y in xy]
    camera_dirs = [v @ p2c[:3, :3].T for v, p2c in zip(pixel_dirs, pix2cam)]
    directions = [v @ c2w[:3, :3].T for v, c2w in zip(camera_dirs, cam2world)]
    origins = [
        np.broadcast_to(c2w[:3, -1], v.shape)
        for v, c2w in zip(directions, cam2world)
    ]
    viewdirs = [
        v / np.linalg.norm(v, axis=-1, keepdims=True) for v in directions
    ]

    def broadcast_scalar_attribute(x):
        return [
            np.broadcast_to(x[i], origins[i][..., :1].shape)
            for i in range(n_examples)
        ]

    lossmult = broadcast_scalar_attribute(meta['lossmult'])
    near = broadcast_scalar_attribute(meta['near'])
    far = broadcast_scalar_attribute(meta['far'])

    # Distance from each unit-norm direction vector to its x-axis neighbor.
    dx = [
        np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :])**2, -1))
        for v in directions
    ]
    dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
    # Cut the distance in half, and then round it out so that it's
    # halfway between inscribed by / circumscribed about the pixel.
    radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

    rays = dict(rays_o=origins,
                rays_d=directions,
                viewdirs=viewdirs,
                radii=radii,
                lossmult=lossmult,
                near=near,
                far=far)
    return rays
