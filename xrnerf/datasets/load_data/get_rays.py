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
