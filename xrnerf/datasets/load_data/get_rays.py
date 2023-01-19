import numpy as np
import torch


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)],
        -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d


def load_rays(H, W, K, poses, images, i_data):
    # [N, ro+rd, H, W, 3]
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)
    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None]], 1)
    # [N, H, W, ro+rd+rgb, 3]
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = np.stack([rays_rgb[i] for i in i_data], 0)
    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
    rays_rgb = rays_rgb.astype(np.float32)
    np.random.shuffle(rays_rgb)
    return rays_rgb


def get_rays_np_hash(H, W, K, c2w):

    c2w = c2w.transpose(1, 0)

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    i, j = i + 0.5, j + 0.5  # tmp

    dirs = np.stack([(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1],
                     np.ones_like(i)], -1)

    rays_d = np.matmul(c2w[:3, :3], dirs[:, :, :, np.newaxis])[..., 0]
    # print('rays_d', rays_d.max(), rays_d.min(), rays_d.shape)
    # exit(0)
    # print(dirs.shape, dirs[..., 0].min(), dirs[..., 0].max(), dirs[..., 0].mean())
    # print(dirs.shape, dirs[..., 1].min(), dirs[..., 1].max(), dirs[..., 1].mean())

    # rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # print('rays_d', rays_d.max(), rays_d.min(), rays_d.shape)
    # exit(0)

    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    # print('rays_d', rays_d.max(), rays_d.min(), rays_d.shape)
    # exit(0)

    # print('dirs',dirs.max(), dirs.min(), dirs.shape)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    # print('rays_d',rays_d.max(), rays_d.min(), rays_d.shape)
    # print('rays_o',rays_o.max(), rays_o.min(), rays_o.shape)
    # exit(0)

    return rays_o, rays_d


def load_rays_hash(H, W, K, poses, images):

    # 1. do not shuffle   2. add img_index   3. no 'i_data'
    # [N, H, W, ro3+rd3] get ray_o_d
    print('start get rays...', flush=True)
    rays = np.stack(
        [np.concatenate(get_rays_np_hash(H, W, K, p), 2) for p in poses], 0)
    print('get rays ok', flush=True)

    # rays_d = rays[:,:,:,3:]
    # print('rays_d',rays_d.max(), rays_d.min(), rays_d.mean(), rays_d.shape)
    # rays_o = rays[:,:,:,:3]
    # print('rays_o',rays_o.max(), rays_o.min(), rays_o.mean(), rays_o.shape)
    # exit(0)

    # [N, H, W, ro3+rd3+rgba4] add rgba
    rays_rgb = np.concatenate([rays, images], 3)
    # [N, 1, 1, 1]
    img_ids = np.array(range(images.shape[0])).reshape((-1, 1, 1, 1))
    # [N, H, W, 1]
    img_ids = np.broadcast_to(img_ids, list(rays_rgb.shape[:3]) + [1])
    # [N, H, W, 10+1]
    rays_rgb = np.concatenate([rays_rgb, img_ids], 3)
    # [N*H*W, 10+1]
    rays_rgb = np.reshape(rays_rgb, [-1, 11])
    rays_rgb = rays_rgb.astype(np.float32)
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


def get_rays_np_bungee(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    dirs = np.stack(
        [(i - W * .5) / focal, -(j - H * .5) / focal, -np.ones_like(i)], -1)
    dirs = dirs / np.linalg.norm(dirs, axis=-1)[..., None]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def load_rays_bungee(H, W, focal, poses, images, i_data, n_images, scale_split,
                     cur_stage):
    # get scale codes
    scale_codes = []
    prev_spl = n_images
    cur_scale = 0
    for spl in scale_split[:cur_stage + 1]:
        scale_codes.append(
            np.tile(
                np.ones(((prev_spl - spl), 1, 1, 1)) * cur_scale,
                (1, H, W, 1)))
        prev_spl = spl
        cur_scale += 1
    scale_codes = np.concatenate(scale_codes, 0)
    scale_codes = scale_codes.astype(np.int64)
    # [N, ro+rd, H, W, 3]
    rays = np.stack([get_rays_np_bungee(H, W, focal, p) for p in poses], 0)
    directions = rays[:, 1, :, :, :]
    dx = np.sqrt(
        np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :])**2, -1))
    dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
    radii = dx[..., None] * 2 / np.sqrt(12)

    # [N, ro+rd+rgb, H, W, 3]
    rays_rgb = np.concatenate([rays, images[:, None]], 1)
    rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
    rays_rgb = np.stack([rays_rgb[i] for i in i_data], 0)
    radii = np.stack([radii[i] for i in i_data], 0)
    scale_codes = np.stack([scale_codes[i] for i in i_data], 0)

    rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
    radii = np.reshape(radii, [-1, 1])
    scale_codes = np.reshape(scale_codes, [-1, 1])

    rand_idx = torch.randperm(rays_rgb.shape[0])
    rays_rgb = rays_rgb[rand_idx.cpu().data.numpy()]
    radii = radii[rand_idx.cpu().data.numpy()]
    scale_codes = scale_codes[rand_idx.cpu().data.numpy()]
    return rays_rgb, radii, scale_codes
