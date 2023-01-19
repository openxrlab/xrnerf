import numpy as np

from .load_blender import load_blender_data
from .load_deepvoxels import load_dv_data
from .load_LINEMOD import load_LINEMOD_data
from .load_llff import load_llff_data
from .load_multiscale import load_multiscale_data
from .load_multiscale_google import load_google_data
from .load_nsvf_dataset import load_nsvf_dataset


def load_data(args):
    # Load data
    K = None
    # print(args.llffhold, args.no_ndc)
    # exit(0)

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, i_test = load_llff_data(
            args.datadir,
            args.factor,
            recenter=True,
            bd_factor=.75,
            spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf,
              args.datadir)
        if not isinstance(i_test, list):
            i_test = [i_test]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            i_test = np.arange(images.shape[0])[::args.llffhold]

        i_val = i_test
        i_train = np.array([
            i for i in np.arange(int(images.shape[0]))
            if (i not in i_test and i not in i_val)
        ])

        print('DEFINING BOUNDS')
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. -
                                                           images[..., -1:])
        else:
            if ('load_alpha' in args) and args.load_alpha:
                images = images
            else:
                images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip)
        print(
            f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}'
        )
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        i_train, i_val, i_test = i_split

        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. -
                                                           images[..., -1:])
        else:
            if ('load_alpha' in args) and args.load_alpha:
                images = images
            else:
                images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(
            scene=args.shape, basedir=args.datadir, testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf,
              args.datadir)
        i_train, i_val, i_test = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R - 1.
        far = hemi_R + 1.

    elif args.dataset_type == 'multiscale':
        meta, images, n_examples = load_multiscale_data(
            args.datadir, args.mode, args.white_bkgd)
        print('Load MultiScale Blender', len(images))
        return meta, images, n_examples

        #nsvf dataset type
    elif args.dataset_type == 'nsvf':
        test_traj_path = args.test_traj_path if 'test_traj_path' in args else None
        images, poses, intrinsics, near, far, background_color, render_poses, i_split = load_nsvf_dataset(
            args.datadir, args.testskip, test_traj_path)
        hwf = [intrinsics.H, intrinsics.W, intrinsics.fx]
        print('Loaded a NSVF-style dataset', images.shape, poses.shape,
              render_poses.shape, args.datadir)

        i_train, i_val, i_test = i_split
        if i_test.size == 0:
            i_test = i_val

        if args.white_bkgd and images.shape[-1] == 4:
            images = images[..., :3] * images[..., -1:] + (1. -
                                                           images[..., -1:])
        else:
            if ('load_alpha' in args) and args.load_alpha:
                images = images
            else:
                images = images[..., :3]

        render_subset = 'custom_path'
        if args.render_test:
            render_subset = 'test'
        if 'render_subset' in args:
            render_subset = args.render_subset

        #render_poses of nsvf is None，need to use poses by render_subset type
        if render_subset == 'train':
            i_render = i_train
        elif render_subset == 'val':
            i_render = i_val
        elif render_subset == 'test':
            i_render = i_test
        if render_subset != 'custom_path':
            render_poses = np.array(poses[i_render])

    elif args.dataset_type == 'mutiscale_google':
        images, poses, scene_scale, scene_origin, scale_split = load_google_data(
            args.datadir, args.factor)
        n_images = len(images)
        print('Load Multiscale Google', n_images)
        if args.white_bkgd:
            images = images[..., :3] * images[..., -1:] + (1. -
                                                           images[..., -1:])
        else:
            images = images[..., :3]
        images = images[scale_split[args.cur_stage]:]
        poses = poses[scale_split[args.cur_stage]:]

        if args.holdout > 0:
            i_test = np.arange(images.shape[0])[::args.holdout]
        i_val = i_test
        i_train = np.array(
            [i for i in np.arange(int(images.shape[0])) if (i not in i_test)])

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])
        render_poses = np.array(poses[i_test])
        return images, poses, render_poses, hwf, K, scene_scale, scene_origin, scale_split, i_train, i_val, i_test, n_images

    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([[focal, 0, 0.5 * W], [0, focal, 0.5 * H], [0, 0, 1]])

    # print(images.shape, poses.shape, render_poses.shape)
    # print(hwf, K, i_train, i_val, i_test)
    # exit(0)

    return images, poses, render_poses, hwf, K, near, far, i_train, i_val, i_test
