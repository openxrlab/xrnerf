import time

import cv2
import numpy as np
import torch
from mmcv.runner import get_dist_info

from ..builder import PIPELINES


@PIPELINES.register_module()
class SelectRays:
    """random select rays when training
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self,
                 enable=True,
                 sel_n=1024,
                 precrop_iters=0,
                 precrop_frac=0.5,
                 include_radius=False,
                 **kwargs):
        self.enable = enable
        self.precrop_iters = precrop_iters
        self.precrop_frac = precrop_frac
        self.kwargs = kwargs
        self.sel_n = sel_n
        self.include_radius = include_radius

    def __call__(self, results):
        """random select rays when training, support precrop
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            H, W, K = self.kwargs['H'], self.kwargs['W'], self.kwargs['K']
            if self.precrop_iters != 0 and results[
                    'iter_n'] < self.precrop_iters:
                # print(results['iter_n'], "precrop now!", flush=True)
                # 在blender数据集的train时，靠前的iter，只取中间部分训练
                dH = int(H // 2 * self.precrop_frac)
                dW = int(W // 2 * self.precrop_frac)
                coords = torch.stack(
                    torch.meshgrid(
                        torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
                        torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)),
                    -1)
            else:
                coords = torch.stack(
                    torch.meshgrid(torch.linspace(0, H - 1, H),
                                   torch.linspace(0, W - 1, W)),
                    -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

            rank, _ = get_dist_info(
            )  # to aviod sampling same rays over different gpu cards in ddp
            np.random.seed(
                int(time.time() + rank)
            )  # fix a bug, for detials please ref to https://github.com/pytorch/pytorch/issues/5059

            select_inds = np.random.choice(coords.shape[0],
                                           size=[self.sel_n],
                                           replace=False)  # (N_rand,)
            select_coords = coords[select_inds].long()  # (N_rand, 2)
            results['rays_o'] = results['rays_o'][
                select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            results['rays_d'] = results['rays_d'][
                select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            results['target_s'] = results['target_s'][
                select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
            if self.include_radius:
                results['radii'] = results['radii'][
                    select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 1)
        return results

    def __repr__(self):
        return '{}:random select rays when training'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class NBSelectRays:
    """random select rays when training
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self,
                 enable=True,
                 sel_n=1024,
                 sel_all=False,
                 sel_rgb=True,
                 **kwargs):
        self.enable = enable
        self.kwargs = kwargs
        self.sel_n = sel_n
        self.sel_all = sel_all
        self.sel_rgb = sel_rgb

    @staticmethod
    def get_bound_2d_mask(bounds, K, pose, H, W):
        min_x, min_y, min_z = bounds[0]
        max_x, max_y, max_z = bounds[1]
        corners_3d = np.array([
            [min_x, min_y, min_z],
            [min_x, min_y, max_z],
            [min_x, max_y, min_z],
            [min_x, max_y, max_z],
            [max_x, min_y, min_z],
            [max_x, min_y, max_z],
            [max_x, max_y, min_z],
            [max_x, max_y, max_z],
        ])

        def project(xyz, K, RT):
            xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
            xyz = np.dot(xyz, K.T)
            xy = xyz[:, :2] / xyz[:, 2:]
            return xy

        corners_2d = project(corners_3d, K, pose)
        corners_2d = np.round(corners_2d).astype(int)

        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
        cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
        cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
        cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
        cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
        cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
        return mask

    @staticmethod
    def get_near_far(bounds, ray_o, ray_d):
        """calculate intersections with 3d bounding box."""
        norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
        viewdir = ray_d / norm_d
        viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
        viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
        tmin = (bounds[:1] - ray_o[:1]) / viewdir
        tmax = (bounds[1:2] - ray_o[:1]) / viewdir
        t1 = np.minimum(tmin, tmax)
        t2 = np.maximum(tmin, tmax)
        near = np.max(t1, axis=-1)
        far = np.min(t2, axis=-1)
        mask_at_box = near < far
        near = near[mask_at_box] / norm_d[mask_at_box, 0]
        far = far[mask_at_box] / norm_d[mask_at_box, 0]
        return near, far, mask_at_box

    def sample_rays(self, results, bounds, bound_mask, human_mask):
        # sample pixels and calculate the ray-box intersections
        ray_o_list = []
        ray_d_list = []
        rgb_list = []
        near_list = []
        far_list = []

        nsampled_rays = 0
        while nsampled_rays < self.sel_n:
            n_body = int((self.sel_n - nsampled_rays) * 0.5)
            n_rand = self.sel_n - nsampled_rays - n_body

            # sample rays on body
            coord_body = np.argwhere(human_mask != 0)
            coord_body = coord_body[np.random.randint(0, len(coord_body),
                                                      n_body)]

            # sample rays in the bound mask
            coord = np.argwhere(bound_mask == 1)
            coord = coord[np.random.randint(0, len(coord), n_rand)]
            coord = np.concatenate([coord_body, coord], axis=0)

            # calculate the ray info
            ray_o_ = results['rays_o'][coord[:, 0], coord[:, 1]]
            ray_d_ = results['rays_d'][coord[:, 0], coord[:, 1]]
            if self.sel_rgb:
                rgb_ = results['img'][coord[:, 0], coord[:, 1]]
            near_, far_, mask_at_box = self.get_near_far(
                bounds, ray_o_, ray_d_)

            ray_o_list.append(ray_o_[mask_at_box])
            ray_d_list.append(ray_d_[mask_at_box])
            rgb_list.append(rgb_[mask_at_box])
            near_list.append(near_)
            far_list.append(far_)
            nsampled_rays += len(near_)

        results['rays_o'] = np.concatenate(ray_o_list).astype(np.float32)
        results['rays_d'] = np.concatenate(ray_d_list).astype(np.float32)
        if self.sel_rgb:
            results['target_s'] = np.concatenate(rgb_list).astype(np.float32)
        results['near'] = np.concatenate(near_list).astype(np.float32)[:, None]
        results['far'] = np.concatenate(far_list).astype(np.float32)[:, None]

        return results

    def select_all_rays(self, results, bounds):
        src_shape = results['rays_d'].shape
        results['src_shape'] = torch.tensor(src_shape)

        ray_o = results['rays_o'].reshape(-1, 3).astype(np.float32)
        ray_d = results['rays_d'].reshape(-1, 3).astype(np.float32)
        near, far, mask_at_box = self.get_near_far(bounds, ray_o, ray_d)
        near = near.astype(np.float32)
        far = far.astype(np.float32)
        ray_o = ray_o[mask_at_box]
        ray_d = ray_d[mask_at_box]

        results['rays_o'] = ray_o.astype(np.float32)
        results['rays_d'] = ray_d.astype(np.float32)
        if self.sel_rgb:
            rgb = results['img'].reshape(-1, 3).astype(np.float32)
            rgb = rgb[mask_at_box]
            results['target_s'] = rgb.astype(np.float32)
        results['near'] = near.astype(np.float32)[:, None]
        results['far'] = far.astype(np.float32)[:, None]
        results['mask_at_box'] = mask_at_box

        return results

    def __call__(self, results):
        """random select rays when training, support precrop
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            # calculate the 3D box that bounds the human
            smpl_verts = results['smpl_verts']
            min_xyz = np.min(smpl_verts, axis=0) - 0.05
            max_xyz = np.max(smpl_verts, axis=0) + 0.05
            bounds = np.stack([min_xyz, max_xyz], axis=0)

            # generate regions for sampling
            cfg = results['cfg']
            if cfg.mode == 'render':
                H, W = cfg.render_H, cfg.render_W
            else:
                H, W = results['img'].shape[:2]
            K, R, T = results['cam_K'], results['cam_R'], results['cam_T']
            pose = np.concatenate([R, T], axis=1)
            bound_mask = self.get_bound_2d_mask(bounds, K, pose, H, W)

            if self.sel_all:
                results = self.select_all_rays(results, bounds)
            else:
                human_mask = results['msk'] * bound_mask
                results = self.sample_rays(results, bounds, bound_mask,
                                           human_mask)

        return results

    def __repr__(self):
        return '{}:random select rays when training'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class PerturbZvals:
    """apply perturb to zvals
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get intervals between samples
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            z_vals = results['z_vals']
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape).to(z_vals.device)
            results['z_vals'] = lower + (upper - lower) * t_rand
        return results

    def __repr__(self):
        return '{}:apply perturb to zvals'.format(self.__class__.__name__)


@PIPELINES.register_module()
class RandomBGColor:
    """random set background color, used in ngp
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        self.kwargs = kwargs

    def __call__(self, results):
        """BatchSlice
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            alpha = results['alpha']
            target_s = results['target_s']
            bg_color = np.random.rand(*list(results['target_s'].shape))
            # bg_color = np.zeros(list(results['target_s'].shape))
            target_s = target_s * alpha + bg_color * (1 - alpha)
            results['target_s'] = target_s.astype(np.float32)
            results['bg_color'] = bg_color.astype(np.float32)
        return results

    def __repr__(self):
        return '{}:sample a batch of rays from all rays'.format(
            self.__class__.__name__)
