import time
import os

import numpy as np
import imageio
import cv2
import torch
from mmcv.runner import get_dist_info

from ..builder import PIPELINES
try:
    import kilonerf_cuda
except:
    print('Please install kilonerf_cuda for training KiloNeRF')

from ..builder import PIPELINES


@PIPELINES.register_module()
class Sample:
    """sample image from dataset
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, N_rand=1024, **kwargs):
        self.enable = enable
        self.kwargs = kwargs

    def __call__(self, results):
        """BatchSlice
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            idx = results['idx']
            img_i = results['i_data'][idx]
            results['pose'] = results['poses'][img_i, :3, :4]
            results['target_s'] = results['images'][img_i]
        return results

    def __repr__(self):
        return '{}:slice a batch of rays from all rays'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class MipMultiScaleSample:
    """sample from dataset
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, N_rand=1024, keys=[], **kwargs):
        self.enable = enable
        self.keys = keys
        self.N_rand = N_rand

    def __call__(self, results):
        """BatchSlice
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            rank, _ = get_dist_info()
            np.random.seed(
                int(time.time()) + rank
            )  # to aviod sampling same rays over different gpu cards in ddp

            ray_indices = np.random.randint(0, results[self.keys[0]].shape[0],
                                            (self.N_rand, ))
            for k in self.keys:
                results[k] = results[k][ray_indices]
        return results

    def __repr__(self):
        return '{}:slice a batch of rays from all rays'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class BatchSample:
    """get slice rays from all rays in batching dataset
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, N_rand=1024, **kwargs):
        self.enable = enable
        self.N_rand = N_rand  # slice how many rays one time
        self.kwargs = kwargs

    def __call__(self, results):
        """BatchSlice
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            start_i = self.N_rand * results['idx']
            batch_rays = results['rays_rgb'][start_i:start_i +
                                             self.N_rand]  # [B, 2+1, 3*?]
            results['rays_o'], results['rays_d'], results[
                'target_s'] = batch_rays[:,
                                         0, :], batch_rays[:,
                                                           1, :], batch_rays[:,
                                                                             2, :]
        return results

    def __repr__(self):
        return '{}:sample a batch of rays from all rays'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class GetRays:
    """get rays from pose
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, include_radius=False, **kwargs):
        self.enable = enable
        self.kwargs = kwargs
        self.include_radius = include_radius

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            pose = results['pose']
            c2w = pose[:3, :4]
            device = pose.device
            H, W, K = self.kwargs['H'], self.kwargs['W'], self.kwargs['K']

            i, j = torch.meshgrid(
                torch.linspace(0, W - 1, W),
                torch.linspace(0, H - 1,
                               H))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i - K[0][2]) / K[0][0],
                                -(j - K[1][2]) / K[1][1], -torch.ones_like(i)],
                               -1).to(device)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(
                dirs[..., np.newaxis, :] * c2w[:3, :3],
                -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3, -1].expand(rays_d.shape)

            results['rays_d'] = rays_d
            results['rays_o'] = rays_o

            if self.include_radius:
                # for mip nerf support
                dx = torch.sqrt(
                    torch.sum((rays_d[:-1, :, :] - rays_d[1:, :, :])**2, -1))
                dx = torch.cat([dx, dx[-2:-1, :]], 0)
                results['radii'] = dx[..., None] * 2 / torch.sqrt(
                    torch.tensor(12)).to(device)

        return results

    def __repr__(self):
        return "{}:get rays from pose and camera's params".format(
            self.__class__.__name__)


@PIPELINES.register_module()
class KilonerfGetRays:
    """get rays from pose using kilonerf cuda
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        self.kwargs = kwargs

    def __call__(self, results):
        """get rays by kilonerf cuda
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            pose = results['pose']
            c2w = pose[:3, :4]
            compute_capability = torch.cuda.get_device_capability(pose.device)
            if compute_capability[0] >= 6:
                # GPU: >= NVIDIA GTX 1080 Ti
                root_num_blocks = 64  # => 4096 blocks
                root_num_threads = 16  # => 256 threads per block
            H, W, K = self.kwargs['H'], self.kwargs['W'], self.kwargs['K']
            rays_d = kilonerf_cuda.get_rays_d(H, W, K[0][2], K[1][2], K[0][0],
                                              K[1][1],
                                              c2w[:3, :3].contiguous(),
                                              root_num_blocks,
                                              root_num_threads)
            '''
            i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
            i = i.t()
            j = j.t()
            dirs = torch.stack([(i - intrinsics.cx) / intrinsics.fx, -(j - intrinsics.cy) / intrinsics.fy, -torch.ones_like(i)], -1)
            # Rotate ray directions from camera frame to the world frame
            rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
            '''
            # Translate camera frame's origin to the world frame. It is the origin of all rays.
            rays_o = c2w[:3, -1].expand(rays_d.shape)
            if self.kwargs['expand_origin']:
                rays_o = rays_o.expand(rays_d.shape)
            else:
                rays_o = rays_o.contiguous()

            results['rays_d'] = rays_d
            results['rays_o'] = rays_o

        return results

    def __repr__(self):
        return "{}:get rays from pose and camera's params".format(
            self.__class__.__name__)


@PIPELINES.register_module()
class NBGetRays:
    """get rays from pose
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable
        self.kwargs = kwargs

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            H, W = results['img'].shape[:2]
            K, R, T = results['cam_K'], results['cam_R'], results['cam_T']

            # calculate the camera origin
            rays_o = -np.dot(R.T, T).ravel()
            # calculate the world coodinates of pixels
            i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                               np.arange(H, dtype=np.float32),
                               indexing='xy')
            xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
            pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
            pixel_world = np.dot(pixel_camera - T.ravel(), R)
            # calculate the ray direction
            rays_d = pixel_world - rays_o[None, None]
            rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
            rays_o = np.broadcast_to(rays_o, rays_d.shape)

            results['rays_d'] = rays_d
            results['rays_o'] = rays_o

        return results

    def __repr__(self):
        return "{}:get rays from pose and camera's params".format(self.__class__.__name__)


@PIPELINES.register_module()
class GetViewdirs:
    """get viewdirs from rays_d
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            viewdirs = results['rays_d'].clone()
            viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
            viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
            results['viewdirs'] = viewdirs
        return results

    def __repr__(self):
        return "{}:get viewdirs from rays' direction".format(
            self.__class__.__name__)


@PIPELINES.register_module()
class GetBounds:
    """get near and far
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, near_new=None, far_new=None, **kwargs):
        self.enable = enable
        # kwargs来自于dataset读取完毕后，记录的datainfo信息
        # use near_new if it's not None, else use 'near' from dataset info
        self.near = near_new if near_new is not None else kwargs['near']
        self.far = far_new if far_new is not None else kwargs['far']

    def __call__(self, results):
        """get bound(near and far)
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            results['near'] = self.near * torch.ones_like(
                results['rays_d'][..., :1])
            results['far'] = self.far * torch.ones_like(
                results['rays_d'][..., :1])
        return results

    def __repr__(self):
        return '{}:get bounds(near and far)'.format(self.__class__.__name__)


@PIPELINES.register_module()
class GetZvals:
    """get intervals between samples
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self,
                 enable=True,
                 lindisp=False,
                 N_samples=64,
                 randomized=False,
                 **kwargs):
        self.enable = enable
        self.lindisp = lindisp
        self.N_samples = N_samples
        self.randomized = randomized

    def __call__(self, results):
        """get intervals between samples
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            device = results['rays_o'].device
            t_vals = torch.linspace(0., 1., steps=self.N_samples).to(device)
            if not self.lindisp:
                z_vals = results['near'] * (1. -
                                            t_vals) + results['far'] * (t_vals)
            else:
                z_vals = 1. / (1. / results['near'] *
                               (1. - t_vals) + 1. / results['far'] * (t_vals))

            if self.randomized:
                mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
                upper = torch.cat([mids, z_vals[..., -1:]], -1)
                lower = torch.cat([z_vals[..., :1], mids], -1)
                z_rand = torch.rand(
                    list(results['rays_o'].shape[:-1]) +
                    [self.N_samples]).to(device)
                z_vals = lower + (upper - lower) * z_rand
            else:
                z_vals = z_vals.expand(
                    list(results['rays_o'].shape[:-1]) + [self.N_samples])

            results['z_vals'] = z_vals
        return results

    def __repr__(self):
        return '{}:get intervals between samples'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class GetPts:
    """get pts
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            results['pts'] = results['rays_o'][..., None, :] + results[
                'rays_d'][..., None, :] * results['z_vals'][..., :, None]
        return results

    def __repr__(self):
        return "{}:get viewdirs from rays' direction".format(
            self.__class__.__name__)


@PIPELINES.register_module()
class DeleteUseless:
    """delete useless params
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, keys=[], **kwargs):
        self.enable = enable
        self.keys = keys

    def __call__(self, results):
        """get viewdirs
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            for k in self.keys:
                if k in results:
                    del results[k]
        return results

    def __repr__(self):
        return '{}:delete useless params'.format(self.__class__.__name__)


@PIPELINES.register_module()
class ExampleSample:
    """sample from examples
    Args:
        keys (Sequence[str]): Required keys to be converted.
    """
    def __init__(self, enable=True, train_batch_size=0, **kwargs):
        self.enable = enable
        self.train_batch_size = train_batch_size

    def __call__(self, results):
        """ExampleSample
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            num_examples_per_network = results['all_examples'].size(1)
            indices = np.random.choice(num_examples_per_network,
                                       size=(self.train_batch_size, ))
            # print("indices",indices)
            results['batch_examples'] = results['all_examples'][:, indices]
        return results

    def __repr__(self):
        return '{}:slice a batch of examples from all examples'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class LoadImageAndCamera:
    """load the image and camera parameter
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            data_root = results['data_root']
            ims = results['ims']
            cams = results['cams']
            idx = results['idx']

            # load data
            img_path = os.path.join(data_root, ims[idx])
            cam_ind = results['cam_inds'][idx]
            K = np.array(cams['K'][cam_ind])
            D = np.array(cams['D'][cam_ind])
            R = np.array(cams['R'][cam_ind])
            T = np.array(cams['T'][cam_ind]) / results['cfg'].unit

            # 此时选择一张图，从该图里面随机选择N_rand个射线
            img = imageio.imread(img_path).astype(np.float32) / 255.

            msk_path = os.path.join(data_root, 'mask',
                                    ims[idx])[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(data_root, 'mask_cihp',
                                        ims[idx])[:-4] + '.png'
            msk = imageio.imread(msk_path)
            msk = (msk != 0).astype(np.uint8)

            # process image and mask
            H, W = img.shape[:2]
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            img = cv2.undistort(img, K, D)
            msk = cv2.undistort(msk, K, D)

            # reduce the image resolution by ratio
            ratio = results['cfg'].ratio
            H, W = int(img.shape[0] * ratio), int(img.shape[1] * ratio)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
            K[:2] = K[:2] * ratio

            # remove the background
            img[msk == 0] = 0
            if results['cfg'].white_bkgd:
                img[msk == 0] = 1

            results.update({'img': img, 'msk': msk, 'cam_K': K, 'cam_R': R, 'cam_T': T,
                            'img_path': img_path})

        return results

    def __repr__(self):
        return '{}:load the image and camera parameter'.format(
            self.__class__.__name__)


@PIPELINES.register_module()
class LoadSmplParam:
    """load the SMPL parameter
    """
    def __init__(self, enable=True, **kwargs):
        self.enable = enable

    def __call__(self, results):
        """
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.enable:
            data_root = results['data_root']
            idx = results['idx']
            cfg = results['cfg']
            num_cams = results['num_cams']
            img_path = results['img_path']

            # load smpl parameters
            smpl_idx = cfg.img_path_to_smpl_idx(img_path)
            vert_path = os.path.join(data_root, cfg.smpl_vertices_dir, '{}.npy'.format(smpl_idx))
            param_path = os.path.join(data_root, cfg.smpl_params_dir, '{}.npy'.format(smpl_idx))

            smpl_verts = np.load(vert_path).astype(np.float32)
            params = np.load(param_path, allow_pickle=True).item()
            Rh = params['Rh']
            smpl_R = cv2.Rodrigues(Rh)[0].astype(np.float32)
            smpl_T = params['Th'].astype(np.float32)
            smpl_pose = params['poses'].astype(np.float32)

            frame_idx = cfg.img_path_to_frame_idx(img_path)
            latent_idx = np.array([idx // num_cams])

            results.update({'smpl_verts': smpl_verts, 'smpl_R': smpl_R,
                            'smpl_T': smpl_T, 'smpl_pose': smpl_pose,
                            'latent_idx': latent_idx})

        return results

    def __repr__(self):
        return '{}:load the SMPL parameter'.format(
            self.__class__.__name__)
