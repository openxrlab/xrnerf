# Copyright (c) OpenMMLab. All rights reserved.
import time

try:
    import kilonerf_cuda
except:
    pass
import torch
from mmcv.runner import get_dist_info
from torch import nn
from tqdm import tqdm

from ..builder import NETWORKS
from .nerf import NerfNetwork
from .utils import img2mse, mse2psnr, recover_shape, unfold_batching


@NETWORKS.register_module()
class KiloNerfNetwork(NerfNetwork):
    """KiloNerfNetwork extends NerfNetwork, but KiloNerfNetwork has  a mlp
    structure which is a multi_network and adds l2_regularization loss."""
    def __init__(self, cfg, mlp=None, mlp_fine=None, render=None):
        super().__init__(cfg, mlp=mlp, mlp_fine=mlp_fine, render=render)

        if 'l2_regularization_lambda' in cfg:
            self.l2_regularization_lambda = cfg.l2_regularization_lambda

    def train_step(self, data, optimizer, **kwargs):
        for k in data:
            data[k] = unfold_batching(data[k])
        ret = self.forward(data, is_test=False)

        img_loss = img2mse(ret['rgb'], data['target_s'])
        psnr = mse2psnr(img_loss)
        loss = img_loss

        if self.l2_regularization_lambda is not None:
            l2_reg_term = self.mlp.get_view_dependent_parameters()[0].norm(2)
            for param in self.mlp.get_view_dependent_parameters()[1:]:
                l2_reg_term = l2_reg_term + param.norm(2)
            l2_loss = self.l2_regularization_lambda * l2_reg_term
            loss = loss + l2_loss

        if 'coarse_rgb' in ret:
            coarse_img_loss = img2mse(ret['coarse_rgb'], data['target_s'])
            loss = loss + coarse_img_loss
            coarse_psnr = mse2psnr(coarse_img_loss)

        log_vars = {
            'loss': loss.item(),
            'psnr': psnr.item(),
            'L2 reg': l2_loss.item()
        }
        outputs = {
            'loss': loss,
            'log_vars': log_vars,
            'num_samples': ret['rgb'].shape[0]
        }
        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        if self.phase == 'test':
            return self.test_step(data, **kwargs)

        rank, world_size = get_dist_info()
        if rank == 0:
            for k in data:
                data[k] = unfold_batching(data[k])
            poses = data['poses']
            images = data['images']
            spiral_poses = data['spiral_poses']
            global_domain_min = data['global_domain_min']
            global_domain_max = data['global_domain_max']

            rgbs, disps, gt_imgs = [], [], []
            elapsed_time_list = []
            for i in tqdm(range(poses.shape[0])):
                start = time.time()
                data = self.val_pipeline({'pose': poses[i]})
                data['global_domain_min'], data[
                    'global_domain_max'] = global_domain_min, global_domain_max
                ret = self.batchify_forward(
                    data, is_test=True)  # when testing, raw_noise_std=False
                end = time.time()
                # elapsed_time includes pipeline time and forward time
                elapsed_time = end - start
                rgb = recover_shape(ret['rgb'], data['src_shape'])
                disp = recover_shape(ret['disp'], data['src_shape'])
                rgbs.append(rgb.cpu().numpy())
                disps.append(disp.cpu().numpy())
                gt_imgs.append(images[i].cpu().numpy())
                elapsed_time_list.append(elapsed_time)

            spiral_rgbs, spiral_disps = [], []
            for i in tqdm(range(spiral_poses.shape[0])):
                data = self.val_pipeline({'pose': spiral_poses[i]})
                data['global_domain_min'], data[
                    'global_domain_max'] = global_domain_min, global_domain_max
                ret = self.batchify_forward(data, is_test=True)
                rgb = recover_shape(ret['rgb'], data['src_shape'])
                disp = recover_shape(ret['disp'], data['src_shape'])
                spiral_rgbs.append(rgb.cpu().numpy())
                spiral_disps.append(disp.cpu().numpy())

            outputs = {
                'spiral_rgbs': spiral_rgbs,
                'spiral_disps': spiral_disps,
                'rgbs': rgbs,
                'disps': disps,
                'gt_imgs': gt_imgs,
                'elapsed_time': elapsed_time_list
            }
        else:
            outputs = {}
        return outputs

    def test_step(self, data, **kwargs):
        """in mmcv's runner, there is only train_step and val_step so use.

        [val_step() + phase=='test'] to represent test.
        """
        rank, world_size = get_dist_info()
        if rank == 0:
            for k in data:
                data[k] = unfold_batching(data[k])

            image = data['image']
            global_domain_min = data['global_domain_min']
            global_domain_max = data['global_domain_max']

            data = self.val_pipeline({'pose': data['pose']})
            data['global_domain_min'], data[
                'global_domain_max'] = global_domain_min, global_domain_max
            ret = self.batchify_forward(data, is_test=True)

            rgb = recover_shape(ret['rgb'], data['src_shape'])
            rgb = rgb.cpu().numpy()
            image = image.cpu().numpy()

            outputs = {'rgb': rgb, 'gt_img': image}

        else:
            outputs = {}
        return outputs
