# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import get_dist_info, load_checkpoint
from torch import nn
from tqdm import tqdm

from .. import builder
from ..builder import NETWORKS
from .nerf import NerfNetwork
from .utils import *
from .utils.gnr import index, init_weights


@NETWORKS.register_module()
class GnrNetwork(NerfNetwork):
    def __init__(self, cfg):

        super(GnrNetwork, self).__init__(cfg)
        self.name = 'gnr'

        self.cfg = cfg
        self.num_views = self.cfg.num_views
        self.use_feat_sr = self.cfg.use_feat_sr
        self.ddp = self.cfg.ddp
        self.feat_dim = 64 if self.use_feat_sr else 256
        self.index = index
        self.error_term = nn.MSELoss()
        self.metrics_dict = {'lpips': [], 'psnr': [], 'ssim': []}

        self.image_filter = builder.build_embedder(cfg.image_filter)

        if self.use_feat_sr:
            self.sr_filter = builder.build_embedder(cfg.sr_filter)

        if not cfg.train_encoder:
            for param in self.image_filter.parameters():
                param.requires_grad = False

        self.nerf = builder.build_mlp(cfg.nerf)

        cfg.nerf_renderer.opt.model = self.nerf

        self.nerf_renderer = builder.build_render(cfg.nerf_renderer)

        init_weights(self)

    def image_rescale(self, images, masks):
        if images.min() < -0.2:
            images = (images + 1) / 2
            images = images * (masks > 0).float()
        return images

    def get_image_feature(self, data):
        if 'feats' not in data.keys():
            images = data['images']
            im_feat = self.image_filter(images[:self.num_views])
            if self.use_feat_sr:
                im_feat = self.sr_filter(im_feat, images[:self.num_views])
            data['images'] = torch.cat([self.image_rescale(images[:self.num_views], data['masks'][:self.num_views]), \
                                        images[self.num_views:]], 0)
            data['feats'] = im_feat
        return data

    def forward(self, data, is_test=False):
        data = self.get_image_feature(data)
        error = self.nerf_renderer.render(**data)
        return error

    def render_path(self, data):
        with torch.no_grad():
            rgbs = None
            data = self.get_image_feature(data)
            rgbs, depths = self.nerf_renderer.render_path(**data)

        return rgbs, depths

    def reconstruct(self, data):
        with torch.no_grad():
            data = self.get_image_feature(data)
            verts, faces, rgbs = self.nerf_renderer.reconstruct(**data)

        return verts, faces, rgbs

    def train_step(self, data, optimizer, **kwargs):

        data = self.prepare_data(self.cfg, data)
        outputs = self.forward(data)

        return outputs

    def val_step(self, data, optimizer=None, **kwargs):
        test_data = data
        data = self.prepare_data(self.cfg, test_data)
        local_rank = 0
        render_gt = test_data['render_gt'][0].to(local_rank)

        data_ren = self.get_image_feature(data)
        rgbs, depths = self.nerf_renderer.render_path(**data_ren)

        if self.cfg.use_attention:
            rgbs, att_rgbs = rgbs[..., :3], rgbs[..., 3:6]
        else:
            att_rgbs = rgbs[..., :3]

        re_att_rgbs = np.array([att_rgb for att_rgb in att_rgbs.cpu().numpy()])
        re_render_gt = np.array(
            [gt for gt in render_gt.permute(0, 2, 3, 1).cpu().numpy()])

        outputs = {
            'rgbs': re_att_rgbs,
            'disps': depths.cpu(),
            'rgb': re_att_rgbs,
            'gt_img': re_render_gt,
            'gt_imgs': re_render_gt,
            'idx': int(test_data['idx'].cpu())
        }

        return outputs

    def cal_metrics(self, metrics, rgbs, gts):
        x = rgbs.clone().permute((0, 3, 1, 2))
        out = {}
        for m_key in metrics.keys():
            out[m_key] = []
            for pred, gt in zip(x, gts):
                metric = metrics[m_key]
                out[m_key].append(metric(pred, gt))
            out[m_key] = torch.stack(out[m_key], dim=0)
        return out

    def prepare_data(self, opt, data, local_rank=0):
        image_tensor = data['img'][0].to(device=local_rank)
        calib_tensor = data['calib'][0].to(device=local_rank)
        mask_tensor = data['mask'][0].to(device=local_rank)
        bbox = list(data['bbox'][0].cpu().numpy().astype(np.int32))
        mesh_param = {
            'center': data['center'][0].to(device=local_rank),
            'spatial_freq': data['spatial_freq'][0].cpu().numpy().item()
        }
        if any([opt.use_smpl_sdf, opt.use_t_pose]):
            smpl = {'rot': data['smpl_rot'].to(device=local_rank)}
            if opt.use_smpl_sdf or opt.use_t_pose:
                smpl['verts'] = data['smpl_verts'][0].to(device=local_rank)
                smpl['faces'] = data['smpl_faces'][0].to(device=local_rank)
            if opt.use_t_pose:
                smpl['t_verts'] = data['smpl_t_verts'][0].to(device=local_rank)
                smpl['t_faces'] = data['smpl_t_faces'][0].to(device=local_rank)
            if opt.use_smpl_depth:
                smpl['depth'] = data['smpl_depth'][0].to(
                    device=local_rank)[:, None, ...]

        else:
            smpl = None

        if 'scan_verts' in data.keys():
            scan = [
                data['scan_verts'][0].to(device=local_rank),
                data['scan_faces'][0].to(device=local_rank)
            ]
        else:
            scan = None

        persps = data['persps'][0].to(
            device=local_rank
        ) if opt.projection_mode == 'perspective' else None

        return {
            'images': image_tensor,
            'calibs': calib_tensor,
            'bbox': bbox,
            'masks': mask_tensor,
            'mesh_param': mesh_param,
            'smpl': smpl,
            'scan': scan,
            'persps': persps
        }

    def to8b(self, img):
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, [1, 2, 0])
        if img.min() < -.2:
            img = (img + 1) * 127.5
        elif img.max() <= 2.:
            img = img * 255.
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)
