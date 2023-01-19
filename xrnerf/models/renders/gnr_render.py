import imp
import logging
from turtle import pd, width

import torch
from cv2 import sepFilter2D

torch.autograd.set_detect_anomaly(True)
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import trimesh
from skimage import measure
from tqdm import tqdm

from extensions.mesh_grid import MeshGridSearcher

from ..builder import RENDERS
from ..networks.utils.gnr import index, orthogonal, perspective

mse = lambda x, y: torch.mean((x - y)**2)
bmse = lambda x, y: torch.sum((x * y - y)**2) / torch.sum(y)
l1 = lambda x, y: torch.mean(torch.abs(x - y))
to8b = lambda x: (np.clip(x.detach().cpu().numpy(), 0, 1) * 255).astype(np.
                                                                        uint8)
eikonal = lambda x: torch.mean(x**2)


@RENDERS.register_module()
class GnrRenderer:
    def __init__(self,
                 opt,
                 nerf_fine=None,
                 projection='perspective',
                 vgg_loss=None,
                 threshold=0.5):
        self.opt = opt
        self.nerf = opt.model
        self.nerf_fine = nerf_fine
        self.use_fine = self.nerf_fine is not None
        self.width = self.opt.loadSize
        self.height = self.opt.loadSize
        self.N_samples = opt.N_samples
        self.num_views = opt.num_views
        self.projection_mode = projection
        self.projection = orthogonal if projection == 'orthogonal' else perspective
        self.N_rand = opt.N_rand
        self.N_grid = opt.N_grid + 1
        self.chunk = opt.chunk
        self.N_rand_infer = opt.N_rand_infer
        self.mse_loss = nn.MSELoss()
        self.alpha_loss = nn.BCELoss()
        self.alpha_loss_bmse = bmse
        self.alpha_grad_loss = eikonal

        self.mesh_searcher = MeshGridSearcher()
        self.use_nml = opt.use_nml
        self.use_attention = opt.use_attention
        self.threshold = threshold
        self.debug = opt.debug

        self.rgb_ch = 6 if self.use_attention else 3
        if self.debug: self.rgb_ch += self.num_views * 3
        self.debug_idx = 0

        self.use_vgg = opt.use_vgg
        self.vgg_loss = vgg_loss
        self.use_smpl_sdf = opt.use_smpl_sdf
        self.use_t_pose = opt.use_t_pose
        self.use_smpl_depth = opt.use_smpl_depth
        self.sel_cords = None
        self.regularization = opt.regularization
        self.angle_diff = opt.angle_diff
        self.use_occlusion = opt.use_occlusion and self.use_smpl_depth
        self.use_occlusion_net = opt.use_occlusion_net
        #self.use_occlusion_net = False

        self.gamma = 1
        self.pts_nml = None
        self.alpha_grad = None
        self.alpha_gt = None
        self.alpha_smpl = None
        self.alpha = None
        self.omega_reg = 0.01

        self.nerf_out_ch = 8 if self.use_attention else 4
        self.use_vh = opt.use_vh
        self.vh_overhead = opt.vh_overhead if self.use_vh else 1
        self.use_vh_free = opt.use_vh_free
        self.use_white_bkgd = opt.use_white_bkgd
        self.default_rgb = torch.zeros if not self.use_white_bkgd else torch.ones
        self.occ = None
        self.occ_gt = None

    def cal_loss(self, rgb, rgb_gt):
        loss = {'nerf': self.mse_loss(rgb[:, :3], rgb_gt)}
        if self.use_attention:
            loss.update({'att': self.mse_loss(rgb[:, 3:6], rgb_gt)})
            x = self.mse_loss(rgb[:, 3:6], rgb_gt)
        if self.alpha_gt is not None and self.alpha is not None:
            # alpha loss has three options, self.alpha_loss (binary cross entropy), self.mse_loss (mean square error)
            # and self.alpha_loss_bmse (one sided mean square error), we find the last performs the best
            loss.update({'alpha': self.mse_loss(self.alpha, self.alpha_gt)})
        if self.regularization:
            loss.update({
                'alpha_reg':
                self.alpha_grad_loss(self.alpha_grad / self.nerf.spatial_freq)
                * self.omega_reg
            })
        if self.angle_diff and self.angle_diff_grad is not None:
            loss.update({'angle_diff': torch.mean(self.angle_diff_grad**2)})
        if self.use_occlusion_net and self.occ is not None and self.occ_gt is not None:
            loss.update({'occ': self.mse_loss(self.occ, self.occ_gt)})
        loss = sum(loss.values())
        return loss

    def get_rays_orthogonal(self, bbox, calib):
        top, bottom, left, right = bbox
        cy, cx, focal = self.height / 2, self.width / 2, self.height / 2
        radian = ((right - left) / 2 + 1) / focal
        i, j = torch.meshgrid(
            torch.linspace(top,
                           bottom - 1,
                           int(bottom - top),
                           device=calib.device),
            torch.linspace(
                left, right - 1, int(right - left),
                device=calib.device))  # pytorch's meshgrid has indexing='ij'

        x = (j - cx) / focal
        y = (i - cy) / focal
        z = torch.sqrt(radian**2 - x**2)
        # z = torch.ones_like(x)
        starts = torch.stack([x, y, z], -1)
        ends = torch.stack([x, y, -z], -1)
        calib = torch.inverse(calib)
        R, t = calib[:3, :3], calib[:3, 3]

        rays_s = torch.sum(starts[..., None, :] * R, -1) + t
        rays_e = torch.sum(ends[..., None, :] * R, -1) + t

        return rays_s, rays_e

    def get_rays_perspective(self, bbox, w2c, cam):
        """
        bbox: bounding box [top, bottom, left, right]
        w2c: 4x4 rotation matrix
        cam: perspective camera parameters [fx, fy, cx, cy, (if distortion), near, far]
        """
        top, bottom, left, right = bbox
        near, far = cam[-2], cam[-1]
        top, bottom, left, right = int(top), int(bottom), int(left), int(right)
        i, j = torch.meshgrid(
            torch.linspace(top,
                           bottom - 1,
                           int(bottom - top),
                           device=w2c.device),
            torch.linspace(left,
                           right - 1,
                           int(right - left),
                           device=w2c.device))
        x = (j - cam[2]) / cam[0]
        y = (i - cam[3]) / cam[1]
        if len(cam) > 6:
            xp, yp = x, y
            for _ in range(3):  # iter to undistort
                x2 = x * x
                y2 = y * y
                xy = x * y
                r2 = x2 + y2
                c = (1 + r2 * (cam[4] + r2 * (cam[5] + r2 * cam[8])))
                x = (xp - cam[6] * 2 * xy - cam[7] *
                     (r2 + 2 * x2)) / (c + 1e-9)
                y = (yp - cam[7] * 2 * xy - cam[6] *
                     (r2 + 2 * y2)) / (c + 1e-9)
        z = torch.ones_like(x)
        starts = torch.stack([x * near, y * near, z * near], -1)
        ends = torch.stack([x * far, y * far, z * far], -1)
        c2w = torch.inverse(w2c)
        R, t = c2w[:3, :3], c2w[:3, 3]

        rays_s = torch.sum(starts[..., None, :] * R, -1) + t
        rays_e = torch.sum(ends[..., None, :] * R, -1) + t
        # rs = rays_s.cpu().numpy().reshape(-1,3)
        # re = rays_e.cpu().numpy().reshape(-1,3)
        return rays_s, rays_e

    def make_att_input(self, pts, viewdirs, calibs, smpl):
        """Prepare input for multiview attention based SSOAB."""
        if self.projection_mode == 'perspective':
            cam_c = torch.inverse(calibs)[:, :3, 3]
            attdirs = cam_c[None, :, :].expand(pts.shape[0], -1,
                                               -1) - pts[:, None, :].expand(
                                                   -1, self.num_views, -1)
            if smpl is not None:
                # print(viewdirs.shape, smpl['rot'].shape)
                viewdirs = viewdirs @ smpl['rot'][0]

                attdirs = (attdirs.view(-1, 3) @ smpl['rot'][0]).view(
                    attdirs.shape)
            attdirs = torch.cat([viewdirs[:, None, :], attdirs], dim=1)
            attdirs = attdirs / torch.clamp(
                torch.norm(attdirs, dim=-1, keepdim=True), min=1e-9)
            if self.angle_diff:
                viewdirs = viewdirs / torch.clamp(
                    torch.norm(viewdirs, dim=-1, keepdim=True), min=1e-9)
                attdirs = torch.sum(attdirs * viewdirs.unsqueeze(1),
                                    dim=-1,
                                    keepdim=True)
        else:
            ## c2w @ [0,0,1] is equvilant to c2w[:3, 2] back tracing attention direction
            attdirs = torch.inverse(calibs)[:, :3, 2]  # [num_views, 3]
            attdirs = attdirs[None, ...].repeat([pts.shape[0], 1, 1])
            if smpl is not None:
                viewdirs = viewdirs @ smpl['rot'][0]
                attdirs = attdirs @ smpl['rot'][0]
            attdirs = torch.cat([viewdirs, attdirs],
                                dim=0)  # [(num_views+1), 3]
            attdirs = attdirs / torch.norm(attdirs, dim=-1, keepdim=True)

        return attdirs

    def make_nerf_input(self,
                        pts,
                        feats,
                        images,
                        smpl,
                        calibs,
                        mesh_param,
                        persps=None,
                        is_train=True):
        """Aggregate Geometric Body Shape Embedding for NeRF input."""
        nerf_input, source_rgb = [], None

        # Convert query point to normalized body coordinate (normalized scale and body orientation)
        center, spatial_freq = mesh_param['center'], mesh_param['spatial_freq']
        if self.use_nml:
            # points normalized to volume [-1,1]^3
            self.pts_nml = ((pts - center) * spatial_freq /
                            (self.width / 2)).requires_grad_()
            if self.use_smpl_sdf:
                self.pts_nml = self.pts_nml @ smpl['rot'][
                    0]  # rotate to smpl volume, with smpl root node facing front
            nerf_input.append(self.pts_nml)
        else:
            nerf_input.append(pts)

        # Body shape embedding
        if self.use_smpl_sdf or self.use_t_pose:
            self.mesh_searcher.set_mesh(smpl['verts'], smpl['faces'])
            closest_pts, closest_idx = self.mesh_searcher.nearest_points(pts)
            pts_first = pts
            closest_pts_first = closest_pts
            vertex_first = smpl['verts']
            faces_first = smpl['faces']

            if self.use_t_pose:
                closest_faces = smpl['faces'][closest_idx.long()]
                t_pose_verts = smpl['t_verts'][closest_faces.long()]
                t_pose_coords = t_pose_verts.mean(dim=1)
                # T-pose correspondance
                nerf_input.append(t_pose_coords)
                tpose_record = t_pose_coords
            if self.use_smpl_sdf:
                reg_vecs = pts - closest_pts
                if self.use_nml:
                    reg_vecs = reg_vecs * spatial_freq / (
                        self.width / 2)  # normalized to volume [-1,1]^3
                    reg_vecs = reg_vecs @ smpl['rot'][
                        0]  # rotate to smpl volume, with smpl root node facing front
                signs = self.mesh_searcher.inside_mesh(pts)
                self.alpha_smpl = (signs + 1) / 2
                norm = torch.norm(reg_vecs, dim=1, keepdim=True) + 1e-8
                sdf = norm * signs[..., None]
                # Normalized SDF Gradient
                nerf_input.append(reg_vecs / norm)
                # SDF (scale for a constant for faster convergence)
                nerf_input.append(torch.tanh(sdf * 20))
                sdf_norm = reg_vecs / norm
                sdf_scale = torch.tanh(sdf * 20)

        # Multiview image feature
        if feats is not None:
            xyz = self.projection(
                pts.permute((1, 0))[None,
                                    ...].expand([calibs.shape[0], -1, -1]),
                calibs, persps)
            xy = xyz[:, :2, :]  # [self.num_views, 2, self.N_samples]
            if persps is not None:
                xy = xy / torch.tensor([[[self.width], [self.height]]], \
                                       dtype=xyz.dtype, device=xyz.device) * 2 - 1
            latent = index(feats,
                           xy)  # [self.num_views, C, self.N_samples(*2)]
            latent = latent.permute(
                (2, 0, 1))  # [self.N_samples(*2), self.num_views, C]
            source_rgb = index(images[:self.num_views], xy)
            source_rgb = source_rgb.permute((2, 0, 1))
            latent = torch.cat([latent, source_rgb], -1)

            nerf_input = [
                inp[:, None, :].expand([-1, self.num_views, -1])
                for inp in nerf_input
            ]  # expand each feature to num_views
            nerf_input += [latent]

        nerf_input = torch.cat(nerf_input,
                               dim=-1)  # [self.N_samples, self.num_views, C]
        return nerf_input, source_rgb

    def make_nerf_output(self,
                         nerf_output,
                         t_vals,
                         norm,
                         source_rgb,
                         is_train=True):
        """Renders ray by integrating sample points."""
        dists = t_vals[..., 1:] - t_vals[..., :-1]
        dists = torch.cat([
            dists,
            torch.tensor([1e10], device=dists.device).expand(
                dists[..., :1].shape)
        ], -1)  # [N_rays, N_samples]
        dists = dists * norm
        N_samples = t_vals.shape[-1]

        rgb = torch.sigmoid(nerf_output[..., :3])  # [N_rays, N_samples, 3]
        noise = torch.randn(nerf_output[..., 3].shape,
                            device=nerf_output.device) if is_train else 0
        alpha = 1. - torch.exp(-F.relu(
            (nerf_output[..., 3] + noise)))  # [N_rays, N_samples]
        weights = alpha * torch.cumprod(
            torch.cat([
                torch.ones((alpha.shape[0], 1), device=nerf_output.device),
                1. - alpha + 1e-10
            ], -1), -1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
        if self.use_attention:
            att = nerf_output[..., 4:]
            source_rgb = source_rgb.reshape([-1, N_samples, self.num_views, 3])
            source_rgb = torch.cat([rgb.unsqueeze(-2), source_rgb], dim=-2)
            source_rgb_att = torch.sum(source_rgb * att[..., None], dim=-2)
            att_rgb_map = torch.sum(weights[..., None] * source_rgb_att,
                                    -2)  # [N_rays, 3]
            rgb_map = torch.cat([rgb_map, att_rgb_map], -1)
            if self.debug:
                for i in range(self.num_views):
                    source_rgb_i = source_rgb[:, :, i, :] * att[:, :, i, None]
                    rgb_map_i = torch.sum(weights[..., None] * source_rgb_i,
                                          -2)
                    rgb_map = torch.cat([rgb_map, rgb_map_i], -1)

        acc_map = torch.sum(weights, -1)
        if self.use_white_bkgd:
            rgb_map = rgb_map + (1. - acc_map[..., None])

        return rgb_map, weights

    def render_rays(self,
                    ray_batch,
                    feats,
                    images,
                    masks,
                    calibs,
                    smpl,
                    mesh_param,
                    scan=None,
                    persps=None,
                    q_persps=None,
                    is_train=True):
        """Volumetric rendering.

        Args:
        ray_batch: array of shape [batch_size, ...]. All information necessary
            for sampling along a ray, including: ray origin, ray direction, min
            dist, max dist, and unit-magnitude viewing direction.
        """
        self.alpha, self.angle_diff_grad = None, None
        eps = 1e-9
        N_rays = ray_batch.shape[0]
        rays_s, rays_e = ray_batch[:, 0:3], ray_batch[:,
                                                      3:6]  # [N_rays, 3] each

        t_vals = torch.linspace(0.,
                                1.,
                                steps=self.N_samples,
                                device=ray_batch.device)
        t_vals = t_vals.repeat([N_rays, 1])
        # perturb during training
        if is_train:
            t_rand = (torch.rand(t_vals.shape, device=ray_batch.device) -
                      0.5) / (self.N_samples - 1)
            t_vals = t_vals + t_rand

        pts = rays_e[:, None, :] * t_vals[..., None] + (
            1 - t_vals[..., None]) * rays_s[:, None, :]
        pts = pts.reshape(-1, 3)

        # Use visual hull to skip sample points outside the body
        inside, smpl_vis, scan_vis = None, None, None
        if self.use_vh:
            inside, smpl_vis, scan_vis = self.inside_pts_vh(
                pts, masks, smpl, calibs, persps)
            try:
                pts = pts[inside]
            except:
                print(inside.sum(), pts.shape)
            if len(pts) == 0:
                return self.default_rgb([N_rays, self.rgb_ch], dtype=torch.float32, device=ray_batch.device), \
                       torch.zeros([N_rays], dtype=torch.float32, device=ray_batch.device)

        # When train RenderPeople with scan ground truth, prepare 3D supervision
        if is_train and scan is not None:
            scan_verts, scan_faces = scan
            self.mesh_searcher.set_mesh(scan_verts, scan_faces)
            self.alpha_gt = (self.mesh_searcher.inside_mesh(pts) + 1) / 2

        # Prepare attention based appereance blending input
        viewdirs = (rays_s - rays_e)[:, None, :].expand(-1, self.N_samples, -1)
        viewdirs = viewdirs.reshape(-1, 3)[inside].requires_grad_()
        attdirs = self.make_att_input(pts, viewdirs, calibs,
                                      smpl) if self.use_attention else []

        # Prepare geometry body shape embedding input for NeRF
        nerf_input, source_rgb = self.make_nerf_input(pts, feats, images, smpl,
                                                      calibs, mesh_param,
                                                      persps)

        # Feed to the network
        nerf_output = torch.cat([self.nerf(nerf_input[i:i + self.chunk], attdirs[i:i + self.chunk], smpl_vis=smpl_vis) \
                                 for i in range(0, nerf_input.shape[0], self.chunk)], 0)
        self.alpha = torch.sigmoid(nerf_output[..., 3] * self.gamma)

        # If RenderPeople available, supervise the occlusion
        if self.use_occlusion_net:
            if is_train and scan is not None:
                self.occ_gt = scan_vis.float()
                self.occ = nerf_output[:, -self.num_views:]
            nerf_output = nerf_output[:, :-self.num_views]

        # Regularize the alpha distribution
        if self.regularization and is_train:
            self.alpha_grad = \
            torch.autograd.grad(self.alpha, self.pts_nml, grad_outputs=torch.ones_like(self.alpha), retain_graph=True)[
                0]

        # use sparse multiplication to aggregate points inside and outside the visual hull for NeRF integration
        if self.use_vh:
            inside_idx = torch.nonzero(inside)
            row_cols = torch.cat([
                inside_idx.view(1, -1),
                torch.arange(len(inside_idx), device=pts.device).view(1, -1)
            ], 0)
            I = torch.sparse_coo_tensor(row_cols,
                                        torch.ones(len(inside_idx),
                                                   dtype=pts.dtype,
                                                   device=pts.device),
                                        size=(N_rays * self.N_samples,
                                              len(inside_idx)))
            nerf_output = torch.sparse.mm(I, nerf_output)
            nerf_output[~inside, :4] = -1e4
            full_source_rgb = torch.zeros(
                [N_rays * self.N_samples, self.num_views, 3],
                device=pts.device)
            full_source_rgb[inside] = source_rgb
            source_rgb = full_source_rgb
        nerf_output = nerf_output.view(N_rays, self.N_samples, -1)

        norm = torch.norm(rays_e - rays_s, dim=-1, keepdim=True)
        if self.use_nml:
            center, spatial_freq = mesh_param['center'], mesh_param[
                'spatial_freq']
            norm = norm * spatial_freq / (self.width / 2)
        rgb_map, weights = self.make_nerf_output(nerf_output,
                                                 t_vals,
                                                 norm,
                                                 source_rgb,
                                                 is_train=is_train)
        z_vals = t_vals * q_persps[-2] + (1 - t_vals) * q_persps[
            -1] if persps is not None and q_persps is not None else 2 * t_vals - 1
        depth = torch.sum(weights * z_vals, -1)

        # Regularize the angle difference of apperance
        if self.angle_diff and is_train:
            self.angle_diff_grad = \
            torch.autograd.grad(rgb_map, viewdirs, grad_outputs=torch.ones_like(rgb_map), retain_graph=True)[0]
        return rgb_map, depth

    def inside_pts_vh(self, pts, masks, smpl, calibs, persps=None):
        """Valid sample point selection via visual hull."""
        '''
        rot torch.Size([3, 3])
        verts torch.Size([10475, 3])
        faces torch.Size([20908, 3])
        betas torch.Size([10])
        t_vert torch.Size([10475, 3])
        t_face torch.Size([20908, 3])
        torch.Size([262144, 3]) torch.Size([4, 1, 512, 512]) torch.Size([4, 4
        , 4]) torch.Size([4, 11])
        '''

        xyz = self.projection(
            pts.permute((1, 0))[None, ...].expand([calibs.shape[0], -1, -1]),
            calibs, persps)
        xy = xyz[:, :2, :]
        if persps is not None:
            xy = xy / torch.tensor([[[self.width], [self.height]]], \
                                   dtype=xyz.dtype, device=xyz.device) * 2 - 1
        inside = index(masks, xy, 'nearest')
        inside = torch.prod(inside, dim=0).squeeze(0) > 0
        if (inside.sum() < self.chunk * 0.7) and self.use_vh_free:
            n_samples = inside.sum() * 0.3
            idx = torch.randperm(len(inside))[:n_samples]
            inside[idx] = True
        smpl_vis, scan_vis = None, None
        if self.use_occlusion:
            smpl_depth = index(smpl['depth'], xy,
                               'nearest').squeeze(1).permute((1, 0))[inside]
            depth = xyz[:, 2, :].permute((1, 0))[inside]
            smpl_vis = ((depth - smpl_depth) <= 0) * (smpl_depth > 0)
        if self.use_occlusion_net and 'scan_depth' in smpl.keys():
            scan_depth = index(smpl['scan_depth'], xy,
                               'nearest').squeeze(1).permute((1, 0))[inside]
            depth = xyz[:, 2, :].permute((1, 0))[inside]
            scan_vis = ((depth - scan_depth) <= 0) * (scan_depth > 0)
        return inside, smpl_vis, scan_vis

    def render(self,
               feats,
               images,
               masks,
               calibs,
               bbox,
               mesh_param,
               smpl=None,
               scan=None,
               persps=None):
        """Render a image from give camera pose."""
        self.debug_idx += 1
        if persps is None:
            rays_s, rays_e = self.get_rays(bbox,
                                           calibs[-1])  # (H, W, 3), (H, W, 3)
        else:
            rays_s, rays_e = self.get_rays_perspective(bbox, calibs[-1],
                                                       persps[-1])

        top, bottom, left, right = bbox
        top = int(top)
        bottom = int(bottom)
        left = int(left)
        right = int(right)
        gt = images[-1].permute((1, 2, 0))[top:bottom, left:right]
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0,
                               bottom - top - 1,
                               int(bottom - top),
                               device=calibs.device),
                torch.linspace(0,
                               right - left - 1,
                               int(right - left),
                               device=calibs.device)), -1)  # (H, W, 2)

        coords = coords.view(-1, 2)  # (H * W, 2)
        select_inds = np.random.choice(coords.shape[0],
                                       size=[self.N_rand * self.vh_overhead],
                                       replace=False)  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_s = rays_s[select_coords[:, 0], select_coords[:,
                                                           1]]  # (N_rand, 3)
        rays_e = rays_e[select_coords[:, 0], select_coords[:,
                                                           1]]  # (N_rand, 3)
        batch_rays = torch.cat([rays_s, rays_e], 1)
        target = gt[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)

        persps = persps[:self.num_views] if persps is not None else None
        rgb, _ = self.render_rays(batch_rays, feats, images[:self.num_views],
                                  masks[:self.num_views],
                                  calibs[:self.num_views], smpl, mesh_param,
                                  scan, persps)
        loss = self.cal_loss(rgb, target)
        outputs = {'loss': loss, 'num_samples': rgb.shape[0]}

        return outputs

    def render_path(self,
                    feats,
                    images,
                    masks,
                    calibs,
                    bbox,
                    mesh_param,
                    smpl=None,
                    scan=None,
                    persps=None):
        """Render a path given trajectory."""
        top, bottom, left, right = bbox
        height, width = max(self.height,
                            bottom - top), max(self.width, right - left)
        calibs_source, calibs_query = calibs[:self.num_views], calibs[
            self.num_views:]
        persps_source = persps[:self.num_views] if persps is not None else None
        persps_query = persps[self.num_views:] if persps is not None else None

        rgbs, depths = [], []
        # inference
        for idx, calib in enumerate(tqdm(calibs_query)):
            if persps is None:
                rays_s, rays_e = self.get_rays(bbox,
                                               calib)  # (H, W, 3), (H, W, 3)
            else:
                persp = persps[self.num_views + idx]
                rays_s, rays_e = self.get_rays_perspective(bbox, calib, persp)
            batch_rays = torch.cat([rays_s.view(-1, 3), rays_e.view(-1, 3)], 1)
            # self.idx = 0
            rgb, depth = [], []
            for i in range(0, batch_rays.shape[0], self.N_rand_infer):
                c, d = self.render_rays(batch_rays[i:i + self.N_rand_infer].detach(), feats, images[:self.num_views], \
                                        masks[:self.num_views], calibs_source, smpl, mesh_param, persps=persps_source,
                                        q_persps=persps_query[idx], is_train=False)
                rgb.append(c[:, :self.rgb_ch])
                depth.append(d)
            rgb = torch.cat(rgb, 0).clone()
            depth = torch.cat(depth, 0).clone()
            img = self.default_rgb((height, width, self.rgb_ch),
                                   dtype=torch.float32,
                                   device=rgb.device)
            dimg = torch.zeros((height, width),
                               dtype=torch.float32,
                               device=rgb.device)
            img[top:bottom, left:right] = rgb.view(int(bottom - top),
                                                   int(right - left),
                                                   self.rgb_ch)
            dimg[top:bottom, left:right] = depth.view(int(bottom - top),
                                                      int(right - left))
            rgbs.append(img)
            depths.append(dimg)
        rgbs = torch.stack(rgbs, dim=0)
        depths = torch.stack(depths, dim=0)

        return rgbs, depths

    def reconstruct(self,
                    feats,
                    images,
                    masks,
                    calibs,
                    bbox,
                    mesh_param,
                    smpl=None,
                    scan=None,
                    persps=None):
        """Mesh Reconstruction borrowed form PIFu."""
        # Deterimine 3D bounding box
        center, spatial_freq = mesh_param['center'].cpu().numpy(
        ), mesh_param['spatial_freq']
        top, bottom, left, right = bbox
        left, right = 0, 512
        bb_min = [
            left - self.width / 2, top - self.height / 2, left - self.width / 2
        ]
        bb_max = [
            right - self.width / 2, bottom - self.height / 2,
            right - self.width / 2
        ]

        # Make mesh grid in normalized body cordinate
        linspaces = [
            np.linspace(bb_min[i], bb_max[i], self.N_grid)
            for i in range(len(bb_min))
        ]
        grids = np.stack(
            np.meshgrid(linspaces[0],
                        linspaces[1],
                        linspaces[2],
                        indexing='ij'), -1)
        sh = grids.shape
        pts = grids / spatial_freq + center
        recon_kwargs = {
            'feats': feats,
            'images': images,
            'smpl': smpl,
            'calibs': calibs[:self.num_views],
            'mesh_param': mesh_param,
            'persps': persps[:self.num_views] if persps is not None else None
        }

        # Reconstruct use progressive octree reconstrution
        sdf = self.octree_reconstruct(pts, masks, **recon_kwargs)
        verts, faces, normals, _ = measure.marching_cubes_lewiner(
            sdf, self.threshold)

        # Convert marching cubes coordinate back to world coordinate
        verts = (verts - self.N_grid / 2) / self.N_grid * np.array(
            [[right - left, bottom - top, right - left]])
        verts = verts / spatial_freq + center

        # use laplacian smooth if the mesh is noisy
        if self.opt.laplacian > 0:
            mesh = trimesh.Trimesh(verts, faces, process=False)
            trimesh.smoothing.filter_laplacian(mesh,
                                               iterations=self.opt.laplacian)
            verts, faces = mesh.vertices, mesh.faces
        pts = torch.tensor(verts, dtype=torch.float32, device=calibs.device)

        viewdirs = torch.from_numpy(normals.astype(np.float32)).to(
            calibs.device)
        attdirs = self.make_att_input(pts, viewdirs, calibs[:self.num_views],
                                      smpl) if self.use_attention else []

        rgbs = []
        for i in range(0, pts.shape[0], self.chunk):
            nerf_input, source_rgb = self.make_nerf_input(
                pts[i:i + self.chunk], **recon_kwargs)
            nerf_output = self.nerf(nerf_input, attdirs[i:i + self.chunk])
            rgb = torch.sigmoid(nerf_output[..., :3])
            if self.use_attention:
                att = nerf_output[..., 4:4 + self.num_views + 1]
                source_rgb = source_rgb.view(-1, self.num_views, 3)
                source_rgb = torch.cat([rgb[:, None], source_rgb], dim=-2)
                rgb = torch.sum(source_rgb * att[..., None], dim=-2)
            rgbs.append(rgb)
        rgbs = torch.cat(rgbs, 0).cpu().numpy()

        return verts, faces, rgbs

    def octree_reconstruct(self, coords, masks, **kwargs):
        """We use Octree recontruction for higher resolution reconstruction
        borrowed form PIFu."""
        device = kwargs['calibs'].device
        calibs = kwargs['calibs']
        persps = kwargs['persps']
        resolution = [self.N_grid, self.N_grid, self.N_grid]
        sdf = np.zeros(resolution)
        notprocessed = np.zeros(resolution, dtype=np.bool)
        notprocessed[:-1, :-1, :-1] = True
        # only voxel grids lies in the visual hull are to processed
        if self.use_vh:
            dilation_kernel = torch.ones((1, 1, 5, 5),
                                         device=device,
                                         dtype=torch.float32)
            masks = torch.clamp(
                torch.nn.functional.conv2d(masks,
                                           dilation_kernel,
                                           padding=(2, 2)), 0, 1)
            masks_np = masks.permute([0, 2, 3, 1]).cpu().numpy()
            pts = coords.reshape(-1, 3)
            notprocessed = notprocessed.reshape(-1)
            for i in range(0, pts.shape[0], self.chunk):
                inside, _, _ = self.inside_pts_vh(
                    torch.tensor(pts[i:i + self.chunk],
                                 dtype=torch.float32,
                                 device=device), masks, kwargs['smpl'], calibs,
                    persps)
                inside = inside.cpu().numpy()
                outside = np.logical_not(inside.astype(np.bool))
                notprocessed_chunk = notprocessed[i:i + self.chunk].copy()
                notprocessed_chunk[outside] = False
                notprocessed[i:i + self.chunk] = notprocessed_chunk
            notprocessed = notprocessed.reshape(resolution)

        grid_mask = np.zeros(resolution, dtype=np.bool)
        reso = self.N_grid // 64

        center = kwargs['mesh_param']['center'].cpu().numpy()
        while reso > 0:
            grid_mask[0:self.N_grid:reso, 0:self.N_grid:reso,
                      0:self.N_grid:reso] = True
            test_mask = np.logical_and(grid_mask, notprocessed)
            pts = coords[test_mask, :]
            if pts.shape[0] == 0:
                print('break')
                break

            pts_tensor = torch.tensor(pts, dtype=torch.float32, device=device)
            nerf_output = []
            for i in range(0, pts_tensor.shape[0], self.chunk):
                nerf_input, _ = self.make_nerf_input(
                    pts_tensor[i:i + self.chunk], **kwargs)
                nerf_output.append(self.nerf(nerf_input, alpha_only=True))
            nerf_output = torch.cat(nerf_output, dim=0)
            sdf[test_mask] = torch.sigmoid(
                nerf_output * self.gamma).detach().cpu().numpy().reshape(-1)

            notprocessed[test_mask] = False

            # do interpolation
            if reso <= 1:
                break
            grid = np.arange(0, self.N_grid, reso)
            v = sdf[tuple(np.meshgrid(grid, grid, grid, indexing='ij'))]
            vs = [
                v[:-1, :-1, :-1], v[:-1, :-1, 1:], v[:-1, 1:, :-1], v[:-1, 1:,
                                                                      1:],
                v[1:, :-1, :-1], v[1:, :-1, 1:], v[1:, 1:, :-1], v[1:, 1:, 1:]
            ]
            grid = grid[:-1] + reso // 2
            nonprocessed_grid = notprocessed[tuple(
                np.meshgrid(grid, grid, grid, indexing='ij'))]

            v = np.stack(vs, 0)
            v_min = v.min(0)
            v_max = v.max(0)
            v = 0.5 * (v_min + v_max)
            skip_grid = np.logical_and(((v_max - v_min) < 0.01),
                                       nonprocessed_grid)
            xs, ys, zs = np.where(skip_grid)
            for x, y, z in zip(xs * reso, ys * reso, zs * reso):
                sdf[x:(x + reso + 1), y:(y + reso + 1),
                    z:(z + reso + 1)] = v[x // reso, y // reso, z // reso]
                notprocessed[x:(x + reso + 1), y:(y + reso + 1),
                             z:(z + reso + 1)] = False
            reso //= 2

        return sdf.reshape(resolution)
