# Copyright (c) OpenMMLab. All rights reserved.

import torch
from torch import nn

from ..builder import SAMPLERS
# from .utils import mark_untrained_density_grid
from .utils import *


@SAMPLERS.register_module()
class NGPGridSampler(nn.Module):
    """perform ray-marching as ngp paper described in Appendix-E."""
    def __init__(
        self,
        update_grid_freq=16,
        update_block_size=5000000,
        n_rays_per_batch=4096,
        cone_angle_constant=0.00390625,
        near_distance=0.2,
        target_batch_size=1 << 18,
        rgb_activation=2,
        density_activation=3,
    ):
        '''
            target_batch_size: n_pts after compacted
        '''
        super().__init__()
        self.update_grid_freq = update_grid_freq
        self.update_block_size = update_block_size
        self.n_rays_per_batch = n_rays_per_batch
        self.target_batch_size = target_batch_size
        self.rgb_activation = rgb_activation
        self.density_activation = density_activation
        self.density_mlp_padded_density_output_width = int(1)

        n_threads_linear = 128
        self.density_grid_ema_step = int(0)
        self.NERF_CASCADES = int(8)  # same as 'raymarch_shared.h'
        self.NERF_GRIDSIZE = int(128)  # same as 'raymarch_shared.h'
        self.near_distance = float(0.05)  # same as 'raymarch_shared.h'
        self.ema_grid_decay = float(0.95)
        self.MAX_STEP = 1024  # same as 'raymarch_shared.h'
        self.cone_angle_constant = float(0.00390625)
        self.NERF_MIN_OPTICAL_THICKNESS = float(
            0.01)  # same as 'raymarch_shared.h'
        self.num_coords_elements = self.n_rays_per_batch * self.MAX_STEP

        self.density_n_elements = self.NERF_CASCADES * \
            self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_GRIDSIZE
        density_grid_bitfield_n_elements = self.NERF_GRIDSIZE * \
            self.NERF_GRIDSIZE*self.NERF_GRIDSIZE
        size_including_mips = self.NERF_GRIDSIZE * \
            self.NERF_GRIDSIZE*self.NERF_GRIDSIZE*self.NERF_CASCADES//8

        self.density_grid_tmp = torch.zeros([self.density_n_elements],
                                            dtype=torch.float32)

        self.density_grid_mean = torch.zeros([
            self.div_round_up(density_grid_bitfield_n_elements,
                              n_threads_linear)
        ],
                                             dtype=torch.float32)

        # density_grid_bitfield need to be reload when test
        # but no need to update by optimizer when train
        density_grid_bitfield = torch.zeros([size_including_mips],
                                            dtype=torch.uint8)
        self.register_buffer('density_grid_bitfield', density_grid_bitfield)

        self.measured_batch_size = torch.zeros((1, ), dtype=torch.int32)

        self.iter_n = 0

    def set_data(self, alldata, datainfo):
        self.resolutions = [datainfo['H'], datainfo['W']]
        self.transforms = torch.tensor(alldata['poses'], dtype=torch.float32)
        self.focal = torch.tensor(alldata['focal'], dtype=torch.float32)
        self.aabb_scale = alldata['aabb_scale']
        self.aabb_range = alldata['aabb_range']
        self.metadata = torch.tensor(alldata['metadata'], dtype=torch.float32)
        self.n_img = alldata['poses'].shape[0]
        self.max_cascade = 0
        while (1 << self.max_cascade) < self.aabb_scale:
            self.max_cascade += 1

    def set_iter(self, iter_n):
        self.iter_n = iter_n

    def update_density_grid_func(self, n_uniform_density_grid_samples,
                                 n_nonuniform_density_grid_samples, mlp):
        n_elements = self.density_n_elements
        n_density_grid_samples = n_uniform_density_grid_samples + \
            n_nonuniform_density_grid_samples
        if not hasattr(self, 'density_grid'):
            self.density_grid = mark_untrained_density_grid(
                self.focal, self.transforms, n_elements, self.n_img,
                self.resolutions, self.device)
        # print('mark_untrained_density_grid', self.density_grid.shape,
        #         self.density_grid.sum())

        positions_uniform, indices_uniform = \
            generate_grid_samples_nerf_nonuniform(
            self.density_grid, n_uniform_density_grid_samples,
            self.density_grid_ema_step, self.max_cascade,
            -0.01, self.aabb_range, self.device)

        # print('positions_uniform/indices_uniform', positions_uniform.shape,
        #     indices_uniform.shape)
        # print('positions_uniform/indices_uniform', positions_uniform.sum(),
        #     indices_uniform.sum())

        positions_nonuniform, indices_nonuniform = \
            generate_grid_samples_nerf_nonuniform(
                self.density_grid, n_nonuniform_density_grid_samples,
                self.density_grid_ema_step, self.max_cascade,
                self.NERF_MIN_OPTICAL_THICKNESS,
                self.aabb_range, self.device)

        # print('positions_nonuniform/indices_uniform', positions_nonuniform.shape,
        #     indices_nonuniform.shape)

        density_grid_positions = torch.cat(
            [positions_uniform, positions_nonuniform])
        density_grid_indices = torch.cat([indices_uniform, indices_nonuniform])
        density_grid_positions = density_grid_positions.reshape(-1, 3)

        with torch.no_grad():
            res = []
            grid_bs = self.update_block_size
            for i in range(0, density_grid_positions.shape[0], grid_bs):
                pts_flat = density_grid_positions[i:i + grid_bs]
                res.append(mlp.run_density(pts_flat))
            density = torch.cat(res, 0)

        self.density_grid_tmp = splat_grid_samples_nerf_max_nearest_neighbor(
            density, density_grid_indices,
            self.density_mlp_padded_density_output_width,
            n_density_grid_samples, self.density_grid_tmp, self.device)

        # print('density', density.shape, density.sum())
        # print('self.density_grid_tmp', self.density_grid_tmp.shape,
        #     self.density_grid_tmp.sum(), self.density_grid_tmp.device)

        self.density_grid = ema_grid_samples_nerf(self.density_grid_tmp,
                                                  self.density_grid,
                                                  n_elements,
                                                  self.ema_grid_decay)

        # print('density_grid', self.density_grid.shape,
        #     self.density_grid.device, self.density_grid.sum())

        self.density_grid = self.density_grid.detach()
        self.density_grid_ema_step += 1

        self.density_grid_bitfield, self.density_grid_mean = update_bitfield(
            self.density_grid, self.density_grid_mean,
            self.density_grid_bitfield, self.device)

        # tmp_bit = torch.tensor(self.density_grid_bitfield, dtype=torch.float32)/255
        # tmp_bit = (self.density_grid_bitfield>0).float()
        # print('tmp_bit', tmp_bit.shape, tmp_bit.sum(), tmp_bit.dtype,
        #     tmp_bit.min(), tmp_bit.max(), flush=True)
        # print('density_grid_mean', self.density_grid_mean.shape,
        #     self.density_grid_mean.sum(), flush=True)
        # exit(0)

    def update_density_grid(self, mlp):
        n_cascades = self.max_cascade + 1
        M = self.NERF_GRIDSIZE * self.NERF_GRIDSIZE * self.NERF_GRIDSIZE * n_cascades
        if self.iter_n < 256:
            self.update_density_grid_func(M, 0, mlp)
        else:
            self.update_density_grid_func(M // 4, M // 4, mlp)

    def check_device(self, data):
        device = data['rays_o'].device
        self.device = device
        if self.density_grid_tmp != device:
            attrs = [
                'transforms', 'focal', 'metadata', 'density_grid_mean',
                'density_grid_bitfield', 'density_grid_tmp',
                'measured_batch_size'
            ]
            for attr in attrs:
                v = getattr(self, attr).to(device).contiguous()
                setattr(self, attr, v)

    def sample(self, data, mlp, is_test=False):
        is_training = (not is_test)
        self.check_device(data)

        if is_training:
            if self.iter_n % self.update_grid_freq == 0 or (not hasattr(
                    self, 'density_grid')):
                self.update_density_grid(mlp)

        rays_o = data['rays_o'].contiguous()
        rays_d = data['rays_d'].contiguous()
        if 'img_ids' in data:
            img_ids = data['img_ids'].to(torch.int32).contiguous()
        if 'bg_color' in data:
            data['bg_color'] = data['bg_color'].to(torch.float32).contiguous()

        coords, rays_index, rays_numsteps, rays_numsteps_counter = rays_sampler(
            rays_o, rays_d, img_ids, self.density_grid_bitfield, self.metadata,
            self.transforms, self.aabb_range, self.near_distance,
            self.cone_angle_constant, self.num_coords_elements, self.device)
        coords_pos = coords[..., :3].detach()
        coords_dir = coords[..., 4:].detach()

        # print('rays_o', rays_o.shape, rays_o.min(), rays_o.max(), rays_o.sum())
        # print('rays_d', rays_d.shape, rays_d.min(), rays_d.max(), rays_d.sum())
        # print('img_ids', img_ids.shape, img_ids.min(), img_ids.max(), img_ids.sum())
        # print('coords_pos', coords_pos.shape, coords_pos.min(), coords_pos.max(),
        #     coords_pos.mean())
        # print('coords_dir', coords_dir.shape, coords_dir.min(), coords_dir.max(),
        #     coords_dir.mean())
        # print('rays_numsteps_counter', rays_numsteps_counter)
        # print('self.num_coords_elements', self.num_coords_elements)
        # exit(0)

        if not is_training:
            self.coords = coords.detach()
            self.rays_numsteps = rays_numsteps.detach()
            data['pts'], data['viewdirs'] = coords_pos, coords_dir
            return data

        tmp_data = {'pts': coords_pos, 'viewdirs': coords_dir}
        nerf_outputs = mlp(tmp_data)['raw'].detach().to(torch.float32)
        coords_compacted, rays_numsteps_compacted, compacted_numstep_counter = \
            compacted_coords(nerf_outputs, coords, rays_numsteps,
                self.target_batch_size, self.aabb_range, self.rgb_activation,
                self.density_activation, self.device)

        # compacted_pos, compacted_dir = coords_compacted[..., :3].detach(), coords_compacted[..., 4:].detach()
        # print('coords_pos', coords_pos.shape, coords_pos.min(), coords_pos.max(),
        #     coords_pos.mean())
        # print('coords_dir', coords_dir.shape, coords_dir.min(), coords_dir.max(),
        #     coords_dir.mean())

        # print('rays_numsteps_compacted', rays_numsteps_compacted.shape,
        #     rays_numsteps_compacted.min(), rays_numsteps_compacted.max(),
        #     rays_numsteps_compacted.sum())
        # print('compacted_numstep_counter', compacted_numstep_counter)
        # print('nerf_outputs', nerf_outputs.shape, nerf_outputs.min(),
        #     nerf_outputs.max(), nerf_outputs.mean())
        # print('rays_numsteps', rays_numsteps.shape, rays_numsteps.min(),
        #     rays_numsteps.max(), rays_numsteps.sum())
        # exit(0)

        self.measured_batch_size += compacted_numstep_counter
        self.update_batch_rays(is_training)

        self.coords = coords_compacted.detach()
        self.rays_numsteps = rays_numsteps.detach()
        self.rays_numsteps_compacted = rays_numsteps_compacted.detach()
        pts, viewdirs = coords_compacted[..., :3], coords_compacted[..., 4:]
        data['pts'], data['viewdirs'] = pts, viewdirs

        # print('pts', pts.shape, pts.min(), pts.max(),
        #     pts.mean())
        # print('viewdirs', viewdirs.shape, viewdirs.min(), viewdirs.max(),
        #     viewdirs.mean())

        return data

    def update_batch_rays(self, is_training):
        if is_training:
            if self.iter_n % self.update_grid_freq == (self.update_grid_freq -
                                                       1):
                measured_batch_size = max(self.measured_batch_size.item() / 16,
                                          1)
                rays_per_batch = int(self.n_rays_per_batch *
                                     self.target_batch_size /
                                     measured_batch_size)
                self.n_rays_per_batch = int(
                    min(
                        self.div_round_up(int(rays_per_batch), 128) * 128,
                        self.target_batch_size))
                self.measured_batch_size.zero_()

    def div_round_up(self, val, divisor):
        return (val + divisor - 1) // divisor
