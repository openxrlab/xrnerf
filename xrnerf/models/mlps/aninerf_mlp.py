from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from .. import builder
from ..builder import MLPS
from ..networks.utils import *


@MLPS.register_module()
class DeformField(nn.Module):
    def __init__(self, phase, smpl_threshold, bw_mlp, novel_pose_bw_mlp):
        super(DeformField, self).__init__()

        self.phase = phase
        self.smpl_threshold = smpl_threshold
        self.bw_mlp = builder.build_mlp(bw_mlp)
        self.novel_pose_bw_mlp = builder.build_mlp(novel_pose_bw_mlp)

    def get_posed_point_viewdir(self, datas):
        num_pixel, num_sample = datas['pts'].shape[:2]
        world_pts = datas['pts'].view(num_pixel * num_sample, -1)
        smpl_R = datas['smpl_R'][None]
        smpl_T = datas['smpl_T'][None]

        # transform points from the world space to the pose space
        world_pts = world_pts[None]
        posed_pts = world_points_to_pose_points(world_pts, smpl_R, smpl_T)
        viewdirs = datas['rays_d']
        viewdirs = viewdirs[:, None].expand(datas['pts'].shape).contiguous()
        viewdirs = viewdirs.view(num_pixel * num_sample, -1)[None]
        posed_dirs = world_dirs_to_pose_dirs(viewdirs, datas['smpl_R'])

        return posed_pts, posed_dirs

    def get_points_near_smpl(self, posed_pts, posed_dirs, datas):
        smpl_R = datas['smpl_R'][None]
        smpl_T = datas['smpl_T'][None]
        smpl_bw = datas['smpl_bw'][None]

        with torch.no_grad():
            smpl_verts = datas['smpl_verts'][None]
            posed_smpl_verts = world_points_to_pose_points(smpl_verts, smpl_R, smpl_T)

            pbw, pnorm = sample_closest_points(posed_pts, posed_smpl_verts, smpl_bw)
            pnorm = pnorm[..., 0]
            norm_th = self.smpl_threshold
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            posed_pts = posed_pts[pind][None]
            posed_dirs = posed_dirs[pind][None]

        return posed_pts, posed_dirs, pind

    def transform_to_tpose(self, pose_pts, pose_dirs, datas):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        world_verts = datas['smpl_verts']
        posed_smpl_verts = torch.matmul(world_verts - datas['smpl_T'], datas['smpl_R'])[None]
        init_pbw, _ = sample_closest_points(pose_pts, posed_smpl_verts, datas['smpl_bw'][None])
        init_pbw = init_pbw.permute(0, 2, 1)

        # neural blend weights of points at i
        if self.phase == 'novel_pose':
            pbw = self.novel_pose_bw_mlp.calculate_neural_blend_weights(pose_pts, init_pbw,
                    datas['bw_latent_idx'])
        else:
            pbw = self.bw_mlp.calculate_neural_blend_weights(
                pose_pts, init_pbw, datas['bw_latent_idx'] + 1)

        # transform points from i to i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, datas['A'][None])
        tpose = tpose_points_to_pose_points(tpose, pbw, datas['big_A'][None])

        init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                             datas['A'][None])
        tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                             datas['big_A'][None])

        return tpose, pbw, tpose_dirs

    def calculate_tpose_tbw(self, tpose, datas):
        smpl_bw = datas['smpl_bw'][None]
        canonical_smpl_verts = datas['canonical_smpl_verts'][None]
        init_tbw, _ = sample_closest_points(tpose, canonical_smpl_verts, smpl_bw)
        init_tbw = init_tbw.permute(0, 2, 1)
        ind = torch.zeros_like(datas['bw_latent_idx'])
        tbw = self.bw_mlp.calculate_neural_blend_weights(tpose, init_tbw, ind)
        return tbw

    def forward(self, datas):
        posed_pts, posed_dirs = self.get_posed_point_viewdir(datas)
        posed_pts, posed_dirs, pind = self.get_points_near_smpl(posed_pts, posed_dirs, datas)
        # transform points from the pose space to the tpose space
        tpose, pbw, tpose_dirs = self.transform_to_tpose(posed_pts, posed_dirs, datas)

        # calculate neural blend weights of points at the tpose space
        tbw = self.calculate_tpose_tbw(tpose, datas)

        deform_ret = {
            'tpose': tpose[0],
            'tpose_dirs': tpose_dirs[0],
            'pind': pind,
            'pbw': pbw,
            'tbw': tbw
        }

        return deform_ret


@MLPS.register_module()
class TPoseHuman(nn.Module):
    def __init__(self, **kwargs):
        super(TPoseHuman, self).__init__()

        self.density_network = builder.build_mlp(kwargs['density_mlp'])
        self.color_network = builder.build_mlp(kwargs['color_mlp'])

    def calculate_alpha(self, tpose):
        nerf_nn_output = self.density_network(tpose[0])
        alpha = nerf_nn_output[:, :1]
        alpha = alpha[None].transpose(1, 2)
        return alpha

    def forward(self, deform_ret, datas):
        wpts = deform_ret['tpose']
        viewdir = deform_ret['tpose_dirs']

        # calculate nerf
        nerf_nn_output = self.density_network(wpts)
        alpha = nerf_nn_output[:, 0]
        feature_vector = nerf_nn_output[:, 1:]

        # calculate color
        ind = datas['color_latent_idx']
        rgb = self.color_network(wpts, viewdir, feature_vector, ind)

        raw = torch.cat((rgb, alpha[:, None]), dim=1)

        return raw

    def filter_and_format_prediction(self, raw, deform_ret, datas):
        tpose = deform_ret['tpose']
        pbw = deform_ret['pbw']
        tbw = deform_ret['tbw']
        canonical_smpl_verts = datas['canonical_smpl_verts'][None]

        min_xyz = torch.min(canonical_smpl_verts[0], dim=0)[0] - 0.05
        max_xyz = torch.max(canonical_smpl_verts[0], dim=0)[0] + 0.05

        inside = tpose > min_xyz
        inside = inside * (tpose < max_xyz)
        outside = torch.sum(inside, dim=1) != 3
        raw[outside] = 0
        alpha = raw[..., -1]

        num_pixel, num_sample = datas['pts'].shape[:2]
        full_raw = torch.zeros([1, num_pixel * num_sample, 4]).to(datas['pts'])
        full_raw[deform_ret['pind']] = raw

        alpha = alpha[None]
        alpha_ind = alpha.detach() > 0
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind]
        tbw = tbw.transpose(1, 2)[alpha_ind]

        num_pixel, num_sample = datas['pts'].shape[:2]
        raw = full_raw.view(num_pixel, num_sample, 4)
        tpose_ret = {'raw': raw, 'pbw': pbw, 'tbw': tbw}
        datas['raw'] = raw

        return datas, tpose_ret


@MLPS.register_module()
class AN_BlendWeightMLP(nn.Module):
    def __init__(self, num_pose, embedder):
        super(AN_BlendWeightMLP, self).__init__()

        self.bw_latent = nn.Embedding(num_pose + 1, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
            self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

        self.embedder = builder.build_embedder(embedder)

    def get_bw_feature(self, pts, ind):
        pts = self.embedder.run_embed(pts, self.embedder.embed_fns)
        pts = pts.transpose(1, 2)
        latent = self.bw_latent(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        features = self.get_bw_feature(pose_pts, latent_index)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw


@MLPS.register_module()
class AN_DensityMLP(nn.Module):
    def __init__(self, embedder):
        super(AN_DensityMLP, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = 8

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        self.embedder = builder.build_embedder(embedder)
        multires = embedder['multires']
        input_ch, _ = self.embedder.get_embed_ch()
        dims[0] = input_ch

        skip_in = [4]
        bias = 0.5
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs):
        inputs = self.embedder.run_embed(inputs, self.embedder.embed_fns)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1], x[:, 1:]], dim=-1)


@MLPS.register_module()
class AN_ColorMLP(nn.Module):
    def __init__(self, num_train_pose, embedder):
        super(AN_ColorMLP, self).__init__()

        self.color_latent = nn.Embedding(num_train_pose, 128)

        d_feature = 256
        d_in = 6
        d_out = 3
        d_hidden = 256
        n_layers = 4

        dims = [d_in + d_feature] + [d_hidden
                                     for _ in range(n_layers)] + [d_out]


        self.embedder = builder.build_embedder(embedder)
        _, input_ch = self.embedder.get_embed_ch()
        dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        self.lin0 = nn.Linear(dims[0], d_hidden)
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden + 128, d_hidden)
        self.lin4 = nn.Linear(d_hidden, d_out)

        weight_norm = True
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.relu = nn.ReLU()

    def forward(self, points, view_dirs, feature_vectors,
                latent_index):
        view_dirs = self.embedder.run_embed(view_dirs, self.embedder.embed_fns_dirs)
        rendering_input = torch.cat(
            [points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input
        net = self.relu(self.lin0(x))
        net = self.relu(self.lin1(net))
        net = self.relu(self.lin2(net))

        latent = self.color_latent(latent_index)
        latent = latent.expand(net.size(0), latent.size(1))
        features = torch.cat((net, latent), dim=1)

        net = self.relu(self.lin3(features))
        x = self.lin4(net)

        return x
