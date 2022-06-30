import torch
import torch.nn.functional as F
import numpy as np

try:
    from pytorch3d.ops.knn import knn_points
except:
    print('Please install pytorch3d')


def sample_closest_points(src: torch.Tensor, ref: torch.Tensor, values: torch.Tensor):
    n_batch, n_points, _ = src.shape
    ret = knn_points(src, ref, K=1)
    dists, vert_ids = ret.dists.sqrt(), ret.idx
    values = values.view(-1, values.shape[-1])  # (n, D)
    sampled = values[vert_ids]  # (s, D)
    return sampled.view(n_batch, n_points, -1), dists.view(n_batch, n_points, 1)


def world_points_to_pose_points(wpts, Rh, Th):
    """
    wpts: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    Th: n_batch, 1, 3
    """
    pts = torch.matmul(wpts - Th, Rh)
    return pts


def world_dirs_to_pose_dirs(wdirs, Rh):
    """
    wdirs: n_batch, n_points, 3
    Rh: n_batch, 3, 3
    """
    pts = torch.matmul(wdirs, Rh)
    return pts


def pose_points_to_tpose_points(ppts, bw, A):
    """transform points from the pose space to the T pose
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ppts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    pts = ppts - A[..., :3, 3]
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * pts[:, :, None], dim=3)
    return pts


def pose_dirs_to_tpose_dirs(ddirs, bw, A):
    """transform directions from the pose space to the T pose
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R_inv = torch.inverse(A[..., :3, :3])
    pts = torch.sum(R_inv * ddirs[:, :, None], dim=3)
    return pts


def tpose_points_to_pose_points(pts, bw, A):
    """transform points from the T pose to the pose space
    ppts: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = pts.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * pts[:, :, None], dim=3)
    pts = pts + A[..., :3, 3]
    return pts


def tpose_dirs_to_pose_dirs(ddirs, bw, A):
    """transform directions from the T pose to the pose space
    ddirs: n_batch, n_points, 3
    bw: n_batch, 24, n_points
    A: n_batch, 24, 4, 4
    """
    sh = ddirs.shape
    bw = bw.permute(0, 2, 1)
    A = torch.bmm(bw, A.view(sh[0], 24, -1))
    A = A.view(sh[0], -1, 4, 4)
    R = A[..., :3, :3]
    pts = torch.sum(R * ddirs[:, :, None], dim=3)
    return pts


class NovelPoseTraining:
    @staticmethod
    def get_sampling_points(bounds):
        sh = bounds.shape
        min_xyz = bounds[:, 0]
        max_xyz = bounds[:, 1]
        N_samples = 1024 * 64
        x_vals = torch.rand([sh[0], N_samples])
        y_vals = torch.rand([sh[0], N_samples])
        z_vals = torch.rand([sh[0], N_samples])
        vals = torch.stack([x_vals, y_vals, z_vals], dim=2)
        vals = vals.to(bounds.device)
        pts = (max_xyz - min_xyz)[:, None] * vals + min_xyz[:, None]
        return pts

    @staticmethod
    def wpts_to_ppts(pts, datas):
        """transform points from the world space to the pose space"""
        Th = datas['smpl_T'][None]
        pts = pts - Th
        R = datas['smpl_R'][None]
        sh = pts.shape
        pts = torch.matmul(pts.view(sh[0], -1, 3), R)
        return pts

    @staticmethod
    def ppts_to_tpose(net, pose_pts, datas, canonical_bounds):
        smpl_bw = datas['smpl_bw'][None]

        # blend weights of points at i
        posed_smpl_verts = NovelPoseTraining.wpts_to_ppts(datas['smpl_verts'][None], datas)
        init_pbw, pnorm = sample_closest_points(pose_pts, posed_smpl_verts, smpl_bw)
        init_pbw = init_pbw.permute(0, 2, 1)
        pnorm = pnorm[..., 0]

        # neural blend weights of points at i
        pbw = net.deform_field.novel_pose_bw_mlp.calculate_neural_blend_weights(pose_pts, init_pbw, datas['bw_latent_idx'])

        # transform points from i to i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, datas['A'][None])
        tpose = tpose_points_to_pose_points(tpose, pbw, datas['big_A'][None])

        # calculate neural blend weights of points at the tpose space
        canonical_smpl_verts = datas['canonical_smpl_verts'][None]
        init_tbw, tnorm = sample_closest_points(tpose, canonical_smpl_verts, smpl_bw)
        init_tbw = init_tbw.permute(0, 2, 1)
        tnorm = tnorm[..., 0]
        ind = torch.zeros_like(datas['bw_latent_idx'])
        tbw = net.deform_field.bw_mlp.calculate_neural_blend_weights(tpose, init_tbw, ind)

        alpha = net.tpose_human.calculate_alpha(tpose)

        inside = tpose > canonical_bounds[:, :1]
        inside = inside * (tpose < canonical_bounds[:, 1:])
        inside = torch.sum(inside, dim=2) == 3
        # inside = inside * (tnorm < cfg.norm_th)
        norm_th = net.cfg['deform_field']['smpl_threshold']
        inside = inside * (pnorm < norm_th)
        outside = ~inside
        alpha = alpha[:, 0]
        alpha[outside] = 0

        alpha_ind = alpha.detach() > 0
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind]
        tbw = tbw.transpose(1, 2)[alpha_ind]

        return pbw, tbw

    @staticmethod
    def tpose_to_ppts(net, tpose, datas):
        smpl_bw = datas['smpl_bw'][None]

        # calculate neural blend weights of points at the tpose space
        canonical_smpl_verts = datas['canonical_smpl_verts'][None]
        init_tbw, tnorm = sample_closest_points(tpose, canonical_smpl_verts, smpl_bw)
        init_tbw = init_tbw.permute(0, 2, 1)
        tnorm = tnorm[..., 0]

        ind = torch.zeros_like(datas['bw_latent_idx'])
        tbw = net.deform_field.bw_mlp.calculate_neural_blend_weights(tpose, init_tbw, ind)

        alpha = net.tpose_human.calculate_alpha(tpose)

        tpose = pose_points_to_tpose_points(tpose, tbw, datas['big_A'][None])
        pose_pts = tpose_points_to_pose_points(tpose, tbw, datas['A'][None])

        # blend weights of points at i
        posed_smpl_verts = NovelPoseTraining.wpts_to_ppts(datas['smpl_verts'][None], datas)
        init_pbw, pnorm = sample_closest_points(pose_pts, posed_smpl_verts, smpl_bw)
        init_pbw = init_pbw.permute(0, 2, 1)

        # neural blend weights of points at i
        pbw = net.deform_field.novel_pose_bw_mlp.calculate_neural_blend_weights(pose_pts, init_pbw, datas['bw_latent_idx'])

        alpha = alpha[:, 0]
        norm_th = net.cfg['deform_field']['smpl_threshold']
        alpha[tnorm > norm_th] = 0

        alpha_ind = alpha.detach() > 0
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind]
        tbw = tbw.transpose(1, 2)[alpha_ind]

        return pbw, tbw

    @staticmethod
    def calculate_bounds(points):
        min_xyz = torch.min(points, dim=0)[0]
        min_xyz = min_xyz - 0.05
        max_xyz = torch.max(points, dim=0)[0]
        max_xyz = max_xyz + 0.05
        bounds = torch.stack([min_xyz, max_xyz])[None]
        return bounds

    @staticmethod
    def calculate_loss(net, datas):
        world_bounds = NovelPoseTraining.calculate_bounds(datas['smpl_verts'])
        canonical_bounds = NovelPoseTraining.calculate_bounds(datas['canonical_smpl_verts'])

        world_points = NovelPoseTraining.get_sampling_points(world_bounds)
        posed_points = NovelPoseTraining.wpts_to_ppts(world_points, datas)
        canonical_points = NovelPoseTraining.get_sampling_points(canonical_bounds)

        pbw0, tbw0 = NovelPoseTraining.ppts_to_tpose(net, posed_points, datas, canonical_bounds)
        pbw1, tbw1 = NovelPoseTraining.tpose_to_ppts(net, canonical_points, datas)

        bw_loss0 = F.smooth_l1_loss(pbw0, tbw0)
        bw_loss1 = F.smooth_l1_loss(pbw1, tbw1)
        loss = bw_loss0 + bw_loss1

        log_vars = {'loss':loss.item()}
        outputs = {'loss':loss, 'log_vars':log_vars, 'num_samples': world_points.shape[1]}

        return outputs
