from math import exp

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import init


def index(feat, uv, mode='bilinear'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    # if torch.__version__ >= "1.3.0":
    #     samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    # else:
    samples = torch.nn.functional.grid_sample(feat, uv, mode=mode)
    return samples[:, :, :, 0]  # [B, C, N]


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d'
        ) != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class LPIPS(torch.nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.net = lpips.LPIPS(net='alex', verbose=False)

    def forward(self, x, gt):
        if torch.max(gt) > 128:
            # [0, 255]
            x = x / 255. * 2 - 1
            gt = gt / 255. * 2 - 1
        elif torch.min(gt) >= 0 and torch.max(gt) <= 1:
            # [0, 1]
            x = x * 2 - 1
            gt = gt * 2 - 1
        with torch.no_grad():
            loss = self.net.forward(x, gt)
        # return loss.item()
        return loss


def psnr(x, gt):
    """
    x: np.uint8, HxWxC, 0 - 255
    gt: np.uint8, HxWxC, 0 - 255
    """
    if torch.max(gt) > 128:
        # [0, 255]
        x = x / 255
        gt = gt / 255
    elif torch.min(gt) < -1:
        # [0, 1]
        x = (x + 1) / 2
        gt = (gt + 1) / 2

    mse = torch.mean((x - gt)**2)
    psnr = -10. * torch.log10(mse)
    return psnr


def gaussian(window_size, sigma):
    gauss = torch.Tensor([
        exp(-(x - window_size // 2)**2 / float(2 * sigma**2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(
        _1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()
    return window


def ssim_(img1,
          img2,
          window_size=11,
          window=None,
          size_average=True,
          full=False,
          val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).\

    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd,
                       groups=channel) - mu1_mu2

    C1 = (0.01 * L)**2
    C2 = (0.03 * L)**2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = v1 / v2  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        cs = cs.mean()
        ret = ssim_map.mean()
    else:
        cs = cs.mean(1).mean(1).mean(1)
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


# Classes to re-use window
class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        if len(list(img1.shape)) < 4:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size,
                                   channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return ssim_(img1,
                     img2,
                     window=window,
                     window_size=self.window_size,
                     size_average=self.size_average)


def rot2euler(R):
    phi = np.arctan2(R[1, 2], R[2, 2])
    theta = -np.arcsin(R[0, 2])
    psi = np.arctan2(R[0, 1], R[0, 0])
    return np.array([phi, theta, psi])


def euler2rot(euler):
    sin, cos = np.sin, np.cos
    phi, theta, psi = euler[0], euler[1], euler[2]
    R1 = np.array([[1, 0, 0], [0, cos(phi), sin(phi)],
                   [0, -sin(phi), cos(phi)]])
    R2 = np.array([[cos(theta), 0, -sin(theta)], [0, 1, 0],
                   [sin(theta), 0, cos(theta)]])
    R3 = np.array([[cos(psi), sin(psi), 0], [-sin(psi), cos(psi), 0],
                   [0, 0, 1]])
    R = R1 @ R2 @ R3
    return R


def batch_rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.

    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(B, 3, 3)
    return rotMat


def index(feat, uv, mode='bilinear'):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    if (len(feat.shape) == 3):
        feat = feat.unsqueeze(1)
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    # if torch.__version__ >= "1.3.0":
    #     samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)  # [B, C, N, 1]
    # else:
    samples = torch.nn.functional.grid_sample(feat, uv, mode=mode)
    return samples[:, :, :, 0]  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    """Compute the orthogonal projections of 3D points into the image plane by
    given projection matrix.

    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, w2c, camera):
    """Compute the perspective projections of 3D points into the image plane by
    given projection matrix.

    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx4/9] Tensor of projection matrix
    :param transforms: [Bx4x4] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    """
    rot = w2c[:, :3, :3]
    trans = w2c[:, :3, 3:4]
    points = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    xy = points[:, :2, :] / torch.clamp(points[:, 2:3, :], 1e-9)
    if camera.shape[1] > 6:
        x2 = xy[:, 0, :] * xy[:, 0, :]
        y2 = xy[:, 1, :] * xy[:, 1, :]
        xy_ = xy[:, 0, :] * xy[:, 1, :]
        r2 = x2 + y2
        c = (1 + r2 * (camera[:, 4:5] + r2 *
                       (camera[:, 5:6] + r2 * camera[:, 8:9])))
        xy = c.unsqueeze(1)*xy + torch.cat([ \
                  (camera[:,6:7]*2*xy_+ camera[:,7:8]*(r2+2*x2)).unsqueeze(1),\
                  (camera[:,7:8]*2*xy_+ camera[:,6:7]*(r2+2*y2)).unsqueeze(1)],1)
    xy = camera[:, 0:2, None] * xy + camera[:, 2:4, None]
    points[:, :2, :] = xy
    return points
