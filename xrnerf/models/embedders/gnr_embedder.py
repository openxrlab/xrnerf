"""This file is directly borrowed from PIFu GNR uses PIFu's Stacked-Hour-Glass
for image encoding."""

# from ..net_util import *

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import EMBEDDERS


def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=strd,
                     padding=padding,
                     bias=bias)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, norm='batch'):
        super(ConvBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if norm == 'batch':
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif norm == 'group':
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                self.bn4,
                nn.ReLU(True),
                nn.Conv2d(in_planes,
                          out_planes,
                          kernel_size=1,
                          stride=1,
                          bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


@EMBEDDERS.register_module()
class PositionalEncoding:
    """GNR uses positional encoding in NeRF for coordinate embedding."""
    def __init__(self,
                 d,
                 num_freqs=10,
                 min_freq=None,
                 max_freq=None,
                 freq_type='linear'):
        self.num_freqs = num_freqs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.freq_type = freq_type
        self.create_embedding_fn(d)

    def create_embedding_fn(self, d):
        embed_fns = []
        out_dim = 0
        embed_fns.append(lambda x: x)
        out_dim += d

        N_freqs = self.num_freqs

        if self.freq_type == 'linear':
            min_freq = 0 if self.min_freq is None else self.min_freq
            max_freq = 2**(self.num_freqs -
                           1) if self.max_freq is None else self.max_freq
            freq_bands = torch.linspace(
                min_freq * math.pi * 2, max_freq * math.pi * 2,
                steps=N_freqs)  # linear freq band, Fourier expansion
        else:
            min_freq = 0 if self.min_freq is None else math.log2(self.min_freq)
            max_freq = self.num_freqs - 1 if self.max_freq is None else math.log2(
                self.max_freq)
            freq_bands = 2.**torch.linspace(min_freq * math.pi * 2,
                                            max_freq * math.pi * 2,
                                            steps=N_freqs)  # log expansion

        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


@EMBEDDERS.register_module()
class SphericalHarmonics:
    """GNR uses Sepherical Harmonics for view direction embedding."""
    def __init__(self, d=3, rank=3):
        assert d % 3 == 0
        self.rank = max([int(rank), 0])
        self.out_dim = self.rank * self.rank * (d // 3)

    def Lengdre_polynormial(self, x, omx=None):
        if omx is None: omx = 1 - x * x
        Fml = [[]] * ((self.rank + 1) * self.rank // 2)
        Fml[0] = torch.ones_like(x)
        for l in range(1, self.rank):
            b = (l * l + l) // 2
            Fml[b + l] = -Fml[b - 1] * (2 * l - 1)
            Fml[b + l - 1] = Fml[b - 1] * (2 * l - 1) * x
            for m in range(l, 1, -1):
                Fml[b + m - 2] = -(omx * Fml[b + m] + \
                                   2 * (m - 1) * x * Fml[b + m - 1]) / ((l - m + 2) * (l + m - 1))
        return Fml

    def SH(self, xyz):
        cs = xyz[..., 0:1]
        sn = xyz[..., 1:2]
        Fml = self.Lengdre_polynormial(xyz[..., 2:3], cs * cs + sn * sn)
        H = [[]] * (self.rank * self.rank)
        for l in range(self.rank):
            b = l * l + l
            attr = np.sqrt((2 * l + 1) / math.pi / 4)
            H[b] = attr * Fml[b // 2]
            attr = attr * np.sqrt(2)
            snM = sn
            csM = cs
            for m in range(1, l + 1):
                attr = -attr / np.sqrt((l + m) * (l + 1 - m))
                H[b - m] = attr * Fml[b // 2 + m] * snM
                H[b + m] = attr * Fml[b // 2 - m] * csM
                snM, csM = snM * cs + csM * sn, csM * cs - snM * sn
        if len(H) > 0:
            return torch.cat(H, -1)
        else:
            return torch.Tensor([])

    def embed(self, inputs):
        return self.SH(inputs)


@EMBEDDERS.register_module()
class SRFilters(nn.Module):
    """Upsample the pixel-aligned feature."""
    def __init__(self, order=2, in_ch=256, out_ch=128):
        super(SRFilters, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.image_factor = [0.5**(order - i) for i in range(0, order + 1)]
        self.convs = nn.ModuleList(
            [nn.Conv2d(in_ch + 3, out_ch, kernel_size=3, padding=1)] + [
                nn.Conv2d(out_ch + 3, out_ch, kernel_size=3, padding=1)
                for i in range(order)
            ])

    def forward(self, feat, images):
        for i, conv in enumerate(self.convs):
            im = F.interpolate(images,
                               scale_factor=self.image_factor[i],
                               mode='bicubic',
                               align_corners=True
                               ) if self.image_factor[i] is not 1 else images
            feat = F.interpolate(
                feat, scale_factor=2, mode='bicubic',
                align_corners=True) if i is not 0 else feat
            feat = torch.cat([feat, im], dim=1)
            feat = self.convs[i](feat)
        return feat


@EMBEDDERS.register_module()
class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features, norm='batch'):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.norm = norm

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module(
            'b1_' + str(level),
            ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module(
            'b2_' + str(level),
            ConvBlock(self.features, self.features, norm=self.norm))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module(
                'b2_plus_' + str(level),
                ConvBlock(self.features, self.features, norm=self.norm))

        self.add_module(
            'b3_' + str(level),
            ConvBlock(self.features, self.features, norm=self.norm))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        # NOTE: for newer PyTorch (1.3~), it seems that training results are degraded due to implementation diff in F.grid_sample
        # if the pretrained model behaves weirdly, switch with the commented line.
        # NOTE: I also found that "bicubic" works better.
        up2 = F.interpolate(low3,
                            scale_factor=2,
                            mode='bicubic',
                            align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


@EMBEDDERS.register_module()
class HGFilter(nn.Module):
    def __init__(self, opt):
        super(HGFilter, self).__init__()
        self.num_modules = opt.num_stack

        self.opt = opt

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        if self.opt['norm'] == 'batch':
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == 'group':
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == 'conv64':
            self.conv2 = ConvBlock(64, 64, self.opt.norm)
            self.down_conv2 = nn.Conv2d(64,
                                        128,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        elif self.opt.hg_down == 'conv128':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
            self.down_conv2 = nn.Conv2d(128,
                                        128,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1)
        elif self.opt.hg_down == 'ave_pool':
            self.conv2 = ConvBlock(64, 128, self.opt.norm)
        else:
            raise NameError('Unknown Fan Filter setting!')

        self.conv3 = ConvBlock(128, 128, self.opt.norm)
        self.conv4 = ConvBlock(128, 256, self.opt.norm)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module(
                'm' + str(hg_module),
                HourGlass(1, opt.num_hourglass, 256, self.opt.norm))

            self.add_module('top_m_' + str(hg_module),
                            ConvBlock(256, 256, self.opt.norm))
            self.add_module(
                'conv_last' + str(hg_module),
                nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            if self.opt.norm == 'batch':
                self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            elif self.opt.norm == 'group':
                self.add_module('bn_end' + str(hg_module),
                                nn.GroupNorm(32, 256))

            self.add_module(
                'l' + str(hg_module),
                nn.Conv2d(256,
                          opt.hourglass_dim,
                          kernel_size=1,
                          stride=1,
                          padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module),
                    nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module(
                    'al' + str(hg_module),
                    nn.Conv2d(opt.hourglass_dim,
                              256,
                              kernel_size=1,
                              stride=1,
                              padding=0))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        # tmpx = x
        if self.opt.hg_down == 'ave_pool':
            x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        elif self.opt.hg_down in ['conv64', 'conv128']:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError('Unknown Fan Filter setting!')

        # normx = x

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        # outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(
                self._modules['bn_end' + str(i)](
                    self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l' + str(i)](ll)
            # outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        # return outputs, tmpx.detach(), normx
        return tmp_out
