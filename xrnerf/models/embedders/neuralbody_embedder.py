import torch
import torch.nn.functional as F
from torch import nn

from ..builder import EMBEDDERS

try:
    import spconv
    if '__version__' in dir(spconv) and spconv.__version__.split(
            '.')[0] == '2':
        import spconv.pytorch as spconv
except:
    print('Please install spconv')


@EMBEDDERS.register_module()
class SmplEmbedder(nn.Module):
    def __init__(self, **kwargs):
        super(SmplEmbedder, self).__init__()

        self.voxel_size = kwargs['voxel_size']

        self.latent_codes = nn.Embedding(6890, 16)
        self.xyzc_net = SparseConvNet()

    @staticmethod
    def interpolate_features(grid_coords, feature_volume):
        features = []
        for volume in feature_volume:
            feature = F.grid_sample(volume,
                                    grid_coords,
                                    padding_mode='zeros',
                                    align_corners=True)
            features.append(feature)
        features = torch.cat(features, dim=1)
        features = features.view(features.size(0), -1, features.size(4))
        return features

    def forward(self, datas):
        # prepare the data related to SparseConvNet
        sparseconv_data = prepare_sparseconv_data(datas, self.voxel_size)

        # encode neural body
        coord = sparseconv_data['coord']
        out_sh = sparseconv_data['out_sh']
        batch_size = sparseconv_data['batch_size']
        code = self.latent_codes(torch.arange(0, 6890).to(coord.device))
        xyzc = spconv.SparseConvTensor(code, coord, out_sh, batch_size)
        feature_volume = self.xyzc_net(xyzc)

        # interpolate features
        pts_idx = sparseconv_data['pts_idx']
        pts_idx = pts_idx[None, None, None]
        xyzc_features = self.interpolate_features(pts_idx, feature_volume)

        return xyzc_features


class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()

        self.conv0 = double_conv(16, 16, 'subm0')
        self.down0 = stride_conv(16, 32, 'down0')

        self.conv1 = double_conv(32, 32, 'subm1')
        self.down1 = stride_conv(32, 64, 'down1')

        self.conv2 = triple_conv(64, 64, 'subm2')
        self.down2 = stride_conv(64, 128, 'down2')

        self.conv3 = triple_conv(128, 128, 'subm3')
        self.down3 = stride_conv(128, 128, 'down3')

        self.conv4 = triple_conv(128, 128, 'subm4')

    def forward(self, x):
        net = self.conv0(x)
        net = self.down0(net)

        net = self.conv1(net)
        net1 = net.dense()
        net = self.down1(net)

        net = self.conv2(net)
        net2 = net.dense()
        net = self.down2(net)

        net = self.conv3(net)
        net3 = net.dense()
        net = self.down3(net)

        net = self.conv4(net)
        net4 = net.dense()

        volumes = [net1, net2, net3, net4]

        return volumes


def single_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          1,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def double_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def triple_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SubMConv3d(in_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
        spconv.SubMConv3d(out_channels,
                          out_channels,
                          3,
                          bias=False,
                          indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
        nn.ReLU(),
    )


def stride_conv(in_channels, out_channels, indice_key=None):
    return spconv.SparseSequential(
        spconv.SparseConv3d(in_channels,
                            out_channels,
                            3,
                            2,
                            padding=1,
                            bias=False,
                            indice_key=indice_key),
        nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01), nn.ReLU())


def prepare_sparseconv_data(datas, voxel_size):
    # calculate the size of bounding box
    world_verts = datas['smpl_verts']
    canonical_verts = torch.matmul(world_verts - datas['smpl_T'],
                                   datas['smpl_R'])
    min_xyz = torch.min(canonical_verts, dim=0)[0]
    min_xyz[2] = min_xyz[2] - 0.05
    max_xyz = torch.max(canonical_verts, dim=0)[0]
    max_xyz[2] = max_xyz[2] + 0.05

    # coordinate, shape, batch size
    sparseconv_data = {}

    # construct the coordinates of SparseConv input data
    # coordinate: [N, 4], batch_idx, z, y, x
    voxel_size = torch.tensor(voxel_size).to(canonical_verts)
    xyz_idx = torch.round((canonical_verts - min_xyz) / voxel_size).int()
    coord = xyz_idx[..., [2, 1, 0]]
    idx = torch.full([len(coord), 1], 0).to(coord)
    sparseconv_data['coord'] = torch.cat([idx, coord], dim=1)

    # construct the output shape
    out_sh = torch.ceil((max_xyz - min_xyz) / voxel_size)[[2, 1, 0]].int()
    x = 32
    out_sh = (out_sh | (x - 1)) + 1
    sparseconv_data['out_sh'] = out_sh.tolist()
    sparseconv_data['batch_size'] = 1

    # convert sampled points to the format for feature interpolation
    num_pixel, num_sample = datas['pts'].shape[:2]
    pts = datas['pts'].view(num_pixel * num_sample, -1)
    canonical_pts = torch.matmul(pts - datas['smpl_T'], datas['smpl_R'])
    pts_idx = (canonical_pts - min_xyz) / voxel_size
    pts_idx = pts_idx / out_sh[[2, 1, 0]] * 2 - 1
    sparseconv_data['pts_idx'] = pts_idx

    return sparseconv_data
