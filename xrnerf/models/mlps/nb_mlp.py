from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from .. import builder
from ..builder import MLPS


@MLPS.register_module()
class NB_NeRFMLP(nn.Module):
    def __init__(self, num_frame, embedder):
        super(NB_NeRFMLP, self).__init__()

        self.appearance_code = nn.Embedding(num_frame, 128)

        self.actvn = nn.ReLU()

        self.fc_0 = nn.Conv1d(352, 256, 1)
        self.fc_1 = nn.Conv1d(256, 256, 1)
        self.fc_2 = nn.Conv1d(256, 256, 1)
        self.alpha_fc = nn.Conv1d(256, 1, 1)

        self.feature_fc = nn.Conv1d(256, 256, 1)
        self.latent_fc = nn.Conv1d(384, 256, 1)
        self.view_fc = nn.Conv1d(346, 128, 1)
        self.rgb_fc = nn.Conv1d(128, 3, 1)

        self.embedder = builder.build_embedder(embedder)

    def forward(self, xyzc_features, datas):
        # calculate density
        net = self.actvn(self.fc_0(xyzc_features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))

        alpha = self.alpha_fc(net)

        # calculate color
        features = self.feature_fc(net)

        latent = self.appearance_code(datas['latent_idx'])
        latent = latent[..., None].expand(*latent.shape, net.size(2))
        features = torch.cat((features, latent), dim=1)
        features = self.latent_fc(features)

        num_pixel, num_sample = datas['pts'].shape[:2]
        viewdirs = datas['rays_d']
        viewdirs = viewdirs[:, None].expand(datas['pts'].shape)
        viewdirs = self.embedder.run_embed(viewdirs, self.embedder.embed_fns_dirs)
        viewdirs = viewdirs.view(num_pixel * num_sample, -1)[None].transpose(1, 2)

        light_pts = self.embedder.run_embed(datas['pts'], self.embedder.embed_fns)
        light_pts = light_pts.view(num_pixel * num_sample, -1)[None].transpose(1, 2)

        features = torch.cat((features, viewdirs, light_pts), dim=1)

        net = self.actvn(self.view_fc(features))
        rgb = self.rgb_fc(net)

        raw = torch.cat((rgb, alpha), dim=1)
        raw = raw.transpose(1, 2)

        datas['raw'] = raw.view(num_pixel, num_sample, 4)

        return datas
