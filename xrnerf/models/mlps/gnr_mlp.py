import torch
import torch.nn.functional as F
from torch import nn

from .. import builder
from ..builder import MLPS
from ..embedders import PositionalEncoding, SphericalHarmonics


@MLPS.register_module()
class GNRMLP(nn.Module):
    def __init__(self,
                 opt,
                 D=8,
                 W=256,
                 input_ch=3,
                 input_ch_atts=3,
                 output_ch=4,
                 activation='relu',
                 pose_freqs=10,
                 att_freqs=6,
                 spatial_freq=1 / 256):
        """"""
        super(GNRMLP, self).__init__()
        self.D = D
        self.W = W

        self.use_smpl_sdf = opt.use_smpl_sdf
        self.use_t_pose = opt.use_t_pose
        self.angle_diff = opt.angle_diff
        self.use_occ_net = opt.use_occlusion_net

        self.input_ch_pos_enc = input_ch
        self.input_ch_smpl = 0
        if self.use_smpl_sdf: self.input_ch_smpl += 4
        if self.use_t_pose: self.input_ch_smpl += 3
        self.use_smpl = self.input_ch_smpl != 0
        self.input_ch_feat = opt.input_ch_feat

        self.input_ch_feat = self.input_ch_feat + 3

        self.skips = opt.skips
        self.use_viewdirs = opt.use_viewdirs and opt.use_attention
        self.num_views = opt.num_views
        # self.input_ch_atts = input_ch_atts if opt.use_attention else 0
        if not opt.use_attention:
            self.input_ch_atts = 0
        elif self.angle_diff:
            self.input_ch_atts = 1
        else:
            self.input_ch_atts = 3
        self.use_sh = opt.use_sh if not self.angle_diff else False

        self.use_attention = opt.use_attention
        self.use_bn = opt.use_bn
        self.spatial_freq = spatial_freq
        self.pose_embeder = PositionalEncoding(self.input_ch_pos_enc,
                                               num_freqs=pose_freqs,
                                               min_freq=spatial_freq * 0.1,
                                               max_freq=spatial_freq * 10)
        self.att_embeder = SphericalHarmonics(
            d=self.input_ch_atts) if self.use_sh else PositionalEncoding(
                self.input_ch_atts, num_freqs=att_freqs)

        self.pose_embed_fn = self.pose_embeder.embed
        self.att_embed_fn = self.att_embeder.embed
        self.weighted_pool = opt.weighted_pool and self.use_attention

        self.alpha_linears = nn.ModuleList([
            nn.Linear(
                self.pose_embeder.out_dim + self.input_ch_smpl +
                self.input_ch_feat, W)
        ] + [
            nn.Linear(W + self.pose_embeder.out_dim + self.input_ch_smpl, W
                      ) if i in self.skips else nn.Linear(W, W)
            for i in range(0, D - 1)
        ])
        self.alpha_out_linear = nn.Linear(W, 1)

        self.rgb_linears = nn.ModuleList([
            nn.Linear(W + self.pose_embeder.out_dim + self.input_ch_smpl, W //
                      4)
        ] + [
            nn.Linear(W // 4 + self.att_embeder.out_dim, W //
                      8) if self.use_viewdirs else nn.Linear(W // 4, W // 8)
        ] + [nn.Linear(W // 8, W // 16)] + [nn.Linear(W // 16, 3)])
        if self.use_bn:
            self.bn_layer_1 = nn.BatchNorm1d(W)
            self.bn_layer_2 = nn.BatchNorm1d(W)
            self.bn_layer_3 = nn.BatchNorm1d(W // 16)

        if self.weighted_pool:
            self.s = nn.Parameter(torch.ones(1))
        # ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'swish':
            if torch.__version__ < '1.7.0':
                swish = lambda x: x * torch.sigmoid(x)
                self.activation_fn = swish
            else:
                self.activation_fn = torch.nn.SiLU

        if self.use_attention:
            self.value_linears = nn.ModuleList([
                nn.Linear(
                    self.pose_embeder.out_dim + self.att_embeder.out_dim + W,
                    W // 4),
                nn.Linear(W // 4 + self.att_embeder.out_dim, W // 8),
                nn.Linear(W // 8 + self.att_embeder.out_dim, W // 16)
            ])
            self.key_linears = nn.ModuleList([
                nn.Linear(
                    self.pose_embeder.out_dim + self.att_embeder.out_dim + W,
                    W // 4),
                nn.Linear(W // 4 + self.att_embeder.out_dim, W // 8),
                nn.Linear(W // 8 + self.att_embeder.out_dim, W // 16)
            ])

        if self.use_occ_net:
            self.occ_linears = nn.ModuleList([
                nn.Linear(self.input_ch_smpl + 6 + self.input_ch_feat, W // 4),
                nn.Linear(W // 4, W // 16),
                nn.Linear(W // 16 + self.input_ch_smpl + 6, 1)
            ])

    def forward(self, x, attdirs=None, alpha_only=False, smpl_vis=None):
        # prepare inputs
        """torch.Size([8590, 3]) torch.Size([8590, 3]) torch.Size([8590, 3])
        torch.Size([8590, 1]) torch.Size([8590, 4, 269]) 3 7 131."""
        #print(x.shape, self.input_ch_pos_enc, self.input_ch_smpl, self.input_ch_feat)
        input_pts, input_smpl, input_feats = torch.split(
            x, [self.input_ch_pos_enc, self.input_ch_smpl, self.input_ch_feat],
            dim=-1)
        unqiue_pts = input_pts[:, 0]
        unqiue_smpl = input_smpl[:, 0] if self.use_smpl else torch.zeros(
            [input_pts.shape[0], 0],
            dtype=torch.float32,
            device=input_pts.device)
        input_pts = input_pts.view([-1, self.input_ch_pos_enc])
        input_smpl = input_smpl.view([
            -1, self.input_ch_smpl
        ]) if self.use_smpl else torch.zeros([input_pts.shape[0], 0],
                                             dtype=torch.float32,
                                             device=input_pts.device)
        input_feats = input_feats.view([-1, self.input_ch_feat])
        if self.use_attention and attdirs is not None:
            qrydirs, srcdirs = torch.split(attdirs, [1, self.num_views],
                                           dim=-2)

        if self.use_occ_net and attdirs is not None:
            # compute plucker coord
            d = srcdirs.reshape([-1, 3])
            m = torch.cross(input_pts, d, dim=-1)
            occ_h = torch.cat([input_smpl, d, m, input_feats], dim=-1)
            for i, l in enumerate(self.occ_linears):
                occ_h = self.occ_linears[i](occ_h)
                if i < len(self.occ_linears) - 1:
                    occ_h = self.activation_fn(occ_h)
                if i == 1:
                    occ_h = torch.cat([input_smpl, d, m, occ_h], dim=-1)
            occ_out = torch.sigmoid(occ_h).view([-1, self.num_views, 1])
            # occ = F.softmax(occ_out, dim=1)

        # alpha mlp
        tmp_h = None
        h = torch.cat([self.pose_embed_fn(input_pts), input_smpl, input_feats],
                      dim=-1)
        for i, l in enumerate(self.alpha_linears):
            h = self.alpha_linears[i](h)
            h = self.activation_fn(h)
            if i in self.skips:
                if i == self.skips[0]:
                    tmp_h = h.clone()
                    h = torch.mean(h.view(-1, self.num_views, self.W), dim=1)
                h = torch.cat([self.pose_embed_fn(unqiue_pts), unqiue_smpl, h],
                              dim=-1)
        alpha = self.alpha_out_linear(h)
        if alpha_only: return alpha

        # rgb mpl
        if self.use_attention and self.weighted_pool:
            weights = torch.exp(self.s *
                                (torch.sum(srcdirs * qrydirs, dim=-1) - 1))
            weights = weights / (torch.sum(weights, dim=-1, keepdim=True) +
                                 1e-8)  # [N_rand*N_sample, 4]
            h = torch.sum(tmp_h.view(-1, self.num_views, self.W) *
                          weights[..., None],
                          dim=1)
            h0 = h.clone()
        else:
            h = torch.mean(tmp_h.view(-1, self.num_views, self.W), dim=1)

        h = torch.cat([self.pose_embed_fn(unqiue_pts), unqiue_smpl, h], -1)
        for i, l in enumerate(self.rgb_linears):
            h = self.rgb_linears[i](h)
            if i < len(self.rgb_linears) - 1:
                h = self.activation_fn(h)
            if i == 0 and self.use_viewdirs:
                h = torch.cat([self.att_embed_fn(-qrydirs.squeeze(1)), h],
                              dim=-1)
        outputs = torch.cat([h, alpha], dim=-1)

        # calculate attention
        if self.use_attention and attdirs is not None:
            attdirs = attdirs.reshape([-1, self.input_ch_atts])
            input_pts_ = torch.cat([unqiue_pts, input_pts], dim=0)
            input_h = torch.cat([h0, tmp_h], dim=0)
            val = torch.cat([
                self.pose_embed_fn(input_pts_),
                self.att_embed_fn(attdirs), input_h
            ],
                            dim=-1)
            for i, l in enumerate(self.value_linears):
                val = self.value_linears[i](val)
                if i < len(self.value_linears) - 1:
                    val = self.activation_fn(val)
                    val = torch.cat([self.att_embed_fn(attdirs), val], dim=-1)
            key = torch.cat([
                self.pose_embed_fn(unqiue_pts),
                self.att_embed_fn(qrydirs.squeeze(1)), h0
            ],
                            dim=-1)
            for i, l in enumerate(self.key_linears):
                key = self.key_linears[i](key)
                if i < len(self.key_linears) - 1:
                    key = self.activation_fn(key)
                    key = torch.cat(
                        [self.att_embed_fn(qrydirs.squeeze(1)), key], dim=-1)
            # attention key (query direction) and val (source view direction)
            key = key.unsqueeze(1)
            val = val.view(unqiue_pts.shape[0], self.num_views + 1, -1)
            attention = torch.matmul(val, key.permute(0, 2, 1)).squeeze(-1)

            if self.use_occ_net:
                attention = self.weighted_softmax(attention,
                                                  occ_out.squeeze(-1))
            elif smpl_vis is not None:
                attention = self.weighted_softmax(attention, smpl_vis.float())
            else:
                attention = F.softmax(attention, dim=-1)

        if self.use_attention and attdirs is not None:
            outputs = torch.cat([outputs, attention], dim=-1)

        if self.use_occ_net:
            outputs = torch.cat([outputs, occ_out.squeeze(-1)], dim=-1)

        return outputs

    def weighted_softmax(self, attention, weight):
        exp_att = torch.exp(attention -
                            torch.max(attention, 1, keepdim=True)[0])
        exp_att = torch.cat([exp_att[:, :1], exp_att[:, 1:].clone() * weight],
                            dim=1)
        exp_att_sum = torch.sum(exp_att, dim=-1, keepdim=True)
        attention = exp_att / (exp_att_sum + 1e-8)

        return attention
