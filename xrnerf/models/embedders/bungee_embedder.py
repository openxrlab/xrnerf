# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import EMBEDDERS


@EMBEDDERS.register_module()
class BungeeEmbedder(nn.Module):
    def __init__(self,
                 i_embed=0,
                 multires=10,
                 multires_dirs=4,
                 input_ch=3,
                 **kwargs):
        super().__init__()  # 对于集成了nn.Module的类型，如果有可学习参数，必须加上这个
        if i_embed == -1:
            self.embed_fns, self.embed_ch = [nn.Identity()], input_ch
            self.embed_fns_dirs, self.embed_ch_dirs = [nn.Identity()], input_ch
        else:
            self.embed_fns, self.embed_ch = self.create_mip_embedding_fn(
                multires, input_ch=input_ch)
            self.embed_fns_dirs, self.embed_ch_dirs = self.create_embedding_fn(
                multires_dirs, input_ch=input_ch)

    def create_mip_embedding_fn(self,
                                multires,
                                input_ch=3,
                                cat_input=True,
                                log_sampling=True,
                                periodic_fns=[torch.sin, torch.cos]):
        num_freqs = multires
        max_freq_log2 = multires - 1
        embed_fns = []
        out_dim = 0
        d = input_ch
        if cat_input:
            embed_fns.append(lambda x: x[:, :d])
            out_dim += d
        N_freqs = num_freqs
        max_freq = max_freq_log2

        if log_sampling:
            freq_bands_y = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            freq_bands_w = 4.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands_y = torch.linspace(2.**0, 2.**max_freq, steps=N_freqs)
            freq_bands_w = torch.linspace(4.**0, 4.**max_freq, steps=N_freqs)
        for freq_y, freq_w in zip(freq_bands_y, freq_bands_w):
            for p_fn in periodic_fns:
                embed_fns.append(lambda inputs, p_fn=p_fn, freq_y=freq_y,
                                 freq_w=freq_w: p_fn(inputs[:, :d] * freq_y) *
                                 torch.exp((-0.5) * freq_w * inputs[:, d:]))
                out_dim += d
        return embed_fns, out_dim

    def create_embedding_fn(self,
                            multires,
                            input_ch=3,
                            cat_input=True,
                            log_sampling=True,
                            periodic_fns=[torch.sin, torch.cos]):
        num_freqs = multires
        max_freq_log2 = multires - 1
        embed_fns = []
        out_dim = 0
        d = input_ch
        if cat_input:
            embed_fns.append(lambda x: x)
            out_dim += d
        N_freqs = num_freqs
        max_freq = max_freq_log2

        if log_sampling:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(
                    lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d
        return embed_fns, out_dim

    def get_embed_ch(self):
        return self.embed_ch, self.embed_ch_dirs

    def forward(self, data):
        means, cov_diags = data['samples']
        means_flat = torch.reshape(means, [-1, means.shape[-1]])
        cov_diags_flat = torch.reshape(cov_diags, [-1, cov_diags.shape[-1]])
        inputs_flat = torch.cat((means_flat, cov_diags_flat), -1)
        embedded = self.run_embed(inputs_flat, self.embed_fns)

        viewdirs = data['viewdirs']
        input_dirs = viewdirs[:, None].expand(means.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = self.run_embed(input_dirs_flat, self.embed_fns_dirs)

        embedded = torch.cat([embedded, embedded_dirs], -1)
        data['unflatten_shape'] = data['samples'][0].shape[:-1]
        data['embedded'] = embedded
        return data

    def run_embed(self, x, embed_fns):
        return torch.cat([fn(x) for fn in embed_fns], -1)
