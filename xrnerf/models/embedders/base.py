# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import EMBEDDERS


@EMBEDDERS.register_module()
class BaseEmbedder(nn.Module):
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
            self.embed_fns, self.embed_ch = self.create_embedding_fn(
                multires, input_ch=input_ch)
            self.embed_fns_dirs, self.embed_ch_dirs = self.create_embedding_fn(
                multires_dirs, input_ch=input_ch)

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
        # pts shape before reshape
        data['unflatten_shape'] = data['pts'].shape[:-1]
        inputs, viewdirs = data['pts'], data['viewdirs']
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        embedded = self.run_embed(inputs_flat, self.embed_fns)

        #如果chunk为None， inputs也是2维，不需要expand
        if len(inputs.shape) > len(viewdirs.shape):
            input_dirs = viewdirs[:, None].expand(inputs.shape)
        else:
            input_dirs = viewdirs

        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = self.run_embed(input_dirs_flat, self.embed_fns_dirs)
        embedded = torch.cat([embedded, embedded_dirs], -1)
        data['embedded'] = embedded
        return data

    def run_embed(self, x, embed_fns):
        return torch.cat([fn(x) for fn in embed_fns], -1)
