import numpy as np
import torch


def unfold_batching(data):
    # 将dataloader叠起来的batching数据，拼成正确的batching格式
    # before: (bs, N_rand_per_sampler ...)
    # after: (bs*N_rand_per_sampler ...)
    if len(data.shape) > 1:
        bs = data.shape[0]
        data = torch.cat([data[b] for b in range(bs)], 0)
    return data
