

import torch
import numpy as np

def unfold_batching(data):
    # 将dataloader叠起来的batching数据，拼成正确的batching格式
    # before: (bs, N_rand_per_samplerm ...)
    # after: (bs*N_rand_per_samplerm ...)
    bs = data.shape[0]
    data = torch.cat([data[b] for b in range(bs)], 0)
    return data
