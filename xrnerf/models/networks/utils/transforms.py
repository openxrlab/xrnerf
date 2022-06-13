import numpy as np
import torch


def recover_shape(data, to_shape):
    # 对于测试数据，回复到(H, W, ...)的格式
    to_shape = list(to_shape[:-1]) + list(data.shape[1:])
    data = torch.reshape(data, to_shape)
    return data


def merge_ret(ret, fine_ret):
    ret['coarse_rgb'] = ret['rgb']
    ret['coarse_disp'] = ret['disp']
    ret['coarse_acc'] = ret['acc']

    ret['rgb'] = fine_ret['rgb']
    ret['disp'] = fine_ret['disp']
    ret['acc'] = fine_ret['acc']
    return ret
