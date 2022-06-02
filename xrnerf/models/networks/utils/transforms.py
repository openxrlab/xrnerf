
import torch
import numpy as np


def recover_shape(data, to_shape):
    # 对于测试数据，回复到(H, W, ...)的格式
    to_shape = list(to_shape[:-1]) + list(data.shape[1:])
    data = torch.reshape(data, to_shape)
    return data

def merge_ret(ret, fine_ret):
    ret['rgb0'] = ret['rgb']
    ret['disp0'] = ret['disp']
    ret['acc0'] = ret['acc']

    ret['rgb'] = fine_ret['rgb']
    ret['disp'] = fine_ret['disp']
    ret['acc'] = fine_ret['acc']
    return ret
