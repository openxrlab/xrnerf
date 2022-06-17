import numpy as np
import torch


def flatten(x):
    # Always flatten out the height x width dimensions
    x = [y.reshape([-1, y.shape[-1]]) for y in x]
    x = np.concatenate(x, axis=0)
    return torch.tensor(x, dtype=torch.float32)
