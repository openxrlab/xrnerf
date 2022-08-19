import torch

img2mse = lambda x, y: torch.mean((x - y)**2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(
    torch.Tensor([10.]).to(x.device))


def HuberLoss(x, y, delta=0.1, reduction='sum'):
    rel = (x - y).abs()
    sqr = 0.5 / delta * rel * rel
    loss = torch.where(rel > delta, rel - 0.5 * delta, sqr)
    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    return loss
