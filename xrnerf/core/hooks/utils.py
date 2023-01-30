import numpy as np
from skimage.metrics import structural_similarity as ssim


def to8b(x):
    """to8b."""
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def img2mse(x, y):
    """img2mse."""
    return np.mean((x - y)**2)


def mse2psnr(x):
    """mse2psnr."""
    return -10. * np.log(x) / np.log(np.array([10.]))


def calculate_ssim(im1, im2, data_range=255, multichannel=True):
    """calculate_ssim."""
    full_ssim = ssim(im1,
                     im2,
                     val_range=data_range,
                     multichannel=multichannel,
                     full=True)[1]
    out_ssim = full_ssim.mean()
    return out_ssim
