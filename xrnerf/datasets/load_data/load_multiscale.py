import json
import os

import numpy as np
from PIL import Image


def load_multiscale_data(datadir, mode, white_bkgd):
    """Load images from disk."""
    with open(os.path.join(datadir, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[mode]
    meta = {k: np.array(meta[k]) for k in meta}
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in meta
    images = []
    for fbase in meta['file_path']:
        fname = os.path.join(datadir, fbase)
        with open(fname, 'rb') as imgin:
            image = np.array(Image.open(imgin), dtype=np.float32) / 255.
        if white_bkgd:
            image = image[..., :3] * \
                image[..., -1:] + (1. - image[..., -1:])
        images.append(image[..., :3])
    images = images
    n_examples = len(images)

    return meta, images, n_examples
