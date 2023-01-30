import json
import os

import cv2
import numpy as np


def load_google_data(datadir, factor=None):
    imgdir = os.path.join(datadir, 'images')
    imgfiles = [
        os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))
        if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        or f.endswith('jpeg')
    ]
    imgs = [
        f for f in imgfiles
        if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])
    ]

    sh = np.array(cv2.imread(imgfiles[0]).shape)
    imgs = []
    for f in imgfiles:
        im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if im.shape[-1] == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)
        im = cv2.resize(im, (sh[1] // factor, sh[0] // factor),
                        interpolation=cv2.INTER_AREA)
        im = im.astype(np.float32) / 255
        imgs.append(im)
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)

    data = json.load(open(os.path.join(datadir, 'poses_enu.json')))
    poses = np.array(data['poses'])[:, :-2].reshape([-1, 3, 5])
    poses[:, :2, 4] = np.array(sh[:2] // factor).reshape([1, 2])
    poses[:, 2, 4] = poses[:, 2, 4] * 1. / factor

    scene_scale = data['scene_scale']
    scene_origin = np.array(data['scene_origin'])
    scale_split = data['scale_split']
    return imgs, poses, scene_scale, scene_origin, scale_split
