import os

import cv2
import numpy as np


def read_video(video_path):
    imgs = []
    video_s = cv2.VideoCapture(video_path)
    while True:
        _, img_si = video_s.read()
        if img_si is not None:
            imgs.append(img_si)
        else:
            break
    video_s.release()
    return imgs


def write_video(imgs, vid_path, fps=25):
    h, w = imgs[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter(vid_path, fourcc, fps, (w, h), True)
    for i, img in enumerate(imgs):
        video.write(img)
        print(i, img.shape, vid_path)
    video.release()


# python run_nerf.py --config configs/instant_ngp/nerf_blender_local01.py \
# --dataname lego --test_only \
# --load_from iter_40000.pth

if __name__ == '__main__':

    vid_folder = '/home/zhengchengyao/Music/FtpShow/0328/vids/ngp'
    vid_names = [
        'lego',
        'chair',
        'drums',
        'ficus',
        'hotdog',
        'materials',
        'mic',
        'ship',
    ]
    imgs_list = []
    for name in vid_names:
        vid_path = os.path.join(vid_folder, '{}.mp4'.format(name))
        imgs = read_video(vid_path)
        imgs_list.append(imgs)
    res_imgs = []
    h, w = imgs_list[0][0].shape[:2]
    for time_i in range(len(imgs_list[0])):
        res_img = np.zeros((h * 2, w * 4, 3)).astype(np.uint8)
        for i, imgs in enumerate(imgs_list):
            img = imgs[time_i]
            y = i // 4
            x = i % 4
            res_img[y * h:(y + 1) * h, x * w:(x + 1) * w, :] = img
        res_imgs.append(res_img)

    write_video(res_imgs, 'ngp_cat.mp4', fps=25)
