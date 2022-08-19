import os

import cv2
import numpy as np


class VideoGenerator:
    def __init__(self, project_dir, work_folder, json_path, config_path, \
                dataname, iters):
        self.iters = iters
        self.dataname = dataname
        self.config_path = config_path
        self.project_dir = project_dir
        self.json_path = os.path.join(work_folder, json_path)
        self.spirals_folder = os.path.join(work_folder,
                                           'visualizations/spirals')

    def run_test(self):
        for i in iters:
            shell = ' cd {} && '.format(self.project_dir)
            shell += '/home/zhengchengyao/Document/apps/anaconda3/envs/hashnerf/bin/python run_nerf.py --config {} '.format(
                self.config_path)
            shell += '--dataname {} --test_only '.format(self.dataname)
            shell += '--load_from iter_{}.pth '.format(i)
            print(shell)
            os.system(shell)

    def read_video(self, video_path):
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

    def write_video(self, imgs, vid_path, fps=25):
        h, w = imgs[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video = cv2.VideoWriter(vid_path, fourcc, fps, (w, h), True)
        for i, img in enumerate(imgs):
            video.write(img)
            print(i, img.shape, vid_path)
        video.release()

    def blend_img(self, imgs_list, vid_i, time_i, win=3):
        n_vid = len(imgs_list)
        print(
            vid_i,
            time_i,
            n_vid,
        )
        imgs_list[vid_i][time_i]
        start_i = vid_i - win // 2
        end_i = start_i + win
        imgs = []
        for i in range(start_i, end_i):
            if i < 0:
                cur_i = 0
            elif i >= n_vid:
                cur_i = n_vid - 1
            else:
                cur_i = i
            imgs.append(imgs_list[cur_i][time_i])
        win_w = [0.2, 0.6, 0.2]
        res = np.zeros_like(imgs[0]).astype(np.float)
        for i, img in enumerate(imgs):
            res += (img * win_w[i]).astype(np.float)
        res = res.astype(np.uint8)
        return res

    def add_text(self, img, i_frame, n_frame, max_iter):
        h, w = img.shape[:2]
        cur_iter = int(i_frame / n_frame * max_iter)
        font = cv2.FONT_HERSHEY_SIMPLEX
        center = (int(w * 0.75), int(h * 0.08))
        cv2.putText(img, 'iter:' + str(cur_iter), center, font, 1, (0, 0, 0),
                    2)
        return img

    def generate_video(self,
                       fps=25,
                       n_cycle=4,
                       out_path='ngp.mp4',
                       max_iter=20000):
        imgs_list = []
        for i in self.iters:
            vid_path = os.path.join(self.spirals_folder,
                                    '{}_rgb.mp4'.format(i))
            imgs = self.read_video(vid_path)
            imgs_list.append(imgs)
        g_imgs = []
        n_img = len(imgs_list[0])
        n_vid = len(imgs_list)
        n_frame = n_img * n_cycle
        for i in range(n_frame):
            time_i = i % n_img
            vid_i = int((i / n_frame) * n_vid)
            img = self.blend_img(imgs_list, vid_i, time_i)
            img = self.add_text(img, i, n_frame, max_iter)
            g_imgs.append(img)
        self.write_video(g_imgs, out_path, fps=25)
        return


# python run_nerf.py --config configs/instant_ngp/nerf_blender_local01.py \
# --dataname lego --test_only \
# --load_from iter_40000.pth

if __name__ == '__main__':

    dataname = 'lego'
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work_folder = os.path.join(
        project_dir, 'work_dirs/instant_ngp/nerf_{}_base01'.format(dataname))
    config_path = os.path.join(project_dir,
                               'configs/instant_ngp/nerf_blender_local01.py')
    json_path = '02-Aug-14-11.log.json'
    print(project_dir)
    iters = list(range(500, 5001, 500))+list(range(6000, 12001, 1500))+ \
            list(range(15000, 20001, 5000))+[30000, 37000]
    print(iters)
    vg = VideoGenerator(project_dir, work_folder, json_path, config_path,
                        dataname, iters)
    # vg.run_test()
    vg.generate_video()

    dataname = 'lego'
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    work_folder = os.path.join(
        project_dir, 'work_dirs/instant_ngp/nerf_{}_base01'.format(dataname))
    config_path = os.path.join(project_dir,
                               'configs/instant_ngp/nerf_blender_local01.py')
    json_path = '02-Aug-14-11.log.json'
    print(project_dir)
    iters = [50000]
    print(iters)
    vg = VideoGenerator(project_dir, work_folder, json_path, config_path,
                        dataname, iters)
    # vg.run_test()
    vg.generate_video()
