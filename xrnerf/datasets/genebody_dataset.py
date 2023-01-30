import os
import sys

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, '..'))

from scipy.optimize import minimize

from .base import BaseDataset
from .builder import DATASETS
from .pipelines import Compose
from .utils.genebody import gen_cam_views, load_obj_mesh, load_ply


@DATASETS.register_module()
class GeneBodyDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        return parser

    def __init__(self, opt, pipeline, phase='eval', root=None, move_cam=0):
        super(GeneBodyDataset, self).__init__()
        self.opt = opt
        self.is_train = phase == 'train'
        self.is_render = phase == 'render'
        self.projection_mode = 'perspective'
        self.eval_skip = self.opt.eval_skip
        self.train_skip = self.opt.train_skip
        self.genebody_seq_len = 150

        self.root = root if root is not None else opt.dataroot
        self.phase = 'val'
        self.load_size = self.opt.loadSize

        self.B_MIN = np.array([-128, -28, -128])
        self.B_MAX = np.array([128, 228, 128])

        self.num_views = self.opt.num_views
        self.input_views = [1, 13, 25, 37]
        self.test_views = sorted(list(range(0, 48)))
        self.split = np.load(os.path.join(self.root, 'genebody_split.npy'),
                             allow_pickle=True).item()
        self.sequences = self.split['train'] if self.is_train else self.split[
            'test']
        self.frames, self.cam_names, self.subjects, self.frames_id = self.get_frames(
        )
        self.load_smpl_param = any(
            [self.opt.use_smpl_sdf, self.opt.use_t_pose])
        self.load_smpl_mesh = any([self.opt.use_smpl_sdf, self.opt.use_t_pose])
        self.smpl_type = self.opt.smpl_type
        self.smpl_t_pose = load_obj_mesh(
            os.path.join(self.opt.t_pose_path, f'{self.smpl_type}.obj'))
        self.use_smpl_depth = opt.use_smpl_depth
        # PIL to tensor
        self.to_tensor_normal = transforms.Compose([
            transforms.Resize(self.load_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.to_tensor = transforms.Compose(
            [transforms.Resize(self.load_size),
             transforms.ToTensor()])

        self.move_cam = move_cam if not self.is_train else 0
        self.use_white_bkgd = self.opt.use_white_bkgd
        self.pipeline = Compose(pipeline)

    def get_frames(self, i=0):
        sequences = self.sequences
        frames, subjects, cam_names, frames_id = [], [], [], []
        i = self.input_views[i % self.num_views]
        for seq in sequences:
            if os.path.exists(os.path.join(self.root, seq)):
                files = sorted([f for f in os.listdir(os.path.join(self.root, \
                        seq, 'param')) \
                        if f[-4:] == '.npy'])
                files = sorted(files)
                cam_names += ['%02d' for i in range(len(files))]
                subjects += [seq for i in range(len(files))]
                frames_id += list(range(len(files)))

                for f in files:
                    f = f[:-4]
                    frames += [f]
        return frames, cam_names, subjects, frames_id

    def get_render_poses(self, annots, move_cam=150):
        height, pitch = [], []
        for view in range(1, 48, 3):
            view = '%02d' % view
            if view in annots.keys():
                height.append(annots[view]['c2w'][1, 3])
                z_rodrigous = annots[view]['c2w'][:3, :3] @ np.array([[0], [0],
                                                                      [1]])
                pitch.append(z_rodrigous[1, 0])
        transl = np.array([0, np.mean(np.array(height)), 0])
        z_pitch = np.mean(np.array(pitch))

        render_poses = gen_cam_views(transl, z_pitch, move_cam)
        return render_poses

    def __len__(self):
        if self.is_train:
            return len(self.frames) * len(self.test_views) // self.train_skip
        else:
            return len(self.frames) // self.eval_skip

    def image_cropping(self, mask):
        a = np.where(mask != 0)
        h, w = list(mask.shape[:2])
        if len(a[0]) > 0:
            top, left, bottom, right = np.min(a[0]), np.min(a[1]), np.max(
                a[0]), np.max(a[1])
        else:
            return 0, 0, mask.shape[0], mask.shape[1]
        bbox_h, bbox_w = bottom - top, right - left

        # padd bbox
        bottom = min(int(bbox_h * 0.1 + bottom), h)
        top = max(int(top - bbox_h * 0.1), 0)
        right = min(int(bbox_w * 0.1 + right), w)
        left = max(int(left - bbox_h * 0.1), 0)
        bbox_h, bbox_w = bottom - top, right - left

        if bbox_h >= bbox_w:
            w_c = (left + right) / 2
            size = bbox_h
            if w_c - size / 2 < 0:
                left = 0
                right = size
            elif w_c + size / 2 >= w:
                left = w - size
                right = w
            else:
                left = int(w_c - size / 2)
                right = left + size
        else:  # bbox_w >= bbox_h
            h_c = (top + bottom) / 2
            size = bbox_w
            if h_c - size / 2 < 0:
                top = 0
                bottom = size
            elif h_c + size / 2 >= h:
                top = h - size
                bottom = h
            else:
                top = int(h_c - size / 2)
                bottom = top + size

        return top, left, bottom, right

    def get_near_far(self, smpl_verts, w2c):
        vp = smpl_verts.dot(w2c[:3, :3].T) + w2c[:3, 3:].T
        vmin, vmax = vp.min(0), vp.max(0)
        near, far = vmin[2], vmax[2]
        near, far = near - (far - near) / 2, far + (far - near) / 2
        return near, far

    def get_realworld_scale(self, smpl_verts, bbox, w2c, K):
        smpl_min, smpl_max = smpl_verts.min(0), smpl_verts.max(0)
        # reprojected smpl verts
        vp = smpl_verts.dot(w2c[:3, :3].T) + w2c[:3, 3:].T
        vp = vp.dot(K.T)
        vp = vp[:, :2] / (vp[:, 2:] + 1e-8)
        vmin, vmax = vp.min(0), vp.max(0)

        # compare with bounding box
        bbox_h = bbox[1][0] - bbox[0][0]
        bbox_w = bbox[1][1] - bbox[0][1]
        long_axis = bbox_h / (vmax[1] - vmin[1]) * (
            smpl_max[1] - smpl_min[1]) if bbox_h > bbox_w else bbox_w / (
                vmax[0] - vmin[0]) * (smpl_max[0] - smpl_min[0])
        spatial_freq = 180 / long_axis / 0.5
        return spatial_freq

    def get_image(self,
                  sid,
                  num_views,
                  view_id=None,
                  random_sample=False,
                  smpl_verts=None):
        frame = self.frames[sid]
        subject = self.subjects[sid]
        # some of the sequence has some view missing
        if subject == 'wuwenyan':
            test_views = list(set(self.test_views) - set([34, 36]))
        elif (subject == 'dannier' or subject == 'Tichinah_jervier'):
            test_views = list(set(self.test_views) - set([32]))
        elif subject == 'joseph_matanda':
            test_views = list(
                set(list(range(48))) - set([39, 40, 42, 43, 44, 45, 46, 47]))
        else:
            test_views = self.test_views
        test_views = sorted(test_views)

        # Select a random view_id from self.max_view_angle if not given
        if self.is_train:
            if view_id is None or random_sample:
                view_id = test_views[np.random.randint(len(test_views))]
            else:
                view_id = test_views[view_id % len(test_views)]
            # The ids are an even distribution of num_views around view_id
            view_ids = self.input_views + [view_id]
        else:
            if self.is_render:
                view_ids = self.input_views
            else:
                view_ids = self.input_views + test_views

        calib_list = []
        image_list = []
        mask_list = []
        extrinsic_list = []
        bbox_list = []
        smpl_depth_list = []
        spatial_freqs = []
        annot_path = os.path.join(self.root, subject, f'annots.npy')
        annots = np.load(annot_path, allow_pickle=True).item()['cams']

        for i, vid in enumerate(view_ids):
            view = '%02d' % vid
            mask_folder = 'mask'
            mask_path = os.path.join(self.root, subject, mask_folder,
                                     self.cam_names[sid] % vid)
            mask_path = [os.path.join(mask_path,f) for f in os.listdir(mask_path) \
                        if frame in f]
            image_path = os.path.join(self.root, subject, 'image',
                                      self.cam_names[sid] % vid)
            image_path = [os.path.join(image_path,f) for f in os.listdir(image_path) \
                          if frame in f]
            image_np = imageio.imread(image_path[0])
            mask_np = imageio.imread(mask_path[0])
            size = image_np.shape
            if self.use_smpl_depth and i < self.num_views:

                smpl_depth_path = os.path.join(self.root, subject,
                                               'smpl_depth',
                                               self.cam_names[sid] % vid)

                smpl_depth_path = [os.path.join(smpl_depth_path,f) for f in os.listdir(smpl_depth_path) \
                                    if frame in f]
                smpl_depth = imageio.imread(smpl_depth_path[0])
                smpl_depth = smpl_depth.astype(np.float32) / 1000.0
            top, left, bottom, right = self.image_cropping(mask_np)

            mask_np = mask_np[top:bottom, left:right]
            image_crop = image_np[top:bottom, left:right]

            mask_np = cv2.resize(mask_np.copy(), (self.load_size,self.load_size), \
                              interpolation = cv2.INTER_NEAREST)
            image_crop = cv2.resize(image_crop.copy(), (self.load_size,self.load_size), \
                              interpolation = cv2.INTER_CUBIC)
            image = Image.fromarray(image_crop)
            mask_np = mask_np > 128
            if self.use_smpl_depth and i < self.num_views:
                smpl_depth = smpl_depth[top:bottom, left:right]
                smpl_depth = cv2.resize(smpl_depth, (self.load_size,self.load_size), \
                                interpolation = cv2.INTER_NEAREST)
                mask_np = np.logical_or(mask_np, smpl_depth > 0)
                smpl_depth_list.append(torch.from_numpy(smpl_depth))

            a = np.where(mask_np != 0)
            try:
                bbox = [[np.min(a[0]), np.min(a[1])], [np.max(a[0]), np.max(a[1])]] if len(a[0]) > 0 else \
                    [[0, 0], [self.load_size, self.load_size]]
            except:
                print(
                    os.path.join(self.root, subject, mask_folder,
                                 self.cam_names[sid] % vid))
                print(top, left, bottom, right)
                print(mask_np)
                exit(0)
            bbox_list.append(bbox)
            mask = torch.from_numpy(mask_np.astype(np.float32)).view(
                1, self.load_size, self.load_size)
            mask_list.append(mask)
            image = self.to_tensor(
                image) if i >= num_views else self.to_tensor_normal(image)
            if i >= self.num_views and self.use_white_bkgd:
                image = image * mask + (1. - mask)
            image = mask.type(image.dtype).expand(3, -1, -1) * image
            rgb = image.cpu().numpy().transpose([1, 2, 0])

            K = np.array(annots[view]['K'], dtype=np.float32)

            K[0, 2] -= left
            K[1, 2] -= top
            K[0, :] *= self.load_size / float(right - left)
            K[1, :] *= self.load_size / float(bottom - top)

            c2w = np.array(annots[view]['c2w'], dtype=np.float32)
            w2c = np.linalg.inv(c2w)

            dist = np.array(annots[view]['D'], dtype=np.float32)
            # determine near far plane from smpl estimation
            near, far = self.get_near_far(smpl_verts, w2c)

            # determine valid body part from smpl and bounding box
            if i < self.num_views:
                spatial_freq = self.get_realworld_scale(
                    smpl_verts, bbox, w2c, K)
                spatial_freqs.append(spatial_freq)

            calib = torch.Tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]] +
                                 list(dist.reshape(-1)) + [near, far]).float()
            extrinsic = torch.from_numpy(w2c)
            image_list.append(image)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)

        if not self.is_train and self.move_cam > 0:
            bboxs = np.array(bbox_list[:self.num_views]).reshape(-1, 2)
        else:
            bboxs = np.array(bbox_list[self.num_views:]).reshape(-1, 2)
        centroid = np.array([mask_np.shape[0], mask_np.shape[1]]) / 2
        bbox = (np.max(np.abs(bboxs - centroid), axis=0) * \
                np.array([1, np.sqrt(2)])).astype(np.int32)
        bbox = np.array([centroid - bbox, centroid + bbox]).T
        bbox = np.clip(bbox.reshape(-1), 0, self.load_size)
        spatial_freq = min(spatial_freqs)

        if self.is_render:
            # render free view point video on full image resolution
            render_id = sid % (self.genebody_seq_len // self.eval_skip)
            render_c2ws = self.get_render_poses(annots, self.move_cam)
            w2c = np.linalg.inv(render_c2ws[render_id])
            K = annots['K'][25]

            render_extrinsics = torch.from_numpy(w2c.astype(np.float32))
            near, far = self.get_near_far(smpl_verts, w2c)
            render_calibs = torch.Tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]] +
                                         list(np.zeros_like(dist)) +
                                         [near, far]).float()
            bbox = np.array([0, size[0], 0, size[1]])
            extrinsic_list = extrinsic_list[:self.num_views] + [
                render_extrinsics
            ]
            calib_list = calib_list[:self.num_views] + [render_calibs]
            mask_list = mask_list[:self.num_views]

        if not self.is_train and self.move_cam is 0:
            gt_list = image_list[num_views:]
            image_list = image_list[:num_views]

        return {
            'img':
            torch.stack(image_list, dim=0),
            'mask':
            torch.stack(mask_list[:num_views], dim=0),
            'persps':
            torch.stack(calib_list, dim=0),
            'calib':
            torch.stack(extrinsic_list, dim=0),
            'bbox':
            bbox,
            'render_gt':
            torch.stack(gt_list, dim=0)
            if not self.is_train and self.move_cam is 0 else torch.tensor([]),
            'smpl_depth':
            torch.stack(smpl_depth_list[:self.num_views], dim=0)
            if self.opt.use_smpl_depth else torch.tensor([]),
            'spatial_freq':
            spatial_freq,
            'center':
            torch.from_numpy((smpl_verts.max(0) + smpl_verts.min(0)) / 2),
        }

    def _fetch_train_data(self, index):
        sid = index % len(self.frames)
        vid = (index // len(self.frames)) % len(self.test_views)
        frame = self.frames[sid]
        subject = self.subjects[sid]

        res = {}

        # load smpl data
        param_dir = os.path.join(self.root, subject, f'param')

        param_path = [
            os.path.join(param_dir, f) for f in os.listdir(param_dir)
            if frame in f
        ]
        param = np.load(os.path.join(param_path[0]), allow_pickle=True).item()
        scale, param = param['smplx_scale'], param['smplx']

        res['body_scale'] = scale
        for key in param.keys():
            if isinstance(param[key], torch.Tensor):
                param[key] = param[key].numpy()

        smpl_dir = os.path.join(self.root, subject, f'smpl')
        smpl_path = [
            os.path.join(smpl_dir, f) for f in os.listdir(smpl_dir)
            if frame in f
        ][0]
        if smpl_path[-4:] == '.obj':
            vert, face = load_obj_mesh(smpl_path)
        else:
            smpl = load_ply(smpl_path)
            vert, face = smpl['vertex'][:, :3], smpl['face']
        vert = vert.astype(np.float32)
        # load image data
        image_data = self.get_image(sid,
                                    num_views=self.num_views,
                                    view_id=vid,
                                    random_sample=self.opt.random_multiview,
                                    smpl_verts=vert)
        res.update(image_data)

        T = cv2.Rodrigues(param['global_orient'].reshape(-1, 3)[:1])[0]
        res['bbox'] = np.array(res['bbox'])
        res['smpl_rot'] = torch.from_numpy(T.astype(np.float32)) \
                          if self.load_smpl_mesh else []
        res['smpl_verts']= torch.from_numpy(vert.astype(np.float32)) \
                          if self.load_smpl_mesh else []
        res['smpl_faces']= torch.from_numpy(face.astype(np.int32)) \
                          if self.load_smpl_mesh else []
        res['smpl_betas']= torch.from_numpy(param['betas'].reshape(-1).astype(np.float32)) \
                          if self.load_smpl_param else []
        if self.load_smpl_param:
            t_vert, t_face = self.smpl_t_pose
            res['smpl_t_verts'] = t_vert
            res['smpl_t_faces'] = t_face
        if self.opt.use_t_pose:
            res['smpl_t_verts'] = torch.from_numpy(res['smpl_t_verts'].astype(
                np.float32))
            res['smpl_t_faces'] = torch.from_numpy(res['smpl_t_faces'].astype(
                np.int32))
        else:
            res['smpl_t_verts'] = []
            res['smpl_t_faces'] = []
        res['idx'] = index

        return res

    def __getitem__(self, index):
        if not self.is_train:
            index *= self.eval_skip
        return self._fetch_train_data(index)
