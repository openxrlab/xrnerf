import os

import numpy as np
import torch
import trimesh
from mesh_grid_searcher import MeshGridSearcher

torch.set_default_tensor_type('torch.cuda.FloatTensor')

data_dir = '../../data/human2/SMPL'
subjects = os.listdir(data_dir)

for subject in subjects:
    mesh_path = os.path.join(data_dir, subject, f'smplx.obj')
    mesh = trimesh.load(mesh_path)

    verts = torch.Tensor(mesh.vertices)
    faces = torch.Tensor(mesh.faces).int()

    mygrid = MeshGridSearcher(verts, faces)

    B_MAX = mesh.vertices.max(0)
    B_MIN = mesh.vertices.min(0)
    length = B_MAX - B_MIN
    points = torch.Tensor(np.random.rand(10, 3) * length + B_MIN)

    nearest_pts, _ = mygrid.nearest_points(points)
    inside = mygrid.inside_mesh(points)
    inside_trimesh = mesh.contains(points.cpu().numpy())

    sdf = (torch.norm(nearest_pts - points, dim=1) *
           inside.float()).cpu().numpy()
    sdf_trimesh = trimesh.proximity.signed_distance(mesh, points.cpu().numpy())
    inside = (inside.cpu().numpy() + 1) / 2

    inside_error = np.abs(inside - inside_trimesh).sum()
    dist_error = np.abs(sdf - sdf_trimesh).sum()
    print('[', subject, '] inside_error: ', inside_error, ' dist_error: ',
          dist_error)
    print('scale: ', length.max())
    print(np.abs(sdf - sdf_trimesh))
