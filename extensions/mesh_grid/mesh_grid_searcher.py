import torch
import trimesh
from mesh_grid import (cumsum, insert_grid_surface, search_inside_mesh,
                       search_intersect, search_nearest_point)


class MeshGridSearcher:
    def __init__(self, verts=None, faces=None):
        if verts is not None and faces is not None:
            self.set_mesh(verts, faces)

    def set_mesh(self, verts, faces):
        self.verts = verts
        self.faces = faces
        _min, _ = torch.min(verts, 0)
        _max, _ = torch.max(verts, 0)
        self.step = (torch.cumprod(_max - _min, 0)[-1] / len(verts))**(1. / 3.)
        l = _max - _min
        c = (_max + _min) / 2
        l = torch.max(torch.floor(l / self.step), torch.zeros_like(l)) + 1
        _min_step = c - self.step * l / 2
        self.num = torch.cat([l, torch.cumprod(l, 0)[-1:]]).int()
        self.minmax = torch.cat([_min_step, _max])

        self.tri_num = torch.zeros(self.num[-1],
                                   dtype=torch.int32).to(verts.device)
        self.tri_idx = insert_grid_surface(self.verts, self.faces, self.minmax,
                                           self.num, self.step, self.tri_num)

    def nearest_points(self, points):
        points = points.to(self.verts.device)
        nearest_faces = torch.zeros(points.shape[-2],
                                    dtype=torch.int32).to(self.verts.device)
        coeff = torch.zeros(points.shape,
                            dtype=torch.float32).to(self.verts.device)
        nearest_pts = torch.zeros_like(coeff)
        search_nearest_point(points, self.verts, self.faces, self.tri_num,
                             self.tri_idx, self.num, self.minmax, self.step,
                             nearest_faces, nearest_pts, coeff)
        return nearest_pts, nearest_faces

    def inside_mesh(self, points):
        points = points.to(self.verts.device)
        inside = torch.zeros(points.shape[-2],
                             dtype=torch.float32).to(self.verts.device)
        search_inside_mesh(points, self.verts, self.faces, self.tri_num,
                           self.tri_idx, self.num, self.minmax, self.step,
                           inside)
        return inside

    def intersects_any(self, origins, directions):
        origins = origins.to(self.verts.device)
        directions = directions.to(self.verts.device)
        intersect = torch.zeros(origins.shape[-2],
                                dtype=torch.bool).to(self.verts.device)
        search_intersect(origins, directions, self.verts, self.faces,
                         self.tri_num, self.tri_idx, self.num, self.minmax,
                         self.step, intersect)
        return intersect
