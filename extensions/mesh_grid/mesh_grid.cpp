#include <torch/torch.h>


at::Tensor insert_grid_surface_cuda(
    at::Tensor verts, at::Tensor faces,
    at::Tensor minmax, at::Tensor num, float step,
    at::Tensor tri_num
);

void search_nearest_point_cuda (
    at::Tensor points, at::Tensor verts, at::Tensor faces,
    at::Tensor tri_num, at::Tensor tri_idx, at::Tensor num,
    at::Tensor minmax, float step, at::Tensor near_faces,
    at::Tensor near_pts, at::Tensor coeff
);

void search_inside_mesh_cuda (
    at::Tensor points, at::Tensor verts, at::Tensor faces,
    at::Tensor tri_num, at::Tensor tri_idx, at::Tensor num,
    at::Tensor minmax, float step, at::Tensor signs
);

void search_intersect_cuda (
	at::Tensor origins, at::Tensor directions, at::Tensor verts,
    at::Tensor faces, at::Tensor tri_num, at::Tensor tri_idx,
    at::Tensor num, at::Tensor minmax, float step, at::Tensor intersect
);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")


at::Tensor insert_grid_surface(
    at::Tensor verts, at::Tensor faces,
    at::Tensor minmax, at::Tensor num, float step,
    at::Tensor tri_num
) {
    CHECK_CUDA(verts);
    CHECK_CUDA(faces);
    CHECK_CUDA(minmax);
    CHECK_CUDA(num);
    CHECK_CUDA(tri_num);

    return insert_grid_surface_cuda(verts, faces, minmax, num, step, tri_num);
}

void search_nearest_point(
    at::Tensor points, at::Tensor verts, at::Tensor faces,
    at::Tensor tri_num, at::Tensor tri_idx, at::Tensor num,
    at::Tensor minmax, float step, at::Tensor near_faces,
    at::Tensor near_pts, at::Tensor coeff
) {
    CHECK_CUDA(points);
    CHECK_CUDA(verts);
    CHECK_CUDA(faces);
    CHECK_CUDA(tri_num);
    CHECK_CUDA(tri_idx);
    CHECK_CUDA(num);
    CHECK_CUDA(minmax);
    CHECK_CUDA(near_faces);
    CHECK_CUDA(coeff);

    search_nearest_point_cuda(points, verts, faces, tri_num, tri_idx, num,
                                minmax, step, near_faces, near_pts, coeff);
}

void search_inside_mesh(
    at::Tensor points, at::Tensor verts, at::Tensor faces,
    at::Tensor tri_num, at::Tensor tri_idx, at::Tensor num,
    at::Tensor minmax, float step, at::Tensor signs
) {
    CHECK_CUDA(points);
    CHECK_CUDA(verts);
    CHECK_CUDA(faces);
    CHECK_CUDA(tri_num);
    CHECK_CUDA(tri_idx);
    CHECK_CUDA(num);
    CHECK_CUDA(minmax);
    CHECK_CUDA(signs);

    search_inside_mesh_cuda(points, verts, faces, tri_num, tri_idx, num,
                                minmax, step, signs);
}

void search_intersect (
	at::Tensor origins, at::Tensor directions, at::Tensor verts,
    at::Tensor faces, at::Tensor tri_num, at::Tensor tri_idx,
    at::Tensor num, at::Tensor minmax, float step, at::Tensor intersect
){
    CHECK_CUDA(origins);
    CHECK_CUDA(directions);
    CHECK_CUDA(verts);
    CHECK_CUDA(faces);
    CHECK_CUDA(tri_num);
    CHECK_CUDA(tri_idx);
    CHECK_CUDA(num);
    CHECK_CUDA(minmax);
    CHECK_CUDA(intersect);

    search_intersect_cuda(origins, directions, verts, faces, tri_num, tri_idx, num,
                                minmax, step, intersect);
}

at::Tensor cumsum(
    at::Tensor input
){
    input.set_(input.cumsum(0));
    // input.set_(at::zeros(input.sizes()));
    // input.zero_();
    input = input.reshape({1,1,-1});
    return input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("insert_grid_surface", &insert_grid_surface, "INSERT_GRID_SURFACE (CUDA)");
    m.def("search_nearest_point", &search_nearest_point, "SEARCH_NEAREST_POINT (CUDA)");
    m.def("search_inside_mesh", &search_inside_mesh, "SEARCH_INSIDE_MESH (CUDA)");
    m.def("search_intersect", &search_intersect, "SEARCH_INTERSECT (CUDA)");
    m.def("cumsum", &cumsum, "RESHAPE_TENSOR");
}
