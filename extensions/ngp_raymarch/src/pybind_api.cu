#include <torch/extension.h>
#include "pybind_api.h"


// you can 'import xxx' to use
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("generate_grid_samples_nerf_nonuniform_api", &generate_grid_samples_nerf_nonuniform_api, "info");
    m.def("mark_untrained_density_grid_api", &mark_untrained_density_grid_api, "info");
    m.def("splat_grid_samples_nerf_max_nearest_neighbor_api", &splat_grid_samples_nerf_max_nearest_neighbor_api, "info");
    m.def("ema_grid_samples_nerf_api", &ema_grid_samples_nerf_api, "info");
    m.def("update_bitfield_api", &update_bitfield_api, "info");
    m.def("rays_sampler_api", &rays_sampler_api, "info");
    m.def("compacted_coord_api", &compacted_coord_api, "info");
    m.def("calc_rgb_forward_api", &calc_rgb_forward_api, "info");
    m.def("calc_rgb_backward_api", &calc_rgb_backward_api, "info");
    m.def("calc_rgb_influence_api", &calc_rgb_influence_api, "info");
}


// you will have to use 'torch.ops.load_library("xxx.so")' to use
// TORCH_LIBRARY(add2, m) {
//     m.def("torch_launch_add2", torch_launch_add2);
// }
