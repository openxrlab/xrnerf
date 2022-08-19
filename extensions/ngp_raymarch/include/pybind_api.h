
// #include "density_grid_sampler_header.h"

void generate_grid_samples_nerf_nonuniform_api(const torch::Tensor &density_grid,
    const int &density_grid_ema_step, const int &n_elements, const int &max_cascade,
    const float &thresh, const float &aabb0, const float &aabb1,
    torch::Tensor &density_grid_positions_uniform,
    torch::Tensor &density_grid_indices_uniform);

void mark_untrained_density_grid_api(const torch::Tensor &focal_lengths,
    const torch::Tensor &transforms,
    const int &n_elements, const int &n_images,
    const int &img_resolution0, const int &img_resolution1,
    torch::Tensor &density_grid);

void splat_grid_samples_nerf_max_nearest_neighbor_api( const torch::Tensor &mlp_out,
    const torch::Tensor &density_grid_indices,  const int &padded_output_width,
    const int &n_density_grid_samples,  torch::Tensor &density_grid_tmp);

void ema_grid_samples_nerf_api(const torch::Tensor &density_grid_tmp,  int &n_elements,
        float &decay, torch::Tensor &density_grid);

void update_bitfield_api(const torch::Tensor &density_grid,
    torch::Tensor &density_grid_mean,  torch::Tensor &density_grid_bitfield);

void rays_sampler_api(
    const torch::Tensor &rays_o,
    const torch::Tensor &rays_d,
    const torch::Tensor &density_grid_bitfield,
    const torch::Tensor &metadata,
    const torch::Tensor &imgs_id,
    const torch::Tensor &xforms,
    const float &aabb0,
    const float &aabb1,
    const float &near_distance,
    const float &cone_angle_constant,
    torch::Tensor &coords_out,
    torch::Tensor &rays_index,
    torch::Tensor &rays_numsteps,
    torch::Tensor &ray_numstep_counter);

void compacted_coord_api(
    const torch::Tensor &network_output,
    const torch::Tensor &coords_in,
    const torch::Tensor &rays_numsteps,
    const torch::Tensor &bg_color_in,
    const int &rgb_activation_i,
    const int &density_activation_i,
    const float &aabb0,
    const float &aabb1,

    torch::Tensor &coords_out,
    torch::Tensor &rays_numsteps_compacted,
    torch::Tensor &compacted_rays_counter,
    torch::Tensor &compacted_numstep_counter);

void calc_rgb_forward_api(
    const torch::Tensor &network_output,
    const torch::Tensor &coords_in,
    const torch::Tensor &rays_numsteps,
    const torch::Tensor &rays_numsteps_compacted,
    const torch::Tensor &training_background_color,

    const int &rgb_activation_i,
    const int &density_activation_i,
    const float &aabb0,
    const float &aabb1,

    torch::Tensor &rgb_output);

void calc_rgb_backward_api(
    const torch::Tensor &network_output,
    const torch::Tensor &rays_numsteps_compacted,
    const torch::Tensor &coords_in,
    const torch::Tensor &grad_x,
    const torch::Tensor &rgb_output,
    const torch::Tensor &density_grid_mean,

    const int &rgb_activation_i,
    const int &density_activation_i,
    const float &aabb0,
    const float &aabb1,
    torch::Tensor &dloss_doutput);

void calc_rgb_influence_api(
    const torch::Tensor &network_output,
    const torch::Tensor &coords_in,
    const torch::Tensor &rays_numsteps,
    const torch::Tensor &bg_color_cpu,
    const int &rgb_activation_i,
    const int &density_activation_i,
    const float &aabb0,
    const float &aabb1,
    torch::Tensor &rgb_output,
    torch::Tensor &alpha_output);
