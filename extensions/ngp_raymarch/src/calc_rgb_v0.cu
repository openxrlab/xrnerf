#include "raymarch_shared.h"
#include "ray_sampler_header.h"
#include "calc_rgb.h"


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
    torch::Tensor &rgb_output){
    /*
     * @brief calc_rgb_forward_api
     * @in-param   'network_output' (n_elements, 1)
     * @in-param   'coords_in' (n_elements,)
     * @in-param   'rays_numsteps'
     * @in-param   'rays_numsteps_compacted'
     * @in-param   'training_background_color'
     * @in-param   'rgb_activation_i'
     * @in-param   'density_activation_i'
     * @in-param   'aabb0'
     * @in-param   'aabb1'
     * @out-param  'rgb_output'
     */
    cudaStream_t stream = 0;
    // input
    float* network_output_p = (float*)network_output.data_ptr();
    float* coords_in_p = (float*)coords_in.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    uint32_t* rays_numsteps_compacted_p = (uint32_t*)rays_numsteps_compacted.data_ptr();
    float* training_background_color_p = (float*)training_background_color.data_ptr();

    // output
    float* rgb_output_p = (float*)rgb_output.data_ptr();

    const uint32_t n_rays = rays_numsteps.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));
    uint32_t padded_output_width = network_output.sizes()[1];
    ENerfActivation rgb_activation = ENerfActivation(rgb_activation_i);
    ENerfActivation density_activation = ENerfActivation(density_activation_i);

    compute_rgbs_fp32(0,stream,
        n_rays, m_aabb,padded_output_width,(float*)network_output_p,
        rgb_activation, density_activation,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),
        (uint32_t*)rays_numsteps_p, (Array3f*)rgb_output_p,
        (uint32_t*)rays_numsteps_compacted_p,(Array3f*)training_background_color_p,
        NERF_CASCADES(),MIN_CONE_STEPSIZE());

    cudaDeviceSynchronize();
}


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

    torch::Tensor &dloss_doutput){
    /*
     * @brief calc_rgb_forward_api
     * @in-param   'network_output'
     * @in-param   'coords_in'
     * @in-param   'rays_numsteps_compacted'
     * @in-param   'training_background_color'
     * @in-param   'rgb_activation_i'
     * @in-param   'density_activation_i'
     * @in-param   'aabb0'
     * @in-param   'aabb1'
     * @out-param  'dloss_doutput'
     */
    cudaStream_t stream = 0;
    // input
    float* network_output_p = (float*)network_output.data_ptr();
    float* coords_in_p = (float*)coords_in.data_ptr();
    float* grad_x_p = (float*)grad_x.data_ptr();
    float* rgb_output_p = (float*)rgb_output.data_ptr();
    float* density_grid_mean_p = (float*)density_grid_mean.data_ptr();
    // uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    uint32_t* rays_numsteps_compacted_p = (uint32_t*)rays_numsteps_compacted.data_ptr();

    // output
    float* dloss_doutput_p = (float*)dloss_doutput.data_ptr();

    // cudaMemsetAsync(out0_p, 0, out0->size);
    const unsigned int num_elements = network_output.sizes()[0];
    const uint32_t n_rays = rays_numsteps_compacted.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));
    uint32_t padded_output_width = network_output.sizes()[1];
    ENerfActivation rgb_activation = ENerfActivation(rgb_activation_i);
    ENerfActivation density_activation = ENerfActivation(density_activation_i);

    compute_rgbs_grad_fp32(0,stream,
        n_rays, m_aabb,padded_output_width,(float*)dloss_doutput_p,
        (float*)network_output_p,(uint32_t*)rays_numsteps_compacted_p,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),
        rgb_activation,density_activation,(Array3f*)grad_x_p,(Array3f*)rgb_output_p,
        (float*)density_grid_mean_p,NERF_CASCADES(),MIN_CONE_STEPSIZE());

    cudaDeviceSynchronize();
}


void calc_rgb_influence_api(
    const torch::Tensor &network_output,
    const torch::Tensor &coords_in,
    const torch::Tensor &rays_numsteps,
    const torch::Tensor &bg_color_cpu,
    const int &rgb_activation_i,
    const int &density_activation_i,
    const float &aabb0,
    const float &aabb1,
    torch::Tensor &rgb_output){
    /*
     * @brief calc_rgb_influence_api
     * @in-param   'network_output' (n_elements, 1)
     * @in-param   'coords_in' (n_elements,)
     * @in-param   'rays_numsteps'
     * @in-param   'bg_color_cpu'
     * @in-param   'rgb_activation_i'
     * @in-param   'density_activation_i'
     * @in-param   'aabb0'
     * @in-param   'aabb1'
     * @out-param  'rgb_output'
     */
    cudaStream_t stream = 0;
    // input
    float* network_output_p = (float*)network_output.data_ptr();
    float* coords_in_p = (float*)coords_in.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    float* bg_color_p = (float*)bg_color_cpu.data_ptr();

    // output
    float* rgb_output_p = (float*)rgb_output.data_ptr();

    const uint32_t n_rays = rays_numsteps.sizes()[0];
    const unsigned int num_elements = network_output.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));
    uint32_t padded_output_width = network_output.sizes()[1];
    ENerfActivation rgb_activation = ENerfActivation(rgb_activation_i);
    ENerfActivation density_activation = ENerfActivation(density_activation_i);
    Array3f bg_color = Array3f(bg_color_p[0], bg_color_p[1], bg_color_p[2]);

    compute_rgbs_inference_fp32(0, stream, n_rays, m_aabb, padded_output_width,
        bg_color, (float*)network_output_p,rgb_activation,density_activation,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),
        (uint32_t*)rays_numsteps_p,(Array3f*)rgb_output_p,
        NERF_CASCADES(),MIN_CONE_STEPSIZE());

    cudaDeviceSynchronize();
}
