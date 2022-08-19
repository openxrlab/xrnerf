
#include "raymarch_shared.h"
extern pcg32 rng;


template <typename T>
__global__ void splat_grid_samples_nerf_max_nearest_neighbor_cuda(const uint32_t n_elements,
    const uint32_t *__restrict__ indices, int padded_output_width,
    const T *network_output, float *__restrict__ grid_out,
     ENerfActivation density_activation)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;

    uint32_t local_idx = indices[i];

    // Current setting: optical thickness of the smallest possible stepsize.
    // Uncomment for:   optical thickness of the ~expected step size when the observer is in the middle of the scene
    uint32_t level = 0; // local_idx / (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

    float mlp = network_to_density(float(network_output[i * padded_output_width]), density_activation);
    float optical_thickness = mlp * scalbnf(MIN_CONE_STEPSIZE(), level);

    // Positive floats are monotonically ordered when their bit pattern is interpretes as uint.
    // uint atomicMax is thus perfectly acceptable.
    atomicMax((uint32_t *)&grid_out[local_idx], __float_as_uint(optical_thickness));
}

void splat_grid_samples_nerf_max_nearest_neighbor_api(const torch::Tensor &mlp_out,
    const torch::Tensor &density_grid_indices,  const int &padded_output_width,
    const int &n_density_grid_samples,  torch::Tensor &density_grid_tmp){
    /*
     * @brief splat_grid_samples_nerf_max_nearest_neighbor_api
     * @in-param   'mlp_out' (n_elements, 1)
     * @in-param   'density_grid_indices' (n_elements,)
     * @in-param   'padded_output_width'
     * @in-param   'n_density_grid_samples'
     * @out-param  'density_grid_tmp'
     */
    cudaStream_t stream=0;
    // input
    uint32_t u_n_density_grid_samples = n_density_grid_samples;
    uint32_t u_padded_output_width = padded_output_width;
    uint32_t* density_grid_indices_p = (uint32_t*)density_grid_indices.data_ptr();
    float* mlp_out_p = (float*)mlp_out.data_ptr();
    // output
    float* density_grid_tmp_p = (float*)density_grid_tmp.data_ptr();

    ENerfActivation density_activation = ENerfActivation::Exponential;
    linear_kernel(splat_grid_samples_nerf_max_nearest_neighbor_cuda<float>,0,stream,
        u_n_density_grid_samples, density_grid_indices_p, u_padded_output_width, mlp_out_p,
        density_grid_tmp_p, density_activation);

    cudaDeviceSynchronize();

}
