
#include "raymarch_shared.h"
extern pcg32 rng;


__global__ void generate_grid_samples_nerf_nonuniform_cuda(const uint32_t n_elements,
    default_rng_t rng, const uint32_t step, BoundingBox aabb,
    const float *__restrict__ grid_in, NerfPosition *__restrict__ out,
    uint32_t *__restrict__ indices, uint32_t n_cascades, const float thresh)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_elements)
        return;
    // 1 random number to select the level, 3 to select the position.
    rng.advance(i * 4);
    uint32_t level = (uint32_t)(random_val(rng) * n_cascades) % n_cascades;

    // Select grid cell that has density
    uint32_t idx;
    // uint32_t step=*step_p; # use input param
    for (uint32_t j = 0; j < 10; ++j)
    {
        idx = ((i + step * n_elements) * 56924617 + j * 19349663 + 96925573) % (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());
        idx += level * NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE();
        if (grid_in[idx] > thresh)
        {
            break;
        }
    }

    // Random position within that cellq
    uint32_t pos_idx = idx % (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());
    uint32_t x = morton3D_invert(pos_idx >> 0);
    uint32_t y = morton3D_invert(pos_idx >> 1);
    uint32_t z = morton3D_invert(pos_idx >> 2);

    Eigen::Vector3f pos = ((Eigen::Vector3f{(float)x, (float)y, (float)z} + random_val_3d(rng)) / NERF_GRIDSIZE() - Eigen::Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Eigen::Vector3f::Constant(0.5f);

    out[i] = {warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE())};
    indices[i] = idx;
};

void generate_grid_samples_nerf_nonuniform_api(const torch::Tensor &density_grid,
        const int &density_grid_ema_step, const int &n_elements, const int &max_cascade,
        const float &thresh, const float &aabb0, const float &aabb1,
        torch::Tensor &density_grid_positions_uniform,
        torch::Tensor &density_grid_indices_uniform){
    /*
     * @brief generate_grid_samples_nerf_nonuniform_api
     * @in-param   'density_grid'
     * @in-param   'density_grid_ema_step' # just use, unchanged
     * @in-param   'n_elements'
     * @in-param   'max_cascade'
     * @in-param   'thresh'
     * @in-param   'aabb0'
     * @in-param   'aabb1'
     * @out-param  'density_grid_positions_uniform'
     * @out-param  'density_grid_indices_uniform'
     */

    // std::cout<<density_grid_ema_step<<std::endl;
    // std::cout<<n_elements<<std::endl;
    // std::cout<<max_cascade<<std::endl;
    // std::cout<<thresh<<std::endl;
    // std::cout<<aabb0<<std::endl;
    // std::cout<<aabb1<<std::endl;

    cudaStream_t stream = 0;

    // input value
    float* density_grid_p = (float*)density_grid.data_ptr();
    BoundingBox m_aabb = BoundingBox(Vector3f::Constant(aabb0), Vector3f::Constant(aabb1));

    // output value
    uint32_t* density_grid_indices_p = (uint32_t*)density_grid_indices_uniform.data_ptr();
    NerfPosition* density_grid_positions_uniform_p = (NerfPosition*)density_grid_positions_uniform.data_ptr();

    linear_kernel(generate_grid_samples_nerf_nonuniform_cuda, 0, stream,
        n_elements, rng, (const uint32_t)density_grid_ema_step, m_aabb,
        density_grid_p, density_grid_positions_uniform_p, density_grid_indices_p,
        max_cascade+1, thresh);

    rng.advance();
    cudaDeviceSynchronize();

}
