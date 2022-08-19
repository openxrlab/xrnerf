
#include "raymarch_shared.h"
// extern pcg32 rng;


__global__ void mark_untrained_density_grid_cuda(const uint32_t n_elements,
                                            float *__restrict__ grid_out,
                                            const uint32_t n_training_images,
                                            const Vector2f *__restrict__ focal_lengths,
                                            const Matrix<float, 3, 4> *training_xforms,
                                            Vector2i resolution)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_elements)
        return;
    uint32_t level = i / (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());
    uint32_t pos_idx = i % (NERF_GRIDSIZE() * NERF_GRIDSIZE() * NERF_GRIDSIZE());

    uint32_t x = morton3D_invert(pos_idx >> 0);
    uint32_t y = morton3D_invert(pos_idx >> 1);
    uint32_t z = morton3D_invert(pos_idx >> 2);

    float half_resx = resolution.x() * 0.5f;
    float half_resy = resolution.y() * 0.5f;

    Vector3f pos = ((Vector3f{(float)x + 0.5f, (float)y + 0.5f, (float)z + 0.5f}) / NERF_GRIDSIZE() - Vector3f::Constant(0.5f)) * scalbnf(1.0f, level) + Vector3f::Constant(0.5f);
    float voxel_radius = 0.5f * SQRT3() * scalbnf(1.0f, level) / NERF_GRIDSIZE();
    int count = 0;
    for (uint32_t j = 0; j < n_training_images; ++j)
    {
        Matrix<float, 3, 4> xform = training_xforms[j];
        Vector3f ploc = pos - xform.col(3);
        float x = ploc.dot(xform.col(0));
        float y = ploc.dot(xform.col(1));
        float z = ploc.dot(xform.col(2));
        if (z > 0.f)
        {
            auto focal = focal_lengths[j];
            // TODO - add a box / plane intersection to stop thomas from murdering me
            if (fabsf(x) - voxel_radius < z / focal.x() * half_resx && fabsf(y) - voxel_radius < z / focal.y() * half_resy)
            {
                count++;
                if (count > 0)
                    break;
            }
        }
    }
    if((grid_out[i] < 0) != (count <= 0))
    {
        grid_out[i] = (count > 0) ? 0.f : -1.f;
    }
}

void mark_untrained_density_grid_api( const torch::Tensor &focal_lengths,
    const torch::Tensor &transforms,  const int &n_elements, const int &n_images,
    const int &img_resolution0, const int &img_resolution1,
    torch::Tensor &density_grid){
    /*
     * @brief mark_untrained_density_grid_api
     * @in-param   'focal_lengths' (n_img, 2)
     * @in-param   'transforms'    (n_img, 4, 3)
     * @in-param   'density_grid'  (n_elements,)
     * @in-param   'n_elements'
     * @in-param   'n_images'
     * @in-param   'img_resolution0'
     * @in-param   'img_resolution1'
     * @out-param  'density_grid'
     */

    cudaStream_t stream=0;
    // input
    Eigen::Vector2f* focal_lengths_p = (Eigen::Vector2f*)focal_lengths.data_ptr();
    Eigen::Matrix<float, 3, 4>* transforms_p = (Eigen::Matrix<float, 3, 4>* )transforms.data_ptr();
    Eigen::Vector2i image_resolution{{img_resolution0, img_resolution1}};
    // output
    float* density_grid_p = (float*)density_grid.data_ptr();

    linear_kernel(mark_untrained_density_grid_cuda, 0, stream, n_elements, density_grid_p,
                    n_images, focal_lengths_p, transforms_p, image_resolution);

    cudaDeviceSynchronize();
}
