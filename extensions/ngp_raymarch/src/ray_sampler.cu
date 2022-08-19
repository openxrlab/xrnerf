#include "raymarch_shared.h"
#include "ray_sampler_header.h"


__global__ void rays_sampler_cuda(
    const uint32_t n_rays,
    BoundingBox aabb,
    const uint32_t max_samples,
    const Vector3f *__restrict__ rays_o,
    const Vector3f *__restrict__ rays_d,
    const uint8_t *__restrict__ density_grid,
    const float cone_angle_constant,
    const TrainingImageMetadata *__restrict__ metadata,
    const uint32_t *__restrict__ imgs_index,
    uint32_t *__restrict__ ray_counter,
    uint32_t *__restrict__ numsteps_counter,
    uint32_t *__restrict__ ray_indices_out,
    uint32_t *__restrict__ numsteps_out,
    PitchedPtr<NerfCoordinate> coords_out,
    const Matrix<float, 3, 4> *training_xforms,
    float near_distance,
    default_rng_t rng

)
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    // i (0,n_rays)
    if (i >= n_rays)
        return;
    uint32_t img = imgs_index[i];
    rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
    // float max_level = 1.0f; // Multiply by 2 to ensure 50% of training is at max level
    // float max_level = max_level_rand_training ? (0.6 * 2.0f) : 1.0f;
    const Matrix<float, 3, 4> xform = training_xforms[img];
    const Vector2f focal_length = metadata[img].focal_length;
    // const Vector2f principal_point = metadata[img].principal_point;
    const Vector3f light_dir_warped = warp_direction(metadata[img].light_dir);
    // const CameraDistortion camera_distortion = metadata[img].camera_distortion;
    Vector3f ray_o = rays_o[i];
    Vector3f ray_d = rays_d[i];

    Vector2f tminmax = aabb.ray_intersect(ray_o, ray_d);
    // float cone_angle = calc_cone_angle(ray_d.dot(xform.col(2)), focal_length, cone_angle_constant);
    float cone_angle = cone_angle_constant;
    // // The near distance prevents learning of camera-specific fudge right in front of the camera
    tminmax.x() = fmaxf(tminmax.x(), near_distance);

    float startt = tminmax.x();
    // // TODO:change
    startt += calc_dt(startt, cone_angle) * random_val(rng);

    Vector3f idir = ray_d.cwiseInverse();

    // // first pass to compute an accurate number of steps
    uint32_t j = 0;
    float t = startt;
    Vector3f pos;
    while (aabb.contains(pos = ray_o + t * ray_d) && j < NERF_STEPS())
    {
        float dt = calc_dt(t, cone_angle);
        uint32_t mip = mip_from_dt(dt, pos);
        if (density_grid_occupied_at(pos, density_grid, mip))
        {
            ++j;
            t += dt;
        }
        else
        {
            uint32_t res = NERF_GRIDSIZE() >> mip;
            t = advance_to_next_voxel(t, cone_angle, pos, ray_d, idir, res);
        }
    }

    uint32_t numsteps = j;
    uint32_t base = atomicAdd(numsteps_counter, numsteps); // first entry in the array is a counter
    if (base + numsteps > max_samples)
    {
        // printf("over max sample!!!!!!!!!!!!!!\n");
        numsteps_out[2 * i + 0] = 0;
        numsteps_out[2 * i + 1] = base;
        return;
    }

    coords_out += base;

    uint32_t ray_idx = atomicAdd(ray_counter, 1);
    ray_indices_out[i] = ray_idx;
    // TODO:
    numsteps_out[2 * i + 0] = numsteps;
    numsteps_out[2 * i + 1] = base;
    if (j == 0)
    {
        ray_indices_out[i] = -1;
        return;
    }
    Vector3f warped_dir = warp_direction(ray_d);
    t = startt;
    j = 0;
    while (aabb.contains(pos = ray_o + t * ray_d) && j < numsteps)
    {
        float dt = calc_dt(t, cone_angle);
        uint32_t mip = mip_from_dt(dt, pos);
        if (density_grid_occupied_at(pos, density_grid, mip))
        {

            coords_out(j)->set_with_optional_light_dir(warp_position(pos, aabb), warped_dir, warp_dt(dt), light_dir_warped, coords_out.stride_in_bytes);
            ++j;
            t += dt;
        }
        else
        {
            uint32_t res = NERF_GRIDSIZE() >> mip;
            t = advance_to_next_voxel(t, cone_angle, pos, ray_d, idir, res);
        }
    }
}


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
    torch::Tensor &ray_numstep_counter
    ){
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
    // float* rays_o_p = (float*)rays_o.data_ptr();
    // float* rays_d_p = (float*)rays_d.data_ptr();
    // float* metadata_p = (float*)metadata.data_ptr();
    // float* xforms_p = (float*)xforms.data_ptr();

    Vector3f* rays_o_p = (Vector3f*)rays_o.data_ptr();
    Vector3f* rays_d_p = (Vector3f*)rays_d.data_ptr();
    Eigen::Matrix<float, 3, 4>* xforms_p = (Eigen::Matrix<float, 3, 4>* )xforms.data_ptr();
    TrainingImageMetadata* metadata_p = (TrainingImageMetadata*)metadata.data_ptr();

    uint8_t* density_grid_bitfield_p = (uint8_t*)density_grid_bitfield.data_ptr();
    uint32_t* imgs_id_p = (uint32_t*)imgs_id.data_ptr();

    // output
    // float* coords_out_p = (float*)coords_out.data_ptr();
    uint32_t* rays_index_p = (uint32_t*)rays_index.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    uint32_t* ray_numstep_counter_p = (uint32_t*)ray_numstep_counter.data_ptr();
    NerfCoordinate* coords_out_p = (NerfCoordinate*)coords_out.data_ptr();

    // remember set to zero
    // int coords_n = (coords_out.sizes()[0]) * (coords_out.sizes()[1]);
    // cudaMemsetAsync(coords_out_p, 0, sizeof(float)*coords_n, stream);

    const unsigned int num_elements = coords_out.sizes()[0];
    const uint32_t n_rays = rays_o.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0),
        Eigen::Vector3f::Constant(aabb1));

    //
    linear_kernel(rays_sampler_cuda, 0, stream,
        n_rays, m_aabb, num_elements, (Vector3f*)rays_o_p, (Vector3f*)rays_d_p,
        (uint8_t*)density_grid_bitfield_p, cone_angle_constant,
        metadata_p, (uint32_t*)imgs_id_p, (uint32_t*)ray_numstep_counter_p,
        ((uint32_t*)ray_numstep_counter_p)+1,
        (uint32_t*)rays_index_p,
        (uint32_t*)rays_numsteps_p,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_out_p, 1, 0, 0),
        xforms_p, near_distance, rng);


    // //
    // linear_kernel(rays_sampler_cuda, 0, stream,
    //     n_rays, m_aabb, num_elements, (Vector3f*)rays_o_p, (Vector3f*)rays_d_p,
    //     (uint8_t*)density_grid_bitfield_p, cone_angle_constant,
    //     (TrainingImageMetadata *)metadata_p, (uint32_t*)imgs_id_p,
    //     (uint32_t*)ray_numstep_counter_p, ((uint32_t*)ray_numstep_counter_p)+1,
    //     (uint32_t*)rays_index_p,(uint32_t*)rays_numsteps_p,
    //     PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_out_p, 1, 0, 0),
    //     (Eigen::Matrix<float, 3, 4>*) xforms_p,
    //     near_distance,rng);

    rng.advance();
    cudaDeviceSynchronize();
}
