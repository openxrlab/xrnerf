#include "raymarch_shared.h"
#include "ray_sampler_header.h"


template <typename TYPE>
__global__ void compute_rgbs(
    const uint32_t n_rays,                      //batch total rays number
    BoundingBox aabb,                           //boundingbox range
    int padded_output_width,                    //network output width
    const TYPE *network_output,                 //network output
    ENerfActivation rgb_activation,             //activation of rgb in output
    ENerfActivation density_activation,         //activation of density in output
    PitchedPtr<NerfCoordinate> coords_in,       //network input,(xyz,dt,dir)
    uint32_t *__restrict__ numsteps_in,         //rays offset and base counter before compact
    Array3f *rgb_output,                        //rays rgb output
    uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
    const Array3f *bg_color_ptr                //background color
    )
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays)
    {
        return;
    }
    Array3f background_color=bg_color_ptr[i];
    uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
    uint32_t base = numsteps_compacted_in[i * 2 + 1];
    if (numsteps == 0)
    {
        rgb_output[i] = background_color;
        return;
    }
    coords_in += base;
    network_output += base * padded_output_width;

    float T = 1.f;

    float EPSILON = 1e-4f;

    Array3f rgb_ray = Array3f::Zero();

    uint32_t compacted_numsteps = 0;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
        const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
        const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
        const float dt = unwarp_dt(coords_in.ptr->dt);

        float density = network_to_density(float(local_network_output[3]), density_activation);

        const float alpha = 1.f - __expf(-density * dt);
        const float weight = alpha * T;
        rgb_ray += weight * rgb;

        T *= (1.f - alpha);
        network_output += padded_output_width;
        coords_in += 1;
    }

    if (compacted_numsteps == numsteps_in[i * 2 + 0])
    {
        rgb_ray += T * background_color;
    }

    rgb_output[i] = rgb_ray;
}


template <typename TYPE>
__global__ void compute_rgbs_grad(
    const uint32_t n_rays,                      //batch total rays number
    BoundingBox aabb,                           //boundingbox range
    int padded_output_width,                    //network output width
    TYPE *__restrict__ dloss_doutput,           //dloss_dnetworkoutput,shape same as network output
    const TYPE *network_output,                 //network output
    uint32_t *__restrict__ numsteps_compacted_in,//rays offset and base counter after compact
    PitchedPtr<NerfCoordinate> coords_in,       //network input,(xyz,dt,dir)
    ENerfActivation rgb_activation,             //activation of rgb in output
    ENerfActivation density_activation,         //activation of density in output
    Array3f *__restrict__ loss_grad,            //dloss_dRGBoutput
    Array3f *__restrict__ rgb_ray,              //RGB from forward calculation
    float *__restrict__ density_grid_mean      //density_grid mean value,
    )
{

    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n_rays)
    {
        return;
    }
    float loss_scale = 128;
    loss_scale /= n_rays;
    uint32_t numsteps = numsteps_compacted_in[i * 2 + 0];
    uint32_t base = numsteps_compacted_in[i * 2 + 1];

    coords_in += base;
    network_output += base * padded_output_width;
    dloss_doutput += base * padded_output_width;
    loss_grad += i;
    rgb_ray += i;

    const float output_l2_reg = rgb_activation == ENerfActivation::Exponential ? 1e-4f : 0.0f;
    const float output_l1_reg_density = *density_grid_mean < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

    float T = 1.f;
    uint32_t compacted_numsteps = 0;
    Array3f rgb_ray2 = Array3f::Zero();
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {

        const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
        const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
        float dt = unwarp_dt(coords_in.ptr->dt);
        float density = network_to_density(float(local_network_output[3]), density_activation);
        const float alpha = 1.f - __expf(-density * dt);
        const float weight = alpha * T;
        rgb_ray2 += weight * rgb;
        T *= (1.f - alpha);

        const Array3f suffix = *rgb_ray - rgb_ray2;
        const Array3f dloss_by_drgb = weight * (*loss_grad);

        vector_t<TYPE, 4> local_dL_doutput;

        // chain rule to go from dloss/drgb to dloss/dmlp_output
        local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x() * network_to_rgb_derivative(local_network_output[0], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[0])); // Penalize way too large color values
        local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y() * network_to_rgb_derivative(local_network_output[1], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[1]));
        local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z() * network_to_rgb_derivative(local_network_output[2], rgb_activation) + fmaxf(0.0f, output_l2_reg * (float)local_network_output[2]));

        float density_derivative = network_to_density_derivative(float(local_network_output[3]), density_activation);
        float dloss_by_dmlp = density_derivative * (dt * (*loss_grad).matrix().dot((T * rgb - suffix).matrix()));
        local_dL_doutput[3] = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0 ? -output_l1_reg_density : 0.0f);
        *(vector_t<TYPE, 4> *)dloss_doutput = local_dL_doutput;

        network_output += padded_output_width;
        dloss_doutput += padded_output_width;
        coords_in += 1;
    }
}


template <typename TYPE>
__global__ void compute_rgbs_inference(
    const uint32_t n_rays,                      //batch total rays number
    BoundingBox aabb,                           //boundingbox range
    int padded_output_width,                    //network output width
    Array3f background_color,                   //background color
    const TYPE *network_output,                 //network output
    ENerfActivation rgb_activation,             //activation of rgb in output
    ENerfActivation density_activation,         //activation of density in output
    PitchedPtr<NerfCoordinate> coords_in,       //network input,(xyz,dt,dir)
    uint32_t *__restrict__ numsteps_in,         //rays offset and base counter
    Array3f *__restrict__ rgb_output,                       //rays rgb output
    float* __restrict__ alpha_output
    )
{
    const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i >= n_rays)
    {
        return;
    }

    uint32_t numsteps = numsteps_in[i * 2 + 0];
    uint32_t base = numsteps_in[i * 2 + 1];
    if (numsteps == 0)
    {
        rgb_output[i] = background_color;
        alpha_output[i] = 0;
        return;
    }
    coords_in += base;
    network_output += base * padded_output_width;

    float T = 1.f;

    float EPSILON = 1e-4f;

    Array3f rgb_ray = Array3f::Zero();

    uint32_t compacted_numsteps = 0;
    for (; compacted_numsteps < numsteps; ++compacted_numsteps)
    {
        const vector_t<TYPE, 4> local_network_output = *(vector_t<TYPE, 4> *)network_output;
        const Array3f rgb = network_to_rgb(local_network_output, rgb_activation);
        const Vector3f pos = unwarp_position(coords_in.ptr->pos.p, aabb);
        const float dt = unwarp_dt(coords_in.ptr->dt);

        float density = network_to_density(float(local_network_output[3]), density_activation);

        const float alpha = 1.f - __expf(-density * dt);
        const float weight = alpha * T;
        rgb_ray += weight * rgb;

        T *= (1.f - alpha);
        network_output += padded_output_width;
        coords_in += 1;
    }
    if (compacted_numsteps == numsteps)
    {
        rgb_ray += T * background_color;
    }
    rgb_output[i] = rgb_ray;
    alpha_output[i] = 1-T;
}

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
    if (CHECK_TENSOR_HALF(network_output))
    {
        #define data_t at::Half
    }else{
        #define data_t float
    }
    data_t* network_output_p = (data_t*)network_output.data_ptr();
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

    linear_kernel(compute_rgbs<data_t>, 0, stream, n_rays, m_aabb,
        padded_output_width,(data_t*)network_output_p,
        rgb_activation, density_activation,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),
        (uint32_t*)rays_numsteps_p,(Array3f*)rgb_output_p,
        (uint32_t*)rays_numsteps_compacted_p,
        (Array3f*)training_background_color_p);

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
    if (CHECK_TENSOR_HALF(network_output))
    {
        #define data_t at::Half
    }else{
        #define data_t float
    }
    data_t* network_output_p = (data_t*)network_output.data_ptr();
    float* coords_in_p = (float*)coords_in.data_ptr();
    float* grad_x_p = (float*)grad_x.data_ptr();
    float* rgb_output_p = (float*)rgb_output.data_ptr();
    float* density_grid_mean_p = (float*)density_grid_mean.data_ptr();
    uint32_t* rays_numsteps_compacted_p = (uint32_t*)rays_numsteps_compacted.data_ptr();

    // output
    data_t* dloss_doutput_p = (data_t*)dloss_doutput.data_ptr();

    // cudaMemsetAsync(out0_p, 0, out0->size);
    const unsigned int num_elements = network_output.sizes()[0];
    const uint32_t n_rays = rays_numsteps_compacted.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));
    uint32_t padded_output_width = network_output.sizes()[1];
    ENerfActivation rgb_activation = ENerfActivation(rgb_activation_i);
    ENerfActivation density_activation = ENerfActivation(density_activation_i);

    linear_kernel(compute_rgbs_grad<data_t>, 0, stream,
        n_rays, m_aabb, padded_output_width, (data_t*)dloss_doutput_p,
        (data_t*)network_output_p, (uint32_t*)rays_numsteps_compacted_p,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),
        rgb_activation, density_activation,(Array3f*)grad_x_p,(Array3f*)rgb_output_p,
        (float*)density_grid_mean_p);

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
    torch::Tensor &rgb_output,
    torch::Tensor &alpha_output
    ){
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
     * @out-param  'alpha_output'
     */
    cudaStream_t stream = 0;
    // input
    if (CHECK_TENSOR_HALF(network_output))
    {
        #define data_t at::Half
    }else{
        #define data_t float
    }
    data_t* network_output_p = (data_t*)network_output.data_ptr();
    float* coords_in_p = (float*)coords_in.data_ptr();
    uint32_t* rays_numsteps_p = (uint32_t*)rays_numsteps.data_ptr();
    float* bg_color_p = (float*)bg_color_cpu.data_ptr();

    // output
    float* rgb_output_p = (float*)rgb_output.data_ptr();
    float* alpha_output_p = (float*)alpha_output.data_ptr();

    const uint32_t n_rays = rays_numsteps.sizes()[0];
    const unsigned int num_elements = network_output.sizes()[0];
    BoundingBox m_aabb = BoundingBox(Eigen::Vector3f::Constant(aabb0), Eigen::Vector3f::Constant(aabb1));
    uint32_t padded_output_width = network_output.sizes()[1];
    ENerfActivation rgb_activation = ENerfActivation(rgb_activation_i);
    ENerfActivation density_activation = ENerfActivation(density_activation_i);
    Array3f bg_color = Array3f(bg_color_p[0], bg_color_p[1], bg_color_p[2]);

    linear_kernel(compute_rgbs_inference<data_t>, 0, stream,
        n_rays, m_aabb,padded_output_width,bg_color,
        (data_t*)network_output_p,rgb_activation,density_activation,
        PitchedPtr<NerfCoordinate>((NerfCoordinate*)coords_in_p, 1, 0, 0),
        (uint32_t*)rays_numsteps_p,
        (Array3f*)rgb_output_p,
        alpha_output_p);

    cudaDeviceSynchronize();
}
