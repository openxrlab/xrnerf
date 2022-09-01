from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

nvcc_flags = [
    '--extended-lambda',
    '--expt-relaxed-constexpr',
]

setup(
    name='raymarch',  # package name, import this to use python API
    include_dirs=[
        'include', 'include/op_include/eigen', 'include/op_include/pcg32'
    ],  # h和c同目录时不需要
    ext_modules=[
        CUDAExtension(
            name='raymarch_cuda',  # extension name, import this to use CUDA API
            sources=[
                'src/pybind_api.cu',
                'src/generate_grid_samples_nerf_nonuniform.cu',
                'src/mark_untrained_density_grid.cu',
                'src/splat_grid_samples_nerf_max_nearest_neighbor.cu',
                'src/ema_grid_samples_nerf.cu',
                'src/update_bitfield.cu',
                'src/ray_sampler.cu',
                'src/compacted_coord.cu',
                'src/calc_rgb.cu',
            ],
            extra_compile_args={
                'nvcc': nvcc_flags,
            },
            # extra_link_args=nvcc_link_flags,
        )
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)})
