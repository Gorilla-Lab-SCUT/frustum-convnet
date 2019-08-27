from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='query_depth_point_cuda',
    ext_modules=[
        CUDAExtension('query_depth_point_cuda', [
            'query_depth_point_cuda.cpp',
            'query_depth_point_cuda_kernel.cu'
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
