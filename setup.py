from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os


csrc_dir = os.path.join(os.path.dirname(__file__), 'csrc')

setup(
    name='fused_ops_backend',
    packages=[],
    ext_modules=[
        CUDAExtension(
            name='fused_ops_backend',
            sources=[
                'csrc/ops.cpp',
                'csrc/kernels/add_relu.cu',
            ],
            include_dirs=[os.path.join(csrc_dir, 'include')],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)