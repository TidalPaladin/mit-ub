from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path

setup(
    name="mit-ub",
    ext_modules=[
        CUDAExtension(
            name="noise_cuda",
            sources=[str(Path("csrc") / "noise.cu")],
            extra_compile_args={"nvcc": ["-O3"]}
        ),
        CUDAExtension(
            name="mixup_cuda",
            sources=[str(Path("csrc") / "mixup.cu")],
            extra_compile_args={"nvcc": ["-O3"]}
        ),
        CUDAExtension(
            name="invert_cuda",
            sources=[str(Path("csrc") / "invert.cu")],
            extra_compile_args={"nvcc": ["-O3"]}
        ),
        CUDAExtension(
            name="posterize_cuda",
            sources=[str(Path("csrc") / "posterize.cu")],
            extra_compile_args={"nvcc": ["-O3"]}
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)