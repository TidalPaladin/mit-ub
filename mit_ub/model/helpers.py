import os
from typing import Tuple, TypeVar


SpatialDims = TypeVar("SpatialDims", bound=Tuple[int, ...])
Dims1D = Tuple[int]
Dims2D = Tuple[int, int]
Dims3D = Tuple[int, int, int]


def compile_is_disabled() -> bool:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE", "1").lower() == "0"


def compile_backend() -> str:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE_BACKEND", "inductor")
