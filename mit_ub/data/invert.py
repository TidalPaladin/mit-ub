from pathlib import Path

import torch
from torch import Tensor
from torch.utils.cpp_extension import load


if torch.cuda.is_available():
    _invert_cuda = load(
        name="invert_cuda",
        sources=[str(Path(__file__).parents[2] / "csrc" / "invert.cu")],
        extra_cuda_cflags=["-O3"],
    )
else:
    _invert_cuda = None


def invert(input: Tensor, invert_prob: float, seed: int | None = None) -> Tensor:
    if _invert_cuda is None:
        raise RuntimeError("Invert is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _invert_cuda.invert(input, invert_prob, seed)


def invert_(input: Tensor, invert_prob: float, seed: int | None = None) -> Tensor:
    if _invert_cuda is None:
        raise RuntimeError("Invert is not available on this system")

    if seed is None:
        seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())
    return _invert_cuda.invert_(input, invert_prob, seed)
