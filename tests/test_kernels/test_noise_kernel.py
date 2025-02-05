from typing import Any, Dict, Tuple

import pytest
import torch
from torch import Tensor

from mit_ub.data.noise import (
    DEFAULT_NOISE_PROB,
    MULTIPLICATIVE_NOISE_MAX,
    MULTIPLICATIVE_NOISE_MIN,
    SALT_PEPPER_NOISE_MAX,
    SALT_PEPPER_NOISE_MIN,
    SALT_PEPPER_NOISE_PROB,
    UNIFORM_NOISE_MAX,
    UNIFORM_NOISE_MIN,
    apply_noise_batched,
)
from mit_ub.data.noise_cuda import apply_noise_batched_cuda


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def input_tensor(device: torch.device) -> Tensor:
    return torch.linspace(0, 1, 1000, device=device).reshape(10, 10, 10)


@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (10,),
        (3, 32, 32),
        (2, 3, 64, 64),
        (16, 3, 224, 224),
    ],
)
def test_noise_kernel_shapes(shape: Tuple[int, ...], device: torch.device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.rand(*shape, device=device)
    y = apply_noise_batched_cuda(x)
    assert y.shape == x.shape
    assert y.device == x.device
    assert y.dtype == x.dtype


def test_noise_kernel_deterministic(input_tensor: Tensor, device: torch.device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed = 42
    output1 = apply_noise_batched_cuda(input_tensor, seed=seed)
    output2 = apply_noise_batched_cuda(input_tensor, seed=seed)
    assert torch.allclose(output1, output2)


def test_noise_kernel_clipping(device: torch.device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.ones(1000, device=device)
    y = apply_noise_batched_cuda(x, uniform_scale=(0.5, 1.0), clip=True)
    assert torch.all(y <= 1.0)
    assert torch.all(y >= 0.0)

    y = apply_noise_batched_cuda(x, uniform_scale=(0.5, 1.0), clip=False)
    assert not torch.all(y <= 1.0)


def test_noise_kernel_cpu_error(device: torch.device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    x = torch.rand(1000)
    with pytest.raises(ValueError, match="Input tensor must be on CUDA device"):
        apply_noise_batched_cuda(x)
