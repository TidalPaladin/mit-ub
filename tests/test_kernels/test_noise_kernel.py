from typing import Tuple

import pytest
import torch
from torch import Tensor

from mit_ub.data.noise import apply_noise_batched_cuda


@pytest.fixture
def input_tensor() -> Tensor:
    return torch.linspace(0, 1, 1000, device="cuda").reshape(10, 10, 10)


@pytest.mark.cuda
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
def test_noise_kernel_shapes(shape: Tuple[int, ...]):
    x = torch.rand(*shape, device="cuda")
    y = apply_noise_batched_cuda(x)
    assert y.shape == x.shape
    assert y.device == x.device
    assert y.dtype == x.dtype


@pytest.mark.cuda
def test_noise_kernel_deterministic(input_tensor: Tensor):
    seed = 42
    output1 = apply_noise_batched_cuda(input_tensor, seed=seed)
    output2 = apply_noise_batched_cuda(input_tensor, seed=seed)
    assert torch.allclose(output1, output2)


@pytest.mark.cuda
def test_noise_kernel_clipping():
    x = torch.ones(1000, device="cuda")
    y = apply_noise_batched_cuda(x, uniform_scale=(0.5, 1.0), clip=True)
    assert torch.all(y <= 1.0)
    assert torch.all(y >= 0.0)

    y = apply_noise_batched_cuda(x, uniform_scale=(0.5, 1.0), clip=False)
    assert not torch.all(y <= 1.0)


@pytest.mark.cuda
def test_noise_kernel_cpu_error():
    x = torch.rand(1000)
    with pytest.raises(ValueError, match="Input tensor must be on CUDA device"):
        apply_noise_batched_cuda(x)
