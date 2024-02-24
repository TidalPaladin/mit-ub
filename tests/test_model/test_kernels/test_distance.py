import pytest
import torch
from torch import Tensor

from mit_ub.model.kernels.distance import euclidean_distance


def reference_forward(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape[-1] == b.shape[-1]
    return (a.view(-1, 1, 2) - b.view(1, -1, 2)).pow(2).sum(-1).sqrt()


@pytest.mark.slow
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.float32, 1e-4, id="float32"),
        pytest.param(torch.bfloat16, 1e-1, id="bfloat16"),
    ],
)
def test_euclidean_distance_forward(dtype: torch.dtype, tol: float):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)
    a = torch.randn((512, 2), device="cuda", dtype=dtype)
    b = torch.randn((512, 2), device="cuda", dtype=dtype)
    torch_output = reference_forward(a, b)
    triton_output = euclidean_distance(a, b)
    assert triton_output.dtype == dtype
    assert torch.allclose(triton_output, torch_output, atol=tol, rtol=0)
