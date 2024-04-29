import pytest
import torch

from mit_ub.model.kernels.distance.__main__ import Baseline
from mit_ub.model.kernels.distance.kernel import euclidean_distance


def reference_forward(a, b, c=None):
    func = Baseline("baseline")
    return func.forward(a, b, c)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["matmul", "matmul-nodiag"])
@pytest.mark.parametrize("with_self", [False, True], ids=["cross", "self"])
@pytest.mark.parametrize(
    "L1, L2, k",
    [
        (16, 16, 1),
        (24, 24, 2),
        (32, 32, 3),
        (256, 128, 2),
    ],
    ids=["L1=L2=16,k=1", "L1=L2=24,k=2", "L1=L2=32,k=3", "L1=256,L2=128,k=2"],
)
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(torch.float32, 1e-3, id="float32"),
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.bfloat16, 5e-1, id="bfloat16"),
    ],
)
def test_euclidean_distance_forward(
    dtype: torch.dtype, tol: float, k: int, method: str, with_self: bool, L1: int, L2: int
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    a = torch.randn((L1, k), device="cuda", dtype=dtype)
    b = a if with_self else torch.randn((L2, k), device="cuda", dtype=dtype)

    torch_output = reference_forward(a, b)
    triton_output = euclidean_distance(a, b, method=method)
    assert triton_output.dtype == dtype
    torch.testing.assert_close(triton_output, torch_output, rtol=0, atol=tol)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["matmul", "matmul-nodiag"])
def test_euclidean_distance_init_accumulator(method: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    k = 3
    L = 16
    a = torch.randn((L, k), device="cuda", dtype=torch.float16)
    b = torch.randn((L, k), device="cuda", dtype=torch.float16)
    c = torch.zeros((L, L), device="cuda", dtype=torch.float16)
    c[0, 0] = float("inf")

    torch_output = reference_forward(a, b, c)
    triton_output = euclidean_distance(a, b, c, method=method)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    assert triton_output[0, 0] == float("inf")
