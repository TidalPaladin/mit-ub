import pytest
import torch

from mit_ub.model.kernels.distance.__main__ import Baseline
from mit_ub.model.kernels.distance.kernel import euclidean_distance


def reference_forward(a, b, w, c=None):
    func = Baseline("baseline")
    return func.forward(a, b, w, c)


@pytest.mark.slow
@pytest.mark.parametrize("matmul", [False, True], ids=["pointwise", "matmul"])
@pytest.mark.parametrize("has_weight", [False, True], ids=["no_weight", "with_weight"])
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
        pytest.param(torch.float32, 1e-2, id="float32"),
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.bfloat16, 1e-1, id="bfloat16"),
    ],
)
def test_euclidean_distance_forward(
    dtype: torch.dtype, tol: float, has_weight: bool, k: int, matmul: bool, with_self: bool, L1: int, L2: int
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    a = torch.randn((L1, k), device="cuda", dtype=dtype)
    b = a if with_self else torch.randn((L2, k), device="cuda", dtype=dtype)
    w = torch.randn((k,), device="cuda", dtype=dtype).abs() if has_weight else None

    torch_output = reference_forward(a, b, w)
    triton_output = euclidean_distance(a, b, w, matmul=matmul)
    assert triton_output.dtype == dtype
    torch.testing.assert_close(triton_output, torch_output, rtol=0, atol=tol)


@pytest.mark.slow
@pytest.mark.parametrize("matmul", [True, False], ids=["matmul", "pointwise"])
def test_euclidean_distance_init_accumulator(matmul: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    k = 3
    L = 16
    a = torch.randn((L, k), device="cuda", dtype=torch.float16)
    b = torch.randn((L, k), device="cuda", dtype=torch.float16)
    c = torch.zeros((L, L), device="cuda", dtype=torch.float16)
    w = torch.randn((k,), device="cuda", dtype=torch.float16).abs()
    c[0, 0] = float("inf")

    torch_output = reference_forward(a, b, w, c)
    triton_output = euclidean_distance(a, b, w, c, matmul=matmul)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    assert triton_output[0, 0] == float("inf")
