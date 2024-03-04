import pytest
import torch

from mit_ub.model.kernels.distance import _reference_forward, euclidean_distance


@pytest.mark.slow
@pytest.mark.parametrize("matmul", [False, True], ids=["pointwise", "matmul"])
@pytest.mark.parametrize("has_weight", [False, True], ids=["no_weight", "with_weight"])
@pytest.mark.parametrize("with_self", [True, False], ids=["self", "cross"])
@pytest.mark.parametrize(
    "L, k",
    [
        (16, 1),
        (24, 2),
        (32, 3),
    ],
    ids=["L=16,k=1", "L=24,k=2", "L=32,k=3"],
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
    dtype: torch.dtype, tol: float, has_weight: bool, k: int, matmul: bool, with_self: bool, L: int
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    a = torch.randn((L, k), device="cuda", dtype=dtype)
    b = a if with_self else torch.randn((L, k), device="cuda", dtype=dtype)
    w = torch.randn((k,), device="cuda", dtype=dtype).abs() if has_weight else None

    torch_output = _reference_forward(a, b, w)
    triton_output = euclidean_distance(a, b, w, matmul=matmul)
    assert triton_output.dtype == dtype
    torch.testing.assert_close(triton_output, torch_output, rtol=0, atol=tol)


# @pytest.mark.slow
# @pytest.mark.parametrize("matmul", [True, False], ids=["matmul", "pointwise"])
# def test_euclidean_distance_inf_weight(matmul: bool):
#    if not torch.cuda.is_available():
#        pytest.skip("CUDA is not available")
#    torch.manual_seed(0)
#
#    k = 3
#    L = 8
#    a = torch.randn((L, k), device="cuda", dtype=torch.float16)
#    w = torch.randn((k,), device="cuda", dtype=torch.float16).abs()
#    #w[-1] = float("inf")
#    w[-1] = 1e4
#    a[..., -1] = torch.arange(L, device="cuda", dtype=torch.float16) * 1000
#
#    torch_output = _reference_forward(a, a, w)
#    triton_output = euclidean_distance(a, a, w, matmul=matmul)
#    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
#    assert triton_output[0, 0] == float("inf")


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

    torch_output = _reference_forward(a, b, w, c)
    triton_output = euclidean_distance(a, b, w, c, matmul=matmul)
    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
    assert triton_output[0, 0] == float("inf")


# @pytest.mark.slow
# def test_euclidean_distance_backward():
#    if not torch.cuda.is_available():
#        pytest.skip("CUDA is not available")
#    torch.manual_seed(0)
#    a = torch.randn((512, 2), device="cuda", dtype=torch.float16, requires_grad=True)
#    b = torch.randn((512, 2), device="cuda", dtype=torch.float16, requires_grad=True)
#
#    # Forward passes
#    torch_output = reference_forward(a, b)
#    triton_output = euclidean_distance(a, b)
#    assert torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0)
#
#    # Torch backward pass
#    do = 0.1 * torch.randn_like(torch_output, requires_grad=False)
#    torch_output.backward(do, retain_graph=True)
#    da_torch, db_torch = [_.grad.clone() for _ in [a, b]]
#
#    # Triton backward pass
#    triton_output.backward(do, retain_graph=True)
#    da_triton, db_triton = [_.grad.clone() for _ in [a, b]]
#    (
#        a.grad,
#        b.grad,
#    ) = (
#        None,
#        None,
#    )
#
#    assert torch.allclose(da_torch, da_triton, atol=1e-2, rtol=0)
#    assert torch.allclose(db_torch, db_triton, atol=1e-2, rtol=0)
