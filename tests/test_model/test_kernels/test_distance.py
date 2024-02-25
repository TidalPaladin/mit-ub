import pytest
import torch

from mit_ub.model.kernels.distance import _reference_forward, euclidean_distance


@pytest.mark.slow
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.float32, 1e-4, id="float32"),
        pytest.param(torch.bfloat16, 1e-1, id="bfloat16"),
    ],
)
def test_euclidean_distance_forward(dtype: torch.dtype, tol: float, has_weight: bool):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    L, K = 256, 2
    a = torch.randn((L, K), device="cuda", dtype=dtype)
    b = torch.randn((L, K), device="cuda", dtype=dtype)
    w = torch.randn((K,), device="cuda", dtype=dtype).abs() if has_weight else None

    torch_output = _reference_forward(a, b, w)
    triton_output = euclidean_distance(a, b, w)
    assert triton_output.dtype == dtype
    assert torch.allclose(triton_output, torch_output, atol=tol, rtol=0)


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
