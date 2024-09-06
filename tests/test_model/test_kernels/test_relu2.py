import pytest
import torch
import torch.nn as nn

from mit_ub.model.kernels.relu2.__main__ import Baseline
from mit_ub.model.kernels.relu2.kernel import relu2
from mit_ub.model.kernels.relu2.module import ReLU2


def reference_forward(x):
    func = Baseline("baseline")
    return func.forward(x)


@pytest.mark.cuda
@pytest.mark.parametrize("L", [1, 8, 16, 32])
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(torch.float32, 1e-3, id="float32"),
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.bfloat16, 5e-1, id="bfloat16"),
    ],
)
def test_relu2_forward(dtype: torch.dtype, tol: float, L: int):
    torch.manual_seed(0)
    x = torch.randn((L,), device="cuda", dtype=dtype)

    torch_output = reference_forward(x)
    triton_output = relu2(x)
    assert triton_output.dtype == dtype
    torch.testing.assert_close(triton_output, torch_output, rtol=0, atol=tol)


@pytest.mark.cuda
@pytest.mark.parametrize("L", [1, 8, 16, 32])
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(torch.float32, 1e-3, id="float32"),
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.bfloat16, 5e-1, id="bfloat16"),
    ],
)
def test_relu2_backward(dtype: torch.dtype, tol: float, L: int):
    torch.manual_seed(0)
    x = torch.randn((L,), device="cuda", dtype=dtype, requires_grad=True)

    torch_output = reference_forward(x)
    torch_output.sum().backward()
    baseline_grad = x.grad
    x.grad = None

    triton_output = relu2(x)
    triton_output.sum().backward()
    triton_grad = x.grad

    assert triton_output.dtype == dtype
    torch.testing.assert_close(triton_grad, baseline_grad, rtol=0, atol=tol)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(torch.float32, 1e-3, id="float32"),
        pytest.param(torch.float16, 1e-2, id="float16"),
        pytest.param(torch.bfloat16, 5e-1, id="bfloat16"),
    ],
)
def test_relu2_module(dtype: torch.dtype, tol: float):
    L = 1024
    x = torch.randn((L,), device="cuda", dtype=dtype, requires_grad=True)
    baseline = nn.ReLU()
    triton = ReLU2()

    baseline_output = baseline(x) * baseline(x)
    baseline_output.sum().backward()
    baseline_grad = x.grad
    x.grad = None

    triton_output = triton(x)
    triton_output.sum().backward()
    triton_grad = x.grad

    torch.testing.assert_close(triton_output, baseline_output, rtol=0, atol=tol)
    torch.testing.assert_close(triton_grad, baseline_grad, rtol=0, atol=tol)
