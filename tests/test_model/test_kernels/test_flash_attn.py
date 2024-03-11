from typing import Final

import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.flash_attn import attention


L: int = 32
D: int = 16


EASY_SHAPE_PARAMS: Final = (
    # Head sizes
    pytest.param(1, L, L, D, 1, id=f"b=1,lq={L},lk={L},dhead={D},nhead=1"),
    pytest.param(1, L, L, 4 * D, 1, id=f"b=1,lq={L},lk={L},dhead={4*D},nhead=1"),
    # Batch and head sizes
    pytest.param(4, L, L, D, 1, id=f"b=4,lq={L},lk={L},dhead={D},nhead=1"),
    pytest.param(1, L, L, D, 4, id=f"b=1,lq={L},lk={L},dhead={D},nhead=4"),
    # Sequence lengths
    pytest.param(1, 2 * L, L, D, 1, id=f"b=1,lq={2*L},lk={L},dhead={D},nhead=1"),
    pytest.param(1, L, 2 * L, D, 1, id=f"b=1,lq={L},lk={2*L},dhead={D},nhead=1"),
)

HARD_SHAPE_PARAMS: Final = (pytest.param(4, 2 * L, 4 * L, 4 * D, 4, id=f"b=4,lq={2*L},lk={4*L},dhead={4*D},nhead=4"),)


DATA_TYPE_PARAMS: Final = (
    pytest.param(True, torch.float16, 0.01, id="float16"),
    pytest.param(True, torch.bfloat16, 0.03, id="bfloat16"),
    pytest.param(False, torch.float16, 0.01, id="float16-fast"),
    pytest.param(False, torch.bfloat16, 0.03, id="bfloat16-fast"),
)


@pytest.fixture(autouse=True)
def warn_spills():
    import mit_ub.model.kernels.helpers as helpers

    helpers.WARN_SPILLS = True


@pytest.fixture(autouse=True)
def autotune():
    from mit_ub.model.kernels.flash_attn import _bwd_kernel, _bwd_preprocess_do_o_dot, _fwd_kernel

    for kernel in [_fwd_kernel, _bwd_kernel, _bwd_preprocess_do_o_dot]:
        kernel.rep = 1


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", EASY_SHAPE_PARAMS + HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_forward(b, lq, lk, dhead, nhead, full_precision, dtype, atol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)

    baseline_output = F.scaled_dot_product_attention(q, k, v)
    triton_output = attention(q, k, v, full_precision=full_precision)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("dpos", [2, 3], ids=lambda v: f"dpos={v}")
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_forward_bias(b, lq, lk, dhead, nhead, dpos, dtype, atol, full_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)
    pos_q = torch.randn((b, nhead, lq, dpos), device="cuda", dtype=dtype)
    pos_k = torch.randn((b, nhead, lk, dpos), device="cuda", dtype=dtype)
    slopes = torch.randint(-10, -1, (b, nhead), device="cuda", dtype=dtype).div_(10)

    bias = slopes[..., None, None] * (
        (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(b, nhead, lq, lk)
    )
    baseline_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    triton_output = attention(q, k, v, pos_q, pos_k, slopes, full_precision=full_precision)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", EASY_SHAPE_PARAMS + HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_backward(b, lq, lk, dhead, nhead, dtype, atol, full_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)

    # Baseline
    o = F.scaled_dot_product_attention(q, k, v)
    o.sum().backward()
    grad_q_baseline = q.grad
    grad_k_baseline = k.grad
    grad_v_baseline = v.grad

    # Triton
    q.grad = k.grad = v.grad = None
    o = attention(q, k, v, full_precision=full_precision)
    o.sum().backward()
    grad_q_triton = q.grad
    grad_k_triton = k.grad
    grad_v_triton = v.grad

    # Test
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_backward_bias(b, lq, lk, dhead, nhead, dtype, atol, full_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    pos_q = torch.randn((b, nhead, lq, 2), device="cuda", dtype=dtype)
    pos_k = torch.randn((b, nhead, lk, 2), device="cuda", dtype=dtype)
    slopes = torch.randint(-10, -1, (b, nhead), device="cuda", dtype=dtype).div_(10)
    bias = slopes[..., None, None] * (
        (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(b, nhead, lq, lk)
    )

    # Baseline
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    o.sum().backward()
    grad_q_baseline = q.grad
    grad_k_baseline = k.grad
    grad_v_baseline = v.grad

    # Triton
    q.grad = k.grad = v.grad = None
    o = attention(q, k, v, pos_q, pos_k, slopes, full_precision=full_precision)
    o.sum().backward()
    grad_q_triton = q.grad
    grad_k_triton = k.grad
    grad_v_triton = v.grad

    # Test
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)
