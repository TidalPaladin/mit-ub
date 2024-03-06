from typing import Final

import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.flash_attn import attention


EASY_SHAPE_PARAMS: Final = (
    # Head sizes
    pytest.param(1, 32, 32, 16, 1, id="b=1,lq=32,lk=32,dhead=16,nhead=1"),
    pytest.param(1, 32, 32, 64, 1, id="b=1,lq=32,lk=32,dhead=64,nhead=1"),
    # Batch and head sizes
    pytest.param(4, 32, 32, 16, 1, id="b=4,lq=32,lk=32,dhead=16,nhead=1"),
    pytest.param(1, 32, 32, 16, 4, id="b=1,lq=32,lk=32,dhead=16,nhead=4"),
    # Sequence lengths
    pytest.param(1, 128, 32, 16, 1, id="b=1,lq=128,lk=32,dhead=16,nhead=1"),
    pytest.param(1, 32, 256, 16, 1, id="b=1,lq=32,lk=256,dhead=16,nhead=1"),
)

HARD_SHAPE_PARAMS: Final = (
    pytest.param(4, 512, 512, 64, 4, id="b=4,lq=512,lk=512,dhead=64,nhead=4"),
)


DATA_TYPE_PARAMS: Final = (
    pytest.param(torch.bfloat16, 0.02, id="bfloat16"),
    pytest.param(torch.float16, 0.01, id="float16"),
)


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", EASY_SHAPE_PARAMS + HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_forward(b, lq, lk, dhead, nhead, dtype, atol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)

    baseline_output = F.scaled_dot_product_attention(q, k, v)
    triton_output = attention(q, k, v, softmax_scale=dhead**-0.5)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("dpos", [2, 3], ids=lambda v: f"dpos={v}")
@pytest.mark.parametrize("slope", [-1, -2], ids=lambda v: f"slope={v}")
@pytest.mark.parametrize("dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_forward_bias(b, lq, lk, dhead, nhead, dpos, dtype, atol, slope):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype)
    pos_q = torch.randn((b, nhead, lq, dpos), device="cuda", dtype=dtype)
    pos_k = torch.randn((b, nhead, lk, dpos), device="cuda", dtype=dtype)
    slopes = torch.full((b, nhead), slope, device="cuda", dtype=dtype)

    bias = slopes[..., None, None] * (
        (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(b, nhead, lq, lk)
    )
    baseline_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    triton_output = attention(q, k, v, pos_q, pos_k, slopes, softmax_scale=dhead**-0.5)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.parametrize("b, lq, lk, dhead, nhead", EASY_SHAPE_PARAMS + HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_backward(b, lq, lk, dhead, nhead, dtype, atol):
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
    o = attention(q, k, v)
    o.sum().backward()
    grad_q_triton = q.grad
    grad_k_triton = k.grad
    grad_v_triton = v.grad

    # Test
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)


@pytest.mark.skip
@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_backward_bias(b, lq, lk, dhead, nhead, dtype, atol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, lq, nhead, dhead), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype, requires_grad=True)
    pos_q = torch.randn((b, lq, nhead, 2), device="cuda", dtype=dtype)
    pos_k = torch.randn((b, lk, nhead, 2), device="cuda", dtype=dtype)
    mask = -1 * (
        (pos_q[:, :, None, ...] - pos_k[:, None, ...]).pow(2).sum(-1).sqrt_().movedim(-1, 1).view(b, nhead, lq, lk)
    )

    # Baseline
    o = F.scaled_dot_product_attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2), attn_mask=mask).movedim(2, 1)
    o.sum().backward()
    grad_q_baseline = q.grad
    grad_k_baseline = k.grad
    grad_v_baseline = v.grad

    # Triton
    q.grad = k.grad = v.grad = None
    o = attention(q, k, v, pos_q, pos_k)
    o.sum().backward()
    grad_q_triton = q.grad
    grad_k_triton = k.grad
    grad_v_triton = v.grad

    # Test
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)
