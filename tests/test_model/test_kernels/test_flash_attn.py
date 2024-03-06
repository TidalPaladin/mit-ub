import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.flash_attn import attention
from mit_ub.model.kernels.distance import _reference_forward


@pytest.mark.parametrize("dhead", [16, 24, 64], ids=lambda v: f"dhead={v}")
@pytest.mark.parametrize("lq", [32, 64, 256], ids=lambda v: f"lq={v}")
@pytest.mark.parametrize("lk", [32, 64, 256], ids=lambda v: f"lk={v}")
@pytest.mark.parametrize("nhead", [1, 2], ids=lambda v: f"nhead={v}")
@pytest.mark.parametrize("b", [1, 2], ids=lambda v: f"b={v}")
@pytest.mark.parametrize("dtype,atol", [(torch.bfloat16, 0.01), (torch.float16, 0.01)], ids=lambda v: str(v))
def test_flash_attn_forward(b, lq, lk, dhead, nhead, dtype, atol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, lq, nhead, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype)

    baseline_output = F.scaled_dot_product_attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)).movedim(2, 1)
    triton_output = attention(q, k, v, softmax_scale=dhead**-0.5)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.parametrize("dtype,atol", [(torch.bfloat16, 0.02), (torch.float16, 0.01)], ids=lambda v: str(v))
@pytest.mark.parametrize("slope", [-1, -2], ids=lambda v: f"slope={v}")
def test_flash_attn_forward_bias(dtype, slope, atol):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)
    b = 2
    lq = 128
    lk = 128
    dhead = 64
    dpos=2
    nhead = 2

    q = torch.randn((b, lq, nhead, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype)
    pos_q = torch.randn((b, lq, nhead, dpos), device="cuda", dtype=dtype)
    pos_k = torch.randn((b, lq, nhead, dpos), device="cuda", dtype=dtype)
    slopes = torch.full((b, nhead), slope, device="cuda", dtype=dtype)

    bias = slopes[..., None, None] * (
        (pos_q[:, :, None, ...] - pos_k[:, None, ...])
        .pow(2).sum(-1).sqrt_()
        .movedim(-1, 1).view(b, nhead, lq, lk)
    )
    baseline_output = F.scaled_dot_product_attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2), attn_mask=bias).movedim(2, 1)

    triton_output = attention(q, k, v, pos_q, pos_k, slopes, softmax_scale=dhead**-0.5)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.parametrize("b,lq,lk,dhead,nhead,bias", [
    (1, 32, 32, 16, 1, False),
    (1, 32, 32, 48, 1, False),
    (4, 32, 32, 16, 1, False),
    (4, 32, 32, 16, 4, False),
    (1, 128, 32, 16, 1, False),
    (1, 32, 256, 16, 1, False),
    (4, 128, 256, 48, 4, False),
])
@pytest.mark.parametrize("dtype,atol", [(torch.bfloat16, 0.02), (torch.float16, 0.01)], ids=lambda v: str(v))
def test_flash_attn_backward(b, lq, lk, dhead, nhead, dtype, atol, bias):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, lq, nhead, dhead), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype, requires_grad=True)

    if bias:
        pos_q = torch.randn((b, lq, nhead, 2), device="cuda", dtype=dtype)
        pos_k = torch.randn((b, lk, nhead, 2), device="cuda", dtype=dtype)
        mask = -1 * (
            (pos_q[:, :, None, ...] - pos_k[:, None, ...])
            .pow(2).sum(-1).sqrt_()
            .movedim(-1, 1).view(b, nhead, lq, lk)
        )
    else:
        pos_q = pos_k = mask = None

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
