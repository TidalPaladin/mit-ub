from typing import Final

import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.attention.kernel import attention
from mit_ub.model.kernels.attention.module import MultiheadAttention


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

HARD_SHAPE_PARAMS: Final = (
    pytest.param(1, 10, 10, D, 1, id=f"b=1,lq=10,lk=10,dhead={D},nhead=1"),
    pytest.param(1, 18, 18, D, 1, id=f"b=1,lq=18,lk=18,dhead={D},nhead=1"),
    pytest.param(1, 10, 18, D, 1, id=f"b=1,lq=10,lk=18,dhead={D},nhead=1"),
    pytest.param(1, 18, 10, D, 1, id=f"b=1,lq=18,lk=10,dhead={D},nhead=1"),
    pytest.param(2, L, L, D, 2, id=f"b=2,lq={L},lk={L},dhead={D},nhead=2"),
    pytest.param(2, 10, 10, D, 2, id=f"b=2,lq=10,lk=10,dhead={D},nhead=2"),
    pytest.param(4, int(3.5 * L), L, 4 * D, 4, id=f"b=4,lq={int(3.5*L)},lk={4*L},dhead={4*D},nhead=4"),
    pytest.param(4, 2 * L, 4 * L, 4 * D, 4, id=f"b=4,lq={2*L},lk={4*L},dhead={4*D},nhead=4"),
    pytest.param(4, 2 * L, int(4.5 * L), 4 * D, 4, id=f"b=4,lq={2*L},lk={int(4.5*L)},dhead={4*D},nhead=4"),
    pytest.param(4, int(3.5 * L), 4 * L, 4 * D, 4, id=f"b=4,lq={int(3.5*L)},lk={4*L},dhead={4*D},nhead=4"),
)


DATA_TYPE_PARAMS: Final = (
    pytest.param(True, torch.float16, 0.01, id="float16"),
    pytest.param(True, torch.bfloat16, 0.05, id="bfloat16"),
    pytest.param(False, torch.float16, 0.01, id="float16-fast"),
    pytest.param(False, torch.bfloat16, 0.05, id="bfloat16-fast"),
)


@pytest.mark.cuda
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


@pytest.mark.cuda
@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("dpos", [2, 3], ids=lambda v: f"dpos={v}")
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
@pytest.mark.parametrize("contiguous", [False], ids=lambda v: f"contiguous={v}")
def test_flash_attn_forward_bias(b, lq, lk, dhead, nhead, dpos, dtype, atol, full_precision, contiguous):
    torch.manual_seed(0)

    def reshape_qkv(x):
        return x.view(b, -1, nhead, dhead).movedim(-2, 1)

    def reshape_pos(x):
        return x.view(1, 1, -1, dpos).expand(b, nhead, -1, -1)

    q = reshape_qkv(torch.randn((b, lq, dhead * nhead), device="cuda", dtype=dtype))
    k = reshape_qkv(torch.randn((b, lk, dhead * nhead), device="cuda", dtype=dtype))
    v = reshape_qkv(torch.randn((b, lk, dhead * nhead), device="cuda", dtype=dtype))
    pos_q = reshape_pos(torch.randn((lq, dpos), device="cuda", dtype=dtype))
    pos_k = reshape_pos(torch.randn((lk, dpos), device="cuda", dtype=dtype))

    if contiguous:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pos_q = pos_q.contiguous()
        pos_k = pos_k.contiguous()

    slopes = torch.randint(-10, -1, (b, nhead), device="cuda", dtype=dtype).div_(10)

    bias = slopes[..., None, None] * (
        (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(b, nhead, lq, lk)
    )
    baseline_output = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    triton_output = attention(q, k, v, pos_q, pos_k, slopes, full_precision=full_precision)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_flash_attn_backward(b, lq, lk, dhead, nhead, dtype, atol, full_precision):
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
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
@pytest.mark.parametrize("contiguous", [False], ids=lambda v: f"contiguous={v}")
def test_flash_attn_backward_bias(b, lq, lk, dhead, nhead, dtype, atol, full_precision, contiguous):
    torch.manual_seed(0)
    dpos = 2

    def reshape_qkv(x):
        return x.view(b, -1, nhead, dhead).movedim(-2, 1)

    def reshape_pos(x):
        return x.view(1, 1, -1, dpos).expand(b, nhead, -1, -1)

    q = reshape_qkv(torch.randn((b, lq, dhead * nhead), device="cuda", dtype=dtype))
    k = reshape_qkv(torch.randn((b, lk, dhead * nhead), device="cuda", dtype=dtype))
    v = reshape_qkv(torch.randn((b, lk, dhead * nhead), device="cuda", dtype=dtype))
    pos_q = reshape_pos(torch.randn((lq, dpos), device="cuda", dtype=dtype))
    pos_k = reshape_pos(torch.randn((lk, dpos), device="cuda", dtype=dtype))

    if contiguous:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pos_q = pos_q.contiguous()
        pos_k = pos_k.contiguous()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

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


@pytest.mark.cuda
@pytest.mark.parametrize(
    "b, l, dhead, nhead",
    [
        pytest.param(1, 10, D, 1, id=f"b=1,l=10,dhead={D},nhead=1"),
        pytest.param(1, 18, D, 1, id=f"b=1,l=18,dhead={D},nhead=1"),
        pytest.param(2, L, D, 2, id=f"b=2,l={L},dhead={D},nhead=2"),
        pytest.param(2, 10, D, 2, id=f"b=2,l=10,dhead={D},nhead=2"),
    ],
)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_mask_threshold_forward(b, l, dhead, nhead, dtype, atol, full_precision):
    torch.manual_seed(0)

    dpos = 2
    lq = lk = l
    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    pos_q = torch.rand(*(b, nhead, lq, dpos), device="cuda", dtype=dtype)
    pos_k = torch.rand(*(b, nhead, lk, dpos), device="cuda", dtype=dtype)
    slopes = torch.full((b, nhead), -1, device="cuda", dtype=dtype)

    # Q and K positions are on [0, 1]. Generate a mask and apply a shift to positions based on the mask
    threshold = 3
    groups = 5
    atol *= groups
    mask = torch.randint(0, groups, (b, nhead, l, 1), device="cuda", dtype=dtype)
    pos_q += mask * threshold
    pos_k += mask * threshold

    baseline_mask = mask == mask.swapdims(-1, -2)
    bias = slopes[..., None, None] * (
        (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(b, nhead, lq, lk)
    )
    baseline_mask = torch.where(baseline_mask, bias, torch.tensor(float("-inf"), device="cuda", dtype=dtype))
    baseline_output = F.scaled_dot_product_attention(q, k, v, attn_mask=baseline_mask)

    triton_output = attention(q, k, v, pos_q, pos_k, slopes, full_precision=full_precision, mask_threshold=threshold)
    torch.testing.assert_close(baseline_output, triton_output, rtol=0, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "b, l, dhead, nhead",
    [
        pytest.param(1, 10, D, 1, id=f"b=1,l=10,dhead={D},nhead=1"),
        pytest.param(1, 18, D, 1, id=f"b=1,l=18,dhead={D},nhead=1"),
        pytest.param(2, L, D, 2, id=f"b=2,l={L},dhead={D},nhead=2"),
        pytest.param(2, 10, D, 2, id=f"b=2,l=10,dhead={D},nhead=2"),
    ],
)
@pytest.mark.parametrize("full_precision, dtype, atol", DATA_TYPE_PARAMS)
def test_mask_threshold_backward(b, l, dhead, nhead, dtype, atol, full_precision):
    torch.manual_seed(0)

    lq = lk = l
    dpos = 2
    q = torch.randn((b, nhead, lq, dhead), device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn((b, nhead, lk, dhead), device="cuda", dtype=dtype, requires_grad=True)
    pos_q = torch.rand((b, nhead, lq, dpos), device="cuda", dtype=dtype)
    pos_k = torch.rand((b, nhead, lk, dpos), device="cuda", dtype=dtype)
    slopes = torch.full((b, nhead), -1, device="cuda", dtype=dtype)

    # Q and K positions are on [0, 1]. Generate a mask and apply a shift to positions based on the mask
    threshold = 3
    groups = 5
    atol *= groups
    mask = torch.randint(0, groups, (b, nhead, l, 1), device="cuda", dtype=dtype)
    pos_q += mask * threshold
    pos_k += mask * threshold

    baseline_mask = mask == mask.swapdims(-1, -2)
    bias = slopes[..., None, None] * (
        (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(b, nhead, lq, lk)
    )
    baseline_mask = torch.where(baseline_mask, bias, torch.tensor(float("-inf"), device="cuda", dtype=dtype))

    # Baseline
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=baseline_mask)
    o.sum().backward()
    grad_q_baseline = q.grad
    grad_k_baseline = k.grad
    grad_v_baseline = v.grad

    # Triton
    q.grad = k.grad = v.grad = None
    o = attention(q, k, v, pos_q, pos_k, slopes, full_precision=full_precision, mask_threshold=threshold)
    o.sum().backward()
    grad_q_triton = q.grad
    grad_k_triton = k.grad
    grad_v_triton = v.grad

    # Test
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)


@pytest.mark.cuda
@pytest.mark.parametrize("b, lq, lk, dhead, nhead", HARD_SHAPE_PARAMS)
@pytest.mark.parametrize("full_precision", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast(b, lq, lk, dhead, nhead, full_precision, dtype):
    torch.manual_seed(0)
    atol = 0.05

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
    with torch.autocast(device_type="cuda", dtype=dtype):
        o = attention(q, k, v, pos_q, pos_k, slopes, full_precision=full_precision)
    o.sum().backward()
    grad_q_triton = q.grad
    grad_k_triton = k.grad
    grad_v_triton = v.grad

    # Test
    torch.testing.assert_close(grad_v_baseline, grad_v_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_k_baseline, grad_k_triton, rtol=0, atol=atol)
    torch.testing.assert_close(grad_q_baseline, grad_q_triton, rtol=0, atol=atol)


class TestModule:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_self_attention(self, device):
        torch.manual_seed(0)

        B, H, L, D = 2, 8, 32, 16
        x = torch.randn((B, L, D * H), device=device, requires_grad=True)
        pos = torch.randn((B, H, L, 2), device=device)
        slopes = -1 * torch.rand((B, H), device=device)

        attn = MultiheadAttention(D * H, H, batch_first=True).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            o = attn(x, x, x, pos, pos, slopes)
        assert o.shape == x.shape

        o.sum().backward()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_cross_attention(self, device):
        torch.manual_seed(0)

        B, H, L, D = 2, 8, 32, 16
        q = torch.randn((B, L, D * H), device=device, requires_grad=True)
        kv = torch.randn((B, L, D * H), device=device, requires_grad=True)
        pos_q = torch.randn((B, H, L, 2), device=device)
        pos_kv = torch.randn((B, H, L, 2), device=device)
        slopes = -1 * torch.rand((B, H), device=device)

        attn = MultiheadAttention(D * H, H, batch_first=True).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            o = attn(q, kv, kv, pos_q, pos_kv, slopes)
        assert o.shape == q.shape

        o.sum().backward()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_cross_attention_qkv_dim_diff(self, device):
        torch.manual_seed(0)

        B, H, L, D = 2, 8, 32, 16
        Dk = 12
        Dv = 14
        q = torch.randn((B, L, D * H), device=device, requires_grad=True)
        k = torch.randn((B, L, Dk * H), device=device, requires_grad=True)
        v = torch.randn((B, L, Dv * H), device=device, requires_grad=True)
        pos_q = torch.randn((B, H, L, 2), device=device)
        pos_kv = torch.randn((B, H, L, 2), device=device)
        slopes = -1 * torch.rand((B, H), device=device)

        attn = MultiheadAttention(D * H, H, kdim=Dk * H, vdim=Dv * H, batch_first=True).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            o = attn(q, k, v, pos_q, pos_kv, slopes)
        assert o.shape == q.shape

        o.sum().backward()


def test_flash_attn_cpu():
    torch.manual_seed(0)
    dtype = torch.float16
    full_precision = True
    contiguous = True
    atol = 1e-2
    dpos = 2
    b, nhead, lq, lk, dhead = 2, 8, 32, 32, 16

    def reshape_qkv(x):
        return x.view(b, -1, nhead, dhead).movedim(-2, 1)

    def reshape_pos(x):
        return x.view(1, 1, -1, dpos).expand(b, nhead, -1, -1)

    q = reshape_qkv(torch.randn((b, lq, dhead * nhead), dtype=dtype))
    k = reshape_qkv(torch.randn((b, lk, dhead * nhead), dtype=dtype))
    v = reshape_qkv(torch.randn((b, lk, dhead * nhead), dtype=dtype))
    pos_q = reshape_pos(torch.randn((lq, dpos), dtype=dtype))
    pos_k = reshape_pos(torch.randn((lk, dpos), dtype=dtype))

    if contiguous:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        pos_q = pos_q.contiguous()
        pos_k = pos_k.contiguous()

    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    slopes = torch.randint(-10, -1, (b, nhead), dtype=dtype).div_(10)
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


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_mask_threshold_infinity(device):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)
    dtype = torch.float16
    atol = 1e-2
    dpos = 2
    b, nhead, lq, lk, dhead = 2, 8, 32, 32, 16

    q = torch.randn((b, nhead, lq, dhead), device=device, dtype=dtype, requires_grad=True)
    k = torch.randn((b, nhead, lk, dhead), device=device, dtype=dtype, requires_grad=True)
    v = torch.randn((b, nhead, lk, dhead), device=device, dtype=dtype, requires_grad=True)
    pos_q = torch.rand(*(b, nhead, lq, dpos), device=device, dtype=dtype)
    pos_k = torch.rand(*(b, nhead, lk, dpos), device=device, dtype=dtype)
    slopes = torch.full((b, nhead), -1, device=device, dtype=dtype)
    pos_q[..., -1, :] = float("inf")
    pos_k[..., -1, :] = float("inf")

    baseline = attention(
        q[..., :-1, :],
        k[..., :-1, :],
        v[..., :-1, :],
        pos_q[..., :-1, :],
        pos_k[..., :-1, :],
        slopes,
    )
    actual = attention(q, k, v, pos_q, pos_k, slopes)
    torch.testing.assert_close(baseline, actual[..., :-1, :], rtol=0, atol=atol)
