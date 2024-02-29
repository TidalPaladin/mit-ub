import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.flash_attn import _flash_attn_forward

@pytest.mark.parametrize("dhead", [
    16, 
    24,
    pytest.param(8, marks=pytest.mark.xfail(strict=True)),
], ids=lambda v: f"dhead={v}")
@pytest.mark.parametrize("lq", [
    32, 
    64, 
    128,
    33,
], ids=lambda v: f"lq={v}")
@pytest.mark.parametrize("lk", [
    32, 
    64, 
    pytest.param(30, marks=pytest.mark.xfail(strict=True)),
], ids=lambda v: f"lk={v}")
@pytest.mark.parametrize("nhead", [1, 2], ids=lambda v: f"nhead={v}")
@pytest.mark.parametrize("b", [1, 2, 3], ids=lambda v: f"b={v}")
def test_flash_attn_forward(b, lq, lk, dhead, nhead):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, lq, nhead, dhead), device="cuda", dtype=torch.float16)
    k = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=torch.float16)
    v = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=torch.float16)

    baseline_output = F.scaled_dot_product_attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)).movedim(2, 1)
    triton_output, _, _ = _flash_attn_forward(q, k, v, causal=False, softmax_scale=dhead**-0.5)
    assert torch.allclose(baseline_output, triton_output, atol=0.01, rtol=0)
