import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.flash_attn import _flash_attn_forward


@pytest.mark.parametrize("dhead", [16, 24, 64], ids=lambda v: f"dhead={v}")
@pytest.mark.parametrize("lq", [32, 64, 256], ids=lambda v: f"lq={v}")
@pytest.mark.parametrize("lk", [32, 64, 256], ids=lambda v: f"lk={v}")
@pytest.mark.parametrize("nhead", [1, 2], ids=lambda v: f"nhead={v}")
@pytest.mark.parametrize("b", [1, 2], ids=lambda v: f"b={v}")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=lambda v: str(v))
def test_flash_attn_forward(b, lq, lk, dhead, nhead, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    q = torch.randn((b, lq, nhead, dhead), device="cuda", dtype=dtype)
    k = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype)
    v = torch.randn((b, lk, nhead, dhead), device="cuda", dtype=dtype)

    baseline_output = F.scaled_dot_product_attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)).movedim(2, 1)
    triton_output, _, _ = _flash_attn_forward(q, k, v, causal=False, softmax_scale=dhead**-0.5)
    assert torch.allclose(baseline_output, triton_output, atol=0.01, rtol=0)
