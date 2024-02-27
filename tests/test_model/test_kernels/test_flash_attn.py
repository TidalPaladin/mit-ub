import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.kernels.flash_attn import _flash_attn_forward


def test_flash_attn_forward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    torch.manual_seed(0)

    B = 2
    Lq = 32
    Lk = 32
    D_head = 32
    N_head = 2
    q = torch.randn((B, Lq, N_head, D_head), device="cuda", dtype=torch.float16)
    k = torch.randn((B, Lk, N_head, D_head), device="cuda", dtype=torch.float16)
    v = torch.randn((B, Lk, N_head, D_head), device="cuda", dtype=torch.float16)

    baseline_output = F.scaled_dot_product_attention(q.movedim(1, 2), k.movedim(1, 2), v.movedim(1, 2)).movedim(2, 1)
    triton_output, _, _ = _flash_attn_forward(q, k, v, causal=False, softmax_scale=D_head**-0.5)
    assert torch.allclose(baseline_output, triton_output, atol=0.01, rtol=0)
