import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from mit_ub.model.transformer import apply_lora, freeze_non_lora


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "quantize_base, use_bias",
    [
        (False, True),
        (False, False),
        (True, False),
        pytest.param(True, True, marks=pytest.mark.xfail(raises=NotImplementedError, strict=True)),
    ],
)
def test_apply_lora(quantize_base, use_bias, dtype, device):
    if device == "cuda" and quantize_base:
        pytest.skip("torchao seems to have issues running quantized on CUDA")

    torch.random.manual_seed(0)
    # NOTE: D must be >= 256 in the quanitzed case
    B, L, D_in, D_out = 2, 10, 256, 512
    x = torch.randn(B, L, D_in, requires_grad=True, device=device)
    linear = nn.Linear(D_in, D_out, bias=use_bias).to(device)
    linear_lora = apply_lora(linear, rank=4, alpha=16, quantize_base=quantize_base)

    with torch.autocast(device_type="cuda", dtype=dtype):
        o1 = linear_lora(x)
        o2 = linear(x)
    if not quantize_base:
        assert_close(o1, o2, rtol=0, atol=1e-3)


@pytest.mark.parametrize(
    "quantize_base, use_bias",
    [
        (False, True),
        (False, False),
        (True, False),
    ],
)
def test_freeze_non_lora(use_bias, quantize_base):
    torch.random.manual_seed(0)
    device = "cpu"

    # NOTE: D must be >= 256 in the quanitzed case
    B, L, D_in, D_out = 2, 10, 256, 512
    x = torch.randn(B, L, D_in, requires_grad=True, device=device)
    linear = nn.Linear(D_in, D_out, bias=use_bias).to(device)
    linear_lora = apply_lora(linear, rank=4, alpha=16, quantize_base=quantize_base)
    freeze_non_lora(linear_lora)

    for name, param in linear_lora.named_parameters():
        assert param.requires_grad == ("lora_a" in name or "lora_b" in name)
