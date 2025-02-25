import math

import pytest
import torch

from mit_ub.model.layers.patch_embed import PatchEmbed2d


class TestPatchEmbed2d:

    @pytest.mark.cuda
    def test_forward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    @pytest.mark.cuda
    def test_backward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None
