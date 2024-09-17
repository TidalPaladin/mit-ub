import pytest
import torch

from mit_ub.model.mlp import MLP
from mit_ub.model.soft_moe import SoftMoE


class TestSoftMoE:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, L, D = 1, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = L // 2

        x = torch.randn(B, L, D, device=device)
        expert = MLP(D, D, D).to(device)
        layer = SoftMoE(expert, D, num_experts, num_slots, nhead=nhead).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, L, D = 1, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = L

        x = torch.randn(B, L, D, device=device, requires_grad=True)
        expert = MLP(D, D, D).to(device)
        layer = SoftMoE(expert, D, num_experts, num_slots, nhead=nhead).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad is not None
