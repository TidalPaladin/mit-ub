import pytest
import torch
import torch.nn.functional as F

from mit_ub.model.layers.mlp import NormType
from mit_ub.model.layers.soft_moe import SoftMoE


class TestSoftMoE:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm_type", [NormType.LAYER_NORM, NormType.RMS_NORM])
    def test_forward(self, device, norm_type):
        B, L, D = 1, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = L // 2

        x = torch.randn(B, L, D, device=device)
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead, norm_type=norm_type).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        assert not out.isnan().any()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    def test_backward(self, device, norm):
        B, L, D = 1, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = L

        x = torch.randn(B, L, D, device=device, requires_grad=True)
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead, norm=norm).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad is not None
        for p in layer.parameters():
            assert p.grad is not None

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward_deterministic(self, device):
        B, L, D = 8, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = L // 2

        x = torch.randn(B, L, D, device=device)
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead, dropout=0.1).to(device)

        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x)
            out2 = layer(x)
        assert not torch.allclose(out1, out2)

        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x)
            out2 = layer(x)
        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize("gate_activation", [None, F.relu])
    def test_reset_parameters(self, mocker, gate_activation):
        torch.random.manual_seed(0)
        D = 32
        nhead = D // 32
        num_experts = 4
        num_slots = 128
        spy = mocker.spy(SoftMoE, "reset_parameters")
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead, gate_activation=gate_activation)
        spy.assert_called_once()

        weight_init = {k: v.clone() for k, v in layer.named_parameters()}
        layer.reset_parameters()
        weight_reset = {k: v.clone() for k, v in layer.named_parameters()}

        for k, v in weight_init.items():
            if (v == 0).all() or (v == 1).all():
                continue
            assert not torch.allclose(v, weight_reset[k], equal_nan=True)

    def test_extra_repr(self):
        layer = SoftMoE(128, 256, 4, 16, nhead=4)
        result = str(layer)
        assert "num_experts=4, num_slots=16" in result
