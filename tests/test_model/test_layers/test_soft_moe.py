import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.layers.mlp import mlp_forward
from mit_ub.model.layers.soft_moe import SoftMoE, forward_experts


class TestSoftMoE:

    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    @pytest.mark.parametrize("output_dropout", [False, True])
    def test_forward_single_expert(self, dropout, output_dropout):
        B, S, D, E = 2, 128, 128, 1
        w_in = torch.randn(E, 2 * D, D)
        b_in = torch.randn(E, 2 * D)
        w_out = torch.randn(E, D, 2 * D)
        b_out = torch.randn(E, D)
        w_gate = torch.randn(E, 2 * D, D)
        b_gate = torch.randn(E, 2 * D)
        x = torch.randn(B, S, D)

        torch.manual_seed(0)
        out1 = forward_experts(x, w_in, b_in, w_out, b_out, F.relu, dropout, w_gate, b_gate, F.relu, training=True)
        torch.manual_seed(0)
        out2 = mlp_forward(
            x,
            w_in.squeeze(0),
            w_out.squeeze(0),
            b_in.squeeze(0),
            b_out.squeeze(0),
            w_gate.squeeze(0),
            b_gate.squeeze(0),
            dropout,
            F.relu,
            F.relu,
            training=True,
        )
        assert_close(out1, out2, atol=0.01, rtol=0)

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
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead).to(device)

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
    def test_backward(self, device):
        B, L, D = 1, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = L

        x = torch.randn(B, L, D, device=device, requires_grad=True)
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        out.sum().backward()
        assert x.grad is not None

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward_deterministic(self, device):
        B, L, D = 1, 128, 128
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

    def test_fused_norm(self):
        torch.random.manual_seed(0)
        B, L, D = 1, 128, 128
        nhead = D // 32
        num_experts = 4
        num_slots = 16

        x = torch.randn(B, L, D)
        layer = SoftMoE(D, D, num_experts, num_slots, nhead=nhead, norm=True)

        y_norm = layer(x)
        layer.w_pre_norm = None  # type: ignore
        layer.b_pre_norm = None  # type: ignore
        y_no_norm = layer(x)
        assert not torch.allclose(y_norm, y_no_norm)

    def test_extra_repr(self):
        layer = SoftMoE(32, 64, 4, 16, nhead=8)
        result = str(layer)
        assert (
            result
            == "SoftMoE(in=32, hidden=64, out=32, experts=4, slots=16, nhead=8, dropout=0.0, act=relu2, gate_act=None, bias=True, norm=False, qk_norm=False)"
        )
