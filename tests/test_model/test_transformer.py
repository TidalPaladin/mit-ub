import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.transformer import (
    TransformerConvDecoderLayer,
    TransformerConvEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class TestTransformerEncoderLayer:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize(
        "activation, gate_activation",
        [
            pytest.param(F.relu, None, id="relu"),
            pytest.param(lambda x: x, F.sigmoid, id="glu"),
            pytest.param(lambda x: x, F.silu, id="swiglu"),
            pytest.param(F.silu, F.sigmoid, id="sigsilu"),
        ],
    )
    def test_forward(self, device, activation, gate_activation):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerEncoderLayer(
            D, nhead, D, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        assert isinstance(layer.mlp.activation, type(activation))
        assert gate_activation is None or (
            layer.mlp.w_gate is not None and layer.mlp.gate_activation == gate_activation
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_experts, num_slots", [(None, None), (8, 8)])
    def test_forward_moe(self, device, num_experts, num_slots):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D, num_experts=num_experts, num_slots=num_slots).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape

    def test_forward_multi_query(self):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D)
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D, num_kv_heads=nhead // 2)
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
        x = torch.randn(B, L, D, device=device, requires_grad=True)
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
            out = out.sum()
        out.backward()

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        D = 32
        nhead = 2
        layer = TransformerEncoderLayer(D, nhead, D)

        spies = {
            name: mocker.spy(module, "reset_parameters")
            for name, module in layer.named_modules()
            if hasattr(module, "reset_parameters")
        }
        layer.reset_parameters()
        for spy in spies.values():
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, F.relu])
    def test_forward_deterministic(self, device, gate_activation):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D, gate_activation=gate_activation).to(device)

        # Training, non-determinstic
        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x)
            out2 = layer(x)
            assert not torch.allclose(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x)
            out2 = layer(x)
            assert_close(out1, out2)

    @pytest.mark.parametrize("num_experts", [None, 2])
    def test_norm(self, num_experts):
        B, L, D = 1, 128, 128
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D, num_experts=num_experts, num_slots=num_experts)
        assert layer.mlp.norm
        assert layer.self_attn.norm


class TestTransformerDecoderLayer:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize(
        "activation, gate_activation",
        [
            pytest.param(F.relu, None, id="relu"),
            pytest.param(lambda x: x, F.sigmoid, id="glu"),
            pytest.param(lambda x: x, F.silu, id="swiglu"),
            pytest.param(F.silu, F.sigmoid, id="sigsilu"),
        ],
    )
    def test_forward(self, device, activation, gate_activation):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(
            Dq, nhead, Dk, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
        assert out.shape == q.shape
        assert isinstance(layer.mlp.activation, type(activation))
        assert gate_activation is None or (
            layer.mlp.w_gate is not None and layer.mlp.gate_activation == gate_activation
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_experts, num_slots", [(None, None), (8, 8)])
    def test_forward_moe(self, device, num_experts, num_slots):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, num_experts=num_experts, num_slots=num_slots).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
        assert out.shape == q.shape

    def test_forward_multi_query(self):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq)
        k = torch.randn(B, Lk, Dk)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, Dq, num_kv_heads=nhead // 2)
        out = layer(q, k)
        assert out.shape == q.shape

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device, requires_grad=True)
        k = torch.randn(B, Lk, Dk, device=device, requires_grad=True)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, Dq).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
            out = out.sum()
        out.backward()
        assert q.grad is not None
        assert k.grad is not None

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        Dq = Dk = 32
        nhead = 2
        layer = TransformerDecoderLayer(Dq, nhead, Dk, Dq)

        spies = {
            name: mocker.spy(module, "reset_parameters")
            for name, module in layer.named_modules()
            if hasattr(module, "reset_parameters")
        }
        layer.reset_parameters()
        for spy in spies.values():
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_forward_deterministic(self, device, gate_activation):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, Dq, gate_activation=gate_activation).to(device)

        # Training, non-determinstic
        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(q, k)
            out2 = layer(q, k)
            assert not torch.allclose(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(q, k)
            out2 = layer(q, k)
            assert_close(out1, out2)

    @pytest.mark.parametrize("num_experts", [None, 2])
    def test_norm(self, num_experts):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, Dq, num_experts=num_experts, num_slots=num_experts)
        assert layer.mlp.norm
        assert layer.self_attn.norm


class TestTransformerConvEncoderLayer:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize(
        "activation, gate_activation",
        [
            pytest.param(F.relu, None, id="relu"),
            pytest.param(lambda x: x, F.sigmoid, id="glu"),
            pytest.param(lambda x: x, F.silu, id="swiglu"),
            pytest.param(F.silu, F.sigmoid, id="sigsilu"),
        ],
    )
    def test_forward(self, device, activation, gate_activation):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerConvEncoderLayer(
            D, nhead, D, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x, (16, 8))
        assert out.shape == x.shape
        assert isinstance(layer.mlp.activation, type(activation))
        assert gate_activation is None or (
            layer.mlp.w_gate is not None and layer.mlp.gate_activation == gate_activation
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device, requires_grad=True)
        nhead = D // 16
        layer = TransformerConvEncoderLayer(D, nhead, D).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x, (16, 8))
            out = out.sum()
        out.backward()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, F.relu])
    def test_forward_deterministic(self, device, gate_activation):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerConvEncoderLayer(D, nhead, D, gate_activation=gate_activation).to(device)

        # Training, non-determinstic
        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x, (16, 8))
            out2 = layer(x, (16, 8))
            assert not torch.allclose(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x, (16, 8))
            out2 = layer(x, (16, 8))
            assert_close(out1, out2)


class TestTransformerConvDecoderLayer:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize(
        "activation, gate_activation",
        [
            pytest.param(F.relu, None, id="relu"),
            pytest.param(lambda x: x, F.sigmoid, id="glu"),
            pytest.param(lambda x: x, F.silu, id="swiglu"),
            pytest.param(F.silu, F.sigmoid, id="sigsilu"),
        ],
    )
    def test_forward(self, device, activation, gate_activation):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerConvDecoderLayer(
            Dq, nhead, Dk, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k, (8, 8))
        assert out.shape == q.shape
        assert isinstance(layer.mlp.activation, type(activation))
        assert gate_activation is None or (
            layer.mlp.w_gate is not None and layer.mlp.gate_activation == gate_activation
        )

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device, requires_grad=True)
        k = torch.randn(B, Lk, Dk, device=device, requires_grad=True)
        nhead = Dq // 16
        layer = TransformerConvDecoderLayer(Dq, nhead, Dk, Dq).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k, (8, 8))
            out = out.sum()
        out.backward()
        assert q.grad is not None
        assert k.grad is not None

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        Dq = Dk = 32
        nhead = 2
        layer = TransformerDecoderLayer(Dq, nhead, Dk, Dq)

        spies = {
            name: mocker.spy(module, "reset_parameters")
            for name, module in layer.named_modules()
            if hasattr(module, "reset_parameters")
        }
        layer.reset_parameters()
        for spy in spies.values():
            spy.assert_called_once()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_forward_deterministic(self, device, gate_activation):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerConvDecoderLayer(Dq, nhead, Dk, Dq, gate_activation=gate_activation).to(device)

        # Training, non-determinstic
        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(q, k, (8, 8))
            out2 = layer(q, k, (8, 8))
            assert not torch.allclose(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(q, k, (8, 8))
            out2 = layer(q, k, (8, 8))
            assert_close(out1, out2)
