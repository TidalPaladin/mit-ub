import pytest
import torch
import torch.nn as nn
from torchtune.modules.peft.lora import LoRALinear

from mit_ub.model.lora import LoRATarget
from mit_ub.model.transformer import TransformerDecoderLayer, TransformerEncoderLayer


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
            pytest.param(nn.ReLU(), None, id="relu"),
            pytest.param(nn.Identity(), nn.Sigmoid(), id="glu"),
            pytest.param(nn.Identity(), nn.SiLU(), id="swiglu"),
            pytest.param(nn.SiLU(), nn.Sigmoid(), id="sigsilu"),
        ],
    )
    def test_forward(self, device, activation, gate_activation):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D, activation=activation, gate_activation=gate_activation).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        assert isinstance(layer.activation, type(activation))
        assert gate_activation is None or (layer.gate is not None and isinstance(layer.gate[-1], type(gate_activation)))

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
            pytest.param(nn.ReLU(), None, id="relu"),
            pytest.param(nn.Identity(), nn.Sigmoid(), id="glu"),
            pytest.param(nn.Identity(), nn.SiLU(), id="swiglu"),
            pytest.param(nn.SiLU(), nn.Sigmoid(), id="sigsilu"),
        ],
    )
    @pytest.mark.parametrize(
        "target",
        [
            [LoRATarget.ATTENTION],
            [LoRATarget.FEEDFORWARD],
            [LoRATarget.ATTENTION, LoRATarget.FEEDFORWARD],
        ],
    )
    def test_lora(self, device, activation, gate_activation, target):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerEncoderLayer(D, nhead, D, activation=activation, gate_activation=gate_activation).to(device)
        layer = layer.apply_lora(target=target, rank=4, alpha=16)

        if LoRATarget.FEEDFORWARD in target:
            assert isinstance(layer.linear1, LoRALinear)
            assert isinstance(layer.linear2, LoRALinear)
            if gate_activation is not None:
                assert isinstance(layer.gate[0], LoRALinear)
        if LoRATarget.ATTENTION in target:
            assert isinstance(layer.self_attn.q_proj, LoRALinear)
            assert isinstance(layer.self_attn.k_proj, LoRALinear)
            assert isinstance(layer.self_attn.v_proj, LoRALinear)
            assert isinstance(layer.self_attn.output_proj, LoRALinear)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape

        for name, param in layer.named_parameters():
            assert param.requires_grad == ("lora_a" in name or "lora_b" in name)


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
            pytest.param(nn.ReLU(), None, id="relu"),
            pytest.param(nn.Identity(), nn.Sigmoid(), id="glu"),
            pytest.param(nn.Identity(), nn.SiLU(), id="swiglu"),
            pytest.param(nn.SiLU(), nn.Sigmoid(), id="sigsilu"),
        ],
    )
    def test_forward(self, device, activation, gate_activation):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, activation=activation, gate_activation=gate_activation).to(
            device
        )
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
        assert out.shape == q.shape
        assert isinstance(layer.activation, type(activation))
        assert gate_activation is None or (layer.gate is not None and isinstance(layer.gate[-1], type(gate_activation)))

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
        layer = TransformerDecoderLayer(Dq, nhead, Dk).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
            out = out.sum()
        out.backward()
        assert q.grad is not None
        assert k.grad is not None

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
            pytest.param(nn.ReLU(), None, id="relu"),
            pytest.param(nn.Identity(), nn.Sigmoid(), id="glu"),
            pytest.param(nn.Identity(), nn.SiLU(), id="swiglu"),
            pytest.param(nn.SiLU(), nn.Sigmoid(), id="sigsilu"),
        ],
    )
    @pytest.mark.parametrize(
        "target",
        [
            LoRATarget.ATTENTION,
            LoRATarget.FEEDFORWARD,
            [LoRATarget.ATTENTION, LoRATarget.FEEDFORWARD],
        ],
    )
    def test_lora(self, device, activation, gate_activation, target):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, activation=activation, gate_activation=gate_activation).to(
            device
        )
        layer = layer.apply_lora(target=target, rank=4, alpha=16)

        if LoRATarget.FEEDFORWARD in target:
            assert isinstance(layer.linear1, LoRALinear)
            assert isinstance(layer.linear2, LoRALinear)
            if gate_activation is not None:
                assert isinstance(layer.gate[0], LoRALinear)
        if LoRATarget.ATTENTION in target:
            for attn in (layer.self_attn, layer.cross_attn):
                assert isinstance(attn.q_proj, LoRALinear)
                assert isinstance(attn.k_proj, LoRALinear)
                assert isinstance(attn.v_proj, LoRALinear)
                assert isinstance(attn.output_proj, LoRALinear)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
        assert out.shape == q.shape

        for name, param in layer.named_parameters():
            assert param.requires_grad == ("lora_a" in name or "lora_b" in name)
