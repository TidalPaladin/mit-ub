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
        layer = TransformerEncoderLayer(
            D, nhead, D, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == x.shape
        assert isinstance(layer.mlp.activation, type(activation))
        assert gate_activation is None or (
            layer.mlp.gate is not None and isinstance(layer.mlp.gate[-1], type(gate_activation))
        )

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
            assert isinstance(layer.mlp.fc1, LoRALinear)
            assert isinstance(layer.mlp.fc2, LoRALinear)
            if gate_activation is not None:
                assert isinstance(layer.mlp.gate[0], LoRALinear)
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

    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_reset_parameters(self, gate_activation):
        torch.random.manual_seed(0)
        D = 32
        nhead = 2
        layer = TransformerEncoderLayer(D, nhead, D, gate_activation=gate_activation)

        # Clone weights and biases before resetting parameters
        attn_weights = {
            "q_proj": layer.self_attn.q_proj.weight.clone(),
            "k_proj": layer.self_attn.k_proj.weight.clone(),
            "v_proj": layer.self_attn.v_proj.weight.clone(),
            "output_proj": layer.self_attn.output_proj.weight.clone(),
        }
        attn_biases = {
            "q_proj": layer.self_attn.q_proj.bias.clone() if layer.self_attn.q_proj.bias is not None else None,
            "k_proj": layer.self_attn.k_proj.bias.clone() if layer.self_attn.k_proj.bias is not None else None,
            "v_proj": layer.self_attn.v_proj.bias.clone() if layer.self_attn.v_proj.bias is not None else None,
            "output_proj": (
                layer.self_attn.output_proj.bias.clone() if layer.self_attn.output_proj.bias is not None else None
            ),
        }
        mlp_weights = {"fc1": layer.mlp.fc1.weight.clone(), "fc2": layer.mlp.fc2.weight.clone()}
        mlp_biases = {
            "fc1": layer.mlp.fc1.bias.clone() if layer.mlp.fc1.bias is not None else None,
            "fc2": layer.mlp.fc2.bias.clone() if layer.mlp.fc2.bias is not None else None,
        }
        # NOTE: Norm weights aren't tested because they reset to 1.0

        layer.reset_parameters()

        # Check that weights and biases have been reset
        for name, weight in attn_weights.items():
            assert not torch.allclose(weight, getattr(layer.self_attn, name).weight)
        for name, bias in attn_biases.items():
            if bias is not None:
                assert not torch.allclose(bias, getattr(layer.self_attn, name).bias)
        for name, weight in mlp_weights.items():
            assert not torch.allclose(weight, getattr(layer.mlp, name).weight)
        for name, bias in mlp_biases.items():
            if bias is not None:
                assert not torch.allclose(bias, getattr(layer.mlp, name).bias)


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
        layer = TransformerDecoderLayer(
            Dq, nhead, Dk, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
        assert out.shape == q.shape
        assert isinstance(layer.mlp.activation, type(activation))
        assert gate_activation is None or (
            layer.mlp.gate is not None and isinstance(layer.mlp.gate[-1], type(gate_activation))
        )

    def test_forward_multi_query(self):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq)
        k = torch.randn(B, Lk, Dk)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk, num_kv_heads=nhead // 2)
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
            assert isinstance(layer.mlp.fc1, LoRALinear)
            assert isinstance(layer.mlp.fc2, LoRALinear)
            if gate_activation is not None:
                assert isinstance(layer.mlp.gate[0], LoRALinear)
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

    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_reset_parameters(self, gate_activation):
        torch.random.manual_seed(0)
        D = 32
        nhead = 2
        layer = TransformerDecoderLayer(D, nhead, D, D, gate_activation=gate_activation)

        # Clone weights and biases before resetting parameters
        attn_weights = {
            "q_proj": layer.self_attn.q_proj.weight.clone(),
            "k_proj": layer.self_attn.k_proj.weight.clone(),
            "v_proj": layer.self_attn.v_proj.weight.clone(),
            "output_proj": layer.self_attn.output_proj.weight.clone(),
        }
        attn_biases = {
            "q_proj": layer.self_attn.q_proj.bias.clone() if layer.self_attn.q_proj.bias is not None else None,
            "k_proj": layer.self_attn.k_proj.bias.clone() if layer.self_attn.k_proj.bias is not None else None,
            "v_proj": layer.self_attn.v_proj.bias.clone() if layer.self_attn.v_proj.bias is not None else None,
            "output_proj": (
                layer.self_attn.output_proj.bias.clone() if layer.self_attn.output_proj.bias is not None else None
            ),
        }
        cross_attn_weights = {
            "q_proj": layer.cross_attn.q_proj.weight.clone(),
            "k_proj": layer.cross_attn.k_proj.weight.clone(),
            "v_proj": layer.cross_attn.v_proj.weight.clone(),
            "output_proj": layer.cross_attn.output_proj.weight.clone(),
        }
        cross_attn_biases = {
            "q_proj": layer.cross_attn.q_proj.bias.clone() if layer.cross_attn.q_proj.bias is not None else None,
            "k_proj": layer.cross_attn.k_proj.bias.clone() if layer.cross_attn.k_proj.bias is not None else None,
            "v_proj": layer.cross_attn.v_proj.bias.clone() if layer.cross_attn.v_proj.bias is not None else None,
            "output_proj": (
                layer.cross_attn.output_proj.bias.clone() if layer.cross_attn.output_proj.bias is not None else None
            ),
        }
        mlp_weights = {"fc1": layer.mlp.fc1.weight.clone(), "fc2": layer.mlp.fc2.weight.clone()}
        mlp_biases = {
            "fc1": layer.mlp.fc1.bias.clone() if layer.mlp.fc1.bias is not None else None,
            "fc2": layer.mlp.fc2.bias.clone() if layer.mlp.fc2.bias is not None else None,
        }
        # NOTE: Norm weights aren't tested because they reset to 1.0

        layer.reset_parameters()

        # Check that weights and biases have been reset
        for name, weight in attn_weights.items():
            assert not torch.allclose(weight, getattr(layer.self_attn, name).weight)
        for name, bias in attn_biases.items():
            if bias is not None:
                assert not torch.allclose(bias, getattr(layer.self_attn, name).bias)
        for name, weight in cross_attn_weights.items():
            assert not torch.allclose(weight, getattr(layer.cross_attn, name).weight)
        for name, bias in cross_attn_biases.items():
            if bias is not None:
                assert not torch.allclose(bias, getattr(layer.cross_attn, name).bias)
        for name, weight in mlp_weights.items():
            assert not torch.allclose(weight, getattr(layer.mlp, name).weight)
        for name, bias in mlp_biases.items():
            if bias is not None:
                assert not torch.allclose(bias, getattr(layer.mlp, name).bias)
