import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close
from torchtune.modules.peft.lora import LoRALinear

from mit_ub.model.lora import LoRATarget
from mit_ub.model.mlp import MLP


class TestMLP:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_forward(self, device, gate_activation, bias, dropout):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D).to(device)
        layer = MLP(D, 2 * D, D, gate_activation=gate_activation, bias=bias, dropout=dropout).to(device)
        y = layer(x.clone())
        assert y.shape == x.shape

    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_reset_parameters(self, gate_activation):
        torch.random.manual_seed(0)
        D = 32
        layer = MLP(D, D, D)
        w1 = layer.fc1.weight.clone()
        w2 = layer.fc2.weight.clone()
        wg = layer.gate[0].weight.clone() if layer.gate is not None else None
        layer.reset_parameters()
        assert not torch.allclose(w1, layer.fc1.weight)
        assert not torch.allclose(w2, layer.fc2.weight)
        if layer.gate is not None and wg is not None:
            assert not torch.allclose(wg, layer.gate[0].weight)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D, requires_grad=True).to(device)
        layer = MLP(D, D, D).to(device)
        y = layer(x)
        y.sum().backward()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_lora(self, device, gate_activation):
        B, L, D = 2, 8, 32
        torch.random.manual_seed(0)
        layer = MLP(D, 2 * D, D, gate_activation=gate_activation).to(device)
        x = torch.randn(B, L, D, device=device)

        layer = layer.apply_lora(target=[LoRATarget.FEEDFORWARD], rank=4, alpha=16)
        assert isinstance(layer.fc1, LoRALinear)
        assert isinstance(layer.fc2, LoRALinear)
        if gate_activation is not None:
            assert isinstance(layer.gate[0], LoRALinear)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
        assert out.shape == (B, L, D)

        for name, param in layer.named_parameters():
            assert param.requires_grad == ("lora_a" in name or "lora_b" in name)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_forward_deterministic(self, device, gate_activation):
        B, L, D = 2, 8, 32
        torch.random.manual_seed(0)
        layer = MLP(D, 2 * D, D, dropout=0.1, gate_activation=gate_activation).to(device)
        x = torch.randn(B, L, D, device=device)

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
