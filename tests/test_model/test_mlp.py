import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.mlp import MLP


class TestMLP:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, F.relu])
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_forward(self, device, gate_activation, bias, dropout):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D).to(device)
        layer = MLP(D, 2 * D, D, gate_activation=gate_activation, bias=bias, dropout=dropout).to(device)
        y = layer(x.clone())
        assert y.shape == x.shape

    @pytest.mark.parametrize("gate_activation", [None, F.relu])
    def test_reset_parameters(self, mocker, gate_activation):
        torch.random.manual_seed(0)
        D = 32
        spy = mocker.spy(MLP, "reset_parameters")
        layer = MLP(D, D, D, gate_activation=gate_activation)
        spy.assert_called_once()

        weight_init = {k: v.clone() for k, v in layer.named_parameters()}
        layer.reset_parameters()
        weight_reset = {k: v.clone() for k, v in layer.named_parameters()}

        for k, v in weight_init.items():
            if (v == 0).all() or (v == 1).all():
                continue
            assert not torch.allclose(v, weight_reset[k], equal_nan=True)

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
