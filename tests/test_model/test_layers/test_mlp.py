import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.layers.mlp import MLP, NormType, mlp_forward


@pytest.mark.parametrize("dropout", [0.0, 0.1])
@pytest.mark.parametrize("training", [False, True])
def test_mlp_forward(dropout, training):
    torch.random.manual_seed(0)
    B, L, D = 2, 8, 32
    x = torch.randn(B, L, D)

    layer = nn.Sequential(
        nn.Linear(D, 2 * D),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(2 * D, D),
        nn.Dropout(dropout),
    )
    layer.train(training)

    torch.random.manual_seed(0)
    baseline = layer(x)
    torch.random.manual_seed(0)
    actual = mlp_forward(
        x,
        layer[0].weight,  # type: ignore
        layer[3].weight,  # type: ignore
        layer[0].bias,  # type: ignore
        layer[3].bias,  # type: ignore
        dropout=dropout,
        activation=F.relu,
        training=training,
    )
    assert_close(baseline, actual, atol=0.001, rtol=0)


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
    @pytest.mark.parametrize("dropout,stochastic_depth", [(0.0, 0.0), (0.1, 0.25)])
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("norm_type", [NormType.LAYER_NORM, NormType.RMS_NORM])
    @pytest.mark.parametrize("layer_scale", [None, 0.1])
    def test_forward(self, device, gate_activation, bias, dropout, norm, norm_type, layer_scale, stochastic_depth):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D).to(device)
        layer = MLP(
            D,
            2 * D,
            D,
            gate_activation=gate_activation,
            bias=bias,
            dropout=dropout,
            norm=norm,
            norm_type=norm_type,
            layer_scale=layer_scale,
            stochastic_depth=stochastic_depth,
        ).to(device)
        y = layer(x.clone())
        assert y.shape == x.shape
        assert not y.isnan().any()

    @pytest.mark.parametrize("gate_activation", [None, F.relu])
    @pytest.mark.parametrize("layer_scale", [None, 1.0])
    @pytest.mark.parametrize("norm_type", [NormType.LAYER_NORM, NormType.RMS_NORM])
    def test_reset_parameters(self, mocker, gate_activation, layer_scale, norm_type):
        torch.random.manual_seed(0)
        D = 32
        spy = mocker.spy(MLP, "reset_parameters")
        layer = MLP(D, D, D, gate_activation=gate_activation, layer_scale=layer_scale, norm=True, norm_type=norm_type)
        spy.assert_called_once()

        if norm_type == NormType.LAYER_NORM:
            assert layer.w_norm is not None
            assert layer.b_norm is not None

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
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("layer_scale", [None, 0.1])
    def test_backward(self, device, norm, checkpoint, layer_scale):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D, requires_grad=True).to(device)
        layer = MLP(D, D, D, dropout=0.1, norm=norm, layer_scale=layer_scale).to(device)
        layer.checkpoint = checkpoint
        layer.train()
        y = layer(x)
        y.sum().backward()
        for p in layer.parameters():
            assert p.grad is not None

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("gate_activation", [None, nn.ReLU()])
    def test_forward_deterministic(self, device, gate_activation):
        B, L, D = 8, 8, 32
        torch.random.manual_seed(0)
        layer = MLP(D, 2 * D, D, dropout=0.1, gate_activation=gate_activation, stochastic_depth=0.25).to(device)
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

    def test_fused_norm(self):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D)
        norm = nn.LayerNorm(D)
        layer = MLP(D, 2 * D, D)

        baseline = layer(norm(x))

        layer.w_norm = norm.weight
        layer.b_norm = norm.bias
        actual = layer(x)
        assert_close(baseline, actual)

    def test_extra_repr(self):
        layer = MLP(32, 64, 32)
        result = str(layer)
        exp = "in=32, hidden=64, out=32, dropout=0.0, act=relu2, gate_act=None, bias=True, norm=False, norm_type=layernorm"
        assert exp in result
