import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.testing import assert_close

from mit_ub.model.conv import ConvEncoderLayer2d, ConvEncoderLayer3d, conv_2d, conv_3d


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv_2d(device, kernel_size, stride, padding):
    B, C, H, W = 2, 128, 16, 16
    x = torch.randn(B, C, H, W, device=device)
    seq = rearrange(x, "b c h w -> b (h w) c")
    conv = nn.Conv2d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, groups=C).to(device)

    expected = rearrange(conv(x), "b c h w -> b (h w) c")
    actual = conv_2d(seq, (H, W), conv.weight, conv.bias, stride=stride, padding=padding, groups=C)
    assert_close(actual, expected)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv_3d(device, kernel_size, stride, padding):
    B, C, D, H, W = 2, 128, 16, 16, 16
    x = torch.randn(B, C, D, H, W, device=device)
    seq = rearrange(x, "b c d h w -> b (d h w) c")
    conv = nn.Conv3d(C, C, kernel_size=kernel_size, stride=stride, padding=padding, groups=C).to(device)

    expected = rearrange(conv(x), "b c d h w -> b (d h w) c")
    actual = conv_3d(seq, (D, H, W), conv.weight, conv.bias, stride=stride, padding=padding, groups=C)
    assert_close(actual, expected)


class TestConvEncoderLayer2d:

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
        B, D, H, W = 1, 128, 16, 16
        x = torch.randn(B, H * W, D, device=device)
        layer = ConvEncoderLayer2d(
            D, 3, 1, D, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x, (H, W))
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
        B, D, H, W = 1, 128, 16, 16
        x = torch.randn(B, H * W, D, device=device, requires_grad=True)
        layer = ConvEncoderLayer2d(D, 3, 1, D).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x, (H, W))
            out = out.sum()
        out.backward()

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        D = 32
        layer = ConvEncoderLayer2d(D, 3, 1, D)

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
        B, D, H, W = 1, 128, 16, 16
        x = torch.randn(B, H * W, D, device=device)
        layer = ConvEncoderLayer2d(D, 3, 1, D, gate_activation=gate_activation).to(device)

        # Training, non-determinstic
        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x, (H, W))
            out2 = layer(x, (H, W))
            assert not torch.allclose(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x, (H, W))
            out2 = layer(x, (H, W))
            assert_close(out1, out2)


class TestConvEncoderLayer3d:

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
        B, C, D, H, W = 1, 128, 16, 16, 16
        x = torch.randn(B, H * W * D, C, device=device)
        layer = ConvEncoderLayer3d(
            C, 3, 1, C, activation=activation, gate_activation=gate_activation, layer_scale=1e-5
        ).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x, (H, W, D))
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
        B, C, D, H, W = 1, 128, 16, 16, 16
        x = torch.randn(B, H * W * D, C, device=device, requires_grad=True)
        layer = ConvEncoderLayer3d(C, 3, 1, C).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x, (H, W, D))
            out = out.sum()
        out.backward()

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        C = 32
        layer = ConvEncoderLayer3d(C, 3, 1, C)

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
        B, C, D, H, W = 1, 128, 16, 16, 16
        x = torch.randn(B, H * W * D, C, device=device)
        layer = ConvEncoderLayer3d(C, 3, 1, C, gate_activation=gate_activation).to(device)

        # Training, non-determinstic
        layer.train()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x, (H, W, D))
            out2 = layer(x, (H, W, D))
            assert not torch.allclose(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = layer(x, (H, W, D))
            out2 = layer(x, (H, W, D))
            assert_close(out1, out2)
