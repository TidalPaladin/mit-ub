import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.layers.convnext import ConvNextBlock


class TestConvNext:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1.0])
    @pytest.mark.parametrize("stochastic_depth", [0.0, 0.25])
    def test_forward(self, device, dtype, layer_scale, stochastic_depth):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W)
        layer = ConvNextBlock(C, layer_scale=layer_scale, stochastic_depth=stochastic_depth)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        assert out.shape == (B, H * W, C)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("layer_scale", [None, 1.0])
    def test_backward(self, device, dtype, checkpoint, layer_scale):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, requires_grad=True)
        layer = ConvNextBlock(C, layer_scale=layer_scale)
        layer.checkpoint = checkpoint
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None

    def test_forward_deterministic(self):
        B, C, H, W = 8, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W)
        layer = ConvNextBlock(C, dropout=0.1, stochastic_depth=0.25)

        layer.train()
        out1 = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        out2 = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        assert not torch.allclose(out1, out2)

        layer.eval()
        out1 = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        out2 = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        assert_close(out1, out2)
