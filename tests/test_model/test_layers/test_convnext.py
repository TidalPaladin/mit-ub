import pytest
import torch

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
    def test_forward(self, device, dtype):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W)
        layer = ConvNextBlock(C)
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
    def test_backward(self, device, dtype, checkpoint):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, requires_grad=True)
        layer = ConvNextBlock(C)
        layer.checkpoint = checkpoint
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None
