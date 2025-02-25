import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.layers.convnext import ConvNextBlock2d, ConvNextBlock2dCPU


class TestConvNextBlock2d:

    @pytest.mark.cuda
    def test_forward(self):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, device="cuda")
        layer = ConvNextBlock2d(C, 4 * C).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        assert out.shape == (B, H * W, C)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, checkpoint):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        layer = ConvNextBlock2d(C, 4 * C).to("cuda")
        layer.checkpoint = checkpoint
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None


class TestConvNextBlock2dCPU:

    def test_forward(self):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, device="cuda")
        layer = ConvNextBlock2dCPU(C, 4 * C).to("cuda")
        out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        assert out.shape == (B, H * W, C)

    def test_backward(self):
        B, C, H, W = 1, 32, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, requires_grad=True)
        layer = ConvNextBlock2dCPU(C, 4 * C)
        out = layer(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        out.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None

    @pytest.mark.parametrize("bias", [True, False])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_weights_compatible(self, bias, normalization, zero_centered_gamma):
        hidden_size = 128
        baseline = ConvNextBlock2d(
            hidden_size,
            hidden_size,
            bias=bias,
            normalization=normalization,
            zero_centered_gamma=zero_centered_gamma,
            device="cpu",
        )
        cpu = ConvNextBlock2dCPU(
            hidden_size, hidden_size, bias=bias, normalization=normalization, zero_centered_gamma=zero_centered_gamma
        )

        for name, param in baseline.named_parameters():
            other_param = cpu.get_parameter(name)
            assert other_param.shape == param.shape
            assert other_param.dtype == param.dtype

    @pytest.mark.cuda
    @pytest.mark.parametrize("bias", [False, True])
    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    @pytest.mark.parametrize("zero_centered_gamma", [False, True])
    def test_forward_equal(self, bias, normalization, zero_centered_gamma):
        hidden_size = 128
        baseline = ConvNextBlock2d(
            hidden_size, hidden_size, bias=bias, normalization=normalization, zero_centered_gamma=zero_centered_gamma
        ).cuda()
        cpu = ConvNextBlock2dCPU(
            hidden_size, hidden_size, bias=bias, normalization=normalization, zero_centered_gamma=zero_centered_gamma
        ).cpu()

        for name, param in baseline.named_parameters():
            cpu_param = cpu.get_parameter(name)
            cpu_param.data = param.data.cpu()

        B, C, H, W = 1, hidden_size, 64, 64
        torch.random.manual_seed(42)
        x = torch.randn(B, C, H, W, device="cuda")

        y_baseline = baseline(x.movedim(1, -1).view(B, -1, C), size=(H, W))
        y_cpu = cpu(x.cpu().movedim(1, -1).view(B, -1, C), size=(H, W))

        assert_close(y_baseline, y_cpu, check_device=False, atol=1e-3, rtol=0)
