import torch

from mit_ub.model import ConvNext


class TestConvNext:

    def test_forward(self):
        torch.random.manual_seed(42)
        x = torch.randn(1, 1, 256, 256)
        model = ConvNext(depths=(2, 2, 2), dims=(32, 48, 64))
        out = model(x)
        expected = (1, 64, 16, 16)
        assert out.shape == expected

    def test_backward(self):
        torch.random.manual_seed(42)
        x = torch.randn(1, 1, 256, 256, requires_grad=True)
        model = ConvNext(depths=(2, 2, 2), dims=(32, 48, 64))
        out = model(x)
        out.sum().backward()
