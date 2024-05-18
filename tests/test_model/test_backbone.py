import pytest
import torch

from mit_ub.model.backbone import TransformerBlock, ViT


class TestTransformerBlock:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, L, D = 1, 128, 128
        x = torch.randn(B, L, D, device=device)
        nhead = D // 16
        layer = TransformerBlock(D, nhead, D).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
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
        layer = TransformerBlock(D, nhead, D).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
            out = out.sum()
        out.backward()


class TestViT:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        x = torch.randn(1, 3, 224, 224, device=device)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = model(x)
        assert out.shape[:2] == (1, 128)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = model(x)
            out = out.sum()
        out.sum().backward()
