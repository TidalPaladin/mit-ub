import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.backbone import TransformerBlock, ViT


class TestTransformerBlock:

    @pytest.mark.parametrize(
        "lower,upper,nhead,exp",
        [
            pytest.param(None, 8, 8, None, marks=pytest.mark.xfail(raises=ValueError)),
            pytest.param(8, None, 8, None, marks=pytest.mark.xfail(raises=ValueError)),
            (None, None, 8, None),
            (0, 8, 8, torch.tensor([-1.0000, -0.4529, -0.2051, -0.0929, -0.0421, -0.0190, -0.0086, -0.0039])),
            (0, 8, 4, torch.tensor([-1.0000, -0.1575, -0.0248, -0.0039])),
            (-2, 0, 3, torch.tensor([-4.0000, -2.0000, -1.0000])),
        ],
    )
    def test_alibi_slopes(self, lower, upper, nhead, exp):
        D = nhead
        layer = TransformerBlock(D, nhead, D, alibi_lower=lower, alibi_upper=upper)
        slopes = layer.alibi
        assert_close(slopes, exp, atol=1e-4, rtol=0)

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

    @pytest.mark.parametrize("alibi", [False, True])
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device, alibi):
        x = torch.randn(1, 3, 224, 224, device=device)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead, alibi=alibi).to(device)
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
