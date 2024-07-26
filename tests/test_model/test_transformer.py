import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.transformer import TransformerDecoderLayer, TransformerEncoderLayer


class TestTransformerEncoderLayer:

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
        layer = TransformerEncoderLayer(D, nhead, D, alibi_lower=lower, alibi_upper=upper)
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
        layer = TransformerEncoderLayer(D, nhead, D).to(device)
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
        layer = TransformerEncoderLayer(D, nhead, D).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(x)
            out = out.sum()
        out.backward()


class TestTransformerDecoderLayer:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
        assert out.shape == q.shape

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, Lq, Dq = 1, 64, 128
        B, Lk, Dk = 1, 128, 32
        q = torch.randn(B, Lq, Dq, device=device, requires_grad=True)
        k = torch.randn(B, Lk, Dk, device=device, requires_grad=True)
        nhead = Dq // 16
        layer = TransformerDecoderLayer(Dq, nhead, Dk).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(q, k)
            out = out.sum()
        out.backward()
        assert q.grad is not None
        assert k.grad is not None
        assert False
