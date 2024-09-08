import math

import pytest
import torch

from mit_ub.model.stem.adaptive_tokenizer import AdaptiveTokenizer2d, AdaptiveTokenizer3d


class TestAdaptiveTokenizer2d:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, C, H, W = (2, 3, 64, 64)
        patch_size = (4, 4)
        token_size = (8, 8)
        Dq = 128
        Dkv = 32
        layer = AdaptiveTokenizer2d(C, Dq, Dkv, patch_size, token_size).to(device)
        x = torch.randn(B, C, H, W, device=device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            q, kv = layer(x)
        assert q.shape == (B, math.prod(token_size), Dq)
        assert kv.shape == (B, math.prod(layer.kv_size(x.shape[2:])), Dkv)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, C, H, W = (2, 3, 64, 64)
        patch_size = (4, 4)
        token_size = (8, 8)
        Dq = 128
        Dkv = 32
        layer = AdaptiveTokenizer2d(C, Dq, Dkv, patch_size, token_size).to(device)
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)

        q, kv = layer(x)
        out = q.sum() + kv.sum()
        out.backward()
        assert x.grad is not None


class TestAdaptiveTokenizer3d:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, C, D, H, W = (2, 3, 64, 64, 64)
        patch_size = (4, 4, 4)
        token_size = (8, 8, 8)
        Dq = 128
        Dkv = 32
        layer = AdaptiveTokenizer3d(C, Dq, Dkv, patch_size, token_size).to(device)
        x = torch.randn(B, C, D, H, W, device=device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            q, kv = layer(x)
        assert q.shape == (B, math.prod(token_size), Dq)
        assert kv.shape == (B, math.prod(layer.kv_size(x.shape[2:])), Dkv)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, C, D, H, W = (2, 3, 64, 64, 64)
        patch_size = (4, 4, 4)
        token_size = (8, 8, 8)
        Dq = 128
        Dkv = 32
        layer = AdaptiveTokenizer3d(C, Dq, Dkv, patch_size, token_size).to(device)
        x = torch.randn(B, C, D, H, W, device=device, requires_grad=True)

        q, kv = layer(x)
        out = q.sum() + kv.sum()
        out.backward()
        assert x.grad is not None