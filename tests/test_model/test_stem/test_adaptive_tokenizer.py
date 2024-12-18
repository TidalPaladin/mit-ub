import math
from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close

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
        target_size = cast(Any, tuple(t * p for t, p in zip(token_size, patch_size)))
        D = 128
        layer = AdaptiveTokenizer2d(C, D, patch_size, target_size).to(device)
        x = torch.randn(B, C, H, W, device=device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            q, kv = layer(x)
        assert q.shape == (B, math.prod(token_size), D)
        assert kv.shape == (B, math.prod(layer.tokenized_size(x.shape[2:])), D)

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
        target_size = cast(Any, tuple(t * p for t, p in zip(token_size, patch_size)))
        D = 128
        layer = AdaptiveTokenizer2d(C, D, patch_size, target_size).to(device)
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)

        q, kv = layer(x)
        out = q.sum() + kv.sum()
        out.backward()
        assert x.grad is not None

    def test_reset_parameters(self):
        C = 3
        D = 64
        model = AdaptiveTokenizer2d(C, D, (4, 4), (8, 8))

        weights_original = {name: param.clone() for name, param in model.named_parameters()}
        model.reset_parameters()
        weights_reset = {name: param for name, param in model.named_parameters()}

        for name, param in weights_original.items():
            # Ignore constant weights or biases
            if (param == 0).all() or (param == 1).all():
                continue
            assert not torch.allclose(param, weights_reset[name], equal_nan=True)

    def test_forward_deterministic(self):
        B, C, H, W = 1, 3, 64, 64
        D = 128
        patch_size = (4, 4)
        token_size = (8, 8)
        target_size = cast(Any, tuple(t * p for t, p in zip(token_size, patch_size)))
        layer = AdaptiveTokenizer2d(C, D, patch_size, target_size, dropout=0.1)
        x = torch.randn(B, C, H, W)

        # Training, non-determinstic
        layer.train()
        out1_1, out1_2 = layer(x)
        out2_1, out2_2 = layer(x)
        assert not torch.allclose(out1_1, out2_1)
        assert not torch.allclose(out1_2, out2_2)

        # Evaluation, deterministic
        layer.eval()
        out1_1, out1_2 = layer(x)
        out2_1, out2_2 = layer(x)
        assert_close(out1_1, out2_1)
        assert_close(out1_2, out2_2)

    def test_extra_repr(self):
        layer = AdaptiveTokenizer2d(3, 64, (4, 4), (8, 8))
        result = str(layer)
        exp = "AdaptiveTokenizer2d(\n  in=3, embed=64, patch_size=(4, 4), target_shape=(8, 8)\n  (pos_enc): RelativeFactorizedPosition(in=2, hidden=128, out=64, dropout=0.0, act=silu)\n)"
        assert result == exp


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
        D = 128
        target_size = cast(Any, tuple(t * p for t, p in zip(token_size, patch_size)))
        layer = AdaptiveTokenizer3d(C, D, patch_size, target_size).to(device)
        x = torch.randn(B, C, D, H, W, device=device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            q, kv = layer(x)
        assert q.shape == (B, math.prod(token_size), D)
        assert kv.shape == (B, math.prod(layer.tokenized_size(x.shape[2:])), D)

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
        D = 128
        target_size = cast(Any, tuple(t * p for t, p in zip(token_size, patch_size)))
        layer = AdaptiveTokenizer3d(C, D, patch_size, target_size).to(device)
        x = torch.randn(B, C, D, H, W, device=device, requires_grad=True)

        q, kv = layer(x)
        out = q.sum() + kv.sum()
        out.backward()
        assert x.grad is not None

    def test_reset_parameters(self):
        C = 3
        D = 64
        model = AdaptiveTokenizer3d(C, D, (4, 4, 4), (8, 8, 8))

        weights_original = {name: param.clone() for name, param in model.named_parameters()}
        model.reset_parameters()
        weights_reset = {name: param for name, param in model.named_parameters()}

        for name, param in weights_original.items():
            # Ignore constant weights or biases
            if (param == 0).all() or (param == 1).all():
                continue
            assert not torch.allclose(param, weights_reset[name], equal_nan=True)

    def test_forward_deterministic(self):
        B, C, D, H, W = 1, 3, 64, 64, 64
        D = 128
        patch_size = (4, 4, 4)
        token_size = (8, 8, 8)
        target_size = cast(Any, tuple(t * p for t, p in zip(token_size, patch_size)))
        layer = AdaptiveTokenizer3d(C, D, patch_size, target_size, dropout=0.1)
        x = torch.randn(B, C, D, H, W)

        # Training, non-determinstic
        layer.train()
        out1_1, out1_2 = layer(x)
        out2_1, out2_2 = layer(x)
        assert not torch.allclose(out1_1, out2_1)
        assert not torch.allclose(out1_2, out2_2)

        # Evaluation, deterministic
        layer.eval()
        out1_1, out1_2 = layer(x)
        out2_1, out2_2 = layer(x)
        assert_close(out1_1, out2_1)
        assert_close(out1_2, out2_2)

    def test_extra_repr(self):
        layer = AdaptiveTokenizer3d(3, 64, (4, 4, 4), (8, 8, 8))
        result = str(layer)
        exp = "AdaptiveTokenizer3d(\n  in=3, embed=64, patch_size=(4, 4, 4), target_shape=(8, 8, 8)\n  (pos_enc): RelativeFactorizedPosition(in=3, hidden=128, out=64, dropout=0.0, act=silu)\n)"
        assert result == exp
