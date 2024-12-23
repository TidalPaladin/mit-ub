import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.layers.pos_enc import relative_factorized_position_forward
from mit_ub.model.stem import PatchEmbed2d, PatchEmbed3d


class TestPatchEmbed2d:

    def test_equivalence(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4))
        layer.eval()
        x = torch.randn(B, C, H, W)

        actual = layer(x)

        w = layer.w_in.view(D_model, 4, 4, C).movedim(-1, 1)
        b = layer.b_in
        expected = F.conv2d(x, w, b, stride=(4, 4))
        expected = expected.view(B, D_model, -1).movedim(1, 2)
        expected = F.layer_norm(expected, expected.shape[-1:], weight=layer.w_norm, bias=layer.b_norm)
        pos = relative_factorized_position_forward(
            (H // 4, W // 4),
            layer.pos_enc.w_in,
            layer.pos_enc.b_in,
            layer.pos_enc.w_out,
            layer.pos_enc.b_out,
            layer.pos_enc.w_norm,
            layer.pos_enc.b_norm,
            layer.pos_enc.activation,
        )
        expected += pos
        assert_close(actual, expected)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to(device)
        x = torch.randn(B, C, H, W, device=device)
        y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to(device)
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None

    def test_reset_parameters(self):
        C = 3
        D_model = 64
        model = PatchEmbed2d(C, D_model, (4, 4))

        weights_original = {name: param.clone() for name, param in model.named_parameters()}
        model.reset_parameters()
        weights_reset = {name: param for name, param in model.named_parameters()}

        for name, param in weights_original.items():
            # Ignore constant weights or biases
            if (param == 0).all() or (param == 1).all():
                continue
            assert not torch.allclose(param, weights_reset[name], equal_nan=True)

    def test_extra_repr(self):
        layer = PatchEmbed2d(3, 64, (4, 4))
        result = str(layer)
        exp = "PatchEmbed2d(\n  in=3, embed=64, patch_size=(4, 4)\n  (pos_enc): RelativeFactorizedPosition(in=2, hidden=256, out=64, dropout=0.0, act=relu2)\n)"
        assert result == exp


class TestPatchEmbed3d:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, C, D, H, W = 2, 3, 64, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4)).to(device)
        x = torch.randn(B, C, D, H, W, device=device)
        y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4, D // 4)), D_model)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        B, C, D, H, W = 2, 3, 64, 64, 64
        D_model = 64
        layer = PatchEmbed3d(C, D_model, (4, 4, 4)).to(device)
        x = torch.randn(B, C, D, H, W, device=device, requires_grad=True)
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None

    def test_reset_parameters(self):
        C = 3
        D_model = 64
        model = PatchEmbed3d(C, D_model, (4, 4, 4))

        weights_original = {name: param.clone() for name, param in model.named_parameters()}
        model.reset_parameters()
        weights_reset = {name: param for name, param in model.named_parameters()}

        for name, param in weights_original.items():
            # Ignore constant weights or biases
            if (param == 0).all() or (param == 1).all():
                continue
            assert not torch.allclose(param, weights_reset[name], equal_nan=True)

    def test_extra_repr(self):
        layer = PatchEmbed3d(3, 64, (4, 4, 4))
        result = str(layer)
        exp = "PatchEmbed3d(\n  in=3, embed=64, patch_size=(4, 4, 4)\n  (pos_enc): RelativeFactorizedPosition(in=3, hidden=256, out=64, dropout=0.0, act=relu2)\n)"
        assert result == exp
