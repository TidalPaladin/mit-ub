import math

import pytest
import torch

from mit_ub.model.stem import PatchEmbed2d, PatchEmbed3d


class TestPatchEmbed2d:

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
