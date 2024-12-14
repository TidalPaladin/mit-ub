from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
from deep_helpers.tokens import create_mask, apply_mask
from torch.testing import assert_close
from einops import rearrange

from mit_ub.model import BACKBONES
from mit_ub.model.backbone import AdaptiveViT, ViT, add_masked

def test_add_masked_trivial():
    torch.random.manual_seed(0)
    B, D, H1, W1 = 2, 32, 16, 16
    H2, W2 = 8, 8

    x1 = rearrange(torch.randn(B, D, H1, W1), "b d h w -> b (h w) d")
    x2 = rearrange(torch.randn(B, D, H2, W2), "b d h w -> b (h w) d")

    mask1 = torch.ones(B, H1 * W1).bool()
    mask2 = torch.ones(B, H2 * W2).bool()

    actual = add_masked(x1, x2, mask1, mask2, (H1, W1), (H2, W2))
    expected = F.interpolate(
        rearrange(x2, "b (h w) d -> b d h w", h=H2, w=W2),
        size=(H1, W1),
        mode="nearest",
    )
    expected = rearrange(expected, "b d h w -> b (h w) d")
    expected = expected + x1
    assert_close(actual, expected)


def test_add_masked():
    torch.random.manual_seed(1)
    B, D, H1, W1 = 2, 32, 16, 16
    H2, W2 = 8, 8

    x1 = rearrange(torch.randn(B, D, H1, W1), "b d h w -> b (h w) d")
    x2 = rearrange(torch.randn(B, D, H2, W2), "b d h w -> b (h w) d")

    mask1 = create_mask((H1, W1), batch_size=B, mask_ratio=0.5, scale=1)
    mask2 = create_mask((H2, W2), batch_size=B, mask_ratio=0.5, scale=1)

    x1 = apply_mask(mask1, x1)
    x2 = apply_mask(mask2, x2)

    result = add_masked(x1, x2, mask1, mask2, (H1, W1), (H2, W2))
    assert result.shape == x1.shape


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

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_token_mask(self, device):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 224, 224
        D, patch_size, depth = 128, (16, 16), 3

        nhead = 128 // 16
        model = ViT(C, D, patch_size, depth, nhead).to(device)

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        x = torch.randn(B, C, H, W, device=device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = model(x, reshape=False)
            out2 = model(x, mask=mask, reshape=False)
        assert out1.shape != out2.shape

    def test_forward_deterministic(self):
        x = torch.randn(1, 3, 224, 224)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead)

        model.train()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1 = model(x)
            out2 = model(x)
        assert not torch.allclose(out1, out2)

        model.eval()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_trivial_token_mask(self):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 224, 224
        D, patch_size, depth = 128, (16, 16), 3

        nhead = 128 // 16
        model = ViT(C, D, patch_size, depth, nhead)
        model.eval()

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        mask = torch.ones_like(mask).bool()
        x = torch.randn(B, C, H, W)
        out1 = model(x, reshape=False)
        out2 = model(x, mask=mask, reshape=False)
        assert_close(out1, out2)


class TestAdaptiveViT:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dim,latent_dim,nhead", [
        (128, None, 8), 
        (64, 128, 4),
        (64, 256, 4),
    ])
    def test_forward(self, device, dim, latent_dim, nhead):
        x = torch.randn(1, 3, 224, 224, device=device)
        model = AdaptiveViT(3, dim, (16, 16), (4, 4), 3, 3, nhead, latent_dim=latent_dim).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
            out = model(x)
        assert out.shape[:2] == (1, dim)

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
        model = AdaptiveViT(3, 128, (16, 16), (4, 4), 3, 3, nhead).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = model(x)
            out = out.sum()
        out.sum().backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_token_mask(self, device):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        D, patch_size, target_size, depth = 128, (16, 16), (4, 4), 3
        nhead = 128 // 16
        model = AdaptiveViT(C, D, patch_size, target_size, depth, depth, nhead).to(device)

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        x = torch.randn(B, C, H, W, device=device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = model(x, reshape=False)
            out2 = model(x, mask=mask, reshape=False)
        assert out1.shape != out2.shape

    def test_forward_deterministic(self):
        x = torch.randn(1, 3, 224, 224)
        nhead = 128 // 16
        model = AdaptiveViT(3, 128, (16, 16), (4, 4), 3, 3, nhead)

        model.train()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1 = model(x)
            out2 = model(x)
        assert not torch.allclose(out1, out2)

        model.eval()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_load_from_vit(self):
        C, D, D_kv = 3, 128, 32
        depth = 3
        depth_adaptive = 2
        nhead = 128 // 16
        model = AdaptiveViT(C, D, (16, 16), (4, 4), depth, depth_adaptive, nhead)
        model2 = ViT(C, D, (16, 16), depth, nhead)
        for p1, p2 in zip(model.blocks.parameters(), model2.blocks.parameters()):
            assert p1.shape == p2.shape

    def test_trivial_token_mask(self):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        D, patch_size, target_size, depth = 128, (16, 16), (4, 4), 3
        nhead = 128 // 16
        model = AdaptiveViT(C, D, patch_size, target_size, depth, depth, nhead)
        model.eval()

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        mask = torch.ones_like(mask).bool()
        x = torch.randn(B, C, H, W)
        out1 = model(x, reshape=False)
        out2 = model(x, mask=mask, reshape=False)
        assert_close(out1, out2)

    def test_layer_scale(self):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        D, patch_size, target_size, depth = 128, (16, 16), (16, 16), 3
        nhead = 128 // 16
        model = AdaptiveViT(C, D, patch_size, target_size, depth, depth, nhead, dynamic_layer_scale=0.0)
        baseline = ViT(C, D, patch_size, depth, nhead)
        model.stem = baseline.stem
        model.blocks = baseline.blocks
        model.embedding_norm = baseline.embedding_norm

        model.eval()
        baseline.eval()

        x = torch.randn(B, C, H, W)
        out1 = baseline(x, reshape=False)
        out2 = model(x, reshape=False)
        assert_close(out1, out2)


@pytest.mark.ci_skip
@pytest.mark.parametrize("model", BACKBONES.available_keys())
def test_registry(model):
    model = BACKBONES.get(model).instantiate_with_metadata()
    assert model is not None
