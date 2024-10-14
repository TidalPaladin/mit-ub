from typing import Any, cast

import pytest
import torch
from deep_helpers.tokens import create_mask

from mit_ub.model import BACKBONES
from mit_ub.model.backbone import AdaptiveViT, ViT


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


class TestAdaptiveViT:

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
        model = AdaptiveViT(3, 128, 32, (16, 16), (4, 4), 3, 3, nhead).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=True):
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
        model = AdaptiveViT(3, 128, 32, (16, 16), (4, 4), 3, 3, nhead).to(device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = model(x)
            out = out.sum()
        out.sum().backward()

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
        D, D_kv, patch_size, target_size, depth = 128, 32, (16, 16), (4, 4), 3
        nhead = 128 // 16
        model = AdaptiveViT(C, D, D_kv, patch_size, target_size, depth, depth, nhead).to(device)

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
        model = AdaptiveViT(3, 128, 32, (16, 16), (4, 4), 3, 3, nhead)

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
        model = AdaptiveViT(C, D, D_kv, (16, 16), (4, 4), depth, depth_adaptive, nhead)
        model2 = ViT(C, D, (16, 16), depth, nhead)
        for p1, p2 in zip(model.blocks.parameters(), model2.blocks.parameters()):
            assert p1.shape == p2.shape


@pytest.mark.ci_skip
@pytest.mark.parametrize("model", BACKBONES.available_keys())
def test_registry(model):
    model = BACKBONES.get(model).instantiate_with_metadata()
    assert model is not None
