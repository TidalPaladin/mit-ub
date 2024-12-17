from typing import Any, cast

import pytest
import torch
from torch.testing import assert_close

from mit_ub.model import BACKBONES
from mit_ub.model.backbone import AdaptiveViT, ConvViT, ViT
from mit_ub.tokens import create_mask


class TestViT:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = model(x)
        assert out.shape[:2] == (1, 128)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward(self, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        nhead = 128 // 16
        model = ViT(3, 128, (16, 16), 3, nhead).to(device)

        with torch.autocast(device_type=device, dtype=dtype):
            out = model(x)
            out = out.sum()
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

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

        x = torch.randn(B, C, H, W, device=device)
        mask = model.create_mask(x, unmasked_ratio=0.25, scale=1)
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
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device)
        nhead = 128 // 16
        model = AdaptiveViT(3, 128, (16, 16), (64, 64), 3, nhead).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = model(x)
        assert out.shape[:2] == (1, 128)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward(self, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        nhead = 128 // 16
        model = AdaptiveViT(3, 128, (16, 16), (64, 64), 3, nhead).to(device)

        with torch.autocast(device_type=device, dtype=dtype):
            out = model(x)
            out = out.sum()
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

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
        D, patch_size, target_size, depth = 128, (16, 16), (64, 64), 3
        nhead = 128 // 16
        model = AdaptiveViT(C, D, patch_size, target_size, depth, nhead).to(device)

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1, device=device)
        x = torch.randn(B, C, H, W, device=device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = model(x, reshape=False)
            out2 = model(x, mask=mask, reshape=False)
        assert out1.shape != out2.shape

    def test_forward_deterministic(self):
        x = torch.randn(1, 3, 224, 224)
        nhead = 128 // 16
        model = AdaptiveViT(3, 128, (16, 16), (64, 64), 3, nhead)

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
        nhead = 128 // 16
        model = AdaptiveViT(C, D, (16, 16), (64, 64), depth, nhead)
        baseline = ViT(C, D, (16, 16), depth, nhead)
        for name, param in baseline.blocks.named_parameters():
            assert param.shape == model.blocks.get_parameter(name).shape

    def test_initialize_like_vit(self):
        C, D, D_kv = 3, 128, 32
        H, W = 256, 256
        depth = 3
        nhead = 128 // 16

        # Set the adaptive model to process fixed tokens at native resolution.
        # Layer scale at 1e-9 should shut off the contribution of the dynamic tokens.
        model = AdaptiveViT(C, D, (16, 16), (256, 256), depth, nhead, layer_scale_adaptive=1e-9)
        baseline = ViT(C, D, (16, 16), depth, nhead)

        # Put the baseline weights into the model
        set_weights = []
        for name, base_param in baseline.named_parameters():
            try:
                model_param = model.get_parameter(name)
                model_param.data = base_param.data
                set_weights.append(name)
            except Exception:
                pass

        model.eval()
        baseline.eval()
        x = torch.randn(1, C, H, W)
        y = model(x)
        y_baseline = baseline(x)
        assert_close(y, y_baseline)

    def test_trivial_token_mask(self):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        D, patch_size, target_size, depth = 128, (16, 16), (64, 64), 3
        nhead = 128 // 16
        model = AdaptiveViT(C, D, patch_size, target_size, depth, nhead)
        model.eval()

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        mask = torch.ones_like(mask).bool()
        x = torch.randn(B, C, H, W)
        out1 = model(x, reshape=False)
        out2 = model(x, mask=mask, reshape=False)
        assert_close(out1, out2)


class TestConvViT:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device)
        nhead = 128 // 16
        model = ConvViT(3, 128, (16, 16), (64, 64), 3, 3, nhead).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = model(x)
        assert out.shape[:2] == (1, 128)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward(self, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        nhead = 128 // 16
        model = ConvViT(3, 128, (16, 16), (64, 64), 3, 3, nhead).to(device)

        with torch.autocast(device_type=device, dtype=dtype):
            out = model(x)
            out = out.sum()
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_forward_deterministic(self):
        x = torch.randn(1, 3, 224, 224)
        nhead = 128 // 16
        model = ConvViT(3, 128, (16, 16), (64, 64), 3, 3, nhead)

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
        model = ConvViT(C, D, (16, 16), (64, 64), depth, depth_adaptive, nhead)
        model2 = ViT(C, D, (16, 16), depth, nhead)
        for p1, p2 in zip(model.blocks.parameters(), model2.blocks.parameters()):
            assert p1.shape == p2.shape


@pytest.mark.ci_skip
@pytest.mark.parametrize("model", BACKBONES.available_keys())
def test_registry(model):
    model = BACKBONES.get(model).instantiate_with_metadata()
    assert model is not None
