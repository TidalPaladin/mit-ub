import pytest
import torch
from ssl_tasks.tokens import TokenMask

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
        mask = TokenMask.create(size=(H, W), patch_size=patch_size, batch_size=B, mask_ratio=0.25, scale=1)
        x = torch.randn(B, C, H, W, device=device)
        x = mask.apply_to_input(x, fill_value=float("nan"))

        nhead = 128 // 16
        model = ViT(C, D, patch_size, depth, nhead).to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = model(x)
            out2 = model(x, mask=mask, reshape=False)
        assert out1.isnan().any()
        assert not out2.isnan().any()


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
        size = (H, W)
        nhead = 128 // 16
        model = ViT(C, D, patch_size, depth, nhead).to(device)
        model = AdaptiveViT(C, D, D_kv, patch_size, target_size, depth, depth, nhead).to(device)

        mask_size = model.equivalent_size_2d(*size)
        mask = TokenMask.create(size=mask_size, patch_size=patch_size, batch_size=B, mask_ratio=0.25, scale=1)
        kv_mask = mask.resize(size)
        test_mask = kv_mask.resize(mask_size)
        assert torch.allclose(mask.mask, test_mask.mask)

        x = torch.randn(B, C, H, W, device=device)
        x = kv_mask.apply_to_input(x, fill_value=float("nan"))

        with torch.autocast(device_type=device, dtype=torch.float16):
            out1 = model(x)
            out2 = model(x, mask=mask, reshape=False)
        assert out1.isnan().any()
        assert not out2.isnan().any()


@pytest.mark.ci_skip
@pytest.mark.parametrize("model", BACKBONES.available_keys())
def test_registry(model):
    model = BACKBONES.get(model).instantiate_with_metadata()
    assert model is not None
