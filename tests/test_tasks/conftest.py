import pytest

from mit_ub.model import BACKBONES, AdaptiveViT, ViT


@pytest.fixture
def optimizer_init():
    return {
        "class_path": "torch.optim.Adam",
        "init_args": {"lr": 1e-3},
    }


@pytest.fixture(scope="session", params=["vit-dummy", "vit-adaptive-dummy"])
def backbone(request):
    dim = 128
    BACKBONES(
        ViT,
        name="vit-dummy",
        in_channels=1,
        dim=dim,
        patch_size=2,
        depth=2,
        nhead=dim // 32,
        dropout=0.1,
        override=True,
    )
    BACKBONES(
        AdaptiveViT,
        name="vit-adaptive-dummy",
        in_channels=1,
        dim=dim,
        kv_dim=dim // 4,
        patch_size=2,
        target_shape=(4, 4),
        decoder_depth=2,
        encoder_depth=2,
        nhead=dim // 32,
        dropout=0.1,
        override=True,
    )
    return request.param
