import pytest

from mit_ub.model import ConvNextConfig, ViTConfig


@pytest.fixture
def optimizer_init():
    return {
        "class_path": "torch.optim.AdamW",
        "init_args": {"lr": 1e-3, "weight_decay": 0.05},
    }


@pytest.fixture(scope="session")
def vit_dummy():
    dim = 128
    config = ViTConfig(
        in_channels=1,
        dim=dim,
        patch_size=(2, 2),
        depth=2,
        nhead=dim // 32,
        dim_feedforward=dim,
        dropout=0.1,
    )
    return config


@pytest.fixture(scope="session")
def convnext_dummy():
    config = ConvNextConfig(
        in_channels=1,
        depths=(2, 2, 2),
        dims=(32, 48, 64),
        dims_feedforward=(128, 192, 256),
        dropout=0.1,
        stochastic_depth=0.1,
        patch_size=2,
    )
    return config


@pytest.fixture(scope="session")
def vit_distillation():
    dim = 128
    config = ViTConfig(
        in_channels=1,
        dim=dim,
        patch_size=(4, 4),
        depth=2,
        nhead=dim // 32,
        dim_feedforward=dim,
        dropout=0.1,
    )
    return config


@pytest.fixture(scope="session")
def convnext_distillation():
    config = ConvNextConfig(
        in_channels=1,
        depths=(2, 2, 2),
        up_depths=(2, 2),
        dims=(32, 48, 64),
        dims_feedforward=(128, 192, 256),
        dropout=0.1,
        stochastic_depth=0.1,
        patch_size=2,
    )
    return config


@pytest.fixture(scope="session", params=["vit-dummy"])
def backbone(request):
    if request.param == "vit-dummy":
        return request.getfixturevalue("vit_dummy")
    else:
        pytest.fail(f"Unsupported backbone: {request.param}")
