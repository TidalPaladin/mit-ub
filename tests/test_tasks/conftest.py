import pytest
import pytorch_lightning as pl
import torch
from convnext import ConvNextConfig
from vit import ViTConfig


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
        patch_size=(2, 2),
        depth=2,
        hidden_size=dim,
        ffn_hidden_size=dim,
        num_attention_heads=dim // 32,
        hidden_dropout=0.1,
        attention_dropout=0.1,
    )
    return config


@pytest.fixture(scope="session")
def convnext_dummy():
    config = ConvNextConfig(
        in_channels=1,
        depths=(2, 2, 2),
        hidden_sizes=(32, 48, 64),
        ffn_hidden_sizes=(128, 192, 256),
        patch_size=(2, 2),
        kernel_size=(3, 3),
    )
    return config


@pytest.fixture(scope="session")
def vit_distillation():
    dim = 128
    config = ViTConfig(
        in_channels=1,
        patch_size=(4, 4),
        depth=2,
        hidden_size=dim,
        ffn_hidden_size=dim,
        num_attention_heads=dim // 32,
        hidden_dropout=0.1,
        attention_dropout=0.1,
    )
    return config


@pytest.fixture(scope="session")
def convnext_distillation():
    config = ConvNextConfig(
        in_channels=1,
        depths=(2, 2, 2),
        hidden_sizes=(32, 48, 64),
        ffn_hidden_sizes=(128, 192, 256),
        patch_size=(2, 2),
        kernel_size=(3, 3),
    )
    return config


@pytest.fixture(scope="session", params=["vit-dummy"])
def backbone(request):
    if request.param == "vit-dummy":
        return request.getfixturevalue("vit_dummy")
    else:
        pytest.fail(f"Unsupported backbone: {request.param}")


@pytest.fixture
def trainer(logger):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return pl.Trainer(
        accelerator="cuda",
        devices=1,
        fast_dev_run=True,
        logger=logger,
    )
