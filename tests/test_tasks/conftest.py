import pytest
from torch import Tensor
from mit_ub.model import BACKBONES, ViT


@pytest.fixture
def optimizer_init():
    return {
        "class_path": "torch.optim.Adam",
        "init_args": {"lr": 1e-3},
    }


@pytest.fixture(scope="session")
def backbone():
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
    )
    return "vit-dummy"

