import pytest
from torch import Tensor
from mit_ub.model import BACKBONES, ViT
from mit_ub.model.kernels.attention import attention


@pytest.fixture
def mock_attention(mocker):
    def new_attention(q: Tensor, *args, **kwargs):
        return q.clone()
    m = mocker.Mock()
    mocker.patch("mit_ub.model.kernel.attention", new=attention)
    mocker.patch("mit_ub.model.module.attention", new=attention)


@pytest.fixture
def optimizer_init():
    return {
        "class_path": "torch.optim.Adam",
        "init_args": {"lr": 1e-3},
    }


@pytest.fixture
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

