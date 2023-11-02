from typing import Any, Dict, Tuple

import pytest
import pytorch_lightning as pl
import torch
from deep_helpers.testing import handle_cuda_mark
from gpvit import GPViT
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData as TVFakeData
from torchvision.transforms.v2 import ConvertImageDtype, ToImage


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


class FakeData(TVFakeData):
    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, _ = super().__getitem__(index)
        img = ToImage()(img)
        img = ConvertImageDtype(torch.float32)(img)
        return {"img": img}


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, img_size: Tuple[int, int] = (32, 32)):
        super().__init__()
        self.dataset = FakeData(size=100, image_size=(1, *img_size))

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)

    def predict_dataloader(self):
        return DataLoader(self.dataset, batch_size=4)


@pytest.fixture
def backbone():
    return GPViT(
        dim=32,
        num_group_tokens=8,
        img_size=(32, 32),
        patch_size=(4, 4),
        window_size=(2, 2),
        depth=2,
        nhead=4,
        in_channels=1,
    )


@pytest.fixture
def datamodule():
    return DummyDataModule()


@pytest.fixture
def optimizer_init():
    return {
        "class_path": "torch.optim.Adam",
        "init_args": {"lr": 1e-3},
    }


@pytest.fixture
def logger(mocker):
    logger = mocker.MagicMock(name="logger")
    return logger


@pytest.fixture
def trainer(logger):
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        logger=logger,
    )
    return trainer
