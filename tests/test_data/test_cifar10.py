from typing import Any, Dict

import numpy as np
import pytest
import torch
from torchvision.datasets import FakeData as BaseFakeData

from mit_ub.data.cifar10 import CIFAR10DataModule


class FakeData(BaseFakeData):

    def __getitem__(self, index: int) -> Dict[str, Any]:
        img, target = super().__getitem__(index)
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1)
        return {"img": img, "label": torch.tensor(target)}


@pytest.fixture
def fake_cifar10(mocker):
    ds = FakeData(size=100, num_classes=10, image_size=(3, 32, 32))
    return ds


@pytest.fixture
def fake_datamodule(tmp_path, mocker, fake_cifar10):
    dm = CIFAR10DataModule(tmp_path, batch_size=2, num_workers=0)
    mocker.patch.object(dm, "prepare_data", autospec=True, return_value=None)
    mocker.patch("mit_ub.data.cifar10.CIFAR10", return_value=fake_cifar10)
    return dm


def test_cifar10_data_module_fit(fake_datamodule):
    fake_datamodule.setup("fit")
    assert len(fake_datamodule.train_dataloader()) == 50
    assert len(fake_datamodule.val_dataloader()) == 50
    train_batch = next(iter(fake_datamodule.train_dataloader()))
    val_batch = next(iter(fake_datamodule.val_dataloader()))
    assert train_batch["img"].shape == (2, 3, 32, 32)
    assert val_batch["img"].shape == (2, 3, 32, 32)


def test_cifar10_data_module_test(fake_datamodule):
    fake_datamodule.setup("test")
    assert len(fake_datamodule.test_dataloader()) == 50
    test_batch = next(iter(fake_datamodule.test_dataloader()))
    assert test_batch["img"].shape == (2, 3, 32, 32)
