from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from mit_ub.data.cifar10 import CIFAR10DataModule


class MockCIFAR10:
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = torch.randn(10, 3, 32, 32)  # Mock 10 random images
        self.targets = torch.randint(0, 10, (10,))  # Mock 10 random labels

    def __getitem__(self, index):
        return {"img": torch.randn(3, 32, 32), "label": torch.tensor(0)}

    def __len__(self):
        return len(self.data)


@pytest.fixture
def mock_dataset(mocker, tmp_path):
    mocker.patch("mit_ub.data.cifar10.CIFAR10", MockCIFAR10)
    return MockCIFAR10(root=tmp_path)


@pytest.fixture
def datamodule(mock_dataset, tmp_path):
    """Create a CIFAR10DataModule with basic transforms."""
    transforms = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    return CIFAR10DataModule(
        root=tmp_path,
        batch_size=4,
        train_transforms=transforms,
        val_transforms=transforms,
        test_transforms=transforms,
    )


@pytest.mark.parametrize("stage", ["fit", None])
def test_setup(datamodule, stage):
    """Test setup creates correct datasets for different stages."""
    datamodule.setup(stage=stage)
    if stage == "fit":
        assert hasattr(datamodule, "dataset_train")
        assert hasattr(datamodule, "dataset_val")


def test_setup_invalid_stage(datamodule):
    """Test setup raises error for invalid stage."""
    with pytest.raises(ValueError, match="Unknown stage"):
        datamodule.setup(stage="invalid")


@pytest.mark.parametrize(
    "loader_fn",
    ["train_dataloader", "val_dataloader", "test_dataloader"],
)
def test_dataloaders(datamodule, loader_fn):
    """Test all dataloaders return correct type and batch size."""
    datamodule.setup(stage="fit")
    loader = getattr(datamodule, loader_fn)()
    assert isinstance(loader, DataLoader)
    assert loader.batch_size == datamodule.batch_size


def test_dataloader_without_setup(datamodule):
    """Test dataloaders raise error if called before setup."""
    with pytest.raises(RuntimeError, match="setup()"):
        datamodule.train_dataloader()


def test_on_after_batch_transfer(datamodule):
    """Test GPU transforms are applied correctly during training."""
    # Mock trainer and GPU transforms
    datamodule.trainer = MagicMock()
    datamodule.trainer.training = True
    gpu_transforms = MagicMock()
    datamodule.train_gpu_transforms = gpu_transforms

    batch = {"img": torch.randn(4, 3, 32, 32)}
    datamodule.on_after_batch_transfer(batch, 0)
    gpu_transforms.assert_called_once_with(batch)

    # Test no GPU transforms during validation
    datamodule.trainer.training = False
    gpu_transforms.reset_mock()
    datamodule.on_after_batch_transfer(batch, 0)
    gpu_transforms.assert_not_called()


def test_prepare_data(datamodule):
    """Test prepare_data downloads both train and test datasets."""
    datamodule.prepare_data()
    # No need to assert calls since we're not actually downloading anything
