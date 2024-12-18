import os
from pathlib import Path

import pytest
import torch
from pytorch_lightning.loggers import WandbLogger
from torch._dynamo import config
from torchvision.datasets import CIFAR10

from mit_ub.data import CIFAR10DataModule


def cuda_available():
    r"""Checks if CUDA is available and device is ready"""
    if not torch.cuda.is_available():
        return False

    capability = torch.cuda.get_device_capability()
    arch_list = torch.cuda.get_arch_list()
    if isinstance(capability, tuple):
        capability = f"sm_{''.join(str(x) for x in capability)}"

    if capability not in arch_list:
        return False

    return True


def handle_cuda_mark(item):  # pragma: no cover
    has_cuda_mark = any(item.iter_markers(name="cuda"))
    if has_cuda_mark and not cuda_available():
        import pytest

        pytest.skip("Test requires CUDA and device is not ready")


def pytest_runtest_setup(item):
    handle_cuda_mark(item)


@pytest.fixture(scope="session")
def _logger(tmpdir_factory):
    tmp_path = Path(tmpdir_factory.mktemp("wandb_logs"))
    logger = WandbLogger(name="test", save_dir=tmp_path, offline=True, anonymous=True)
    return logger


@pytest.fixture
def logger(_logger, mocker):
    logger = _logger
    logger.experiment.log = mocker.spy(logger.experiment, "log")
    return logger


@pytest.fixture(scope="session", autouse=True)
def triton_cache(tmpdir_factory):
    # Uses a fresh temporary cache directory for each test
    root = Path(tmpdir_factory.mktemp("triton"))
    path = root / ".triton"
    path.mkdir()
    os.environ["TRITON_CACHE_DIR"] = str(path)
    return path


@pytest.fixture(autouse=True, scope="session")
def no_autotune():
    from triton.runtime import Autotuner

    Autotuner.prune_configs = lambda self, *args, **kwargs: [self.configs[0]]


@pytest.fixture
def mock_cifar10(mocker):
    torch.random.manual_seed(0)
    mock_dataset = mocker.MagicMock(spec_set=CIFAR10)
    mock_dataset.__getitem__.side_effect = lambda index: {
        # Intentionally set 1-channels so we don't need a custom backbone
        "img": torch.rand(1, 32, 32),
        "label": torch.randint(0, 10, (1,)).item(),
    }
    mock_dataset.__len__.return_value = 100
    mocker.patch("torchvision.datasets.CIFAR10", return_value=mock_dataset)
    mocker.patch("mit_ub.data.cifar10.CIFAR10", return_value=mock_dataset)
    return mock_dataset


@pytest.fixture
def cifar10_datamodule(tmp_path, mock_cifar10):
    return CIFAR10DataModule(
        root=tmp_path,
        batch_size=4,
        num_workers=0,
    )


@pytest.fixture(scope="session", autouse=True)
def torch_compile_cache_size():
    config.cache_size_limit = 64
