import os
from pathlib import Path

import pytest
from pytorch_lightning.loggers import WandbLogger
from torch_dicom.preprocessing.datamodule import PreprocessedPNGDataModule
from torch_dicom.testing import MammogramTestFactory


@pytest.fixture(scope="session")
def datamodule(tmpdir_factory):
    root = Path(tmpdir_factory.mktemp("preprocessed"))
    factory = MammogramTestFactory(root, dicom_size=(64, 32), num_studies=3)
    return factory(batch_size=2, num_workers=0, datamodule_class=PreprocessedPNGDataModule)


@pytest.fixture
def logger(tmp_path, mocker):
    tmp_path = Path(tmp_path)
    logger = WandbLogger(name="test", save_dir=tmp_path, offline=True, anonymous=True)
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
def triton_debug():
    os.environ["TRITON_DEBUG"] = str(1)


# @pytest.fixture(autouse=True, scope="session")
# def no_autotune():
#    from triton.runtime import Autotuner
#
#    Autotuner.prune_configs = lambda self, *args, **kwargs: [self.configs[0]]
