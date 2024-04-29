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
