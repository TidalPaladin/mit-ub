import pytest
import pytorch_lightning as pl
import torch

from mit_ub.data.chexpert import LABELS, CheXpert, CheXpertDataModule
from mit_ub.tasks.chexpert import JEPAChexpert


@pytest.fixture
def mock_chexpert(mocker):
    torch.random.manual_seed(0)
    mock_dataset = mocker.MagicMock(spec_set=CheXpert)
    mock_dataset.__getitem__.side_effect = lambda index: {
        # Intentionally set 1-channels so we don't need a custom backbone
        "img": torch.rand(1, 32, 32),
        "finding": torch.randint(-1, 2, (1,)).item(),
        "dense_label": torch.rand(len(LABELS)).round(),
    }
    mock_dataset.__len__.return_value = 100
    mocker.patch("mit_ub.data.chexpert.CheXpert", return_value=mock_dataset)
    return mock_dataset


@pytest.fixture
def chexpert_datamodule(tmp_path, mock_chexpert):
    return CheXpertDataModule(
        root=tmp_path,
        batch_size=4,
        num_workers=0,
    )


class TestJEPAChexpert:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return JEPAChexpert(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, chexpert_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=chexpert_datamodule)
