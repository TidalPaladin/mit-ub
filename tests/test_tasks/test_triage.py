import pytest
import pytorch_lightning as pl

from mit_ub.tasks.triage import BreastTriage


class TestBreastTriage:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return BreastTriage(backbone, optimizer_init=optimizer_init)

    @pytest.mark.skip(reason="'malignant' key will be renamed soon")
    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
