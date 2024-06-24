import pytest
import pytorch_lightning as pl

from mit_ub.tasks.view_pos import JEPAWithViewPosition


class TestJEPAWithViewPosition:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return JEPAWithViewPosition(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
