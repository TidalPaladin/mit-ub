import pytest
import pytorch_lightning as pl

from mit_ub.tasks.jepa import JEPA


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return JEPA(backbone, optimizer_init=optimizer_init)

    @pytest.mark.parametrize("dist_gather", [False, True])
    def test_fit(self, task, datamodule, logger, dist_gather):
        task.dist_gather = dist_gather
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
