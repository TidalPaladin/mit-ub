import pytest
import pytorch_lightning as pl

from mit_ub.tasks.jepa import JEPA


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return JEPA(backbone, optimizer_init=optimizer_init, context_scale=1)

    @pytest.mark.parametrize("dist_gather", [False, True])
    @pytest.mark.parametrize("contrastive_dropout", [None, 0.1])
    def test_fit(self, task, datamodule, logger, dist_gather, contrastive_dropout):
        task.dist_gather = dist_gather
        task.contrastive_dropout = contrastive_dropout
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
