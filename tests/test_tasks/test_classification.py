import pytest
import pytorch_lightning as pl

from mit_ub.tasks.classification import ClassificationTask, JEPAWithClassification
from mit_ub.tasks.jepa import JEPAConfig


class TestClassificationTask:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return ClassificationTask(backbone, num_classes=10, optimizer_init=optimizer_init)

    def test_fit(self, task, cifar10_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)


class TestJEPAWithClassification:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        config = JEPAConfig()
        config.context_scale = 1
        config.target_scale = 1
        return JEPAWithClassification(backbone, num_classes=10, optimizer_init=optimizer_init, jepa_config=config)

    def test_fit(self, task, cifar10_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)
