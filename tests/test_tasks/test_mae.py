import pytest

from mit_ub.tasks.mae import MAE


class TestMAE:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return MAE(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, trainer):
        trainer.fit(task, datamodule=datamodule)

    def test_predict(self, task, datamodule, trainer):
        trainer.predict(task, datamodule=datamodule)
