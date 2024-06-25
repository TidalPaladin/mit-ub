import pytest
import pytorch_lightning as pl

from mit_ub.tasks.density import Density


@pytest.fixture(params=[False, True])
def loss_includes_unmasked(request):
    return request.param


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return Density(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
