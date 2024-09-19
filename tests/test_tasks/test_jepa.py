import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.tasks.jepa import JEPA, average_pairwise_cosine_similarity


def test_average_pairwise_cosine_similarity():
    B, L, D = 10, 128, 32
    torch.manual_seed(0)
    x = torch.randn(B, L, D)

    actual = average_pairwise_cosine_similarity(x, 1, 2)
    expected = F.cosine_similarity(x.view(B, L, 1, D), x.view(B, 1, L, D), dim=-1).mean(dim=(1, 2))
    assert_close(expected, actual)


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return JEPA(backbone, optimizer_init=optimizer_init, context_scale=1)

    @pytest.mark.parametrize("dist_gather", [False, True])
    def test_fit(self, task, datamodule, logger, dist_gather):
        task.dist_gather = dist_gather
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
