from typing import Any, Dict, Optional, Tuple, cast

import pytest
import pytorch_lightning as pl
import torch.nn as nn
from torch import Tensor

from mit_ub.tasks.jepa import JEPA
from ssl_tasks.tokens import TokenMask


@pytest.fixture(params=[False, True])
def loss_includes_unmasked(request):
    return request.param


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        return JEPA(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)