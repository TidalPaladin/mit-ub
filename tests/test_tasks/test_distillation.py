import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn

from mit_ub.model import BACKBONES
from mit_ub.tasks.distillation import Distillation


class TestDistillation:
    @pytest.fixture
    def task(self, tmp_path, vit_dummy, convnext_dummy, optimizer_init):
        backbone = vit_dummy
        teacher_checkpoint = tmp_path / "teacher.pth"
        model: nn.Module = BACKBONES.get(backbone).instantiate_with_metadata().fn  # type: ignore
        torch.save(model.state_dict(), teacher_checkpoint)
        return Distillation(convnext_dummy, backbone, teacher_checkpoint, optimizer_init=optimizer_init)

    def test_fit(self, task, cifar10_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)
