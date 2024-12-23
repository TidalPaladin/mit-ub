import pytest
import pytorch_lightning as pl
import torch

from mit_ub.tasks.distillation import Distillation


class TestDistillation:
    @pytest.fixture
    def task(self, tmp_path, vit_distillation, convnext_distillation, optimizer_init):
        student_config = convnext_distillation
        teacher_config = vit_distillation

        teacher_checkpoint = tmp_path / "teacher.pth"
        model = teacher_config.instantiate()
        torch.save(model.state_dict(), teacher_checkpoint)

        return Distillation(student_config, teacher_config, teacher_checkpoint, optimizer_init=optimizer_init)

    def test_fit(self, task, cifar10_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)
