import pytest
import pytorch_lightning as pl
import torch
from deep_helpers.structs import Mode, State

from mit_ub.model.layers.layer_scale import has_layer_scale
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

    @pytest.mark.parametrize(
        "state",
        [
            State(Mode.TRAIN),
            State(Mode.VAL),
            State(Mode.VAL, sanity_checking=True),
            State(Mode.TEST),
        ],
    )
    def test_create_metrics(self, task, state):
        metrics = task.create_metrics(state)
        base_keys = {"distill_loss"}
        train_keys = {"layer_scale_mean", "layer_scale_max"} if has_layer_scale(task.backbone) else set()

        if state.mode == Mode.TRAIN:
            assert set(metrics.keys()) == base_keys | train_keys
        else:
            assert set(metrics.keys()) == base_keys

    def test_fit(self, task, cifar10_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)
