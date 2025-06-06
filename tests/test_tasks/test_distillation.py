import pytest
import torch
from deep_helpers.structs import Mode, State

from mit_ub.tasks.distillation import Distillation, DistillationConfig


@pytest.mark.skip(reason="Broken")
class TestDistillation:

    @pytest.fixture(params=["vit->conv", "vit->vit"])
    def task(self, request, tmp_path, vit_distillation, convnext_distillation, optimizer_init):
        if request.param == "vit->conv":
            student_config = convnext_distillation
            teacher_config = vit_distillation
            distillation_config = DistillationConfig(
                student_pool_type="avg",
                teacher_pool_type=None,
                teacher_resolution=(16, 16),
            )
        elif request.param == "vit->vit":
            student_config = vit_distillation
            teacher_config = vit_distillation
            distillation_config = DistillationConfig(
                student_pool_type=None,
                teacher_pool_type=None,
            )
        else:
            raise ValueError(f"Unsupported task: {request.param}")

        teacher_checkpoint = tmp_path / "teacher.pth"
        model = teacher_config.instantiate()
        torch.save(model.state_dict(), teacher_checkpoint)

        return Distillation(
            student_config, teacher_config, teacher_checkpoint, distillation_config, optimizer_init=optimizer_init
        )

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
        base_keys = {"distill_loss", "distill_loss_cls"}
        train_keys = set()

        if state.mode == Mode.TRAIN:
            assert set(metrics.keys()) == base_keys | train_keys
        else:
            assert set(metrics.keys()) == base_keys

    def test_fit(self, task, cifar10_datamodule, trainer):
        trainer.fit(task, datamodule=cifar10_datamodule)
