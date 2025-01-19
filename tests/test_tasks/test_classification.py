import pytest
import pytorch_lightning as pl
import torch
from deep_helpers.structs import Mode, State

from mit_ub.model.layers import has_layer_scale
from mit_ub.tasks.classification import (
    ClassificationConfig,
    ClassificationTask,
    DistillationWithClassification,
    JEPAWithClassification,
)
from mit_ub.tasks.jepa import JEPAConfig


class TestClassificationTask:
    @pytest.fixture(params=["attention", "avg"])
    def task(self, optimizer_init, backbone, request):
        pool_type = request.param
        config = ClassificationConfig(num_classes=10, pool_type=pool_type)
        return ClassificationTask(backbone, classification_config=config, optimizer_init=optimizer_init)

    @pytest.fixture(params=["attention", "avg"])
    def binary_task(self, optimizer_init, backbone, request):
        pool_type = request.param
        config = ClassificationConfig(num_classes=2, pool_type=pool_type)
        return ClassificationTask(backbone, classification_config=config, optimizer_init=optimizer_init)

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
        base_keys = {"ce_loss", "acc", "macro_acc"}
        assert set(metrics.keys()) == base_keys

    @pytest.mark.parametrize(
        "state",
        [
            State(Mode.TRAIN),
            State(Mode.VAL),
            State(Mode.VAL, sanity_checking=True),
            State(Mode.TEST),
        ],
    )
    def test_create_metrics_binary(self, binary_task, state):
        metrics = binary_task.create_metrics(state)
        base_keys = {"bce_loss", "acc", "macro_acc", "auroc"}
        assert set(metrics.keys()) == base_keys

    def test_fit(self, task, cifar10_datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)

    def test_fit_binary(self, binary_task, cifar10_datamodule_binary, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(binary_task, datamodule=cifar10_datamodule_binary)


class TestJEPAWithClassification:
    @pytest.fixture(params=["attention", "avg"])
    def task(self, optimizer_init, backbone, request):
        pool_type = request.param
        config = JEPAConfig()
        config.scale = 1
        classification_config = ClassificationConfig(num_classes=10, pool_type=pool_type)
        return JEPAWithClassification(
            backbone, classification_config=classification_config, optimizer_init=optimizer_init, jepa_config=config
        )

    @pytest.fixture(params=["attention", "avg"])
    def binary_task(self, optimizer_init, backbone, request):
        pool_type = request.param
        config = JEPAConfig()
        config.scale = 1
        classification_config = ClassificationConfig(num_classes=2, pool_type=pool_type)
        return JEPAWithClassification(
            backbone, classification_config=classification_config, optimizer_init=optimizer_init, jepa_config=config
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
        base_keys = {
            "example_sim",
            "micro_token_sim",
            "macro_token_sim",
            "jepa_loss",
            "ce_loss",
            "acc",
            "macro_acc",
            "example_rms",
            "micro_token_rms",
            "macro_token_rms",
            "siglip_loss",
        }
        train_keys = (
            {"layer_scale_mean", "layer_scale_max", "ema_momentum"}
            if has_layer_scale(task.backbone)
            else {"ema_momentum"}
        )
        if state.mode == Mode.TRAIN:
            assert set(metrics.keys()) == base_keys | train_keys
        else:
            assert set(metrics.keys()) == base_keys

    @pytest.mark.parametrize(
        "state",
        [
            State(Mode.TRAIN),
            State(Mode.VAL),
            State(Mode.VAL, sanity_checking=True),
            State(Mode.TEST),
        ],
    )
    def test_create_metrics_binary(self, binary_task, state):
        metrics = binary_task.create_metrics(state)
        base_keys = {
            "example_sim",
            "micro_token_sim",
            "macro_token_sim",
            "jepa_loss",
            "bce_loss",
            "acc",
            "macro_acc",
            "auroc",
            "example_rms",
            "micro_token_rms",
            "macro_token_rms",
            "siglip_loss",
        }
        train_keys = (
            {"layer_scale_mean", "layer_scale_max", "ema_momentum"}
            if has_layer_scale(binary_task.backbone)
            else {"ema_momentum"}
        )
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

    def test_fit_binary(self, binary_task, cifar10_datamodule_binary, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(binary_task, datamodule=cifar10_datamodule_binary)


class TestDistillationWithClassification:
    @pytest.fixture(params=["attention", "avg"])
    def task(self, tmp_path, vit_distillation, convnext_distillation, optimizer_init, request):
        pool_type = request.param
        student_config = convnext_distillation
        teacher_config = vit_distillation

        teacher_checkpoint = tmp_path / "teacher.pth"
        model = teacher_config.instantiate()
        torch.save(model.state_dict(), teacher_checkpoint)

        classification_config = ClassificationConfig(num_classes=10, pool_type=pool_type)

        return DistillationWithClassification(
            student_config,
            teacher_config,
            teacher_checkpoint,
            classification_config=classification_config,
            optimizer_init=optimizer_init,
        )

    @pytest.fixture(params=["attention", "avg"])
    def binary_task(self, tmp_path, vit_distillation, convnext_distillation, optimizer_init, request):
        pool_type = request.param
        student_config = convnext_distillation
        teacher_config = vit_distillation

        teacher_checkpoint = tmp_path / "teacher.pth"
        model = teacher_config.instantiate()
        torch.save(model.state_dict(), teacher_checkpoint)

        classification_config = ClassificationConfig(num_classes=2, pool_type=pool_type)

        return DistillationWithClassification(
            student_config,
            teacher_config,
            teacher_checkpoint,
            classification_config=classification_config,
            optimizer_init=optimizer_init,
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
        base_keys = {"distill_loss", "ce_loss", "acc", "macro_acc"}
        train_keys = {"layer_scale_mean", "layer_scale_max"} if has_layer_scale(task.backbone) else set()
        if state.mode == Mode.TRAIN:
            assert set(metrics.keys()) == base_keys | train_keys
        else:
            assert set(metrics.keys()) == base_keys

    @pytest.mark.parametrize(
        "state",
        [
            State(Mode.TRAIN),
            State(Mode.VAL),
            State(Mode.VAL, sanity_checking=True),
            State(Mode.TEST),
        ],
    )
    def test_create_metrics_binary(self, binary_task, state):
        metrics = binary_task.create_metrics(state)
        base_keys = {"distill_loss", "bce_loss", "acc", "macro_acc", "auroc"}
        train_keys = {"layer_scale_mean", "layer_scale_max"} if has_layer_scale(binary_task.backbone) else set()
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

    def test_fit_binary(self, binary_task, cifar10_datamodule_binary, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(binary_task, datamodule=cifar10_datamodule_binary)
