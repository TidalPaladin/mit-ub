import pytest
import pytorch_lightning as pl
import torch

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
