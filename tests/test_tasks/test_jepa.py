import math
import os
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from deep_helpers.structs import Mode, State
from torch.multiprocessing import spawn  # type: ignore
from torch.testing import assert_close

from mit_ub.model.layers import has_layer_scale
from mit_ub.tasks.jepa import JEPA, EMAConfig, JEPAConfig


def run_ema_sync(rank, world_size, backbone, optimizer_init):
    torch.random.manual_seed(0)
    task = JEPA(backbone, optimizer_init=optimizer_init)
    trainer = MagicMock(spec_set=pl.Trainer)
    trainer.world_size = world_size
    task.trainer = trainer

    trainer.world_size = world_size
    task.trainer = trainer
    try:
        # Initialize the process group and get rank
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        # Set EMA weights to the rank
        with torch.no_grad():
            for param in task.teacher_backbone.parameters():
                param.fill_(rank)

        # Synchronize EMA weights
        task.synchronize_ema_weights()

        # EMA weights are averaged
        expected = torch.tensor(sum(range(world_size)) / world_size)
        for param in task.teacher_backbone.parameters():
            assert_close(param.data, expected.expand_as(param.data))

    # Clean up the process group
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


class TestJEPA:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        config = JEPAConfig()
        config.scale = 1
        return JEPA(backbone, optimizer_init=optimizer_init, jepa_config=config)

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
            "example_rms",
            "micro_token_rms",
            "macro_token_rms",
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
        "max_steps,current_step,stopped_steps,expected",
        [
            # Initial step starts at initial momentum
            (1000, 0, 0, 1.0),
            # After warmup steps, momentum reaches target momentum
            (1000, 100, 0, 0.98),
            # Before cooldown, momentum is somewhere in between
            (1000, 500, 0, 0.9885),
            # After cooldown steps, momentum reaches target momentum
            (1000, 1000, 0, 1.0),
            # Alternate cooldown with a stopped stage
            (1000, 800, 200, 1.0),
            (1000, 1000, 200, 1.0),
        ],
    )
    def test_ema_momentum_step(
        self, mocker, vit_dummy, optimizer_init, max_steps, current_step, stopped_steps, expected
    ):
        momentum = 0.98
        initial_momentum = 1.0
        ema_config = EMAConfig(
            momentum=momentum,
            initial_momentum=initial_momentum,
            warmup_steps=100,
            stopped_steps=stopped_steps,
            cooldown_steps=200,
            timescale=200,
        )
        config = JEPAConfig(ema_config=ema_config)
        task = JEPA(vit_dummy, optimizer_init=optimizer_init, jepa_config=config)
        trainer = mocker.MagicMock(spec_set=pl.Trainer)
        trainer.max_steps = max_steps
        trainer.global_step = current_step
        trainer.max_epochs = None
        trainer.current_epoch = None
        task.trainer = trainer

        actual = task.get_ema_momentum()
        assert math.isclose(actual, expected, abs_tol=1e-3)

    @pytest.mark.parametrize(
        "current,new,momentum,expected",
        [
            (0.0, 1.0, 0.95, 0.0 * 0.95 + 1.0 * (1 - 0.95)),
            (1.0, 0.0, 0.998, 1.0 * 0.998 + 0.0 * (1 - 0.998)),
        ],
    )
    def test_update_ema(self, mocker, vit_dummy, optimizer_init, current, new, momentum, expected):
        config = JEPAConfig()
        task = JEPA(vit_dummy, optimizer_init=optimizer_init, jepa_config=config)
        mocker.patch.object(task, "get_ema_momentum", return_value=momentum)
        trainer = mocker.MagicMock(spec=pl.Trainer)
        trainer.world_size = 1
        trainer.accumulate_grad_batches = 1
        trainer.global_step = 1
        task.trainer = trainer

        with torch.no_grad():
            # Set EMA backbone weights to current
            for param in task.teacher_backbone.parameters():
                param.fill_(current)

            # Set backbone weights to new
            for param in task.backbone.parameters():
                param.fill_(new)

        # Update EMA
        task.update_ema(0)
        expected = torch.tensor(expected)
        for param in task.teacher_backbone.parameters():
            assert_close(param.data, expected.expand_as(param.data))

    def test_update_weight_decay(self, mocker, task):
        task.parameter_groups = [
            {"params": ("jepa_predictor",), "weight_decay": 1.0},
        ]
        task.jepa_config.weight_decay_final = 0.5
        trainer = mocker.MagicMock(spec_set=pl.Trainer)
        trainer.global_step = 100
        trainer.max_steps = 100
        mocker.patch.object(task, "_trainer", trainer)
        opt = task.configure_optimizers()["optimizer"]
        mocker.patch.object(task, "optimizers", side_effect=lambda: opt)
        task.update_weight_decay()
        # We shouldn't decrease weight decay, only increase it.
        # The first parameter group is the custom one for jepa_predictor.
        assert opt.param_groups[0]["weight_decay"] == 1.0
        assert opt.param_groups[1]["weight_decay"] == 0.5

    def test_fit(self, task, cifar10_datamodule, logger):
        task.weight_decay_final = 4.0
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=cifar10_datamodule)

    @pytest.fixture
    def task_contrastive(self, optimizer_init, backbone):
        config = JEPAConfig()
        config.scale = 1
        config.contrastive_weight = 0.5
        config.contrastive_margin = 0.5
        return JEPA(backbone, optimizer_init=optimizer_init, jepa_config=config)

    def test_fit_contrastive(self, task_contrastive, cifar10_datamodule, logger):
        task_contrastive.weight_decay_final = 4.0
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task_contrastive, datamodule=cifar10_datamodule)
