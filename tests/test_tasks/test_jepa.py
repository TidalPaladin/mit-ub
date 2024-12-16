import os
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.multiprocessing import spawn  # type: ignore
from torch.testing import assert_close

from mit_ub.tasks.jepa import JEPA, JEPAConfig, average_pairwise_cosine_similarity


@pytest.mark.parametrize(
    "batch, tokens",
    [
        (10, 128),
        (1, 128),
        (10, 1),
        (1, 1),
    ],
)
def test_average_pairwise_cosine_similarity(batch, tokens):
    B, L, D = batch, tokens, 32
    torch.manual_seed(0)
    x = torch.randn(B, L, D)

    actual = average_pairwise_cosine_similarity(x, 1, 2)
    expected = F.cosine_similarity(x.view(B, L, 1, D), x.view(B, 1, L, D), dim=-1).mean(dim=(1, 2))
    assert_close(expected, actual)


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
            for param in task.ema_backbone.parameters():
                param.fill_(rank)

        # Synchronize EMA weights
        task.synchronize_ema_weights()

        # EMA weights are averaged
        expected = torch.tensor(sum(range(world_size)) / world_size)
        for param in task.ema_backbone.parameters():
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
        return JEPA(backbone, optimizer_init=optimizer_init)

    @pytest.mark.parametrize(
        "momentum_schedule,max_steps,current_step,ema_alpha,expected",
        [
            (True, 1000, 0, 0.95, 0.95),
            (True, 1000, 1000, 0.95, 1.0),
            (True, 1000, 500, 0.95, 0.975),
            (False, 1000, 1000, 0.95, 0.95),
        ],
    )
    def test_ema_momentum_step(
        self, mocker, vit_dummy, optimizer_init, momentum_schedule, max_steps, current_step, ema_alpha, expected
    ):
        config = JEPAConfig(ema_alpha=ema_alpha, momentum_schedule=momentum_schedule)
        task = JEPA(vit_dummy, optimizer_init=optimizer_init, jepa_config=config)
        trainer = mocker.MagicMock(spec_set=pl.Trainer)
        trainer.max_steps = max_steps
        trainer.global_step = current_step
        trainer.max_epochs = None
        trainer.current_epoch = None
        task.trainer = trainer

        actual = task.get_ema_momentum()
        assert actual == expected

    @pytest.mark.parametrize(
        "momentum_schedule,max_epochs,current_epoch,ema_alpha,expected",
        [
            (True, 100, 0, 0.95, 0.95),
            (True, 100, 100, 0.95, 1.0),
            (True, 100, 50, 0.95, 0.975),
            (False, 100, 100, 0.95, 0.95),
        ],
    )
    def test_ema_momentum_epoch(
        self, mocker, vit_dummy, optimizer_init, momentum_schedule, max_epochs, current_epoch, ema_alpha, expected
    ):
        config = JEPAConfig(ema_alpha=ema_alpha, momentum_schedule=momentum_schedule)
        task = JEPA(vit_dummy, optimizer_init=optimizer_init, jepa_config=config)
        trainer = mocker.MagicMock(spec_set=pl.Trainer)
        trainer.max_steps = None
        trainer.global_step = None
        trainer.max_epochs = max_epochs
        trainer.current_epoch = current_epoch
        task.trainer = trainer

        actual = task.get_ema_momentum()
        assert actual == expected

    @pytest.mark.parametrize(
        "current,new,momentum,expected",
        [
            (0.0, 1.0, 0.95, 0.0 * 0.95 + 1.0 * (1 - 0.95)),
            (1.0, 0.0, 0.998, 1.0 * 0.998 + 0.0 * (1 - 0.998)),
        ],
    )
    def test_update_ema(self, mocker, vit_dummy, optimizer_init, current, new, momentum, expected):
        config = JEPAConfig(ema_alpha=0.95, momentum_schedule=True)
        task = JEPA(vit_dummy, optimizer_init=optimizer_init, jepa_config=config)
        mocker.patch.object(task, "get_ema_momentum", return_value=momentum)
        trainer = mocker.MagicMock(spec_set=pl.Trainer)
        trainer.world_size = 1
        task.trainer = trainer

        with torch.no_grad():
            # Set EMA backbone weights to current
            for param in task.ema_backbone.parameters():
                param.fill_(current)

            # Set backbone weights to new
            for param in task.backbone.parameters():
                param.fill_(new)

        # Update EMA
        task.update_ema()
        expected = torch.tensor(expected)
        for param in task.ema_backbone.parameters():
            assert_close(param.data, expected.expand_as(param.data))

    @pytest.mark.parametrize("world_size", [1, 2])
    def test_synchronize_ema_weights(self, mocker, optimizer_init, world_size):
        if not dist.is_available():
            pytest.skip("Distributed training is not available")

        # NOTE: We must use a backbone that is globally registered as to be accessible by all processes.
        # We choose vit-cifar10 since it is small.
        if world_size > 1:
            spawn(run_ema_sync, args=(world_size, "vit-cifar10", optimizer_init), nprocs=world_size, join=True)

        else:
            torch.random.manual_seed(0)
            task = JEPA("vit-cifar10", optimizer_init=optimizer_init)
            trainer = mocker.MagicMock(spec_set=pl.Trainer)
            trainer.world_size = world_size
            task.trainer = trainer

            # No update will be made
            expected = torch.tensor(1.0)
            with torch.no_grad():
                for param in task.ema_backbone.parameters():
                    param.fill_(expected)
            task.synchronize_ema_weights()
            for param in task.ema_backbone.parameters():
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
