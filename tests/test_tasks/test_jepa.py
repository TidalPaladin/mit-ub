import math
import os
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from deep_helpers.structs import Mode, State
from torch.multiprocessing import spawn  # type: ignore
from torch.testing import assert_close
from torchvision.ops import sigmoid_focal_loss

from mit_ub.tasks.jepa import JEPA, JEPAConfig, compute_siglip_loss, ring_exchange_all


def _run_exchange(rank: int, world_size: int):
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=10))

    try:
        # Create tensor with value equal to rank
        tensor = torch.tensor(float(rank))
        exchanged = torch.zeros_like(tensor)
        total = torch.tensor(0.0)

        # Verify each exchanged tensor matches expected value
        print("Beginning ring exchange")
        for i, exchanged in enumerate(ring_exchange_all(tensor, rank, world_size)):
            print(f"Rank {rank} received {exchanged} on iteration {i}, total now {total}")
            total += exchanged

        expected = torch.arange(world_size, dtype=torch.float32).sum()
        assert_close(total, expected, msg=f"Rank {rank} total {total} does not match expected {expected}")
        assert_close(exchanged, tensor, msg=f"Final exchange {exchanged} does not match initial tensor {tensor}")

    finally:
        dist.destroy_process_group()


@pytest.mark.skip(reason="Prone to deadlocks")
@pytest.mark.parametrize("world_size", [1, 2, 3, 4])
def test_ring_exchange(world_size):
    spawn(_run_exchange, nprocs=world_size, args=(world_size,))


def test_compute_siglip_loss_local():
    torch.random.manual_seed(0)
    B, D = 32, 64
    x1 = torch.randn(B, D)
    x2 = torch.randn(B, D)
    target = torch.eye(B, dtype=torch.float32)
    t = torch.tensor(1.0)
    b = torch.tensor(0.0)
    rank = 0
    world_size = 1
    loss = compute_siglip_loss(x1, x2, target, t, b, rank, world_size, eps=1e-12)
    x1 = F.normalize(x1, dim=-1, eps=1e-12)
    x2 = F.normalize(x2, dim=-1, eps=1e-12)
    expected = F.binary_cross_entropy_with_logits(torch.matmul(x1, x2.T) * t.exp() + b, target, reduction="mean")
    assert_close(loss, expected)


def _compute_siglip_loss(rank: int, world_size: int, expected: float, B: int, D: int):
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12369"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=10))

    try:
        t = torch.tensor(1.0)
        b = torch.tensor(0.0)
        torch.random.manual_seed(rank)
        x1 = torch.randn(B, D)
        x2 = torch.randn(B, D)
        target = torch.eye(B, dtype=torch.float32)
        loss = compute_siglip_loss(x1, x2, target, t, b, rank, world_size)
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss = loss / world_size
        assert_close(loss, loss.new_tensor(expected))

    finally:
        dist.destroy_process_group()


@pytest.mark.skip(reason="Prone to deadlocks")
@pytest.mark.parametrize("world_size", [1, 2, 3])
def test_compute_siglip_loss(world_size):
    t = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # First compute an expected loss locally
    B, D = 32, 64
    x1_list = []
    x2_list = []
    for i in range(world_size):
        torch.random.manual_seed(i)
        x1 = torch.randn(B, D)
        x2 = torch.randn(B, D)
        x1 = F.normalize(x1, dim=-1, eps=1e-12)
        x2 = F.normalize(x2, dim=-1, eps=1e-12)
        x1_list.append(x1)
        x2_list.append(x2)

    x1 = torch.cat(x1_list, 0)
    x2 = torch.cat(x2_list, 0)
    target = torch.eye(x1.shape[0], dtype=torch.float32)
    expected = sigmoid_focal_loss(torch.matmul(x1, x2.T) * t.exp() + b, target, reduction="mean", alpha=-1)

    spawn(
        _compute_siglip_loss,
        nprocs=world_size,
        args=(world_size, expected.float().item(), B, D),
    )


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
    @pytest.fixture(params=["smooth_l1", "cosine"])
    def task(self, optimizer_init, backbone, request):
        config = JEPAConfig()
        config.scale = 1
        config.loss_fn = request.param
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
            "siglip_loss",
        }
        train_keys = {"ema_momentum", "siglip_t", "siglip_b"}

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
            assert_close(param.data, expected.expand_as(param.data), check_device=False)

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

    def test_fit(self, task, cifar10_datamodule, trainer):
        task.weight_decay_final = 4.0
        trainer.fit(task, datamodule=cifar10_datamodule)

    @pytest.fixture
    def task_siglip(self, optimizer_init, backbone):
        config = JEPAConfig()
        config.scale = 1
        config.siglip_weight = 0.5
        return JEPA(backbone, optimizer_init=optimizer_init, jepa_config=config)

    def test_fit_siglip(self, task_siglip, cifar10_datamodule, trainer):
        task_siglip.weight_decay_final = 4.0
        trainer.fit(task_siglip, datamodule=cifar10_datamodule)
