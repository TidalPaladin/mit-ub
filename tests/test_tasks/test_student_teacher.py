import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.multiprocessing import spawn  # type: ignore
from torch.testing import assert_close

from mit_ub.tasks.student_teacher import EMAConfig, get_ema_momentum, synchronize_teacher, update_teacher


def run_ema_sync(rank, world_size, device):
    torch.random.manual_seed(0)
    teacher = nn.Linear(10, 10).to(device)
    try:
        # Initialize the process group and get rank
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        # Set EMA weights to the rank
        with torch.no_grad():
            for param in teacher.parameters():
                param.fill_(rank)

        # Synchronize EMA weights
        synchronize_teacher(teacher, world_size)

        # EMA weights are averaged
        expected = torch.tensor(sum(range(world_size)) / world_size, device=device)
        for param in teacher.parameters():
            assert_close(param.data, expected.expand_as(param.data))

    # Clean up the process group
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def run_update_and_sync(rank, world_size, device):
    torch.random.manual_seed(0)
    teacher = nn.Linear(10, 10).to(device)
    student = nn.Linear(10, 10).to(device)
    try:
        # Initialize the process group and get rank
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        with torch.no_grad():
            # Set EMA weights to the rank
            for param in teacher.parameters():
                param.fill_(rank)

            # Set student weights to the rank + 1
            for param in student.parameters():
                param.fill_(rank + 1)

        update_teacher(
            student,
            teacher,
            momentum=0.5,
            batch_idx=0,
            global_step=0,
            accumulate_grad_batches=1,
            world_size=world_size,
            sync_interval=1,
        )

        ranks = torch.arange(world_size, device=device).float()
        updated = ranks.lerp(ranks + 1, 0.5)
        expected = updated.mean()
        for param in teacher.parameters():
            assert_close(
                param.data,
                expected.expand_as(param.data),
                msg=f"{param.data} != {expected}",
            )

    # Clean up the process group
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def run_update_and_sync_big_model(rank, world_size, device, exp):
    torch.random.manual_seed(0)
    teacher = nn.Sequential(*[nn.Linear(512, 512) for _ in range(5)]).to(device)
    torch.random.manual_seed(rank)
    student = nn.Sequential(*[nn.Linear(512, 512) for _ in range(5)]).to(device)
    try:
        # Initialize the process group and get rank
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

        update_teacher(
            student,
            teacher,
            momentum=0.95,
            batch_idx=0,
            global_step=0,
            accumulate_grad_batches=1,
            world_size=world_size,
            sync_interval=1,
        )
        assert_close(sum(p.sum() for p in teacher.parameters()), exp)

    # Clean up the process group
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


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
def test_get_ema_momentum(max_steps, current_step, stopped_steps, expected):
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
    actual = get_ema_momentum(
        global_step=current_step,
        max_steps=max_steps,
        ema_config=ema_config,
    )

    assert math.isclose(actual, expected, abs_tol=1e-3)


@pytest.mark.parametrize(
    "current,new,momentum,expected",
    [
        (0.0, 1.0, 0.95, 0.0 * 0.95 + 1.0 * (1 - 0.95)),
        (1.0, 0.0, 0.998, 1.0 * 0.998 + 0.0 * (1 - 0.998)),
    ],
)
def test_update_teacher(current, new, momentum, expected):
    student = nn.Linear(10, 10)
    teacher = nn.Linear(10, 10)

    # Set student and teacher weights
    with torch.no_grad():
        for param in teacher.parameters():
            param.fill_(current)
        for param in student.parameters():
            param.fill_(new)

    # Update EMA
    update_teacher(
        student, teacher, momentum, batch_idx=0, global_step=0, accumulate_grad_batches=1, world_size=1, sync_interval=1
    )
    expected = torch.tensor(expected)
    for param in teacher.parameters():
        assert_close(param.data, expected.expand_as(param.data))


def test_momentum_1_shortcut(mocker):
    student = nn.Linear(10, 10)
    teacher = nn.Linear(10, 10)
    spy = mocker.spy(teacher, "parameters")

    # Early return should mean parameters are not iterated over
    update_teacher(
        student,
        teacher,
        momentum=1.0,
        batch_idx=0,
        global_step=0,
        accumulate_grad_batches=1,
        world_size=1,
        sync_interval=1,
    )
    spy.assert_not_called()

    update_teacher(
        student,
        teacher,
        momentum=0.98,
        batch_idx=0,
        global_step=0,
        accumulate_grad_batches=1,
        world_size=1,
        sync_interval=1,
    )
    spy.assert_called_once()


@pytest.mark.parametrize(
    "interval,accumulate_grad_batches,global_step,batch_idx,should_sync",
    [
        (100, 1, 0, 0, False),
        (100, 1, 49, 49, False),
        (50, 1, 49, 49, True),
        (100, 1, 99, 99, True),
        (100, 5, 0, 0 * 5, False),
        (100, 5, 49, 49 * 5, False),
        (50, 5, 49, 49 * 5 - 1, True),
        (50, 5, 49, 49 * 5, False),
        (50, 5, 49, 49 * 5 + 1, False),
        (100, 5, 99, 99 * 5 - 1, True),
        (100, 5, 99, 99 * 5, False),
        (100, 5, 99, 99 * 5 + 1, False),
    ],
)
def test_sync_interval(
    mocker,
    interval,
    accumulate_grad_batches,
    global_step,
    batch_idx,
    should_sync,
):
    student = nn.Linear(10, 10)
    teacher = nn.Linear(10, 10)
    update_sync = mocker.patch("mit_ub.tasks.student_teacher._update_and_sync")
    sync = mocker.patch("mit_ub.tasks.student_teacher.synchronize_teacher")
    update_teacher(
        student,
        teacher,
        momentum=0.95,
        batch_idx=batch_idx,
        global_step=global_step,
        accumulate_grad_batches=accumulate_grad_batches,
        world_size=2,
        sync_interval=interval,
    )
    if should_sync:
        assert update_sync.call_count == 1 or sync.call_count == 1
    else:
        update_sync.assert_not_called()
        sync.assert_not_called()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_synchronize_teacher(world_size, device):
    if not dist.is_available() and world_size > 1:
        pytest.skip("Distributed training is not available")

    if world_size > 1 and device != "cpu":
        spawn(run_ema_sync, args=(world_size, device), nprocs=world_size, join=True)

    else:
        torch.random.manual_seed(0)
        teacher = nn.Linear(10, 10).to(device)
        synchronize_teacher(teacher, world_size)

        # No update will be made
        expected = torch.tensor(1.0, device=device)
        with torch.no_grad():
            for param in teacher.parameters():
                param.fill_(expected)
        for param in teacher.parameters():
            assert_close(param.data, expected.expand_as(param.data))


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_update_and_sync(world_size, device):
    if not dist.is_available():
        pytest.skip("Distributed training is not available")

    if world_size > 1 and device != "cpu":
        spawn(run_update_and_sync, args=(world_size, device), nprocs=world_size, join=True)

    else:
        torch.random.manual_seed(0)
        teacher = nn.Linear(10, 10).to(device)
        student = nn.Linear(10, 10).to(device)
        with torch.no_grad():
            for param in teacher.parameters():
                param.fill_(1.0)
            for param in student.parameters():
                param.fill_(0.0)

        update_teacher(
            student,
            teacher,
            momentum=0.5,
            batch_idx=0,
            global_step=0,
            accumulate_grad_batches=1,
            world_size=world_size,
            sync_interval=1,
        )

        # No update will be made
        expected = torch.tensor(0.5, device=device)
        for param in teacher.parameters():
            assert_close(param.data, expected.expand_as(param.data))


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_update_and_sync_big_model(device):
    if not dist.is_available():
        pytest.skip("Distributed training is not available")

    # Run manually without distribution
    torch.random.manual_seed(0)
    teacher1 = nn.Sequential(*[nn.Linear(512, 512) for _ in range(5)]).to(device)
    torch.random.manual_seed(0)
    teacher2 = nn.Sequential(*[nn.Linear(512, 512) for _ in range(5)]).to(device)
    torch.random.manual_seed(0)
    student1 = nn.Sequential(*[nn.Linear(512, 512) for _ in range(5)]).to(device)
    torch.random.manual_seed(1)
    student2 = nn.Sequential(*[nn.Linear(512, 512) for _ in range(5)]).to(device)

    update_teacher(
        student1,
        teacher1,
        momentum=0.95,
        batch_idx=0,
        global_step=0,
        accumulate_grad_batches=1,
        world_size=1,
        sync_interval=1,
    )
    update_teacher(
        student2,
        teacher2,
        momentum=0.95,
        batch_idx=0,
        global_step=0,
        accumulate_grad_batches=1,
        world_size=1,
        sync_interval=1,
    )
    total = torch.tensor(0.0, device=device)
    for teacher1_param, teacher2_param in zip(teacher1.parameters(), teacher2.parameters()):
        total += (teacher1_param.sum() + teacher2_param.sum()) / 2
    total = total.detach()

    spawn(run_update_and_sync_big_model, args=(2, device, total), nprocs=2, join=True)
