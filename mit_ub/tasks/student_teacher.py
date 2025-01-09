from dataclasses import dataclass, field
from typing import Any, Final, Iterator, List

import torch
import torch.nn as nn
from deep_helpers.optim.rsqrt import get_momentum
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce


BUCKET_SIZE_MB: Final = 25


@dataclass
class EMAConfig:
    """
    Configuration for student-teacher EMA updates.

    Momentum scheduling uses the following strategy:
        * Begin at `initial_momentum`
        * Linear warmup over `warmup_steps` to `momentum`
        * Reciprocal square root increase over `timescale` steps to `momentum`
        * Linear cooldown over `cooldown_steps` back to `initial_momentum`

    Args:
        momentum: Momentum value after warmup.
        warmup_steps: Number of steps over which to warm up the momentum.
        peak_steps: Number of steps at the peak.
        cooldown_steps: Number of steps over which to cooldown the momentum.
        stopped_steps: Number of steps after cooldown at which to hold the momentum at `initial_momentum`.
        timescale: Time scale for the EMA update.
        initial_momentum: Initial momentum value.
        sync_interval: Interval (in steps) at which to synchronize the EMA weights across all processes.
            This should not be needed since consistent student weights are ensured by the DDP wrapper.
            However, if for some reason this isn't happening, manual synchronization can be performed
            at a preset interval by setting this value.
    """

    momentum: float = 0.98
    warmup_steps: int = 1000
    peak_steps: int = 0
    cooldown_steps: int = 25000
    stopped_steps: int = 0
    timescale: int = 10000
    initial_momentum: float = 1.0
    sync_interval: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.momentum < 1.0:
            raise ValueError("momentum must be in the range [0, 1]")
        if self.sync_interval is not None and not 0 < self.sync_interval:
            raise ValueError("sync_interval must be positive")
        if not 0 <= self.warmup_steps:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 <= self.peak_steps:
            raise ValueError("peak_steps must be non-negative")
        if not 0 <= self.cooldown_steps:
            raise ValueError("cooldown_steps must be non-negative")
        if not 0 <= self.timescale:
            raise ValueError("timescale must be non-negative")
        if not 0.0 <= self.initial_momentum <= 1.0:
            raise ValueError("initial_momentum must be in the range [0, 1]")
        if not self.initial_momentum >= self.momentum:
            raise ValueError("initial_momentum must be >= momentum")
        if not 0 <= self.stopped_steps:
            raise ValueError("stopped_steps must be non-negative")

    def validate_schedule(self, max_steps: int) -> None:
        linear_steps = self.warmup_steps + self.cooldown_steps + self.stopped_steps + self.peak_steps
        if linear_steps > max_steps:
            raise ValueError(f"Cannot satisfy EMA momentum schedule for {max_steps} steps: {self}")


@dataclass(repr=False)
class ParameterBucket:
    r"""Data structure for storing and synchronizing buckets of parameters across processes."""

    params: List[Tensor] = field(default_factory=list)
    _handle: Any = field(init=False, default=None)
    _flat_params: Tensor | None = field(init=False, default=None)

    def __repr__(self) -> str:
        return f"ParameterBucket(params={len(self)}, size_mb={self.size_mb():.2f})"

    def __len__(self) -> int:
        return len(self.params)

    def __bool__(self) -> bool:
        return bool(self.params)

    def __iter__(self) -> Iterator[Tensor]:
        return iter(self.params)

    def __getitem__(self, index: int) -> Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of bounds for bucket of size {len(self)}")
        return self.params[index]

    def size_mb(self) -> float:
        return sum(p.data.numel() * p.data.element_size() for p in self.params) / 1024 / 1024

    def size_mb_with(self, new: Tensor) -> float:
        return self.size_mb() + (new.numel() * new.element_size()) / 1024 / 1024

    @property
    def device(self) -> torch.device:
        return self.params[0].device if self.params else torch.device("cpu")

    def flat_params(self) -> Tensor:
        return torch.cat([param.data.flatten() for param in self.params])

    def add(self, param: Tensor) -> None:
        if self.params and param.device != self.device:
            raise ValueError(f"Parameter device {param.device} does not match bucket device {self.device}")
        self.params.append(param)

    def all_reduce(self, world_size: int) -> None:
        if not self or self.device.type == "cpu":
            return
        if self._handle is not None:
            raise RuntimeError("Cannot all_reduce twice")
        self._flat_params = self.flat_params().div_(world_size)
        self._handle = all_reduce(self._flat_params, op=ReduceOp.SUM, async_op=True)

    def join(self) -> None:
        if self.device.type == "cpu" or not self:
            return
        if self._handle is None or self._flat_params is None:
            raise RuntimeError("Cannot join without all_reduce")
        self._handle.wait()
        self._handle = None

        # Restore parameters from flattened tensor
        offset = 0
        for param in self.params:
            numel = param.data.numel()
            param.data.copy_(self._flat_params[offset : offset + numel].reshape_as(param))
            offset += numel
        self._flat_params = None


def get_ema_momentum(
    max_steps: int,
    global_step: int,
    ema_config: EMAConfig,
) -> float:
    r"""Get the momentum for the EMA update based on the current step.

    Args:
        max_steps: Maximum number of steps in the training loop.
        global_step: Current step in the training loop.
        ema_config: Configuration for the EMA update.

    Returns:
        The momentum for the EMA update.
    """
    # Shortcut in the stopped stage
    if max_steps - global_step < (stopped_steps := ema_config.stopped_steps):
        return ema_config.initial_momentum

    # Effective total schedule length accounting for stopped stage duration
    effective_max_steps = max_steps - stopped_steps
    assert effective_max_steps > 0, "Effective max steps must be positive"

    momentum = get_momentum(
        global_step,
        ema_config.momentum,
        ema_config.warmup_steps,
        ema_config.cooldown_steps,
        effective_max_steps,
        ema_config.timescale,
        ema_config.initial_momentum,
        ema_config.peak_steps,
    )
    assert 0 <= momentum <= 1.0
    return momentum


def _update(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
) -> None:
    assert momentum < 1.0, f"Momentum must be less than 1.0, got {momentum}"
    weight = 1 - momentum
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        # Update local teacher weights with EMA
        teacher_param.lerp_(student_param, weight)


def _update_and_sync(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
    world_size: int,
) -> None:
    assert momentum < 1.0, f"Momentum must be less than 1.0, got {momentum}"
    assert world_size > 1, "World size must be greater than 1"
    weight = 1 - momentum

    buckets: List[ParameterBucket] = [ParameterBucket()]
    current_bucket = buckets[0]
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        # Update local teacher weights with EMA
        teacher_param.lerp_(student_param, weight)

        # If adding this parameter would exceed bucket size, reduce current bucket and start a new one
        if current_bucket.size_mb_with(teacher_param) > BUCKET_SIZE_MB and len(current_bucket) > 0:
            current_bucket.all_reduce(world_size)
            current_bucket = ParameterBucket()
            buckets.append(current_bucket)

        current_bucket.add(teacher_param)

    if current_bucket:
        current_bucket.all_reduce(world_size)

    # Collect the reduction results
    for bucket in buckets:
        bucket.join()


@torch.no_grad()
def update_teacher(
    student: nn.Module,
    teacher: nn.Module,
    momentum: float,
    batch_idx: int,
    global_step: int,
    accumulate_grad_batches: int,
    world_size: int,
    sync_interval: int | None,
) -> None:
    """Update the Exponential Moving Average (EMA) of the teacher parameters.

    This function accounts for distributed training and gradient accumulation.
    Updates to local weights are only performed on the batch for which a gradient accumulation
    and weight update occurs. Weight synchronization is performed at the specified interval,
    and follows the strategy used by DDP of bucketing parameters into chunks and interleaving
    computation with reduction.

    Args:
        student: The student model.
        teacher: The teacher model.
        momentum: The momentum for the EMA update.
        batch_idx: The current batch index.
        global_step: The current global step.
        accumulate_grad_batches: The number of batches to accumulate gradients over.
        world_size: The number of processes in the distributed training environment.
        sync_interval: The interval at which to synchronize the EMA weights across processes.
    """
    # Shortcut if momentum=1.0 (no update)
    if momentum == 1.0:
        return

    # Determine what to do
    is_distributed = world_size > 1
    is_sync_step = sync_interval is not None and (global_step + 1) % sync_interval == 0
    is_update_batch_idx = (batch_idx + 1) % accumulate_grad_batches == 0
    if is_distributed and is_sync_step and is_update_batch_idx:
        # Local params need update and weights need to be synced
        _update_and_sync(student, teacher, momentum, world_size)
    elif is_update_batch_idx:
        # Only local params need update
        _update(student, teacher, momentum)
    else:
        return


@torch.no_grad()
def synchronize_teacher(teacher: nn.Module, world_size: int) -> None:
    if world_size == 1:
        return

    buckets: List[ParameterBucket] = [ParameterBucket()]
    current_bucket = buckets[0]
    for teacher_param in teacher.parameters():
        if current_bucket.size_mb_with(teacher_param) > BUCKET_SIZE_MB and len(current_bucket) > 0:
            current_bucket.all_reduce(world_size)
            current_bucket = ParameterBucket()
            buckets.append(current_bucket)
        current_bucket.add(teacher_param)
    current_bucket.all_reduce(world_size)

    # Collect the reduction results
    for bucket in buckets:
        bucket.join()
