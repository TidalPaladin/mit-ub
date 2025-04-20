import math
import os
from typing import Iterable, Literal, Protocol, Sequence, Set, Sized, Tuple, TypeVar, cast, overload, runtime_checkable

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


T = TypeVar("T")
SpatialDims = TypeVar("SpatialDims", bound=Tuple[int, ...])
Dims1D = Tuple[int]
Dims2D = Tuple[int, int]
Dims3D = Tuple[int, int, int]


def compile_is_disabled() -> bool:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE", "1").lower() == "0"


def compile_backend() -> str:
    """Gets state of ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE=0`` to disable ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE_BACKEND", "inductor")


def max_autotune() -> bool:
    """Gets state of "max_autotune" for ``torch.compile`` from environment variable.

    Set ``TORCH_COMPILE_MAX_AUTOTUNE=0`` to disable "max_autotune" for ``torch.compile``.
    """
    return os.getenv("TORCH_COMPILE_MAX_AUTOTUNE", "1").lower() == "1"


@overload
def to_tuple(x: T | Iterable[T], length: Literal[1]) -> Tuple[T]:
    pass


@overload
def to_tuple(x: T | Iterable[T], length: Literal[2]) -> Tuple[T, T]:
    pass


@overload
def to_tuple(x: T | Iterable[T], length: Literal[3]) -> Tuple[T, T, T]:
    pass


def to_tuple(x: T | Iterable[T], length: int) -> Tuple[T, ...]:
    """
    Converts a value or iterable of values to a tuple.

    Args:
        x: The value or iterable of values to convert to a tuple.
        length: The expected length of the tuple.

    Raises:
        * ValueError: If `x` is a non-str iterable and its length does not match `length`.

    Returns:
        The value or iterable of values as a tuple.
    """
    if isinstance(x, Sized) and len(x) == length:
        return tuple(cast(Iterable[T], x))
    elif isinstance(x, Iterable) and not isinstance(x, str):
        result = tuple(x)
        if not len(result) == length:
            raise ValueError(f"Expected an iterable of length {length}, but got {len(result)}.")
        return result
    else:
        return cast(Tuple[T, ...], (x,) * length)


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def tokens_to_grid(x: Tensor, size: Sequence[int]) -> Tensor:
    r"""Convert a channel-last flat token sequence to a channel-first spatial grid.

    Args:
        x: The token sequence to convert to a grid.
        size: The size of the grid to convert to.

    Returns:
        The grid of tokens.

    Raises:
        ValueError: If the token length does not match the grid size.
    """
    _, L, _ = x.shape
    if L != math.prod(size):
        raise ValueError(f"Token length {L} does not match grid size {size}")

    if len(size) == 1:
        return rearrange(x, "b l c -> b c l")
    elif len(size) == 2:
        return rearrange(x, "b (h w) c -> b c h w", h=size[0], w=size[1])
    elif len(size) == 3:
        return rearrange(x, "b (d h w) c -> b c d h w", d=size[0], h=size[1], w=size[2])
    else:
        raise ValueError(f"Invalid size: {size}")


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def grid_to_tokens(x: Tensor) -> Tensor:
    r"""Convert a channel-first spatial grid to a channel-last flat token sequence.

    Args:
        x: The grid to convert to a token sequence.

    Returns:
        The token sequence.
    """
    return rearrange(x, "b c ... -> b (...) c")


@runtime_checkable
class Checkpointable(Protocol):
    checkpoint: bool


def set_checkpointing(module: nn.Module, checkpoint: bool) -> Set[str]:
    r"""Recursively set checkpointing for all modules in the module hierarchy.

    Returns:
        Names of child modules that were set to checkpoint.
    """
    if isinstance(module, Checkpointable):
        module.checkpoint = checkpoint

    names = set()
    for name, child in module.named_modules():
        if isinstance(child, Checkpointable):
            child.checkpoint = checkpoint
            names.add(name)
    return names
