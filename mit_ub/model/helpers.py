import os
from typing import Iterable, Literal, Sized, Tuple, TypeVar, cast, overload


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
