import sys
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Final, TypeVar, cast

import triton
from triton.compiler import CompiledKernel


# In the case of K=16 we will perform the following operation in each tensor core
# (16x16) * (16x8) = (16x8)
# BFloat16 will only support a FP32 accumulator
TENSOR_CORE_K: Final = 16


@dataclass
class IsBlockMultiple:
    r"""Heuristic to determine if a dimension is a multiple of a block dimension.

    Args:
        dim: Input dimension name
        block_dim: Block dimension name
        override_val: If set the heuristic will always return this value

    Returns:
        True if the dimension is a multiple of the block dimension, False otherwise.
        If the `override_val` is set, it will be returned instead.
    """

    dim: str
    block_dim: str
    override_val: bool | None = None

    def __call__(self, args: Dict[str, Any]) -> bool:
        if self.override_val is not None:
            return self.override_val
        return args[self.dim] % args[self.block_dim] == 0


@dataclass
class PowerOfTwoHeuristic:
    r"""Heuristic to select the next power of two for a given dimension.

    Args:
        dim: Input dimension name
        min_val: Minimum value for the output
        max_val: Maximum value for the output
        previous: If True, the previous power of two is returned if the next one is greater than the input.

    Returns:
        The next power of two for the given dimension.
    """

    dim: str
    min_val: int = 1
    max_val: int = sys.maxsize
    previous: bool = False

    def __call__(self, args: Dict[str, Any]) -> int:
        dim = args[self.dim]
        pow_2 = triton.next_power_of_2(dim)
        if self.previous and pow_2 > dim:
            pow_2 //= 2
        return max(self.min_val, min(self.max_val, pow_2))


@dataclass
class DivisorHeuristic:
    r"""Heuristic to select the largest power of two that is a divisor of a given dimension.

    Args:
        dim: Input dimension name
        min_val: Minimum value for the output
        max_val: Maximum value for the output
        error_on_non_divisor: If True, an error is raised if the dimension is not a power of two.

    Returns:
        The next power of two for the given dimension.

    Example:
        >>> DivisorHeuristic("dim", 16, 64)({"dim": 128})
        64
        >>> DivisorHeuristic("dim", 16, 64)({"dim": 100})
        16
        >>> DivisorHeuristic("dim", 16, 64)({"dim": 32})
        32
    """

    dim: str
    min_val: int = 1
    max_val: int = sys.maxsize
    error_on_non_divisor: bool = False

    def __call__(self, args: Dict[str, Any]) -> int:
        dim = args[self.dim]
        largest_divisor_pow_2 = self.min_val
        while dim % (largest_divisor_pow_2 * 2) == 0:
            largest_divisor_pow_2 *= 2

        result = min(self.max_val, largest_divisor_pow_2)
        if self.error_on_non_divisor and dim % result != 0:
            raise ValueError(
                f"Cannot find a divisor for {self.dim} of size {dim} within the range "
                f"[{self.min_val}, {self.max_val}] that is a power of two. "
            )

        return result


T = TypeVar("T")


def spill_warning(func: triton.JITFunction[T], limit: int = 0) -> Callable[..., T]:
    r"""Wrapper to emit a warning if the compiled kernel spills registers.

    This should not be used as a decorator. See the example below.

    Args:
        func: JIT function to wrap
        limit: Maximum number of spills before emitting a warning

    Returns:
        Wrapped JIT function

    Example:
        >>> @triton.jit
        >>> def _my_kernel(x_ptr, y_ptr):
        >>>     ...
        >>> # Wrap the JIT function with grid specialization
        >>> _my_kernel = spill_warning(_my_kernel[(1,)], limit=10)(x, y)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        compiled: CompiledKernel = func(*args, **kwargs)
        if compiled.n_spills > limit:
            warnings.warn(f"{compiled.fn} spilled {compiled.n_spills} times using {compiled.n_regs} registers")

    return cast(Callable[..., T], wrapper)
