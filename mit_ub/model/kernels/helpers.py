import sys
import warnings
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Final, List, TypeVar, cast

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


T = TypeVar("T", bound=Callable)
WARN_SPILLS: bool = False


def spill_warning(limit: int = 0, enable: bool | None = None) -> Callable[[T], T]:
    r"""Wrapper to emit a warning if the compiled kernel spills registers.

    This should not be used as a decorator. See the example below.

    Args:
        limit: Maximum number of spills before emitting a warning
        enable: If set, it will override the global `WARN_SPILLS` variable

    Returns:
        Wrapped JIT function

    Example:
        >>> @triton.jit
        >>> def _my_kernel(x_ptr, y_ptr):
        >>>     ...
        >>> # Wrap the JIT function with grid specialization
        >>> _my_kernel = spill_warning(limit=10)(_my_kernel[(1,)])(x, y)
    """

    def outer(func: T) -> T:
        @wraps(func)
        def inner(*args, **kwargs):
            compiled = cast(CompiledKernel, func(*args, **kwargs))
            if enable is not None:
                global WARN_SPILLS
                WARN_SPILLS = enable
            if WARN_SPILLS and compiled.n_spills > limit:
                warnings.warn(f"{compiled.fn} spilled {compiled.n_spills} times using {compiled.n_regs} registers")

        return cast(T, inner)

    return cast(Any, outer)


@dataclass
class SelectHeuristic:
    r"""Selects between two heuristics based on a condition.

    Args:
        func: Condition to select the heuristic. Should accept `args` dict as input.
        when_true: Minimum value for the output
        when_false: Maximum value for the output

    Returns:
        Selected heuristic based on the condition.
    """

    func: Callable[[Dict[str, Any]], bool]
    when_true: Callable[[Dict[str, Any]], Any]
    when_false: Callable[[Dict[str, Any]], Any]

    def __call__(self, args: Dict[str, Any]) -> Any:
        return self.when_true(args) if self.func(args) else self.when_false(args)


@dataclass
class PruneConfigs:
    r"""Prune autotuner configs based on a condition.

    Args:
        key: Key to check in the config
        low: Either an integer indicating the minimum allowed value of ``key`` or a string indicating the key to use
            as the minimum value
        high: Either an integer indicating the maximum allowed value of ``key`` or a string indicating the key to use
            as the maximum value
    """

    key: str
    low: int | str = 0
    high: int | str = sys.maxsize

    def __call__(self, configs: List[triton.Config], args: Dict[str, Any]) -> List[triton.Config]:
        out: List[triton.Config] = []
        for config in configs:
            val = config.kwargs[self.key]
            low = self.low if isinstance(self.low, int) else args[self.low]
            high = self.high if isinstance(self.high, int) else args[self.high]
            if low <= val <= high:
                out.append(config)

        if not out:
            raise ValueError("All configurations were pruned")
        return out

    @classmethod
    def compose(cls, *pruners: "PruneConfigs") -> Callable[[List[triton.Config], Dict[str, Any]], List[triton.Config]]:
        r"""Compose multiple pruners into a single pruner."""

        def _composed(configs: List[triton.Config], args: Dict[str, Any]) -> List[triton.Config]:
            for pruner in pruners:
                configs = pruner(configs, args)
            return configs

        return _composed
