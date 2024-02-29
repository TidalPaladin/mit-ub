import sys
from dataclasses import dataclass
from typing import Any, Dict, Final

import triton


# In the case of K=16 we will perform the following operation in each tensor core
# (16x16) * (16x8) = (16x8)
# BFloat16 will only support a FP32 accumulator
TENSOR_CORE_K: Final = 16


@dataclass
class BoundaryCheckHeuristic:
    r"""Heuristic to determine if block pointers need boundary checks based on the
    input dimensions and block dimensions.

    Args:
        dims: Input dimension name
        block_dims: Block dimension name

    Returns:
        True if boundary checks are needed, False otherwise.
    """

    dim: str
    block_dim: str
    disable: bool = False

    def __call__(self, args: Dict[str, Any]) -> bool:
        return self.disable or args[self.dim] % args[self.block_dim] != 0


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
