from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from triton_helpers import TENSOR_CORE_K
from triton_helpers.heuristics import BoundaryCheckHeuristic, PowerOfTwoHeuristic


@triton.jit
def euclidean_distance_inner(
    a: tl.tensor,
    b: tl.tensor,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    METHOD: tl.constexpr = "matmul",
    SQRT: tl.constexpr = True,
):
    tl.static_assert(
        (METHOD == "matmul") | (METHOD == "matmul-nodiag") | (METHOD == "pointwise"),
        f"Invalid Euclidean distance METHOD: {METHOD}. Should be 'matmul', 'matmul-nodiag', or 'pointwise'",
    )
    tl.static_assert(tl.constexpr(a.shape[-1] == b.shape[-1]), "Incompatible dimensions to Euclidean distance")
    DOT_DTYPE: tl.constexpr = tl.float32

    # Pointwise
    # NOTE: This method seems to have broken in a recent Triton update. However it is slow and only exists for baseline,
    # so we will leave it as is for now.
    if METHOD == "pointwise":
        diff = a[:, None, :] - b[None, :, :]
        diff = diff * diff
        result = tl.sum(diff, 2)

    # Not clear why but we need a nested if/else
    else:
        # Other choices are matmul. First decide how to compute diagonals
        # NOTE: Force non-TF32, otherwise precision issues arise
        if METHOD == "matmul":
            # Compute diag(a @ a.T)
            block_idx = tl.arange(0, BLOCK_M)
            other = tl.full((1,), 0.0, dtype=DOT_DTYPE)
            diag_a = tl.where(
                block_idx[:, None] == block_idx, tl.dot(a, tl.trans(a), out_dtype=DOT_DTYPE, allow_tf32=False), other
            )
            diag_a = tl.sum(diag_a, 1)

            # Compute diag(b @ b.T)
            block_idx = tl.arange(0, BLOCK_N)
            diag_b = tl.where(
                block_idx[:, None] == block_idx, tl.dot(b, tl.trans(b), out_dtype=DOT_DTYPE, allow_tf32=False), other
            )
            diag_b = tl.sum(diag_b, 1)

        # Compute diagonals for without a matmul
        elif METHOD == "matmul-nodiag":
            diag_a = tl.sum(tl.math.pow(a.to(tl.float32), 2), 1)
            diag_b = tl.sum(tl.math.pow(b.to(tl.float32), 2), 1)

        else:
            tl.static_assert(False, "Unreachable code path in Euclidean distance kernel.")
            diag_a: Any = None
            diag_b: Any = None

        # Compute a @ b.T
        ab = tl.dot(a, tl.trans(b), out_dtype=DOT_DTYPE, allow_tf32=False)

        # Update accumulator -> diag(a @ a.T) - 2 * a @ b + diag(b @ b.T)
        result = diag_a[:, None] - 2 * ab + diag_b[None, :]
        result = tl.maximum(result, 0.0)

    if SQRT:
        result = tl.math.sqrt(result)
    return result


@triton.heuristics(
    {
        "BLOCK_K": PowerOfTwoHeuristic("K", min_val=TENSOR_CORE_K, max_val=16),
        "BLOCK_M": PowerOfTwoHeuristic("M", min_val=TENSOR_CORE_K, max_val=64),
        "BLOCK_N": PowerOfTwoHeuristic("N", min_val=TENSOR_CORE_K, max_val=64),
        "GROUP_SIZE_M": lambda args: 8,
        "BOUNDARY_CHECK_MK": BoundaryCheckHeuristic(["M", "K"], ["BLOCK_M", "BLOCK_K"]),
        "BOUNDARY_CHECK_NK": BoundaryCheckHeuristic(["N", "K"], ["BLOCK_N", "BLOCK_K"]),
        "BOUNDARY_CHECK_MN": BoundaryCheckHeuristic(["M", "N"], ["BLOCK_M", "BLOCK_N"]),
        "BOUNDARY_CHECK_K": BoundaryCheckHeuristic(["K"], ["BLOCK_K"]),
    }
)
@triton.jit
def _euclidean_distance_kernel(
    # fmt: off
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M: int, N: int, K: int,
    # Strides
    stride_am: int, stride_ak: int,
    stride_bn: int, stride_bk: int,
    stride_cm: int, stride_cn: int,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr,
    # Heuristics
    INIT_ACCUMULATOR: tl.constexpr,
    BOUNDARY_CHECK_MK: tl.constexpr, BOUNDARY_CHECK_NK: tl.constexpr, BOUNDARY_CHECK_MN: tl.constexpr, BOUNDARY_CHECK_K: tl.constexpr,
    METHOD: tl.constexpr = "matmul-nodiag",
    # fmt: on
):
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Initialize block pointers for A B and C
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # Set up the accumulator. If requested we will load the initial value from C.
    if INIT_ACCUMULATOR:
        accumulator = tl.load(c_block_ptr, boundary_check=BOUNDARY_CHECK_MN.value).to(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Outer loop over the K dimension.
    # We will accumulate the sum of squared differences across K blocks and then take the square root.
    # NOTE: We accumulate into float32 because sqrt is not supported for float16
    for _ in range(0, K, BLOCK_K):
        # Load A and B blocks
        a = tl.load(a_block_ptr, boundary_check=BOUNDARY_CHECK_MK.value)
        b = tl.load(b_block_ptr, boundary_check=BOUNDARY_CHECK_NK.value)
        accumulator += euclidean_distance_inner(a, b, BLOCK_M, BLOCK_N, METHOD, SQRT=False)

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

    # Apply the square root and store the result
    c = tl.sqrt(accumulator).to(c_ptr.dtype.element_ty)

    # Write out accumulated result to block of C
    tl.store(c_block_ptr, c, boundary_check=BOUNDARY_CHECK_MN.value)


class EuclideanDistance(Function):

    @staticmethod
    def forward(
        ctx,
        a: Tensor,
        b: Tensor,
        c: Tensor | None = None,
        method: str = "matmul-nodiag",
    ) -> Tensor:
        assert a.shape[-1] == b.shape[-1], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"

        M, K = a.shape
        N, K = b.shape
        has_accumulator = c is not None
        c = c if c is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
        assert c.shape == (M, N), "Invalid output matrix shape"

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        cast(Any, _euclidean_distance_kernel)[grid](
            # fmt: off
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            INIT_ACCUMULATOR=has_accumulator,
            METHOD=method,
            # fmt: on
        )

        return c


def euclidean_distance(
    a: Tensor,
    b: Tensor,
    c: Tensor | None = None,
    method: str = "matmul-nodiag",
) -> Tensor:
    r"""Compute the Euclidean distance between two matrices of shape (M, K) and (N, K).

    The result is a matrix of shape (M, N) where each element (i, j) is the Euclidean distance between
    the i-th row of A and the j-th row of B.

    The kernel supports weights, which can be used to scale the squared differences before summing them. If weights
    are provided, the kernel will compute the weighted squared differences and sum them before taking the square root.

    Args:
        a: First input tensor
        b: Second input tensor
        w: Optional weight tensor of shape. If not provided, the kernel will compute the unweighted Euclidean distance.
        c: Optional initial value for the output tensor. If not provided, the kernel will allocate a new tensor.

    Shapes:
        * ``a`` - :math:`(M, K)`
        * ``b`` - :math:`(N, K)`
        * ``c`` - :math:`(M, N)`
        * Output - :math:`(M, N)`
    """
    return cast(Tensor, EuclideanDistance.apply(a, b, c, method))
