from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import math
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from tqdm import tqdm

from .helpers import TENSOR_CORE_K, IsBlockMultiple, PowerOfTwoHeuristic


ROOT_2 = math.sqrt(2)

# We expect K to be relatively small (probably length 1-3) for spacial coordinates
@triton.heuristics(
    {
        "BLOCK_SIZE_K": PowerOfTwoHeuristic("K", min_val=1, max_val=16),
        "BLOCK_SIZE_M": PowerOfTwoHeuristic("M", min_val=TENSOR_CORE_K, max_val=64),
        "BLOCK_SIZE_N": PowerOfTwoHeuristic("N", min_val=TENSOR_CORE_K, max_val=64),
        "GROUP_SIZE_M": lambda args: 8,
        "EVEN_M": IsBlockMultiple("M", "BLOCK_SIZE_M"),
        "EVEN_N": IsBlockMultiple("N", "BLOCK_SIZE_N"),
        "EVEN_K": IsBlockMultiple("K", "BLOCK_SIZE_K"),
    }
)
@triton.jit
def _euclidean_distance_kernel_fwd_pointwise(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    w_ptr,
    # Matrix dimensions
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bn: int,
    stride_bk: int,
    stride_cm: int,
    stride_cn: int,
    stride_w: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    INIT_ACCUMULATOR: tl.constexpr,
):
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    #tl.device_print("m", N)

    # Create block pointers for A B and C
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, K),
        strides=(stride_bn, stride_bk),
        offsets=(pid_n * BLOCK_SIZE_N, 0),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # Create block pointers for weights if it is provided
    if HAS_WEIGHTS:
        w_block_ptr = tl.make_block_ptr(
            base=w_ptr,
            shape=(K,),
            strides=(stride_w,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_K,),
            order=(0,),
        )
    else:
        w_block_ptr = w_ptr

    # Set up the accumulator. If requested we will load the initial value from C.
    if INIT_ACCUMULATOR:
        accumulator = tl.load(
            c_block_ptr,
            boundary_check=(0, 1) if not (EVEN_M and EVEN_N) else (0,) if not EVEN_M else (1,) if not EVEN_N else None,
        ).to(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Outer loop over the K dimension.
    # We will accumulate the sum of squared differences across K blocks and then take the square root.
    # NOTE: We accumulate into float32 because sqrt is not supported for float16
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B into shared memory for this chunk of K
        a = tl.load(
            a_block_ptr,
            boundary_check=(0, 1) if not (EVEN_M and EVEN_K) else (0,) if not EVEN_M else (1,) if not EVEN_K else None,
        )
        b = (
            tl.load(
                b_block_ptr,
                boundary_check=(
                    (0, 1) if not (EVEN_M and EVEN_K) else (0,) if not EVEN_N else (1,) if not EVEN_K else None
                ),
            )
        )

        # Compute the squared differences
        diff = a[:, None] - b[None, :]
        diff = diff * diff

        # Load the weight matrix if it is provided and apply it to the squared differences
        if HAS_WEIGHTS:
            w = tl.load(w_block_ptr, boundary_check=(0,) if not EVEN_K else None).to(tl.float32)
            diff = diff * w
            w_block_ptr = tl.advance(w_block_ptr, (BLOCK_SIZE_K,))

        # Compute the sum of above
        diff = tl.sum(diff, axis=2)

        # Accumulate the result
        accumulator += diff

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K)) 

    # Apply the square root and store the result
    c = tl.sqrt(accumulator).to(c_ptr.dtype.element_ty)

    # Write out accumulated result to block of C
    tl.store(
        c_block_ptr,
        c,
        boundary_check=(0, 1) if not (EVEN_M and EVEN_N) else (0,) if not EVEN_M else (1,) if not EVEN_N else None,
    )


@triton.jit
def _euclidean_distance_matmul_inner(
    a: tl.tensor,
    b: tl.tensor,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Compute a @ b.T
    ab = tl.dot(a, b)

    # Compute diag(a @ a.T)
    block_idx = tl.arange(0, BLOCK_SIZE_M)
    diag_a = tl.where(block_idx[:, None] == block_idx, tl.dot(a, tl.trans(a)), float(0.0))
    diag_a = tl.sum(diag_a, 1)

    # Compute diag(b @ b.T)
    block_idx = tl.arange(0, BLOCK_SIZE_N)
    diag_b = tl.where(block_idx[:, None] == block_idx, tl.dot(tl.trans(b), b), float(0.0))
    diag_b = tl.sum(diag_b, 1)

    # Update accumulator -> diag(a @ a.T) - 2 * a @ b + diag(b @ b.T)
    return diag_a[:, None] - 2*ab + diag_b[None, :]


@triton.heuristics(
    {
        "BLOCK_SIZE_K": PowerOfTwoHeuristic("K", min_val=TENSOR_CORE_K, max_val=16),
        "BLOCK_SIZE_M": PowerOfTwoHeuristic("M", min_val=TENSOR_CORE_K, max_val=64),
        "BLOCK_SIZE_N": PowerOfTwoHeuristic("N", min_val=TENSOR_CORE_K, max_val=64),
        "GROUP_SIZE_M": lambda args: 8,
        "EVEN_M": IsBlockMultiple("M", "BLOCK_SIZE_M"),
        "EVEN_N": IsBlockMultiple("N", "BLOCK_SIZE_N"),
        "EVEN_K": IsBlockMultiple("K", "BLOCK_SIZE_K"),
    }
)
@triton.jit
def _euclidean_distance_kernel_fwd_matmul(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    w_ptr,
    # Matrix dimensions
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bn: int,
    stride_bk: int,
    stride_cm: int,
    stride_cn: int,
    stride_w: int,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    INIT_ACCUMULATOR: tl.constexpr,
):
    # Map program ids `pid` to the block of C it should compute.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
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
        offsets=(pid_m * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
        order=(0, 1),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )

    # Create block pointers for weights and load if it is provided
    if HAS_WEIGHTS:
        w_block_ptr = tl.make_block_ptr(
            base=w_ptr,
            shape=(K,),
            strides=(stride_w,),
            offsets=(0,),
            block_shape=(BLOCK_SIZE_K,),
            order=(0,),
        )
    else:
        w_block_ptr = w_ptr

    # Set up the accumulator. If requested we will load the initial value from C.
    if INIT_ACCUMULATOR:
        accumulator = tl.load(
            c_block_ptr,
            boundary_check=(0, 1) if not (EVEN_M and EVEN_N) else (0,) if not EVEN_M else (1,) if not EVEN_N else None,
        ).to(tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Outer loop over the K dimension.
    # We will accumulate the sum of squared differences across K blocks and then take the square root.
    # NOTE: We accumulate into float32 because sqrt is not supported for float16
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load A and B blocks
        a = tl.load(
            a_block_ptr,
            boundary_check=(0, 1) if not (EVEN_M and EVEN_K) else (0,) if not EVEN_M else (1,) if not EVEN_K else None,
        )
        b = tl.load(
            b_block_ptr,
            boundary_check=(0, 1) if not (EVEN_K and EVEN_N) else (1,) if not EVEN_K else (0,) if not EVEN_N else None,
        )

        # Load the weight matrix if it is provided and apply it to a
        if HAS_WEIGHTS:
            w = tl.load(w_block_ptr, boundary_check=(0,) if not EVEN_K else None).to(tl.float32)
            w = tl.math.sqrt(w)
            a = a * w[None, :]
            b = b * w[:, None]
            w_block_ptr = tl.advance(w_block_ptr, (BLOCK_SIZE_K,))

        accumulator += _euclidean_distance_matmul_inner(a, b, BLOCK_SIZE_M, BLOCK_SIZE_N)

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    # Apply the square root and store the result
    c = tl.sqrt(accumulator).to(c_ptr.dtype.element_ty)

    # Write out accumulated result to block of C
    tl.store(
        c_block_ptr,
        c,
        boundary_check=(0, 1) if not (EVEN_M and EVEN_N) else (0,) if not EVEN_M else (1,) if not EVEN_N else None,
    )


class _EuclideanDistance(Function):

    @staticmethod
    def forward(
        ctx,
        a: Tensor,
        b: Tensor,
        w: Tensor | None = None,
        c: Tensor | None = None,
        matmul: bool | None = None,
    ) -> Tensor:
        assert a.shape[-1] == b.shape[-1], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"

        M, K = a.shape
        N, K = b.shape
        has_accumulator = c is not None
        c = c if c is not None else torch.empty((M, N), device=a.device, dtype=a.dtype)
        assert c.shape == (M, N), "Invalid output matrix shape"

        # The matmul implementation is faster for K >= 2
        # Otherwise pointwise is slightly faster
        matmul = matmul if matmul is not None else K >= 2

        if w is None:
            w = torch.empty((K,), device=a.device, dtype=a.dtype)
            has_weights = False
        else:
            assert w.shape == (K,), "Invalid weight matrix shape"
            assert w.is_contiguous(), "Weight matrix must be contiguous"
            has_weights = True

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

        match a.dtype:
            case torch.float16:
                dtype = tl.float16
            case torch.float32:
                dtype = tl.float32
            case torch.bfloat16:
                dtype = tl.bfloat16
            case _:
                raise ValueError(f"Unsupported dtype: {a.dtype}")

        fn = _euclidean_distance_kernel_fwd_matmul if matmul else _euclidean_distance_kernel_fwd_pointwise
        cast(Any, fn)[grid](
            a,
            b,
            c,
            w,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            w.stride(0),
            HAS_WEIGHTS=has_weights,
            INIT_ACCUMULATOR=has_accumulator,
        )
        # ctx.save_for_backward(a, b, c)

        return c


def euclidean_distance(
    a: Tensor,
    b: Tensor,
    w: Tensor | None = None,
    c: Tensor | None = None,
    matmul: bool = True,
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
        * ``w`` - :math:`(K,)`
        * ``c`` - :math:`(M, N)`
        * Output - :math:`(M, N)`
    """
    return cast(Tensor, _EuclideanDistance.apply(a, b, w, c, matmul))


def _reference_forward(a: Tensor, b: Tensor, w: Tensor | None = None, c: Tensor | None = None) -> Tensor:
    assert a.shape[-1] == b.shape[-1]
    M, K = a.shape
    N, K = b.shape
    has_c = c is not None
    c = c if has_c else torch.empty((M, N), device=a.device, dtype=a.dtype)
    if w is not None:
        assert w.shape == (K,)
        result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).mul(w.view(1, 1, K)).sum(-1).sqrt_()
    else:
        result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).sum(-1).sqrt_()
    if has_c:
        result.add_(c)
    return result


def _benchmark(
    M: int,
    N: int,
    K: int,
    provider: str,
    warmup: int = 25,
    rep: int = 100,
    bar: tqdm | None = None,
    verbose: bool = False,
    weighted: bool = False,
):
    if verbose:
        with tqdm.external_write_mode():
            print(f"Running benchmark for M={M}, N={N}, K={K}, provider={provider}")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    w = torch.randn((K,), device="cuda", dtype=torch.float16).abs() if weighted else None
    quantiles = [0.5, 0.2, 0.8]

    match provider:
        case "cublas":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _reference_forward(a, b, w),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        case "triton" | "triton-matmul":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: euclidean_distance(a, b, w, matmul="matmul" in provider),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    def perf(_ms):
        return (3 * M * N * K + M * N) * 1e-12 / (_ms * 1e-3)

    if bar is not None:
        bar.update(1)

    return perf(ms), perf(max_ms), perf(min_ms)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Benchmark performance of euclidean distance kernel for f((M, K), (N, K)) -> (M, N)",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-MN", type=int, default=8192, help="Max M and N dimension")
    parser.add_argument("-K", type=int, default=[2], nargs="+", help="Fixed size of K dimension")
    parser.add_argument("-s", "--step", type=int, default=128, help="Step size for M and N dimensions")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path to save benchmark results")
    parser.add_argument("--warmup", type=int, default=25, help="`warmup` arg for `triton.testing.do_bench`")
    parser.add_argument("--rep", type=int, default=100, help="`rep` arg for `triton.testing.do_bench`")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print each benchmark result")
    parser.add_argument(
        "-w", "--weighted", default=False, action="store_true", help="Compute weighted euclidean distance"
    )
    return parser.parse_args()


def main(args: Namespace):
    test_configs = list(range(args.step, args.MN + 1, args.step))
    providers = ["cublas", "triton", "triton-matmul"]
    total_tests = len(providers) * len(test_configs) * len(args.K)
    bar = tqdm(total=total_tests, desc="Benchmarking")

    for k in args.K:
        plot_name = f"euclidean-kernel-k={k}" + ("-weighted" if args.weighted else "")
        xlabel = f"Input size (Mx{k}, Mx{k})" + (", (weighted)" if args.weighted else "")
        benchmark = triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["M", "N"],
                x_vals=test_configs,
                line_arg="provider",
                line_vals=providers,
                line_names=["cuBLAS", "Triton", "Triton (Matmul)"],
                styles=[("green", "-"), ("blue", "-"), ("orange", "-")],
                xlabel=xlabel,
                ylabel="TFLOPS",
                plot_name=plot_name,
                args={},
            )
        )(
            partial(
                _benchmark,
                warmup=args.warmup,
                rep=args.rep,
                bar=bar,
                verbose=args.verbose,
                weighted=args.weighted,
            )
        )
        df: Any = benchmark.run(
            show_plots=False,
            print_data=False,
            return_df=True,
            save_path=args.output,
            K=k,
        )
        with tqdm.external_write_mode():
            print(df.to_string(index=False))
    bar.close()


if __name__ == "__main__":
    main(parse_args())
