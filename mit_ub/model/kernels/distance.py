from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, cast

import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd import Function
from tqdm import tqdm


# We expect K to be relatively small (probably length 1-3) for spacial coordinates
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 1, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 2, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 2, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 2, "GROUP_SIZE_M": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 2, "GROUP_SIZE_M": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 4, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 4, "GROUP_SIZE_M": 16}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 4, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 4, "GROUP_SIZE_M": 16}, num_stages=3, num_warps=8
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _euclidean_distance_kernel_fwd(
    # Pointers to matrices
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
    w_ptr: tl.tensor,
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
    DTYPE: tl.constexpr,
    HAS_WEIGHTS: tl.constexpr,
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

    # Create block pointers for A and B
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_N, BLOCK_SIZE_K] pointers
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

    # Outer loop over the K dimension.
    # We will accumulate the sum of squared differences across K blocks and then take the square root.
    # NOTE: We accumulate into float32 because sqrt is not supported for float16
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B into shared memory for this chunk of K
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # Compute the squared differences
        diff = a[:, None] - b[None, :]
        diff = diff * diff

        # Load the weight matrix if it is provided and apply it to the squared differences
        if HAS_WEIGHTS:
            w = tl.load(w_block_ptr)
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
    c = tl.sqrt(accumulator).to(DTYPE)

    # Write out accumulated result to block of C
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


class _EuclideanDistance(Function):

    @staticmethod
    def forward(ctx, a: Tensor, b: Tensor, w: Tensor | None = None) -> Tensor:
        assert a.shape[-1] == b.shape[-1], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"

        M, K = a.shape
        N, K = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

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

        cast(Any, _euclidean_distance_kernel_fwd)[grid](
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
            DTYPE=dtype,
            HAS_WEIGHTS=has_weights,
        )
        # ctx.save_for_backward(a, b, c)

        return c


def euclidean_distance(a: Tensor, b: Tensor, w: Tensor | None = None) -> Tensor:
    r"""Compute the Euclidean distance between two matrices of shape (M, K) and (N, K).

    The result is a matrix of shape (M, N) where each element (i, j) is the Euclidean distance between
    the i-th row of A and the j-th row of B.

    The kernel supports weights, which can be used to scale the squared differences before summing them. If weights
    are provided, the kernel will compute the weighted squared differences and sum them before taking the square root.

    Args:
        a: First input tensor
        b: Second input tensor
        w: Optional weight tensor of shape. If not provided, the kernel will compute the unweighted Euclidean distance.

    Shapes:
        * ``a`` - :math:`(M, K)`
        * ``b`` - :math:`(N, K)`
        * ``w`` - :math:`(K,)`
        * Output - :math:`(M, N)`
    """
    return cast(Tensor, _EuclideanDistance.apply(a, b, w))


def _reference_forward(a: Tensor, b: Tensor, w: Tensor | None = None) -> Tensor:
    assert a.shape[-1] == b.shape[-1]
    K = a.shape[-1]
    if w is not None:
        assert w.shape == (K,)
        result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).mul(w.view(1, 1, K)).sum(-1).sqrt()
    else:
        result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).sum(-1).sqrt()
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
        case "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: euclidean_distance(a, b, w),
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
    total_tests = 2 * len(test_configs) * len(args.K)
    bar = tqdm(total=total_tests, desc="Benchmarking")

    for k in args.K:
        plot_name = f"euclidean-kernel-k={k}" + ("-weighted" if args.weighted else "")
        xlabel = f"Input size (Mx{k}, Mx{k})" + (", (weighted)" if args.weighted else "")
        benchmark = triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["M", "N"],
                x_vals=test_configs,
                line_arg="provider",
                line_vals=["cublas", "triton"],
                line_names=["cuBLAS", "Triton"],
                styles=[("green", "-"), ("blue", "-")],
                xlabel=xlabel,
                ylabel="TFLOPS",
                plot_name=plot_name,
                args={},
            )
        )(partial(_benchmark, warmup=args.warmup, rep=args.rep, bar=bar, verbose=args.verbose, weighted=args.weighted))
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
