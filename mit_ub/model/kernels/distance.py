from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Callable, cast

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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
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

    # Outer loop over the K dimension.
    # We will accumulate the sum of squared differences across K blocks and then take the square root.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B into shared memory for this chunk of K
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # Compute the sum of the squared differences
        diff = a[:, None] - b[None, :]
        diff = diff * diff
        diff = tl.sum(diff, axis=2)

        # Accumulate the result
        accumulator += diff

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K))

    # Apply the square root and store the result
    c = tl.sqrt(accumulator).to(tl.float16)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        order=(1, 0),
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


@triton.jit
def _euclidean_distance_kernel_bwd(
    # Pointers to matrices
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    c_ptr: tl.tensor,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
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

    # Outer loop over the K dimension.
    # We will accumulate the sum of squared differences across K blocks and then take the square root.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B into shared memory for this chunk of K
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))

        # Compute the sum of the squared differences
        diff = a[:, None] - b[None, :]
        diff = diff * diff
        diff = tl.sum(diff, axis=2)

        # Accumulate the result
        accumulator += diff

        # Advance block pointers
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_SIZE_K))

    # Apply the square root and store the result
    c = tl.sqrt(accumulator).to(tl.float16)

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
    def forward(ctx, a: Tensor, b: Tensor) -> Tensor:
        assert a.shape[-1] == b.shape[-1], "Incompatible dimensions"
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"

        M, K = a.shape
        N, K = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

        cast(Any, _euclidean_distance_kernel_fwd)[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
        )
        ctx.save_for_backward(a, b, c)

        return c


euclidean_distance: Callable[[Tensor, Tensor], Tensor] = cast(Any, _EuclideanDistance.apply)


def _benchmark(
    M: int,
    N: int,
    K: int,
    provider: str,
    warmup: int = 25,
    rep: int = 100,
    bar: tqdm | None = None,
    verbose: bool = False,
):
    if verbose:
        with tqdm.external_write_mode():
            print(f"Running benchmark for M={M}, N={N}, K={K}, provider={provider}")
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((N, K), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    match provider:
        case "cublas":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: (a.view(-1, 1, 2) - b.view(1, -1, 2)).pow(2).sum(-1).sqrt(),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        case "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: euclidean_distance(a, b),
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
    parser.add_argument("-w", "--warmup", type=int, default=25, help="`warmup` arg for `triton.testing.do_bench`")
    parser.add_argument("-r", "--rep", type=int, default=100, help="`rep` arg for `triton.testing.do_bench`")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print each benchmark result")
    return parser.parse_args()


def main(args: Namespace):
    test_configs = list(range(args.step, args.MN + 1, args.step))
    total_tests = 2 * len(test_configs) * len(args.K)
    bar = tqdm(total=total_tests, desc="Benchmarking")

    for k in args.K:
        benchmark = triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["M", "N"],
                x_vals=test_configs,
                line_arg="provider",
                line_vals=["cublas", "triton"],
                line_names=["cuBLAS", "Triton"],
                styles=[("green", "-"), ("blue", "-")],
                xlabel=f"Input size (Mx{k}, Mx{k})",
                ylabel="TFLOPS",
                plot_name=f"euclidean-kernel-k={k}",
                args={},
            )
        )(partial(_benchmark, warmup=args.warmup, rep=args.rep, bar=bar, verbose=args.verbose))
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
