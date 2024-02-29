import math
from typing import Final, Tuple
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, cast
from tqdm import tqdm

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from .helpers import TENSOR_CORE_K, BoundaryCheckHeuristic, PowerOfTwoHeuristic


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
    ],
    key=["D", "M", "N"],
)
@triton.heuristics(
    {
        # BLOCK_HEADDIM must go up to the size of D.
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D", min_val=TENSOR_CORE_K),
        #"BLOCK_M": PowerOfTwoHeuristic("M", min_val=TENSOR_CORE_K, max_val=32),
        #"BLOCK_N": PowerOfTwoHeuristic("N", min_val=TENSOR_CORE_K, max_val=32),
        "EVEN_M": BoundaryCheckHeuristic("M", "BLOCK_M", disable=False),
        "EVEN_N": BoundaryCheckHeuristic("N", "BLOCK_N", disable=False),
        "EVEN_D": BoundaryCheckHeuristic("D", "BLOCK_HEADDIM", disable=False),
        "num_warps": lambda args: 4 if args["D"] <= 64 else 8,
        #"num_warps": lambda args: 4,
    }
)
@triton.jit
def _fwd_kernel(
    # Inputs
    q_p,
    k_p,
    v_p,
    # pos_q_p,
    # pos_k_p,
    # pos_slopes_p,
    out_p,
    softmax_scale,
    B,
    H,
    M,
    N,
    D,
    # Sizes
    # D_pos,
    # Q strides
    stride_q_b,
    stride_q_m,
    stride_q_h,
    # K strides
    stride_k_b,
    stride_k_n,
    stride_k_h,
    # V strides
    stride_v_b,
    stride_v_n,
    stride_v_h,
    # Position Q strides
    # stride_posq_b,
    # stride_posq_h,
    # stride_posq_m,
    # Position K strides
    # stride_posk_b,
    # stride_posk_h,
    # stride_posk_n,
    # Ouptut strides
    stride_o_b,
    stride_o_m,
    stride_o_h,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    # Heuristics for boundary checks
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_D: tl.constexpr,
):
    # Grid is (L, H * B)
    # Q of shape (B, Lq, H, D)
    # K of shape (B, Lk, H, D)
    # V of shape (B, Lk, H, D)

    # Initialize offsets
    # Each query block gets its own program
    start_m = tl.program_id(0)
    # Each batch/head gets its own program
    offset_bh = tl.program_id(1)
    offset_b = offset_bh // H
    offset_h = offset_bh % H
    BHN = B * H * N
    BHM = B * H * M

    #if EVEN_M: 
    #    tl.multiple_of(M, BLOCK_M)
    #if EVEN_N:
    #    tl.multiple_of(N, BLOCK_N)
    #if EVEN_D:
    #    tl.multiple_of(D, BLOCK_HEADDIM)

    # Initialize pointers
    # This program's block of queries will be loaded by this program for processing and kept there.
    Q_ptrs = tl.make_block_ptr(
        base=q_p + offset_b * stride_q_b + offset_h * stride_q_h,
        shape=(BHM, D),
        strides=(stride_q_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_HEADDIM),
        order=(1, 0),
    )
    # We will iteratively load blocks of K and V
    K_block_ptr = tl.make_block_ptr(
        base=k_p + offset_b * stride_k_b + offset_h * stride_k_h,
        shape=(D, BHN),
        strides=(1, stride_k_n),
        offsets=(0, 0),
        block_shape=(BLOCK_HEADDIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=v_p + offset_b * stride_v_b + offset_h * stride_v_h,
        shape=(BHN, D),
        strides=(stride_v_n, 1),
        offsets=(0, 0),
        block_shape=(BLOCK_N, BLOCK_HEADDIM),
        order=(1, 0),
    )

    # For each query we will track the maximum unscaled logit that we saw for that query's softmax.
    # We will use this to offset all logits for that query such that numerical overflow is avoided
    # when computing potentially large exponentials.
    #
    # Recall that softmax(x) = softmax(x - c) for any constant c.
    # Recall that softmax(x_i) = exp(x_i) / sum_j exp(x_j)
    # Subtracting a constant c gives softmax(x_i - c) = exp(x_i - c) / sum_j exp(x_j - c)
    #
    # If we need to change the subtracted offset to d, we can apply the following transformation:
    # softmax(x_i - d) = exp(x_i - c + c - d) / sum_j exp(x_j - c + c - d)
    #
    # We can factor this out into a function of the original logit and a function of the maximum logit:
    # softmax(x_i - d) = exp(x_i - c) * exp(c - d) / sum_j exp(x_j - c) * exp(c - d)
    #
    # In the following code we will use the following variables:
    #   * `c` - `query_i_maxdot`, the largest logit seen for each query
    #   * `d` - `query_i_maxdot_new`, the new largest logit seen for each query
    #   * `p` - exp(x_i - c) for each query
    #   * `softmax_denominator` - sum_j exp(x_j - c) for each query
    #   * `alpha` - exp(c - d) for each query
    query_i_maxdot = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    softmax_denominator = tl.zeros([BLOCK_M], dtype=tl.float32)
    value_accumulator = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Q gets loaded into SRAM and stays there. Pre-apply the softmax scale
    q = tl.load(Q_ptrs, boundary_check=(1, 0))
    q = (q * softmax_scale).to(k_p.dtype.element_ty)

    for _ in range(0, N, BLOCK_N):
        # Load K and V blocks into SRAM
        k = tl.load(K_block_ptr, boundary_check=(1, 0))
        v = tl.load(V_block_ptr, boundary_check=(1, 0))

        # Compute QK
        qk = tl.dot(q, k, allow_tf32=True)

        # Determine the maximum logit seen for each query
        query_i_maxdot_new = tl.maximum(query_i_maxdot, tl.max(qk, 1))

        # Compute scaling constant alpha.
        # Multiplying by this constant unscales the previous contributions and rescales according to the new maximum logit
        alpha = tl.exp(query_i_maxdot - query_i_maxdot_new)

        # Compute the softmax numerator for each key, applying the maximum logit offset to avoid numerical overflow
        p = tl.exp(qk - query_i_maxdot_new[:, None])

        # Apply the scaling constant to the accumulated values to rescale previous contributions
        # and accumulate the weighted values for this block of V
        value_accumulator = (value_accumulator * alpha[:, None]) + tl.dot(
            p.to(v_p.dtype.element_ty), v, allow_tf32=True
        )

        # Compute the softmax denominator for each query, applying the maximum logit offset to the existing denominator
        softmax_denominator = softmax_denominator * alpha + tl.sum(p, 1)

        # Update and advance pointers
        query_i_maxdot = query_i_maxdot_new
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    # Compute the final softmax values
    value_accumulator = value_accumulator / softmax_denominator[:, None]

    # l_ptrs = L + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, m_i + tl.math.log2(l_i))

    # Write output
    O_block_ptr = tl.make_block_ptr(
        base=out_p + offset_b * stride_o_b + offset_h * stride_o_h,
        shape=(BHM, D),
        strides=(stride_o_m, 1),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_HEADDIM),
        order=(1, 0),
    )
    tl.store(O_block_ptr, value_accumulator.to(out_p.dtype.element_ty), boundary_check=(1, 0))


def _flash_attn_forward(q, k, v, bias=None, causal=False, softmax_scale=None):
    # shape constraints
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    has_bias = bias is not None
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            pass
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            pass
        else:
            raise RuntimeError("Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)")
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    #lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    lse = None
    #tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    B = q.shape[0]
    H = q.shape[2]
    D = q.shape[3]

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)

    _fwd_kernel[grid](
        q,
        k,
        v,
        o,
        softmax_scale,
        B,
        H,
        seqlen_q,
        seqlen_k,
        D,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


def _benchmark(
    Q: int,
    K: int,
    D: int,
    provider: str,
    warmup: int = 25,
    rep: int = 100,
    bar: tqdm | None = None,
    verbose: bool = False,
):
    if verbose:
        with tqdm.external_write_mode():
            print(f"Running benchmark for Lq={Q}, Lk={K}, D={D}, provider={provider}")
    B = 4
    N_head = 8

    q = torch.randn((B, Q, N_head, D), device="cuda", dtype=torch.float16)
    k = torch.randn((B, K, N_head, D), device="cuda", dtype=torch.float16)
    v = torch.randn((B, K, N_head, D), device="cuda", dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]

    match provider:
        case "torch" | "flash" | "mem-eff":
            q = q.movedim(1, 2)
            k = k.movedim(1, 2)
            v = v.movedim(1, 2)
            with torch.backends.cuda.sdp_kernel(
                enable_flash=provider == "flash", 
                enable_math=provider == "torch",
                enable_mem_efficient=provider == "mem-eff",
            ):
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: F.scaled_dot_product_attention(q, k, v),
                    quantiles=quantiles,
                    warmup=warmup,
                    rep=rep,
                )
        case "triton":
            ms, min_ms, max_ms = triton.testing.do_bench(
                lambda: _flash_attn_forward(q, k, v, causal=False, softmax_scale=D ** -0.5),
                quantiles=quantiles,
                warmup=warmup,
                rep=rep,
            )
        case _:
            raise ValueError(f"Unknown provider: {provider}")

    def perf(_ms):
        return _ms

    if bar is not None:
        bar.update(1)

    return perf(ms), perf(max_ms), perf(min_ms)


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description="",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-QK", type=int, default=8192, help="Max Q and K length")
    parser.add_argument("-D", type=int, default=[32], nargs="+", help="Fixed size of D dimension")
    parser.add_argument("-s", "--step", type=int, default=128, help="Step size for Q and K dimensions")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path to save benchmark results")
    parser.add_argument("--warmup", type=int, default=25, help="`warmup` arg for `triton.testing.do_bench`")
    parser.add_argument("--rep", type=int, default=100, help="`rep` arg for `triton.testing.do_bench`")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Print each benchmark result")
    return parser.parse_args()


def main(args: Namespace):
    test_configs = list(range(args.step, args.QK + 1, args.step))
    providers = ["torch", "flash", "mem-eff", "triton"]
    total_tests = len(providers) * len(test_configs) * len(args.D)
    bar = tqdm(total=total_tests, desc="Benchmarking")

    for d in args.D:
        plot_name = f"flash-attn-d={d}"
        xlabel = f"Input size (Lx{d}, Lx{d})" 
        benchmark = triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["Q", "K"],
                x_vals=test_configs,
                line_arg="provider",
                line_vals=providers,
                line_names=["Torch", "Flash Attn", "ME Attn", "Triton"],
                styles=[("green", "-"), ("orange", "-"), ("blue", "-"), ("red", "-")],
                xlabel=xlabel,
                ylabel="Latency (ms)",
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
            )
        )
        df: Any = benchmark.run(
            show_plots=False,
            print_data=False,
            return_df=True,
            save_path=args.output,
            D=d,
        )
        with tqdm.external_write_mode():
            print(df.to_string(index=False))
    bar.close()


if __name__ == "__main__":
    main(parse_args())
