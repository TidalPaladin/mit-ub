import math
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from functools import partial
from pathlib import Path
from typing import Any, Dict, Final, Tuple, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from tqdm import tqdm
from triton import JITFunction

from .distance import _euclidean_distance_matmul_inner
from .helpers import TENSOR_CORE_K, IsBlockMultiple, PowerOfTwoHeuristic, PruneConfigs, spill_warning


LN_2_RECIP: Final = 1 / math.log(2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ],
    key=["M", "N", "D", "D_pos", "HAS_BIAS", "ACCUMULATOR_DTYPE", "PROB_DTYPE", "DOT_DTYPE"],
    # No boundary checks so don't try configs that are bigger than the input
    prune_configs_by={
        "early_config_prune": PruneConfigs.compose(
            PruneConfigs("BLOCK_M", high="M"),
            PruneConfigs("BLOCK_N", high="N"),
        )
    },
)
@triton.heuristics(
    {
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D", min_val=TENSOR_CORE_K),
        "BLOCK_POSDIM": PowerOfTwoHeuristic("D_pos", min_val=TENSOR_CORE_K),
        "EVEN_POSDIM": IsBlockMultiple("D_pos", "BLOCK_POSDIM"),
        "num_warps": lambda args: 4 if args["D"] <= 64 else 8,
    }
)
@triton.jit
def _fwd_kernel(
    # fmt: off
    # Inputs
    q_p, k_p, v_p, logit_scale_p, pos_q_p, pos_k_p, pos_slopes_p, out_p, 
    softmax_scale: tl.constexpr, qk_scale: tl.constexpr, bias_scale: tl.constexpr,
    # Sizes 
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, D_pos: tl.constexpr,
    # Q strides
    stride_q_b: tl.constexpr, stride_q_h: tl.constexpr, stride_q_m: tl.constexpr,
    # K strides
    stride_k_b: tl.constexpr, stride_k_h: tl.constexpr, stride_k_n: tl.constexpr,
    # V strides
    stride_v_b: tl.constexpr, stride_v_h: tl.constexpr, stride_v_n: tl.constexpr,
    # Logit scale strides
    stride_logit_b: tl.constexpr, stride_logit_h: tl.constexpr,
    # Position Q strides
    stride_posq_b: tl.constexpr, stride_posq_h: tl.constexpr, stride_posq_m: tl.constexpr,
    # Position K strides
    stride_posk_b: tl.constexpr, stride_posk_h: tl.constexpr, stride_posk_n: tl.constexpr,
    # Slopes strides
    stride_slopes_b: tl.constexpr,
    # Output strides
    stride_o_b: tl.constexpr, stride_o_h: tl.constexpr, stride_o_m: tl.constexpr,
    # Dtypes
    ACCUMULATOR_DTYPE: tl.constexpr, PROB_DTYPE: tl.constexpr, DOT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Heuristics for boundary checks
    EVEN_POSDIM: tl.constexpr, 
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, BLOCK_POSDIM: tl.constexpr,
    # fmt: on
):
    # Grid is (L, H * B)
    # Q of shape (B, H, Lq, D)
    # K of shape (B, H, Lk, D)
    # V of shape (B, H, Lk, D)

    # Initialize offsets
    # Each query block gets its own program
    start_m = tl.program_id(0)
    # Each batch/head gets its own program
    offset_bh = tl.program_id(1)
    offset_b = offset_bh // H
    offset_h = offset_bh % H

    # Initialize pointers
    # This program's block of queries will be loaded by this program for processing and kept there.
    # NOTE: Block pointers may contribute to register spilling due to int64 indexing. Checking the
    # compiled kernel.n_spills indiciates no spilling - is this an issue or not?
    q_p += offset_b * stride_q_b + offset_h * stride_q_h + start_m * BLOCK_M * stride_q_m
    k_p += offset_b * stride_k_b + offset_h * stride_k_h
    v_p += offset_b * stride_v_b + offset_h * stride_v_h
    pos_q_p += offset_b * stride_posq_b + offset_h * stride_posq_h + start_m * BLOCK_M * stride_posq_m
    pos_k_p += offset_b * stride_posk_b + offset_h * stride_posk_h
    pos_slopes_p += offset_b * stride_slopes_b
    out_p += offset_b * stride_o_b + offset_h * stride_o_h + start_m * BLOCK_M * stride_o_m
    logit_scale_p += offset_b * stride_logit_b + offset_h * stride_logit_h + start_m * BLOCK_M

    # For each query we will track the maximum unscaled logit that we saw for that query's softmax.
    # We will use this to offset all logits for that query such that numerical overflow is avoided
    # when computing potentially large exponentials. This behavior is togglable with the `NEEDS_SCALE` flag,
    # which is relevant in situations like cosine attention where Q and K are normalized.
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
    # NOTE: We use qk_scale = 1 / (ln(2) * sqrt(D)) so we can compute logs and exponentials in base 2,
    # which is empirically faster. We must also scale biases by 1 / ln(2) to match.
    value_accumulator = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=ACCUMULATOR_DTYPE)
    softmax_denominator = tl.zeros([BLOCK_M], dtype=ACCUMULATOR_DTYPE)
    query_i_maxdot = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # Q gets loaded into SRAM and stays there. Pre-apply the softmax scale
    m_offsets = tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_HEADDIM)
    q = tl.load(q_p + (m_offsets[:, None] * stride_q_m + d_offsets[None, :]))
    q *= tl.full((1,), qk_scale, dtype=q.dtype)

    # If we have bias, we will also load the Q positions and ALiBi slopes
    # NOTE: Since distances are likely < D_pos we must apply masking along D_pos
    if HAS_BIAS:
        # Load the slopes squared
        d_offsets = tl.arange(0, BLOCK_POSDIM)
        if EVEN_POSDIM:
            pos_q = tl.load(pos_q_p + (m_offsets[:, None] * stride_posq_m + d_offsets[None, :]))
        else:
            pos_q_mask = (d_offsets < D_pos)[None, :]
            pos_q = tl.load(pos_q_p + (m_offsets[:, None] * stride_posq_m + d_offsets[None, :]), mask=pos_q_mask)
        pos_slope = (tl.load(pos_slopes_p) * bias_scale).to(DOT_DTYPE)
    else:
        pos_slope = None
        pos_q = None

    # Iterate over KV blocks
    for _ in range(0, N, BLOCK_N):
        # Load K block and compute QK
        k = tl.load(k_p + (tl.arange(0, BLOCK_N)[None, :] * stride_k_n + tl.arange(0, BLOCK_HEADDIM)[:, None]))
        qk = tl.dot(q, k, allow_tf32=True)

        # Compute the bias
        if HAS_BIAS:
            n_offsets = tl.arange(0, BLOCK_N)
            d_offsets = tl.arange(0, BLOCK_POSDIM)
            if EVEN_POSDIM:
                pos_k = tl.load(pos_k_p + (n_offsets[None, :] * stride_posk_n + d_offsets[:, None]))
            else:
                pos_k_mask = (d_offsets < D_pos)[:, None]
                pos_k = tl.load(pos_k_p + (n_offsets[None, :] * stride_posk_n + d_offsets[:, None]), mask=pos_k_mask)
            bias = _euclidean_distance_matmul_inner(pos_q, pos_k, BLOCK_M, BLOCK_N, DOT_DTYPE)
            bias = tl.sqrt(bias.to(tl.float32)).to(DOT_DTYPE)
            qk = bias * pos_slope + qk

        # Determine the maximum logit seen for each query
        query_i_maxdot_new = tl.maximum(query_i_maxdot, tl.max(qk, 1))

        # Compute scaling constant alpha and rescale the previous contributions, updating the maximum logit
        alpha = tl.math.exp2(query_i_maxdot - query_i_maxdot_new).to(ACCUMULATOR_DTYPE)
        query_i_maxdot = query_i_maxdot_new

        # Compute the softmax numerator for each key, applying the maximum logit offset to avoid numerical overflow
        p = tl.math.exp2(qk - query_i_maxdot_new[:, None]).to(PROB_DTYPE)

        # Compute the softmax denominator for each query, applying the maximum logit offset to the existing denominator
        softmax_denominator = softmax_denominator * alpha + tl.sum(p, 1).to(ACCUMULATOR_DTYPE)

        # Accumulate the weighted values for this block of V
        v = tl.load(v_p + (tl.arange(0, BLOCK_N)[:, None] * stride_v_n + tl.arange(0, BLOCK_HEADDIM)[None, :]))
        value_accumulator = value_accumulator * alpha[:, None] + tl.dot(
            p.to(v.dtype), v, allow_tf32=True, out_dtype=cast(tl.dtype, DOT_DTYPE)
        ).to(ACCUMULATOR_DTYPE)

        # Advance pointers
        k_p += BLOCK_N * stride_k_n
        v_p += BLOCK_N * stride_v_n
        if HAS_BIAS:
            pos_k_p += BLOCK_N * stride_posk_n

    # Compute the final softmax values
    value_accumulator = value_accumulator / softmax_denominator.to(ACCUMULATOR_DTYPE)[:, None]

    # Per Flash Attention 2, we store only logsumexp for the backward pass
    tl.store(
        logit_scale_p + tl.arange(0, BLOCK_M),
        (query_i_maxdot + tl.math.log2(softmax_denominator.to(tl.float32))),
    )

    # Write output
    tl.store(
        out_p + (tl.arange(0, BLOCK_M)[:, None] * stride_o_m + tl.arange(0, BLOCK_HEADDIM)[None, :]),
        value_accumulator.to(out_p.dtype.element_ty),
    )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128}),
        triton.Config({"BLOCK_M": 64}),
        triton.Config({"BLOCK_M": 32}),
    ],
    key=["M", "D"],
    prune_configs_by={
        "early_config_prune": PruneConfigs.compose(
            PruneConfigs("BLOCK_M", high="M"),
        )
    },
)
@triton.heuristics(
    {
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D", min_val=TENSOR_CORE_K),
        "EVEN_D": IsBlockMultiple("D", "BLOCK_HEADDIM"),
        "num_warps": lambda args: 4 if args["D"] <= 64 else 8,
    }
)
@triton.jit
def _bwd_preprocess_do_o_dot(
    # fmt: off
    # Inputs
    Out, DO, Delta,
    # Strides
    stride_o_b: tl.constexpr, stride_o_h: tl.constexpr, stride_o_m: tl.constexpr,
    stride_do_b: tl.constexpr, stride_do_h: tl.constexpr, stride_do_m: tl.constexpr,
    stride_delta_b: tl.constexpr, stride_delta_h: tl.constexpr,
    # Sizes
    H: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
    # Blocks
    BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    # Derived values
    BHM: tl.constexpr, EVEN_D: tl.constexpr,
    # fmt: on
):
    # Initialize offsets
    start_m = tl.program_id(0)
    offset_bh = tl.program_id(1)
    offset_b = offset_bh // H
    offset_h = offset_bh % H

    # Load O
    o_block_ptr = tl.make_block_ptr(
        Out + offset_b * stride_o_b + offset_h * stride_o_h,
        (BHM, D),
        (stride_o_m, 1),
        (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    o = tl.load(o_block_ptr).to(Delta.dtype.element_ty)

    # Load DO
    do_block_ptr = tl.make_block_ptr(
        DO + offset_b * stride_do_b + offset_h * stride_do_h,
        (BHM, D),
        (stride_do_m, 1),
        (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    do = tl.load(do_block_ptr).to(Delta.dtype.element_ty)

    # Compute
    delta = tl.sum(o * do, axis=1)

    # Write output
    delta_offset = Delta + offset_b * stride_delta_b + offset_h * stride_delta_h + start_m * BLOCK_M
    delta_ptrs = delta_offset + tl.arange(0, BLOCK_M)
    tl.store(delta_ptrs, delta)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
    ],
    key=["D", "M", "N", "ACCUMULATOR_DTYPE", "DOT_DTYPE"],
    prune_configs_by={
        "early_config_prune": PruneConfigs.compose(
            PruneConfigs("BLOCK_M", high="M"),
            PruneConfigs("BLOCK_N", high="N"),
        )
    },
    reset_to_zero=["dq_p", "lock_p"],
)
@triton.heuristics(
    {
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D", min_val=TENSOR_CORE_K),
        "BLOCK_POSDIM": PowerOfTwoHeuristic("D_pos", min_val=TENSOR_CORE_K),
        "EVEN_POSDIM": IsBlockMultiple("D_pos", "BLOCK_POSDIM"),
        "num_warps": lambda args: 4 if args["D"] <= 64 else 8,
    }
)
@triton.jit
def _bwd_kernel(
    # fmt: off
    # Inputs
    q_p, k_p, v_p, logit_scale_p, pos_q_p, pos_k_p, pos_slopes_p,
    # Derivatives
    do_p, dq_p, dk_p, dv_p, delta_p,
    softmax_scale: tl.constexpr, qk_scale: tl.constexpr, bias_scale: tl.constexpr, 
    lock_p,
    # Strides
    stride_q_b: tl.constexpr, stride_q_h: tl.constexpr , stride_q_m: tl.constexpr,
    stride_k_b: tl.constexpr, stride_k_h: tl.constexpr , stride_k_n: tl.constexpr, 
    stride_v_b: tl.constexpr, stride_v_h: tl.constexpr , stride_v_n: tl.constexpr, 
    stride_posq_b: tl.constexpr, stride_posq_h: tl.constexpr , stride_posq_m: tl.constexpr,
    stride_posk_b: tl.constexpr, stride_posk_h: tl.constexpr , stride_posk_n: tl.constexpr,
    stride_slopes_b: tl.constexpr,
    # Sizes
    B: tl.constexpr, H: tl.constexpr, M: tl.constexpr, N: tl.constexpr, D: tl.constexpr, D_pos: tl.constexpr,
    # Dtypes
    ACCUMULATOR_DTYPE: tl.constexpr, DOT_DTYPE: tl.constexpr, PROB_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # Blocks
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, BLOCK_POSDIM: tl.constexpr,
    # Heuristics
    EVEN_POSDIM: tl.constexpr,
    # fmt: on
):
    # Initialize offsets.
    # Grid will be over (Lk, B*H)
    start_n = tl.program_id(0)
    offset_bh = tl.program_id(1)
    offset_b = offset_bh // H
    offset_h = offset_bh % H

    # Seek pointers
    # NOTE: This kernel has significant register pressure. To avoid spilling we favor manual pointers
    # over block pointers (for int32 indexing) and rematerialization over storing intermediate results.
    q_p += offset_b * stride_q_b + offset_h * stride_q_h
    k_p += offset_b * stride_k_b + offset_h * stride_k_h
    v_p += offset_b * stride_v_b + offset_h * stride_v_h
    do_p += offset_b * stride_q_b + offset_h * stride_q_h
    dq_p += offset_b * stride_q_b + offset_h * stride_q_h
    dk_p += offset_b * stride_k_b + offset_h * stride_k_h
    dv_p += offset_b * stride_v_b + offset_h * stride_v_h
    pos_q_p += offset_b * stride_posq_b + offset_h * stride_posq_h
    pos_k_p += offset_b * stride_posk_b + offset_h * stride_posk_h
    pos_slopes_p += offset_b * stride_slopes_b
    delta_p += offset_bh * M
    logit_scale_p += offset_bh * M
    lock_p += offset_bh * tl.cdiv(M, BLOCK_M)

    # Load K and V - stay in SRAM for the entire kernel
    n_offsets = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_HEADDIM)
    k_ptrs = k_p + (n_offsets[:, None] * stride_k_n + d_offsets[None, :])
    k = tl.load(k_ptrs)
    v_ptrs = v_p + (n_offsets[:, None] * stride_v_n + d_offsets[None, :])
    v = tl.load(v_ptrs)

    # Init dv and dk
    # These will be accumulated at the same precision as their destination.
    # Likewise, since dQ is accumulated to HBM every loop iteration, we will accumulate it at the same precision.
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=ACCUMULATOR_DTYPE)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=ACCUMULATOR_DTYPE)

    # Init pos_k
    if HAS_BIAS:
        # Load the slopes squared
        pos_d_offsets = tl.arange(0, BLOCK_POSDIM)
        if EVEN_POSDIM:
            pos_k = tl.load(pos_k_p + (n_offsets[None, :] * stride_posk_n + pos_d_offsets[:, None]))
        else:
            pos_k_mask = (pos_d_offsets < D_pos)[:, None]
            pos_k = tl.load(pos_k_p + (n_offsets[None, :] * stride_posk_n + pos_d_offsets[:, None]), mask=pos_k_mask)
        pos_slope = (tl.load(pos_slopes_p) * bias_scale).to(DOT_DTYPE)
    else:
        pos_slope = None
        pos_k = None

    # Recall that for matrix multiplication C = A * B,
    # * dC/dA = B.T
    # * dC/dB = A.T
    for _ in range(0, M, BLOCK_M):
        m_offsets = tl.arange(0, BLOCK_M) * stride_q_m
        d_offsets = tl.arange(0, BLOCK_HEADDIM)

        # Recompute p = softmax(qk), p is (MxN)
        # NOTE: Keep qk in fp32 to avoid overflow and because exponentiation requires FP32
        q = tl.load(q_p + (m_offsets[:, None] + d_offsets[None, :]))
        logit_scale = tl.load(logit_scale_p + tl.arange(0, BLOCK_M))
        qk = tl.dot(q, tl.trans(k))
        if HAS_BIAS:
            pos_m_offsets = tl.arange(0, BLOCK_M) * stride_posq_m
            pos_d_offsets = tl.arange(0, BLOCK_POSDIM)
            if EVEN_POSDIM:
                pos_q = tl.load(pos_q_p + (pos_m_offsets[:, None] + pos_d_offsets[None, :]))
            else:
                pos_q_mask = (pos_d_offsets < D_pos)[None, :]
                pos_q = tl.load(pos_q_p + (pos_m_offsets[:, None] + pos_d_offsets[None, :]), mask=pos_q_mask)
            # NOTE: Compute this in FP32, it's overflow prone
            bias = _euclidean_distance_matmul_inner(pos_q, pos_k, BLOCK_M, BLOCK_N, tl.float32)
            bias = pos_slope * tl.sqrt(bias.to(tl.float32)).to(DOT_DTYPE)
            qk = qk * qk_scale + bias
            p = tl.math.exp2(qk - logit_scale[:, None]).to(do_p.dtype.element_ty)
        else:
            p = tl.math.exp2(qk * qk_scale - logit_scale[:, None]).to(do_p.dtype.element_ty)

        # compute dL/dv = dL/do * do/dv = dL/do * p
        # Shape do = (MxD)
        # NOTE: `do` is pre-divided by `l`; no normalization here
        do = tl.load(do_p + (m_offsets[:, None] + d_offsets[None, :]))  # (MxD)
        dv += tl.dot(tl.trans(p), do, allow_tf32=True, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(ACCUMULATOR_DTYPE)

        # compute dL/dp = dL/do * do/dp = dL/do * v
        # Shape dp = (MxN)
        dp = tl.dot(do, tl.trans(v), allow_tf32=True, out_dtype=cast(tl.dtype, DOT_DTYPE))

        # compute dL/ds = dL/dp * dp/ds = p * (dp - delta[:, None])
        # Shape ds = (MxN)
        delta = tl.load(delta_p + tl.arange(0, BLOCK_M))
        ds = ((p * (dp - delta[:, None])) * softmax_scale).to(q_p.dtype.element_ty)

        # compute dL/dk = dL/ds * ds/dk = dot(ds.T, q)
        q = tl.load(q_p + (m_offsets[:, None] + d_offsets[None, :]))
        dk += tl.dot(tl.trans(ds), q, allow_tf32=True, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(ACCUMULATOR_DTYPE)

        # compute dL/dq = dL/ds * ds/dq = dot(ds, k)
        # NOTE: We do an atomic add here since multiple threads may be writing to the dq location.
        # For some reason tl.atomic_add is much slower than what we have here.
        # Using a counter to avoid initializing dq to 0 results in register spilling, so we just start with
        # zero initialization and add at each step
        dq = tl.dot(ds, k, allow_tf32=True, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(dq_p.dtype.element_ty)
        # tl.atomic_add(dq_p + (m_offsets[:, None] + d_offsets[None, :]), dq)
        while tl.atomic_cas(lock_p, 0, 1) == 1:
            pass
        dq += tl.load(dq_p + (m_offsets[:, None] + d_offsets[None, :]), eviction_policy="evict_last")
        tl.store(dq_p + (m_offsets[:, None] + d_offsets[None, :]), dq, eviction_policy="evict_last")
        tl.atomic_xchg(lock_p, 0)

        # advance pointers
        q_p += BLOCK_M * stride_q_m
        dq_p += BLOCK_M * stride_q_m
        do_p += BLOCK_M * stride_q_m
        delta_p += BLOCK_M
        logit_scale_p += BLOCK_M
        lock_p += 1

    # write-back
    start_n = tl.program_id(0)
    n_offsets = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_HEADDIM)
    tl.store(dk_p + (n_offsets[:, None] * stride_k_n + d_offsets[None, :]), dk.to(dk_p.dtype.element_ty))
    tl.store(dv_p + (n_offsets[:, None] * stride_v_n + d_offsets[None, :]), dv.to(dv_p.dtype.element_ty))


class _attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        pos_q=None,
        pos_k=None,
        slopes=None,
        softmax_scale=None,
        full_precision: bool = True,
    ) -> Tensor:
        # shape constraints
        B, H, Lq, D = q.shape
        _, _, Lk, _ = k.shape
        assert k.shape == (B, H, Lk, D)
        assert v.shape == (B, H, Lk, D)
        assert D <= 128, "FlashAttention only support head dimensions up to 128"
        assert D >= 16, "FlashAttention requires head dimensions of at least 16"
        assert triton.next_power_of_2(D) == D, "FlashAttention requires head dimensions to be a power of 2"
        assert Lq % TENSOR_CORE_K == 0, f"Lq must be a multiple of {TENSOR_CORE_K}"
        assert Lk % TENSOR_CORE_K == 0, f"Lk must be a multiple of {TENSOR_CORE_K}"

        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda
        softmax_scale = softmax_scale or 1.0 / math.sqrt(D)

        # Check bias
        if pos_q is not None:
            _, _, _, D_pos = pos_q.shape
            assert pos_q.shape == (B, H, Lq, D_pos), "Query position shape must be (B, H, Lq, D_pos)"
            pos_k = pos_k if pos_k is not None else pos_q
            assert pos_k.shape == (B, H, Lk, D_pos), "Key position must be (B, H, Lk, D_pos)"
            slopes = slopes if slopes is not None else torch.full((B, H), -1, device=q.device, dtype=q.dtype)
            assert slopes.shape == (B, H)
            assert pos_q.dtype == pos_k.dtype == slopes.dtype == q.dtype
            has_bias = True
        else:
            pos_q = torch.empty((0, 0, 0, 0), device=q.device, dtype=q.dtype)
            pos_k = torch.empty((0, 0, 0, 0), device=k.device, dtype=k.dtype)
            slopes = torch.empty((0, 0), device=q.device, dtype=q.dtype)
            D_pos = D
            has_bias = False

        logit_scale = torch.empty((B, H, Lq), device=q.device, dtype=torch.float32)
        o = torch.empty_like(q)
        # NOTE: It seems important that we seed to at most 2**30. Otherwise we randomly get huge latency spikes.
        # seed = torch.randint(0, 2**30, (1,), dtype=torch.int64).item()

        def grid(META):
            return (triton.cdiv(Lq, META["BLOCK_M"]), B * H)

        # fmt: off
        spill_warning()(cast(JITFunction, _fwd_kernel)[grid])(
            q, k, v, logit_scale, pos_q, pos_k, slopes, o,
            softmax_scale, softmax_scale * LN_2_RECIP, LN_2_RECIP,
            B, H, Lq, Lk, D, D_pos,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            logit_scale.stride(0), logit_scale.stride(1),
            pos_q.stride(0), pos_q.stride(1), pos_q.stride(2),
            pos_k.stride(0), pos_k.stride(1), pos_k.stride(2),
            slopes.stride(0),
            o.stride(0), o.stride(1), o.stride(2),
            HAS_BIAS=has_bias,
            **_attention._select_dtypes(o.dtype, full_precision),
        )
        # fmt: on

        ctx.save_for_backward(q, k, v, o, logit_scale, pos_q, pos_k, slopes)
        ctx.softmax_scale = softmax_scale
        ctx.full_precision = full_precision
        ctx.has_bias = has_bias
        return o

    @staticmethod
    @torch.inference_mode()
    def backward(ctx, do):
        q, k, v, o, logit_scale, pos_q, pos_k, slopes = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        B, H, Lk, D = k.shape
        _, _, Lq, _ = q.shape
        D_pos = pos_q.shape[-1] if ctx.has_bias else 0
        assert logit_scale.shape == (B, H, Lq)

        do = do.contiguous() if do.stride(-1) != 1 else do
        # NOTE: Using a leading dimension and reducing with sum is faster than atomic_add
        # NOTE: We get away with using an empty dq because all dimensions are multiples of their block sizes
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(logit_scale, dtype=dq.dtype)
        # TODO: We don't know how many locks we need until the autotuner chooses a config.
        # The best we can do is to allocate enough for the worst case. Can this be improved?
        MIN_AUTOTUNER_N = 32
        locks = torch.zeros(B * H * triton.cdiv(Lq, MIN_AUTOTUNER_N), dtype=torch.int32, device=q.device)

        def pre_grid(META):
            return (triton.cdiv(Lq, META["BLOCK_M"]), B * H)

        # fmt: off
        spill_warning()(cast(JITFunction, _bwd_preprocess_do_o_dot)[pre_grid])(
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2),
            delta.stride(0), delta.stride(1),
            H, D, Lq,
            BHM=B*H*Lq,
        )
        # fmt: on

        def grid(META):
            return (triton.cdiv(Lk, META["BLOCK_N"]), B * H)

        # fmt: off
        spill_warning()(cast(JITFunction, _bwd_kernel)[grid])(
            q, k, v, logit_scale, pos_q, pos_k, slopes,
            do, dq, dk, dv, delta,
            ctx.softmax_scale, ctx.softmax_scale * LN_2_RECIP, LN_2_RECIP,
            locks,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            pos_q.stride(0), pos_q.stride(1), pos_q.stride(2),
            pos_k.stride(0), pos_k.stride(1), pos_k.stride(2),
            slopes.stride(0),
            B, H, Lq, Lk, D, D_pos,
            HAS_BIAS=ctx.has_bias,
            **_attention._select_dtypes(dq.dtype, ctx.full_precision),
        )
        # fmt: on

        return dq, dk, dv, None, None, None, None, None, None

    @staticmethod
    def _select_dtypes(dtype: torch.dtype, full_precision: bool) -> Dict[str, Any]:
        if full_precision:
            return dict(
                ACCUMULATOR_DTYPE=tl.float32,
                PROB_DTYPE=tl.float32,
                DOT_DTYPE=tl.float32,
            )
        else:
            # BF16 is slower at reduced precision and it doesn't seem possible to fix.
            # This is probably because dot products do not support BF16 accumulators so a lot of casting is required.
            if dtype is torch.bfloat16:
                warnings.warn(
                    "Reducing precision for BF16 inputs is slower than full precision. "
                    "Consider using full precision or working with FP16 inputs"
                )
            return dict(
                ACCUMULATOR_DTYPE=tl.float16,
                PROB_DTYPE=tl.float16,
                DOT_DTYPE=tl.float16,
            )


attention_fn = _attention.apply


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_q: Tensor | None = None,
    pos_k: Tensor | None = None,
    slopes: Tensor | None = None,
    softmax_scale: float | None = None,
    full_precision: bool = True,
) -> Tensor:
    r"""Computes scaled dot product attention using Flash Attention 2.

    Supports positional encodings for queries and keys that will be constructed using the Euclidean distance.
    This is implemented such that the memory requirement is linear.

    Args:
        q: Query tensor of shape `(B, H, Lq, D)`
        k: Key tensor of shape `(B, H, Lk, D)`
        v: Value tensor of shape `(B, H, Lk, D)`
        pos_q: Query position tensor of shape `(B, H, Lq, D_pos)`. Defaults to None.
        pos_k: Key position tensor of shape `(B, H, Lk, D_pos)`. Defaults to None.
        slopes: ALiBi slopes tensor of shape `(B, H)`. Defaults to None.
        softmax_scale: Scale factor for the softmax. Defaults to None, in which case it is set to `1 / sqrt(D)`.
        full_precision: Whether to use full precision for intermediate computations. Defaults to True.
            Setting to falsle will perform intermediate computations in FP16, yielding a speedup at the cost
            of precision and numerical stability.
    """
    return cast(Tensor, attention_fn(q, k, v, pos_q, pos_k, slopes, softmax_scale, full_precision))


def _benchmark(
    Q: int,
    K: int,
    D: int,
    provider: str,
    warmup: int = 25,
    rep: int = 100,
    bar: tqdm | None = None,
    verbose: bool = False,
    bias: bool = False,
    mode: str = "fwd",
    dtype: str = "fp16",
):
    if verbose:
        with tqdm.external_write_mode():
            print(f"Running benchmark for Lq={Q}, Lk={K}, D={D}, provider={provider}")
    B = 4
    H = 12
    D_pos = 2
    torch_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16

    q = torch.randn((B, H, Q, D), device="cuda", dtype=torch_dtype, requires_grad=mode != "fwd")
    k = torch.randn((B, H, K, D), device="cuda", dtype=torch_dtype, requires_grad=mode != "fwd")
    v = torch.randn((B, H, K, D), device="cuda", dtype=torch_dtype, requires_grad=mode != "fwd")
    quantiles = [0.5, 0.2, 0.8]

    if bias and provider not in ["flash", "mem-eff"]:
        pos_q = torch.randn((B, H, Q, D_pos), device="cuda", dtype=torch_dtype)
        pos_k = torch.randn((B, H, K, D_pos), device="cuda", dtype=torch_dtype)
        attn_mask = -1 * ((pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(B, H, Q, K))
    else:
        pos_q = pos_k = attn_mask = None

    match provider:
        case "torch" | "flash" | "mem-eff":
            with torch.backends.cuda.sdp_kernel(
                enable_flash=provider == "flash",
                enable_math=provider == "torch",
                enable_mem_efficient=provider == "mem-eff",
            ):
                if mode == "fwd":
                    ms, min_ms, max_ms = triton.testing.do_bench(
                        lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask),
                        quantiles=quantiles,
                        warmup=warmup,
                        rep=rep,
                    )
                else:
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask).mul(2).sum()
                    ms, min_ms, max_ms = triton.testing.do_bench(
                        lambda: o.backward(retain_graph=True),
                        quantiles=quantiles,
                        warmup=warmup,
                        rep=rep,
                    )

        case "triton" | "triton-fast":
            kwargs: Dict[str, Any] = dict(full_precision=(provider != "triton-fast"))
            if mode == "fwd":
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: attention(q, k, v, pos_q, pos_k, **kwargs),
                    quantiles=quantiles,
                    warmup=warmup,
                    rep=rep,
                )
            else:
                o = attention(q, k, v, pos_q, pos_k, **kwargs).mul(2).sum()
                ms, min_ms, max_ms = triton.testing.do_bench(
                    lambda: o.backward(retain_graph=True),
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
    parser.add_argument("-b", "--bias", default=False, action="store_true", help="Benchmark with ALiBi style bias")
    parser.add_argument("--mode", default="fwd", choices=["fwd", "bwd"], help="Mode to benchmark")
    parser.add_argument("-d", "--dtype", default="fp16", choices=["fp16", "bf16"], help="Data type to test")
    parser.add_argument("-t", "--torch", default=False, action="store_true", help="Benchmark Torch (slow)")
    parser.add_argument(
        "-p",
        "--providers",
        default=["triton", "triton-fast", "flash"],
        nargs="+",
        choices=["triton", "triton-fast", "torch", "flash"],
        help="Additional variations to benchmark",
    )
    return parser.parse_args()


def main(args: Namespace):
    test_configs = list(range(args.step, args.QK + 1, args.step))
    PROVIDER_NAMES = {
        "triton": "Triton",
        "triton-fast": "Triton (FP16 intermediate)",
        "torch": "Torch",
        "flash": f"Flash Attention 2" + (" (no bias)" if args.bias else ""),
    }
    line_names = [PROVIDER_NAMES[p] for p in args.providers]

    total_tests = len(args.providers) * len(test_configs) * len(args.D)
    bar = tqdm(total=total_tests, desc="Benchmarking")
    for d in args.D:
        plot_name = f"flash-attn-d={d}-{args.mode}-{args.dtype}" + ("-bias" if args.bias else "")
        xlabel = f"Input size (Lx{d}, Lx{d})"
        ylabel = f"{'Forward' if args.mode == 'fwd' else 'Backward'} Pass Latency @ {args.dtype.upper()} (ms)"
        benchmark = triton.testing.perf_report(
            triton.testing.Benchmark(
                x_names=["Q", "K"],
                x_vals=test_configs,
                line_arg="provider",
                line_vals=args.providers,
                line_names=line_names,
                xlabel=xlabel,
                ylabel=ylabel,
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
                bias=args.bias,
                mode=args.mode,
                dtype=args.dtype,
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
