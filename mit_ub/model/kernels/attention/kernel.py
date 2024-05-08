import math
from typing import Any, Dict, Final, Tuple, cast

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor
from triton import JITFunction
from triton_helpers import TENSOR_CORE_K
from triton_helpers.heuristics import BoundaryCheckHeuristic, IsBlockMultiple, PowerOfTwoHeuristic, SMHeuristic
from triton_helpers.ops import to_tensor

from ..distance.kernel import euclidean_distance_inner


LN_2_RECIP: Final = 1 / math.log(2)


# @triton.autotune(
#    configs=[
#        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}),
#        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=1),
#        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2),
#        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
#        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}),
#        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
#    ],
#    key=["M", "N", "D", "D_pos", "HAS_BIAS", "HAS_MASK_THRESH", "ACCUMULATOR_DTYPE", "PROB_DTYPE", "DOT_DTYPE"],
#    # No boundary checks so don't try configs that are bigger than the input
#    prune_configs_by={
#        "early_config_prune": PruneConfigs.compose(
#            PruneConfigs("BLOCK_M", high="M"),
#            PruneConfigs("BLOCK_N", high="N"),
#        )
#    },
# )
@triton.heuristics(
    {
        "BLOCK_M": SMHeuristic("q_p", ("B", "H", "M"), min_size=TENSOR_CORE_K, max_size=128),
        "BLOCK_N": lambda args: max(
            # FP16 beats flash when BLOCK_M == BLOCK_N at max 128
            # BF16 seems to benefit from half BLOCK_N size
            args["BLOCK_M"] // (2 if args["q_p"].dtype == torch.bfloat16 else 1),
            TENSOR_CORE_K,
        ),
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D", min_val=TENSOR_CORE_K),
        "BLOCK_POSDIM": PowerOfTwoHeuristic("D_pos", min_val=TENSOR_CORE_K),
        "BOUND_CHECK_Q": BoundaryCheckHeuristic(["M", "D"], ["BLOCK_M", "BLOCK_HEADDIM"]),
        "BOUND_CHECK_K": BoundaryCheckHeuristic(["N", "D"], ["BLOCK_N", "BLOCK_HEADDIM"]),
        "BOUND_CHECK_POSQ": BoundaryCheckHeuristic(["M", "D_pos"], ["BLOCK_M", "BLOCK_POSDIM"]),
        "BOUND_CHECK_POSK": BoundaryCheckHeuristic(["N", "D_pos"], ["BLOCK_N", "BLOCK_POSDIM"]),
        "BOUND_CHECK_LOGIT": BoundaryCheckHeuristic("M", "BLOCK_M"),
        "EVEN_K": IsBlockMultiple("N", "BLOCK_N"),
        "num_warps": lambda args: 4 if args["D"] <= 64 else 8,
    }
)
@triton.jit
def _fwd_kernel(
    # fmt: off
    # Inputs
    q_p, k_p, v_p, logit_scale_p, pos_q_p, pos_k_p, pos_slopes_p, out_p, 
    SOFTMAX_SCALE: tl.constexpr,
    # Sizes 
    B: int, H: tl.constexpr, M: int, N: int, D: tl.constexpr, D_pos: tl.constexpr,
    # Q strides
    stride_q_b: int, stride_q_h: int, stride_q_m: int,
    # K strides
    stride_k_b: int, stride_k_h: int, stride_k_n: int,
    # V strides
    stride_v_b: int, stride_v_h: int, stride_v_n: int,
    # Logit scale strides
    stride_logit_b: int, stride_logit_h: int,
    # Position Q strides
    stride_posq_b: int, stride_posq_h: int, stride_posq_m: int,
    # Position K strides
    stride_posk_b: int, stride_posk_h: int, stride_posk_n: int,
    # Slopes strides
    stride_slopes_b: int,
    # Output strides
    stride_o_b: int, stride_o_h: int, stride_o_m: int,
    # Dtypes
    ACCUMULATOR_DTYPE: tl.constexpr, PROB_DTYPE: tl.constexpr, DOT_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_MASK_THRESH: tl.constexpr, MASK_THRESH: float,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, BLOCK_POSDIM: tl.constexpr,
    # Heuristics
    BOUND_CHECK_Q: tl.constexpr, BOUND_CHECK_K: tl.constexpr, BOUND_CHECK_POSQ: tl.constexpr, BOUND_CHECK_POSK: tl.constexpr, BOUND_CHECK_LOGIT: tl.constexpr,
    EVEN_K: tl.constexpr,
    # fmt: on
):
    QK_SCALE: tl.constexpr = SOFTMAX_SCALE * tl.constexpr(LN_2_RECIP)
    BIAS_SCALE: tl.constexpr = tl.constexpr(LN_2_RECIP)

    # Grid is (L, H * B)
    # Q of shape (B, H, Lq, D)
    # K of shape (B, H, Lk, D)
    # V of shape (B, H, Lk, D)

    # Initialize offsets
    # Each query block gets its own program
    offset_b = tl.program_id(0)
    offset_h = tl.program_id(1)
    start_m = tl.program_id(2)

    # Initialize base pointers to this batch / head
    # This program's block of queries will be loaded by this program for processing and kept there.
    # NOTE: Block pointers may contribute to register spilling due to int64 indexing. Checking the
    # compiled kernel.n_spills indiciates no spilling - is this an issue or not?
    q_p += offset_b * stride_q_b + offset_h * stride_q_h
    k_p += offset_b * stride_k_b + offset_h * stride_k_h
    v_p += offset_b * stride_v_b + offset_h * stride_v_h
    pos_q_p += offset_b * stride_posq_b + offset_h * stride_posq_h
    pos_k_p += offset_b * stride_posk_b + offset_h * stride_posk_h
    pos_slopes_p += offset_b * stride_slopes_b + offset_h
    out_p += offset_b * stride_o_b + offset_h * stride_o_h
    logit_scale_p += offset_b * stride_logit_b + offset_h * stride_logit_h

    # Initialize pointer blocks
    Q_block_ptr = tl.make_block_ptr(
        q_p,
        (M, D),
        (stride_q_m, 1),
        (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        k_p,
        (N, D),
        (stride_k_n, 1),
        (0, 0),
        (BLOCK_N, BLOCK_HEADDIM),
        (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        v_p,
        (N, D),
        (stride_v_n, 1),
        (0, 0),
        (BLOCK_N, BLOCK_HEADDIM),
        (1, 0),
    )
    if HAS_BIAS:
        Posq_block_ptr = tl.make_block_ptr(
            pos_q_p,
            (M, D_pos),
            (stride_posq_m, 1),
            (start_m * BLOCK_M, 0),
            (BLOCK_M, BLOCK_POSDIM),
            (1, 0),
        )
        # For some reason we must load this transposed. Calling `tl.trans` on the block pointer
        # results in a compile error.
        Posk_block_ptr = tl.make_block_ptr(
            pos_k_p,
            (N, D_pos),
            (stride_posk_n, 1),
            (0, 0),
            (BLOCK_N, BLOCK_POSDIM),
            (1, 0),
        )
    else:
        Posq_block_ptr = None
        Posk_block_ptr = None

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
    # NOTE: We use QK_SCALE = 1 / (ln(2) * sqrt(D)) so we can compute logs and exponentials in base 2,
    # which is empirically faster. We must also scale biases by 1 / ln(2) to match.
    value_accumulator = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=ACCUMULATOR_DTYPE)
    softmax_denominator = tl.zeros([BLOCK_M], dtype=ACCUMULATOR_DTYPE)
    query_i_maxdot = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)

    # Q gets loaded into SRAM and stays there. Pre-apply the softmax scale
    # TODO: Specify boundary_check using a conditional tl.constexpr. This is not currently
    # supported by triton, see https://github.com/openai/triton/issues/1946
    # Performance tests indicate no significant loss in performance from doing unnecessary
    # boundary checks. Also, a "-inf" padding option is needed for keys.
    q = tl.load(Q_block_ptr, boundary_check=BOUND_CHECK_Q.value, eviction_policy="evict_first")
    q *= to_tensor(QK_SCALE, q.dtype)

    # If we have bias, we will also load the Q positions and ALiBi slopes
    # NOTE: Pre-apply bias scale into the ALiBi slopes
    if HAS_BIAS:
        pos_q = tl.load(Posq_block_ptr, boundary_check=BOUND_CHECK_POSQ.value)
        pos_slope = (tl.load(pos_slopes_p) * BIAS_SCALE).to(DOT_DTYPE)
    else:
        pos_slope = None
        pos_q = None

    # Iterate over KV blocks
    for i in range(0, N, BLOCK_N):
        # Load K block and compute QK
        k = tl.load(K_block_ptr, boundary_check=BOUND_CHECK_K.value)
        qk = tl.dot(q, tl.trans(k), out_dtype=cast(tl.dtype, DOT_DTYPE))

        # Compute the bias
        if HAS_BIAS:
            pos_k = tl.load(Posk_block_ptr, boundary_check=BOUND_CHECK_POSK.value)
            bias = euclidean_distance_inner(pos_q, pos_k, BLOCK_M, BLOCK_N, SQRT=True)
            # Apply mask if threshold given
            if HAS_MASK_THRESH:
                bias = tl.where(bias <= MASK_THRESH, bias, to_tensor(float("inf"), bias.dtype))
            bias *= pos_slope
            tl.device_assert(bias <= 0, "ALiBi bias must be negative")
            qk += bias

        # Key masking to avoid including garbage in the max logit calculation
        if not EVEN_K:
            if i + BLOCK_N > N:
                mask = (tl.arange(0, BLOCK_N) < N - i)[None, :]
                qk = tl.where(mask, qk, to_tensor(float("-inf"), qk.dtype))

        # Determine the maximum logit seen for each query
        query_i_maxdot_new = tl.maximum(query_i_maxdot, tl.max(qk, 1))

        # Compute scaling constant alpha and rescale the previous contributions, updating the maximum logit
        alpha = tl.math.exp2(query_i_maxdot - query_i_maxdot_new).to(ACCUMULATOR_DTYPE)
        alpha = tl.where(tl.math.isnan(alpha), to_tensor(1, alpha.dtype), alpha)
        tl.device_assert((alpha >= 0) & (alpha <= 1), "alpha must be in [0, 1]")
        query_i_maxdot = query_i_maxdot_new

        # Compute the softmax numerator for each key, applying the maximum logit offset to avoid numerical overflow
        p = tl.math.exp2(qk - query_i_maxdot_new[:, None])
        p = tl.where(tl.math.isnan(p), to_tensor(0, p.dtype), p)
        p = p.to(PROB_DTYPE)
        tl.device_assert((p >= 0) & (p <= 1), "p must be in [0, 1]")

        # Compute the softmax denominator for each query, applying the maximum logit offset to the existing denominator
        softmax_denominator = softmax_denominator * alpha + tl.sum(p, 1).to(ACCUMULATOR_DTYPE)

        # Accumulate the weighted values for this block of V
        v = tl.load(V_block_ptr, boundary_check=BOUND_CHECK_K.value)
        p = p.to(v_p.dtype.element_ty)
        value_accumulator = value_accumulator * alpha[:, None] + tl.dot(p, v, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(
            ACCUMULATOR_DTYPE
        )

        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        if HAS_BIAS:
            Posk_block_ptr = tl.advance(Posk_block_ptr, (BLOCK_N, 0))

    # Compute the final softmax values
    value_accumulator = value_accumulator / softmax_denominator.to(ACCUMULATOR_DTYPE)[:, None]

    # Per Flash Attention 2, we store only logsumexp for the backward pass
    start_m = tl.program_id(2)
    Logit_block_ptr = tl.make_block_ptr(logit_scale_p, (M,), (1,), (start_m * BLOCK_M,), (BLOCK_M,), (0,))
    final_logit_scale = query_i_maxdot + tl.math.log2(softmax_denominator.to(tl.float32))
    tl.store(Logit_block_ptr, final_logit_scale, boundary_check=BOUND_CHECK_LOGIT.value, eviction_policy="evict_first")

    # Write output
    O_block_ptr = tl.make_block_ptr(
        out_p,
        (M, D),
        (stride_o_m, 1),
        (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    tl.store(
        O_block_ptr,
        value_accumulator.to(out_p.dtype.element_ty),
        boundary_check=BOUND_CHECK_Q.value,
        eviction_policy="evict_first",
    )


# @triton.autotune(
#    configs=[
#        triton.Config({"BLOCK_M": 128}),
#        triton.Config({"BLOCK_M": 64}),
#        triton.Config({"BLOCK_M": 32}),
#    ],
#    key=["M", "D"],
#    prune_configs_by={
#        "early_config_prune": PruneConfigs.compose(
#            PruneConfigs("BLOCK_M", high="M"),
#        )
#    },
# )
@triton.heuristics(
    {
        "BLOCK_M": SMHeuristic("Out", ("B", "H", "M"), min_size=1, max_size=256),
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D"),
        "BOUND_CHECK_Q": BoundaryCheckHeuristic(["M", "D"], ["BLOCK_M", "BLOCK_HEADDIM"]),
        "BOUND_CHECK_LOGIT": BoundaryCheckHeuristic("M", "BLOCK_M"),
        "num_warps": lambda args: 4 if args["D"] <= 64 else 8,
    }
)
@triton.jit
def _bwd_do_o_dot(
    # fmt: off
    # Inputs
    Out: tl.pointer_type, DO: tl.pointer_type, Delta: tl.pointer_type,
    # Strides
    stride_o_b: int, stride_o_h: int, stride_o_m: int,
    stride_do_b: int, stride_do_h: int, stride_do_m: int, stride_do_d: int,
    stride_delta_b: int, stride_delta_h: int, stride_delta_m: int,
    # Sizes
    B: int, H: tl.constexpr, D: tl.constexpr, M: int,
    # Blocks
    BLOCK_M: tl.constexpr, BLOCK_HEADDIM: tl.constexpr,
    # Heuristics
    BOUND_CHECK_Q: tl.constexpr, BOUND_CHECK_LOGIT: tl.constexpr,
    # fmt: on
):
    # Initialize offsets
    offset_b = tl.program_id(0)
    offset_h = tl.program_id(1)
    start_m = tl.program_id(2)

    # Seek pointers
    Out += offset_b * stride_o_b + offset_h * stride_o_h
    DO += offset_b * stride_do_b + offset_h * stride_do_h
    Delta += offset_b * stride_delta_b + offset_h * stride_delta_h

    # Load O
    o_block_ptr = tl.make_block_ptr(
        Out,
        (M, D),
        (stride_o_m, 1),
        (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    o = tl.load(o_block_ptr, boundary_check=BOUND_CHECK_Q.value).to(tl.float32)

    # Load DO
    do_block_ptr = tl.make_block_ptr(
        DO,
        (M, D),
        (stride_do_m, stride_do_d),
        (start_m * BLOCK_M, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    do = tl.load(do_block_ptr, boundary_check=BOUND_CHECK_Q.value).to(tl.float32)

    # Compute
    delta = tl.sum(o * do, axis=1).to(Delta.dtype.element_ty)

    # Write output
    delta_block_ptr = tl.make_block_ptr(Delta, (M,), (stride_delta_m,), (start_m * BLOCK_M,), (BLOCK_M,), (0,))
    tl.store(delta_block_ptr, delta, boundary_check=BOUND_CHECK_LOGIT.value, cache_modifier=".wt")


# @triton.autotune(
#    configs=[
#        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}),
#        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}),
#        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}),
#        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32}),
#        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16}),
#    ],
#    key=["D", "M", "N", "ACCUMULATOR_DTYPE", "DOT_DTYPE"],
#    prune_configs_by={
#        "early_config_prune": PruneConfigs.compose(
#            PruneConfigs("BLOCK_M", high="M"),
#            PruneConfigs("BLOCK_N", high="N"),
#        )
#    },
#    reset_to_zero=["dq_p", "lock_p"],
# )
@triton.heuristics(
    {
        "BLOCK_HEADDIM": PowerOfTwoHeuristic("D", min_val=TENSOR_CORE_K),
        "BLOCK_POSDIM": PowerOfTwoHeuristic("D_pos", min_val=TENSOR_CORE_K),
        "BLOCK_N": SMHeuristic("q_p", ("B", "H", "N"), min_size=TENSOR_CORE_K, max_size=128),
        "BLOCK_M": lambda args: max(args["BLOCK_N"] // 2, TENSOR_CORE_K),
        "BOUND_CHECK_Q": BoundaryCheckHeuristic(["M", "D"], ["BLOCK_M", "BLOCK_HEADDIM"]),
        "BOUND_CHECK_K": BoundaryCheckHeuristic(["N", "D"], ["BLOCK_N", "BLOCK_HEADDIM"]),
        "BOUND_CHECK_POSQ": BoundaryCheckHeuristic(["M", "D_pos"], ["BLOCK_M", "BLOCK_POSDIM"]),
        "BOUND_CHECK_POSK": BoundaryCheckHeuristic(["N", "D_pos"], ["BLOCK_N", "BLOCK_POSDIM"]),
        "BOUND_CHECK_LOGIT": BoundaryCheckHeuristic("M", "BLOCK_M"),
        "EVEN_Q": IsBlockMultiple("M", "BLOCK_M"),
        "EVEN_K": IsBlockMultiple("N", "BLOCK_N"),
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
    SOFTMAX_SCALE: tl.constexpr,
    lock_p,
    # Strides
    stride_q_b: int, stride_q_h: int , stride_q_m: int,
    stride_k_b: int, stride_k_h: int , stride_k_n: int, 
    stride_v_b: int, stride_v_h: int , stride_v_n: int, 
    stride_posq_b: int, stride_posq_h: int , stride_posq_m: int,
    stride_posk_b: int, stride_posk_h: int , stride_posk_n: int,
    stride_slopes_b: int,
    stride_do_b: int, stride_do_h: int , stride_do_m: int, stride_do_d: int,
    stride_logit_b: int, stride_logit_h: int,
    # Sizes
    B: int, H: tl.constexpr, M: int, N: int, D: tl.constexpr, D_pos: tl.constexpr,
    # Dtypes
    ACCUMULATOR_DTYPE: tl.constexpr, DOT_DTYPE: tl.constexpr, PROB_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_MASK_THRESH: tl.constexpr, MASK_THRESH: float,
    # Blocks
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_HEADDIM: tl.constexpr, BLOCK_POSDIM: tl.constexpr,
    # Heuristics
    EVEN_Q: tl.constexpr, EVEN_K: tl.constexpr,
    BOUND_CHECK_Q: tl.constexpr, BOUND_CHECK_K: tl.constexpr, BOUND_CHECK_POSQ: tl.constexpr, BOUND_CHECK_POSK: tl.constexpr, BOUND_CHECK_LOGIT: tl.constexpr,
    # Hparams
    ATOMIC_ADD: tl.constexpr = False,
    # fmt: on
):
    QK_SCALE: tl.constexpr = SOFTMAX_SCALE * tl.constexpr(LN_2_RECIP)
    BIAS_SCALE: tl.constexpr = tl.constexpr(LN_2_RECIP)

    # Initialize offsets.
    # Grid will be over (Lk, B*H)
    offset_b = tl.program_id(0)
    offset_h = tl.program_id(1)
    start_n = tl.program_id(2)

    # Seek pointers
    # NOTE: This kernel has significant register pressure. To avoid spilling we favor manual pointers
    # over block pointers (for int32 indexing) and rematerialization over storing intermediate results.
    q_p += offset_b * stride_q_b + offset_h * stride_q_h
    k_p += offset_b * stride_k_b + offset_h * stride_k_h
    v_p += offset_b * stride_v_b + offset_h * stride_v_h
    do_p += offset_b * stride_do_b + offset_h * stride_do_h
    dq_p += offset_b * stride_q_b + offset_h * stride_q_h
    dk_p += offset_b * stride_k_b + offset_h * stride_k_h
    dv_p += offset_b * stride_v_b + offset_h * stride_v_h
    pos_q_p += offset_b * stride_posq_b + offset_h * stride_posq_h
    pos_k_p += offset_b * stride_posk_b + offset_h * stride_posk_h
    pos_slopes_p += offset_b * stride_slopes_b + offset_h
    delta_p += offset_b * stride_logit_b + offset_h * stride_logit_h
    logit_scale_p += offset_b * stride_logit_b + offset_h * stride_logit_h
    if not ATOMIC_ADD:
        lock_p += offset_b * stride_logit_b + offset_h * stride_logit_h

    # Initialize pointer blocks
    Q_block_ptr = tl.make_block_ptr(
        q_p,
        (M, D),
        (stride_q_m, 1),
        (0, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        k_p,
        (N, D),
        (stride_k_n, 1),
        (start_n * BLOCK_N, 0),
        (BLOCK_N, BLOCK_HEADDIM),
        (1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        v_p,
        (N, D),
        (stride_v_n, 1),
        (start_n * BLOCK_N, 0),
        (BLOCK_N, BLOCK_HEADDIM),
        (1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        do_p,
        (M, D),
        (stride_do_m, stride_do_d),
        (0, 0),
        (BLOCK_M, BLOCK_HEADDIM),
        (1, 0),
    )
    Delta_block_ptr = tl.make_block_ptr(delta_p, (M,), (1,), (0,), (BLOCK_M,), (0,))
    Logit_block_ptr = tl.make_block_ptr(logit_scale_p, (M,), (1,), (0,), (BLOCK_M,), (0,))
    if HAS_BIAS:
        Posq_block_ptr = tl.make_block_ptr(
            pos_q_p,
            (M, D_pos),
            (stride_posq_m, 1),
            (0, 0),
            (BLOCK_M, BLOCK_POSDIM),
            (1, 0),
        )
        # For some reason we must load this transposed. Calling `tl.trans` on the block pointer
        # results in a compile error.
        Posk_block_ptr = tl.make_block_ptr(
            pos_k_p,
            (N, D_pos),
            (stride_posk_n, 1),
            (start_n * BLOCK_N, 0),
            (BLOCK_N, BLOCK_POSDIM),
            (1, 0),
        )
    else:
        Posq_block_ptr = None
        Posk_block_ptr = None

    # Load K and V - stay in SRAM for the entire kernel
    k = tl.load(K_block_ptr, boundary_check=BOUND_CHECK_K.value)
    v = tl.load(V_block_ptr, boundary_check=BOUND_CHECK_K.value)

    # Init dv and dk
    # These will be accumulated at the same precision as their destination.
    # Likewise, since dQ is accumulated to HBM every loop iteration, we will accumulate it at the same precision.
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=ACCUMULATOR_DTYPE)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=ACCUMULATOR_DTYPE)

    # Init pos_k
    if HAS_BIAS:
        pos_k = tl.load(Posk_block_ptr, boundary_check=BOUND_CHECK_POSK.value)
        pos_slope = tl.load(pos_slopes_p) * BIAS_SCALE
    else:
        pos_slope = None
        pos_k = None

    # Recall that for matrix multiplication C = A * B,
    # * dC/dA = B.T
    # * dC/dB = A.T
    for i in range(0, tl.cdiv(M, BLOCK_M)):
        q = tl.load(Q_block_ptr, boundary_check=BOUND_CHECK_Q.value)
        do = tl.load(DO_block_ptr, boundary_check=BOUND_CHECK_Q.value)
        logit_scale = tl.load(Logit_block_ptr, boundary_check=BOUND_CHECK_LOGIT.value)
        if HAS_BIAS:
            pos_q = tl.load(Posq_block_ptr, boundary_check=BOUND_CHECK_POSQ.value)
        else:
            pos_q = None

        # Recompute p = softmax(qk), p is (MxN)
        # NOTE: Keep qk in fp32 to avoid overflow and because exponentiation requires FP32
        qk = tl.dot(q, tl.trans(k))
        if HAS_BIAS:
            # NOTE: Compute this in FP32, it's overflow prone
            bias = euclidean_distance_inner(pos_q, pos_k, BLOCK_M, BLOCK_N, METHOD="matmul-nodiag", SQRT=True)
            # Apply mask if threshold given
            if HAS_MASK_THRESH:
                bias = tl.where(bias <= MASK_THRESH, bias, to_tensor(float("inf"), bias.dtype))
            bias = pos_slope * bias
            qk = qk * QK_SCALE + bias
            p = tl.math.exp2(qk - logit_scale[:, None]).to(do_p.dtype.element_ty)
        else:
            p = tl.math.exp2(qk * QK_SCALE - logit_scale[:, None]).to(do_p.dtype.element_ty)

        # compute dL/dv = dL/do * do/dv = dL/do * p
        # Shape do = (MxD)
        # NOTE: `do` is pre-divided by `l`; no normalization here
        dv += tl.dot(tl.trans(p), do, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(ACCUMULATOR_DTYPE)

        # compute dL/dp = dL/do * do/dp = dL/do * v
        # Shape dp = (MxN)
        delta = tl.load(Delta_block_ptr, boundary_check=BOUND_CHECK_LOGIT.value)
        dp = tl.dot(do, tl.trans(v), out_dtype=cast(tl.dtype, DOT_DTYPE))

        # compute dL/ds = dL/dp * dp/ds = p * (dp - delta[:, None])
        # Shape ds = (MxN)
        ds = ((p * (dp - delta[:, None])) * SOFTMAX_SCALE).to(q_p.dtype.element_ty)

        # compute dL/dk = dL/ds * ds/dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(ACCUMULATOR_DTYPE)

        # compute dL/dq = dL/ds * ds/dq = dot(ds, k)
        # NOTE: We do an atomic add here since multiple threads may be writing to the dq location.
        # For some reason tl.atomic_add is much slower than what we have here.
        # Using a counter to avoid initializing dq to 0 results in register spilling, so we just start with
        # zero initialization and add at each step
        offs_m = tl.arange(0, BLOCK_M) * stride_q_m
        offs_d = tl.max_contiguous(tl.arange(0, BLOCK_HEADDIM), BLOCK_HEADDIM)
        q_grid = dq_p + offs_m[:, None] + offs_d[None, :]

        dq = tl.dot(ds, k, out_dtype=cast(tl.dtype, DOT_DTYPE)).to(dq_p.dtype.element_ty)
        if ATOMIC_ADD:
            if EVEN_Q:
                tl.atomic_add(q_grid, dq)
            else:
                q_mask = (tl.arange(0, BLOCK_M)[:, None] < M) & (tl.arange(0, BLOCK_HEADDIM)[None, :] < D)
                tl.atomic_add(dq_p + q_grid, dq, mask=q_mask)
        else:
            while tl.atomic_cas(lock_p, 0, 1) != 0:
                pass
            if EVEN_Q:
                dq += tl.load(q_grid)
                tl.store(q_grid, dq)
            else:
                q_mask = (tl.arange(0, BLOCK_M)[:, None] < M) & (tl.arange(0, BLOCK_HEADDIM)[None, :] < D)
                dq += tl.load(q_grid, mask=q_mask)
                tl.store(q_grid, dq, mask=q_mask)
            tl.atomic_xchg(lock_p, 0)

        # advance pointers
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
        Delta_block_ptr = tl.advance(Delta_block_ptr, BLOCK_M)
        Logit_block_ptr = tl.advance(Logit_block_ptr, BLOCK_M)
        dq_p += BLOCK_M * stride_q_m
        if not ATOMIC_ADD:
            lock_p += 1
        if HAS_BIAS:
            Posq_block_ptr = tl.advance(Posq_block_ptr, (BLOCK_M, 0))

    start_n = tl.program_id(2)
    dk = dk.to(dk_p.dtype.element_ty)
    dv = dv.to(dv_p.dtype.element_ty)

    DK_block_ptr = tl.make_block_ptr(
        dk_p,
        (N, D),
        (stride_k_n, 1),
        (start_n * BLOCK_N, 0),
        (BLOCK_N, BLOCK_HEADDIM),
        (1, 0),
    )
    tl.store(DK_block_ptr, dk, boundary_check=BOUND_CHECK_K.value, cache_modifier=".wt")
    DV_block_ptr = tl.make_block_ptr(
        dv_p,
        (N, D),
        (stride_v_n, 1),
        (start_n * BLOCK_N, 0),
        (BLOCK_N, BLOCK_HEADDIM),
        (1, 0),
    )
    tl.store(DV_block_ptr, dv, boundary_check=BOUND_CHECK_K.value, cache_modifier=".wt")


class SDPA(torch.autograd.Function):

    @torch.cuda.amp.custom_fwd()
    @staticmethod
    def forward(
        # fmt: off
        ctx,
        q, k, v,
        pos_q=None, pos_k=None, slopes=None,
        softmax_scale=None,
        full_precision: bool = True,
        mask_threshold: float | None = None,
        # fmt: on
    ) -> Tensor:
        # shape constraints
        B, H, Lq, D = q.shape
        _, _, Lk, _ = k.shape
        assert k.shape == (B, H, Lk, D)
        assert v.shape == (B, H, Lk, D)
        assert D <= 128, "FlashAttention only support head dimensions up to 128"
        assert D >= 16, "FlashAttention requires head dimensions of at least 16"
        assert triton.next_power_of_2(D) == D, "FlashAttention requires head dimensions to be a power of 2"

        assert q.dtype == k.dtype == v.dtype, "All tensors must have the same type"
        assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
        assert q.is_cuda and k.is_cuda and v.is_cuda
        softmax_scale = softmax_scale or 1.0 / math.sqrt(D)

        # TODO: Benchmark this conversion vs supporting non-contiguous inputs.
        with torch.no_grad():
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        # Check bias
        if pos_q is not None:
            _, _, _, D_pos = pos_q.shape
            assert pos_q.shape == (B, H, Lq, D_pos), "Query position shape must be (B, H, Lq, D_pos)"
            pos_k = pos_k if pos_k is not None else pos_q
            assert pos_k.shape == (B, H, Lk, D_pos), "Key position must be (B, H, Lk, D_pos)"
            slopes = slopes if slopes is not None else torch.full((B, H), -1, device=q.device, dtype=q.dtype)
            assert slopes.shape == (B, H)
            has_bias = True

            # Make positions contiguous if they are non-contiguous along a non batch or head dimension
            with torch.no_grad():
                if pos_q.stride(-1) != 1 or pos_q.stride(-2) != D_pos:
                    pos_q = pos_q.contiguous()
                if pos_k.stride(-1) != 1 or pos_k.stride(-2) != D_pos:
                    pos_k = pos_k.contiguous()
                slopes = slopes.contiguous()

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
            return (B, H, triton.cdiv(Lq, META["BLOCK_M"]))

        # fmt: off
        cast(JITFunction, _fwd_kernel)[grid](
            q, k, v, logit_scale, pos_q, pos_k, slopes, o,
            softmax_scale,
            B, H, Lq, Lk, D, D_pos,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            logit_scale.stride(0), logit_scale.stride(1),
            pos_q.stride(0), pos_q.stride(1), pos_q.stride(2),
            pos_k.stride(0), pos_k.stride(1), pos_k.stride(2),
            slopes.stride(0),
            o.stride(0), o.stride(1), o.stride(2),
            HAS_BIAS=has_bias, HAS_MASK_THRESH=has_bias and mask_threshold is not None, MASK_THRESH=mask_threshold,
            **SDPA._select_dtypes(o.dtype, full_precision),
        )
        # fmt: on

        ctx.save_for_backward(q, k, v, o, logit_scale, pos_q, pos_k, slopes)
        ctx.softmax_scale = softmax_scale
        ctx.full_precision = full_precision
        ctx.has_bias = has_bias
        ctx.mask_threshold = mask_threshold if has_bias else None
        return o

    @torch.cuda.amp.custom_bwd
    @torch.no_grad()
    @staticmethod
    def backward(ctx, do: Tensor):
        q, k, v, o, logit_scale, pos_q, pos_k, slopes = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        B, H, Lk, D = k.shape
        _, _, Lq, _ = q.shape
        D_pos = pos_q.shape[-1] if ctx.has_bias else 0
        assert logit_scale.shape == (B, H, Lq)
        ATOMIC_ADD = False

        do = do.contiguous()

        delta = torch.empty_like(logit_scale, dtype=q.dtype)
        # TODO: We don't know how many locks we need until the autotuner chooses a config.
        # The best we can do is to allocate enough for the worst case. Can this be improved?
        if ATOMIC_ADD:
            locks = torch.empty(1, dtype=torch.int32, device=q.device)
        else:
            locks = torch.zeros(B, H, Lq, dtype=torch.int32, device=q.device)

        ctx.full_precision if D < 64 else True

        def pre_grid(META):
            return (B, H, triton.cdiv(Lq, META["BLOCK_M"]))

        # fmt: off
        cast(JITFunction, _bwd_do_o_dot)[pre_grid](
            o, do, delta,
            o.stride(0), o.stride(1), o.stride(2),
            do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            delta.stride(0), delta.stride(1), delta.stride(2),
            B, H, D, Lq,
        )
        # fmt: on

        def grid(META):
            return (B, H, triton.cdiv(Lk, META["BLOCK_N"]))

        # NOTE: Using a leading dimension and reducing with sum is faster than atomic_add
        # NOTE: We get away with using an empty dq because all dimensions are multiples of their block sizes
        dq = torch.zeros_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        # fmt: off
        cast(JITFunction, _bwd_kernel)[grid](
           q, k, v, logit_scale, pos_q, pos_k, slopes,
           do, dq, dk, dv, delta,
           ctx.softmax_scale,
           locks,
           q.stride(0), q.stride(1), q.stride(2),
           k.stride(0), k.stride(1), k.stride(2),
           v.stride(0), v.stride(1), v.stride(2),
           pos_q.stride(0), pos_q.stride(1), pos_q.stride(2),
           pos_k.stride(0), pos_k.stride(1), pos_k.stride(2),
           slopes.stride(0),
           do.stride(0), do.stride(1), do.stride(2), do.stride(3),
           logit_scale.stride(0), logit_scale.stride(1),
           B, H, Lq, Lk, D, D_pos,
           HAS_BIAS=ctx.has_bias, HAS_MASK_THRESH=ctx.mask_threshold is not None, MASK_THRESH=ctx.mask_threshold,
           **SDPA._select_dtypes(dq.dtype, ctx.full_precision),
           ATOMIC_ADD=ATOMIC_ADD,
        )
        # fmt: on

        return dq, dk, dv, None, None, None, None, None, None

    @staticmethod
    def _select_dtypes(dtype: torch.dtype, full_precision: bool) -> Dict[str, Any]:
        # BF16 is slower at reduced precision and it doesn't seem possible to fix.
        # This is probably because dot products do not support BF16 accumulators so a lot of casting is required.
        if full_precision or dtype is torch.bfloat16:
            return dict(
                ACCUMULATOR_DTYPE=tl.float32,
                PROB_DTYPE=tl.float32,
                DOT_DTYPE=tl.float32,
            )
        else:
            return dict(
                # No performance hit from accumulating in FP32
                ACCUMULATOR_DTYPE=tl.float32,
                PROB_DTYPE=tl.float16,
                DOT_DTYPE=tl.float16,
            )


def attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    pos_q: Tensor | None = None,
    pos_k: Tensor | None = None,
    slopes: Tensor | None = None,
    softmax_scale: float | None = None,
    full_precision: bool = True,
    mask_threshold: float | None = None,
) -> Tensor:
    r"""Computes scaled dot product attention using Flash Attention 2.

    Supports positional encodings for queries and keys that will be constructed using the Euclidean distance.
    This is implemented such that the memory requirement is linear. When inputs are non-CUDA tensors, the function
    will fallback to the baseline flash attention implementation in PyTorch. In this case there will be a quadratic memory
    requirement to store the ALiBi mask.

    .. note::
        This implementation has been optimized on a RTX 3090 and may not be optimal on other GPUs.

    .. note::
        Passing ``float('inf')`` to a positional encoding will result in that token being masked out in the attention
        computation, regardless of the value of the mask threshold.

    Args:
        q: Query tensor of shape `(B, H, Lq, D)`
        k: Key tensor of shape `(B, H, Lk, D)`
        v: Value tensor of shape `(B, H, Lk, D)`
        pos_q: Query position tensor of shape `(B, H, Lq, D_pos)`. Defaults to None.
        pos_k: Key position tensor of shape `(B, H, Lk, D_pos)`. Defaults to None.
        slopes: ALiBi slopes tensor of shape `(B, H)`. Defaults to None.
        softmax_scale: Scale factor for the softmax. Defaults to None, in which case it is set to `1 / sqrt(D)`.
        full_precision: Whether to use full precision for intermediate computations. Defaults to True.
            Setting to false will perform intermediate computations in FP16, yielding a speedup at the cost
            of precision and numerical stability.
        mask_threshold: Threshold for masking. If set, the attention weights will be set to 0 for all query key
            pairs with positional encodings that are further apart than this threshold. By shifting positional encodings
            for all tokens in an attention group, this can be used to implement attention masking while still retaining
            positional encodings
    """
    if q.is_cuda:
        return cast(Tensor, SDPA.apply(q, k, v, pos_q, pos_k, slopes, softmax_scale, full_precision, mask_threshold))
    else:
        B, H, Lq, D = q.shape
        _, _, Lk, _ = k.shape
        if pos_q is not None and pos_k is not None and slopes is not None:
            distance = (pos_q[..., None, :] - pos_k[..., None, :, :]).pow(2).sum(-1).sqrt_().view(B, H, Lq, Lk)
            bias = slopes[..., None, None] * distance
            if mask_threshold is not None:
                bias = torch.where(distance > mask_threshold, bias, bias.new_tensor(float("-inf")))
        else:
            bias = None
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
