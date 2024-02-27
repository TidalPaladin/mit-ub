import math
from typing import Final, Tuple

import torch
import triton
import triton.language as tl


# In the case of K=16 we will perform the following operation in each tensor core
# (16x16) * (16x8) = (16x8)
# BFloat16 will only support a FP32 accumulator
TENSOR_CORE_MAX_K: Final = 16


def _boundary_check(l, d, block_l, block_d, flip) -> Tuple[int, ...] | None:
    result = []
    if l % block_l != 0:
        result.append(0)
    if d % block_d != 0:
        result.append(1)
    if flip:
        result = result[::-1]
    return tuple(result) if result else None


# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )
# Boundary checks take time. We will avoid them when possible by using heuristics to determine if they are necessary.
@triton.heuristics(
    {
        "BOUNDARY_CHECK_Q": lambda args: _boundary_check(
            args["M"], args["D"], args["BLOCK_M"], args["BLOCK_HEADDIM"], False
        ),
        "BOUNDARY_CHECK_K": lambda args: _boundary_check(
            args["N"], args["D"], args["BLOCK_N"], args["BLOCK_HEADDIM"], True
        ),
        "BOUNDARY_CHECK_V": lambda args: _boundary_check(
            args["N"], args["D"], args["BLOCK_N"], args["BLOCK_HEADDIM"], False
        ),
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
    BOUNDARY_CHECK_Q: tl.constexpr,
    BOUNDARY_CHECK_K: tl.constexpr,
    BOUNDARY_CHECK_V: tl.constexpr,
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
    query_i_maxdot = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    softmax_denominator = tl.zeros([BLOCK_M], dtype=tl.float32)
    value_accumulator = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Q gets loaded into SRAM and stays there. Pre-apply the softmax scale
    q = tl.load(Q_ptrs, boundary_check=BOUNDARY_CHECK_Q)
    q = (q * softmax_scale).to(k_p.dtype.element_ty)

    for _ in range(0, N, BLOCK_N):
        # Load K and V blocks into SRAM
        k = tl.load(K_block_ptr, boundary_check=BOUNDARY_CHECK_K)
        v = tl.load(V_block_ptr, boundary_check=BOUNDARY_CHECK_V)

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
    tl.store(O_block_ptr, value_accumulator.to(out_p.dtype.element_ty), boundary_check=BOUNDARY_CHECK_Q)


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
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    B = q.shape[0]
    H = q.shape[2]
    D = q.shape[3]

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), TENSOR_CORE_MAX_K)
    BLOCK = 16
    num_warps = 4 if d <= 64 else 8

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
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        # num_warps=num_warps,
        # num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated
