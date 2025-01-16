from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..activations import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, Activation
from ..helpers import compile_is_disabled, max_autotune
from .layer_scale import LayerScale
from .mlp import MLP, NormType


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": max_autotune(),
        "triton.cudagraph_trees": max_autotune(),
        "shape_padding": True,
    },
    disable=compile_is_disabled(),
)
def dispatch(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_k: Tensor,
    b_k: Tensor | None,
    num_heads: int,
    dropout: float = 0.0,
    training: bool = False,
    # fmt: on
) -> Tensor:
    # In dispatch queries are slots, keys and values are tokens
    # Q -> directly use the slot vectors
    # K -> project tokens to keys
    # V -> directly use the token vectors
    q = rearrange(q, "b l (h d) -> b h l d", h=num_heads)
    k = rearrange(F.linear(kv, w_k, b_k), "b l (h d) -> b h l d", h=num_heads)
    v = rearrange(kv, "b l (h d) -> b h l d", h=num_heads)

    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if training else 0.0)

    return rearrange(o, "b h l d -> b l (h d)")


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": max_autotune(),
        "triton.cudagraph_trees": max_autotune(),
        "shape_padding": True,
    },
    disable=compile_is_disabled(),
)
def combine(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_q: Tensor, w_k: Tensor,
    b_q: Tensor | None, b_k: Tensor | None,
    num_heads: int,
    dropout: float = 0.0,
    training: bool = False,
    # fmt: on
) -> Tensor:
    # In combine queries are tokens, keys and values are slots
    # Q -> project tokens to queries
    # K -> project slot to keys
    # V -> directly use the slot output vectors
    q = rearrange(F.linear(q, w_q, b_q), "b l (h d) -> b h l d", h=num_heads)
    k = rearrange(F.linear(kv, w_k, b_k), "b l (h d) -> b h l d", h=num_heads)
    v = rearrange(kv, "b l (h d) -> b h l d", h=num_heads)

    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if training else 0.0)

    return rearrange(o, "b h l d -> b l (h d)")


class SoftMoE(nn.Module):
    """
    Implements a mixture of experts inspired by the paper
    "From Sparse to Soft Mixtures of Experts" (https://arxiv.org/abs/2308.00951).

    This implementation differs in that it formulates routing as multi-head attention between slots and tokens.
    It is suggested to set ``num_slots = num_experts = num_tokens``. Experiments suggest adding a small
    number of such MoE layers to the end of the network (e.g. 1-2 for a 12 layer transformer).

    Args:
        expert: The expert module to be used. This module will be duplicated `num_experts` times.
            The expert should provide a :meth:`reset_parameters` method to initialize the parameters uniquely
            for each expert.
        dim: The dimension of the input tokens.
        num_experts: The number of expert modules.
        num_slots: The number of slots for routing tokens. Must be greater than or equal to `num_experts`.
        nhead: The number of attention heads.
        dropout: Dropout probability for the attention mechanism.
        activation: Activation function.
        gate_activation: Gate activation function (optional).
        bias: Whether to use bias in the experts.
        qk_norm: Whether to use QK normalization in the dispatch and combine steps.
        norm: Whether to use layer normalization on the input.
        norm_type: The type of normalization to use.

    Raises:
        - ValueError: If `num_slots` is less than `num_experts`
        - ValueError: If `num_slots` is not divisible by `num_experts`.

    Shapes:
        - Input: :math:`(B, L, D)` where B is the batch size, L is the sequence length, and D is the token dimension.
        - Output: :math:`(B, L, D)`

    """

    def __init__(
        self,
        dim: int,
        dim_feedfoward: int,
        num_experts: int,
        num_slots: int,
        nhead: int,
        dropout: float = 0.0,
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        bias: bool = True,
        norm: bool = False,
        norm_type: NormType = NormType.LAYER_NORM,
        layer_scale: float | None = None,
    ):
        super().__init__()
        if num_slots < num_experts:
            raise ValueError("num_slots must be greater than or equal to num_experts")
        if not num_slots % num_experts == 0:
            raise ValueError(f"num_slots must be divisible by num_experts, got {num_slots} and {num_experts}")
        self.nhead = nhead
        self.dropout = dropout

        self.slots = nn.Parameter(torch.empty(1, num_slots, dim))
        self.w_k_dispatch = nn.Parameter(torch.empty(dim, dim))
        self.w_q_combine = nn.Parameter(torch.empty(dim, dim))
        self.w_k_combine = nn.Parameter(torch.empty(dim, dim))
        if bias:
            self.b_k_dispatch = nn.Parameter(torch.empty(dim))
            self.b_q_combine = nn.Parameter(torch.empty(dim))
            self.b_k_combine = nn.Parameter(torch.empty(dim))
        else:
            self.register_parameter("b_k_dispatch", None)
            self.register_parameter("b_q_combine", None)
            self.register_parameter("b_k_combine", None)

        if norm:
            self.pre_norm = nn.LayerNorm(dim)
        else:
            self.pre_norm = nn.Identity()

        self.experts = nn.ModuleList(
            [
                MLP(
                    dim,
                    dim_feedfoward,
                    dim,
                    dropout,
                    activation,
                    gate_activation,
                    bias,
                    norm=True,
                    norm_type=norm_type,
                )
                for _ in range(num_experts)
            ]
        )
        self.layer_scale = LayerScale(dim, layer_scale) if layer_scale is not None else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.slots, std=0.02)
        nn.init.xavier_uniform_(self.w_k_dispatch)
        nn.init.xavier_uniform_(self.w_q_combine)
        nn.init.xavier_uniform_(self.w_k_combine)
        if self.b_k_dispatch is not None:
            nn.init.zeros_(self.b_k_dispatch)
        if self.b_q_combine is not None:
            nn.init.zeros_(self.b_q_combine)
        if self.b_k_combine is not None:
            nn.init.zeros_(self.b_k_combine)
        for expert in self.experts:
            expert.reset_parameters()
        if not isinstance(self.pre_norm, nn.Identity):
            self.pre_norm.reset_parameters()
        if not isinstance(self.layer_scale, nn.Identity):
            self.layer_scale.reset_parameters()

    @property
    def num_slots(self) -> int:
        return self.slots.shape[1]

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def norm(self) -> bool:
        return not isinstance(self.pre_norm, nn.Identity)

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, " f"num_slots={self.num_slots}, "

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # Dispatch (pre-normalized by MHA, residual on slots)
        B = x.shape[0]
        q = self.slots.expand(B, -1, -1)
        o = dispatch(q, x, self.w_k_dispatch, self.b_k_dispatch, self.nhead, self.dropout, self.training)

        # Experts (pre-normalized by MLP, residual on slots)
        num_slots = o.shape[1]
        num_experts = len(self.experts)
        slots_per_expert = num_slots // num_experts
        output: List[Tensor] = []
        for i in range(num_experts):
            start = i * slots_per_expert
            end = start + slots_per_expert
            x_i = o[..., start:end, :]
            o_i = self.experts[i](x_i)
            output.append(o_i)
        o = o + torch.cat(output, dim=1)

        # Combine (pre-normalized by MHA, no residual)
        o = combine(
            x,
            o,
            self.w_q_combine,
            self.w_k_combine,
            self.b_q_combine,
            self.b_k_combine,
            self.nhead,
            self.dropout,
            self.training,
        )

        return self.layer_scale(o)
