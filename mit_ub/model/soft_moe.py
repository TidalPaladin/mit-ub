from typing import Callable, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .helpers import compile_backend
from .mlp import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, mlp_forward


@torch.compile(
    options={
        "max_autotune": True,
        "shape_padding": True,
        "triton.cudagraph_trees": True,
    },
    # This seems to be bugged when compiled, gives nan loss
    disable=True,
    backend=compile_backend(),
)
def dispatch(
    tokens: Tensor,
    slot_query: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    nhead: int,
    dropout: float = 0.0,
    norm_w: Tensor | None = None,
    norm_b: Tensor | None = None,
    eps: float = 1e-5,
    training: bool = False,
) -> Tensor:
    """Dispatch tokens to expert slots using attention mechanism.

    Args:
        tokens: Input tokens.
        slot_query: Query vectors for expert slots.
        w_k: Weight matrix for key projection.
        w_v: Weight matrix for value projection.
        nhead: Number of attention heads.
        dropout: Dropout probability.
        norm_w: Weight matrix for normalization.
        norm_b: Bias vector for normalization.
        eps: Epsilon value for normalization.
        training: State of the training flag.
    Shapes:
        - tokens: :math:`(B, L, D)` where :math:`B` is batch size, :math:`L` is sequence length, and :math:`D` is the embedding dimension.
        - slot_query: :math:`(H, S, D / H)` where :math:`H` is the number of heads and :math:`S` is the number of slots
        - w_k: :math:`(D, D)`
        - w_v: :math:`(D, D)`
        - Output: :math:`(B, S, D)`
    """
    q = slot_query
    k = rearrange(F.linear(tokens, w_k), "b l (h d) -> b h l d", h=nhead)
    v = rearrange(F.linear(tokens, w_v), "b l (h d) -> b h l d", h=nhead)
    if norm_w is not None:
        q = F.layer_norm(q, q.shape[-1:], norm_w, norm_b, eps)
        k = F.layer_norm(k, k.shape[-1:], norm_w, norm_b, eps)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if training else 0.0)
    return rearrange(o, "b h l d -> b l (h d)")


@torch.compile(
    options={
        "max_autotune": True,
        "shape_padding": True,
        "triton.cudagraph_trees": True,
    },
    disable=True,
    backend=compile_backend(),
)
def combine(
    tokens: Tensor,
    slot_key: Tensor,
    expert_out: Tensor,
    w_q: Tensor,
    nhead: int,
    dropout: float = 0.0,
    norm_w: Tensor | None = None,
    norm_b: Tensor | None = None,
    eps: float = 1e-5,
    training: bool = False,
) -> Tensor:
    """Combine expert outputs using attention mechanism.

    Args:
        tokens: Input tokens.
        slot_key: Key vectors for expert slots.
        expert_out: Output from experts.
        w_q: Weight matrix for query projection.
        nhead: Number of attention heads.
        dropout: Dropout probability.
        norm_w: Weight matrix for normalization.
        norm_b: Bias vector for normalization.
        eps: Epsilon value for normalization.
        training: State of the training flag.

    Shapes:
        - tokens: :math:`(B, L, D)` where :math:`B` is batch size, :math:`L` is sequence length, and :math:`D` is the embedding dimension.
        - slot_key: :math:`(H, S, D / H)` where :math:`H` is the number of heads and :math:`S` is the number of slots.
        - expert_out: :math:`(B, S, D)` where :math:`S` is the number of slots.
        - w_q: :math:`(D, D)`
        - Output: :math:`(B, L, D)`
    """
    q = rearrange(F.linear(tokens, w_q), "b l (h d) -> b h l d", h=nhead)
    k = slot_key
    v = rearrange(expert_out, "b l (h d) -> b h l d", h=nhead)
    if norm_w is not None:
        q = F.layer_norm(q, q.shape[-1:], norm_w, norm_b, eps)
        k = F.layer_norm(k, k.shape[-1:], norm_w, norm_b, eps)
    o = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout if training else 0.0)
    return rearrange(o, "b h l d -> b l (h d)")


def forward_experts(
    x: Tensor,
    w_in: Tensor,
    b_in: Tensor | None,
    w_out: Tensor,
    b_out: Tensor | None,
    activation: Callable[[Tensor], Tensor] = DEFAULT_MLP_ACTIVATION,
    dropout: float = 0.0,
    w_gate: Tensor | None = None,
    b_gate: Tensor | None = None,
    gate_activation: Callable[[Tensor], Tensor] | None = DEFAULT_MLP_GATE_ACTIVATION,
    training: bool = False,
) -> Tensor:
    """Perform forward pass through experts.

    Args:
        x: Input tensor.
        w_in: Input weight matrices for each expert.
        b_in: Input bias vectors for each expert (optional).
        w_out: Output weight matrices for each expert.
        b_out: Output bias vectors for each expert (optional).
        activation: Activation function.
        dropout: Dropout probability.
        w_gate: Gate weight matrices for each expert (optional).
        b_gate: Gate bias vectors for each expert (optional).
        gate_activation: Gate activation function (optional).
        training: State of the training flag.
    Returns:
        Output tensor after passing through experts.

    Shapes:
        - x: :math:`(B, S, D)` where B is batch size, S is number of slots, and D is embedding dimension.
        - w_in: :math:`(E, D, D_{hidden})` where E is number of experts.
        - b_in: :math:`(E, D_{hidden})` if provided.
        - w_out: :math:`(E, D_{hidden}, D)`
        - b_out: :math:`(E, D)` if provided.
        - w_gate: :math:`(E, D, D_{hidden})` if provided.
        - b_gate: :math:`(E, D_{hidden})` if provided.
        - Output: :math:`(B, S, D)`
    """
    num_slots = x.shape[1]
    num_experts = w_in.shape[0]
    slots_per_expert = num_slots // num_experts
    output: List[Tensor] = []
    for i in range(num_experts):
        start = i * slots_per_expert
        end = start + slots_per_expert
        x_i = x[..., start:end, :]
        w_in_i = w_in[i]
        b_in_i = b_in[i] if b_in is not None else None
        w_out_i = w_out[i]
        b_out_i = b_out[i] if b_out is not None else None
        w_gate_i = w_gate[i] if w_gate is not None else None
        b_gate_i = b_gate[i] if b_gate is not None else None
        o_i = mlp_forward(
            x_i,
            w_in_i,
            w_out_i,
            b_in_i,
            b_out_i,
            w_gate_i,
            b_gate_i,
            dropout,
            activation,
            gate_activation,
            training=training,
        )
        output.append(o_i)
    return torch.cat(output, dim=1)


def soft_moe_forward(
    x: Tensor,
    slot_query: Tensor,
    slot_key: Tensor,
    w_dispatch_k: Tensor,
    w_dispatch_v: Tensor,
    w_combine_q: Tensor,
    w_in: Tensor,
    b_in: Tensor | None,
    w_out: Tensor,
    b_out: Tensor | None,
    activation: Callable[[Tensor], Tensor] = DEFAULT_MLP_ACTIVATION,
    dropout: float = 0.0,
    w_gate: Tensor | None = None,
    b_gate: Tensor | None = None,
    gate_activation: Callable[[Tensor], Tensor] | None = DEFAULT_MLP_GATE_ACTIVATION,
    norm_w: Tensor | None = None,
    norm_b: Tensor | None = None,
    pre_norm_w: Tensor | None = None,
    pre_norm_b: Tensor | None = None,
    eps: float = 1e-5,
    training: bool = False,
) -> Tensor:
    if pre_norm_w is not None:
        x = F.layer_norm(x, x.shape[-1:], pre_norm_w, pre_norm_b, eps)

    nhead = slot_query.shape[0]
    y = dispatch(x, slot_query, w_dispatch_k, w_dispatch_v, nhead, dropout, norm_w, norm_b, eps, training)
    y = forward_experts(y, w_in, b_in, w_out, b_out, activation, dropout, w_gate, b_gate, gate_activation, training)
    y = combine(x, slot_key, y, w_combine_q, nhead, dropout, norm_w, norm_b, eps, training)
    return y


class SoftMoE(nn.Module):
    """
    Implements a mixture of experts inspired by the paper
    "From Sparse to Soft Mixtures of Experts" (https://arxiv.org/abs/2308.00951).

    We use a custom attention router defined in :class:`AttentionRouter` to perform the routing step.
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
        activation: Callable[[Tensor], Tensor] = DEFAULT_MLP_ACTIVATION,
        gate_activation: Callable[[Tensor], Tensor] | None = DEFAULT_MLP_GATE_ACTIVATION,
        bias: bool = True,
        qk_norm: bool = False,
        norm: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.gate_activation = gate_activation

        if num_slots < num_experts:
            raise ValueError("num_slots must be greater than or equal to num_experts")
        if not num_slots % num_experts == 0:
            raise ValueError(f"num_slots must be divisible by num_experts, got {num_slots} and {num_experts}")

        # Register parameters
        for prefix in ("w_", "b_"):
            for suffix in ("pre_norm", "in", "out", "gate", "norm"):
                self.register_parameter(f"{prefix}{suffix}", None)

        if norm:
            self.w_pre_norm = nn.Parameter(torch.empty(dim))
            self.b_pre_norm = nn.Parameter(torch.empty(dim))

        # Dispatch
        self.slot_query = nn.Parameter(torch.empty(nhead, num_slots, dim // nhead))
        self.w_dispatch_k = nn.Parameter(torch.empty(dim, dim))
        self.b_dispatch_k = nn.Parameter(torch.empty(dim)) if bias else None
        self.w_dispatch_v = nn.Parameter(torch.empty(dim, dim))
        self.b_dispatch_v = nn.Parameter(torch.empty(dim)) if bias else None

        # Combine
        self.slot_key = nn.Parameter(torch.empty(nhead, num_slots, dim // nhead))
        self.w_combine_q = nn.Parameter(torch.empty(dim, dim))
        self.b_combine_q = nn.Parameter(torch.empty(dim)) if bias else None

        # QK normalization for dispatch and combine
        if qk_norm:
            self.w_norm = nn.Parameter(torch.empty(dim // nhead))
            self.b_norm = nn.Parameter(torch.empty(dim // nhead))

        # Experts
        self.w_in = nn.Parameter(torch.empty(num_experts, dim_feedfoward, dim))
        self.b_in = nn.Parameter(torch.empty(num_experts, dim_feedfoward)) if bias else None
        self.w_out = nn.Parameter(torch.empty(num_experts, dim, dim_feedfoward))
        self.b_out = nn.Parameter(torch.empty(num_experts, dim)) if bias else None
        if gate_activation is not None:
            self.w_gate = nn.Parameter(torch.empty(num_experts, dim_feedfoward, dim))
            self.b_gate = nn.Parameter(torch.empty(num_experts, dim_feedfoward)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("b_"):
                nn.init.zeros_(param)
            elif "norm" in name:
                nn.init.ones_(param)
            elif "slot" in name:
                nn.init.trunc_normal_(param, std=0.02)
            elif name.startswith("w_"):
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError(f"Unsure how to initialize {name}")

    @property
    def num_experts(self) -> int:
        return self.w_in.shape[0]

    @property
    def num_slots(self) -> int:
        return self.slot_query.shape[1]

    @property
    def nhead(self) -> int:
        return self.slot_query.shape[0]

    @property
    def qk_norm(self) -> bool:
        return self.w_norm is not None

    @property
    def norm(self) -> bool:
        return self.w_pre_norm is not None

    def forward(self, x: Tensor) -> Tensor:
        return soft_moe_forward(
            x,
            self.slot_query,
            self.slot_key,
            self.w_dispatch_k,
            self.w_dispatch_v,
            self.w_combine_q,
            self.w_in,
            self.b_in,
            self.w_out,
            self.b_out,
            self.activation,
            self.dropout if self.training else 0.0,
            self.w_gate,
            self.b_gate,
            self.gate_activation,
            self.w_norm,
            self.b_norm,
            self.w_pre_norm,
            self.b_pre_norm,
            training=self.training,
        )
