from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from ..activations import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, Activation
from .attention import MultiHeadAttention
from .mlp import MLP, NormType


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
        qk_norm: bool = False,
        norm: bool = False,
        norm_type: NormType = NormType.LAYER_NORM,
        layer_scale: float | None = None,
    ):
        super().__init__()
        if num_slots < num_experts:
            raise ValueError("num_slots must be greater than or equal to num_experts")
        if not num_slots % num_experts == 0:
            raise ValueError(f"num_slots must be divisible by num_experts, got {num_slots} and {num_experts}")

        self.dispatch = MultiHeadAttention(
            dim,
            nhead,
            nhead,
            dropout,
            qk_norm,
            kdim=dim,
            vdim=dim,
            bias=bias,
            norm=norm,
            norm_type=norm_type,
            kv_norm=norm,
        )
        self.combine = MultiHeadAttention(
            dim,
            nhead,
            nhead,
            dropout,
            qk_norm,
            kdim=dim,
            vdim=dim,
            bias=bias,
            norm=norm,
            norm_type=norm_type,
            kv_norm=norm,
            # Layer scale only on the combine step
            layer_scale=layer_scale,
        )
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
        self.slot_query = nn.Parameter(torch.empty(1, num_slots, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.dispatch.reset_parameters()
        self.combine.reset_parameters()
        for expert in self.experts:
            expert.reset_parameters()
        nn.init.trunc_normal_(self.slot_query, std=0.01)

    @property
    def num_slots(self) -> int:
        return self.slot_query.shape[1]

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def norm(self) -> bool:
        return self.dispatch.norm

    def extra_repr(self) -> str:
        return f"num_experts={self.num_experts}, " f"num_slots={self.num_slots}, "

    def forward(self, x: Tensor) -> Tensor:
        # Dispatch (pre-normalized by MHA, residual on slots)
        B = x.shape[0]
        q = self.slot_query.expand(B, -1, -1)
        o = q + self.dispatch(q, x, x)

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
        o = self.combine(x, o, o)

        return o
