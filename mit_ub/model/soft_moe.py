from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class AttentionRouter(nn.Module):
    """
    Implements a mixture of experts routing mechanism inspired by the paper
    "From Sparse to Soft Mixtures of Experts" (https://arxiv.org/abs/2308.00951).

    This class reformulates routing as multi-head scaled dot product attention in both the dispatch and combine steps.
    Under this reformulation we can leverage efficient attention mechanisms to compute the routing.
    Likewise, we limit the head dimension as to mitigate the decay to one-hot routing vectors that
    occurs at large dimension.

    Attributes:
        dim: Dimension of the input tokens.
        num_slots: Number of slots (experts) to route the tokens to.
        nhead: Number of attention heads.
        dropout: Dropout probability for attention weights.
    """

    def __init__(
        self,
        dim: int,
        num_slots: int,
        nhead: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._num_slots = num_slots
        self._nhead = nhead
        self._dim = dim
        self.dropout = dropout

        # Dispatch
        self.slot_query = nn.Parameter(torch.empty(1, nhead, num_slots, self.head_dim))
        self.dispatch_k_proj = nn.Linear(dim, dim, bias=False)

        # Combine
        self.slot_key = nn.Parameter(torch.empty(1, nhead, num_slots, self.head_dim))
        self.combine_q_proj = nn.Linear(dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.slot_query, std=0.02)
        nn.init.trunc_normal_(self.slot_key, std=0.02)
        self.dispatch_k_proj.reset_parameters()
        self.combine_q_proj.reset_parameters()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def head_dim(self) -> int:
        return self.dim // self.nhead

    def _dispatch(self, tokens: Tensor) -> Tensor:
        # Dispatch, queries are slots and key-values are tokens
        q = self.slot_query
        k = self.dispatch_k_proj(tokens)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.nhead)
        v = rearrange(tokens, "b l (h d) -> b h l d", h=self.nhead)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        return rearrange(o, "b h l d -> b l (h d)")

    def _combine(self, tokens: Tensor, expert_out: Tensor) -> Tensor:
        # Combine, queries are tokens, keys are slots, values are slot expert outputs
        q = self.combine_q_proj(tokens)
        q = rearrange(q, "b l (h d) -> b h l d", h=self.nhead)
        k = self.slot_key
        v = rearrange(expert_out, "b l (h d) -> b h l d", h=self.nhead)
        o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0)
        o = rearrange(o, "b h l d -> b l (h d)")
        return o

    def forward(self, tokens: Tensor, expert_out: Tensor | None = None) -> Tensor:
        return self._dispatch(tokens) if expert_out is None else self._combine(tokens, expert_out)


class SoftMoE(nn.Module):
    """
    Implements a soft mixture of experts (soft MoE).

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

    Raises:
        - ValueError: If `num_slots` is less than `num_experts`
        - ValueError: If `num_slots` is not divisible by `num_experts`.

    Shapes:
        - Input: :math:`(B, L, D)` where B is the batch size, L is the sequence length, and D is the token dimension.
        - Output: :math:`(B, L, D)`

    """

    def __init__(
        self,
        expert: nn.Module,
        dim: int,
        num_experts: int,
        num_slots: int,
        nhead: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_slots < num_experts:
            raise ValueError("num_slots must be greater than or equal to num_experts")
        if not num_slots % num_experts == 0:
            raise ValueError(f"num_slots must be divisible by num_experts, got {num_slots} and {num_experts}")

        self.router = AttentionRouter(dim, num_slots, nhead, dropout=dropout)
        self.experts = nn.ModuleList([deepcopy(expert) for _ in range(num_experts)])
        self.reset_parameters()

    def reset_parameters(self):
        self.router.reset_parameters()
        for expert in self.experts:
            if hasattr(expert, "reset_parameters"):
                expert.reset_parameters()

    @property
    def dim(self) -> int:
        return self.router.dim

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    @property
    def num_slots(self) -> int:
        return self.router.num_slots

    @property
    def slots_per_expert(self) -> int:
        return self.num_slots // self.num_experts

    def forward(self, x: Tensor) -> Tensor:
        B, L, D = x.shape
        S, P = self.num_slots, self.slots_per_expert

        # Dispatch tokens to slots
        slots = self.router(x).split(P, -2)
        assert len(slots) == self.num_experts
        assert slots[0].shape == (B, P, D)

        # Run each expert on its respective slots
        output = torch.cat([expert(slot) for expert, slot in zip(self.experts, slots)], dim=-2)
        assert output.shape == (B, S, D)

        # Combine slots to tokens
        output = self.router(x, output)
        assert output.shape == (B, L, D)

        return output
