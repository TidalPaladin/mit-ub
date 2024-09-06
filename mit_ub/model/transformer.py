from copy import deepcopy
from typing import Sequence, cast

import torch
import torch.nn as nn
from torch import Tensor

from .kernels.attention import MultiheadAttention
from .kernels.relu2 import ReLU2
from .lora import LoRATarget, SupportsLoRA, apply_lora, freeze_non_lora


@torch.no_grad()
def init_alibi(lower: int, upper: int, nhead: int) -> Tensor:
    return torch.logspace(lower, upper, nhead, base=2).reciprocal_().neg_()


class TransformerEncoderLayer(nn.Module, SupportsLoRA):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = ReLU2(),
        gate_activation: nn.Module | None = None,
        layer_norm_eps: float = 1e-5,
        alibi_lower: int | None = None,
        alibi_upper: int | None = None,
    ):
        super().__init__()
        self.nhead = nhead

        # Configure ALiBi if bounds are provided
        if alibi_lower is None and alibi_upper is None:
            self.alibi = None
        elif alibi_lower is not None and alibi_upper is not None:
            self.register_buffer("alibi", init_alibi(alibi_lower, alibi_upper, nhead))
        else:
            raise ValueError("Either both `alibi_lower` and `alibi_upper` must be provided, or neither")

        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # MLP up-project is a GLU-variant -> F.silu(W1x + b1) * F.sigmoid(W2x + b2).
        # Improves probe accuracy, convergence rate, and reduces feature variance
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        if gate_activation is not None:
            self.gate = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                deepcopy(gate_activation),
            )
        else:
            self.gate = None

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = deepcopy(activation)

    def forward(
        self,
        x: Tensor,
        pos: Tensor | None = None,
        full_precision: bool = True,
        mask_threshold: float | None = None,
    ) -> Tensor:
        # Self attention
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        if self.alibi is not None:
            slopes = self.alibi.view(1, H).expand(B, -1).contiguous()
            y = self.self_attn(y, y, y, pos, pos, slopes, mask_threshold=mask_threshold)
        else:
            y = self.self_attn(y, y, y)
        x = x + y

        # MLP
        y = self.norm2(x)
        if self.gate is not None:
            y = self.activation(self.linear1(y)) * self.gate(y)
        else:
            y = self.activation(self.linear1(y))
        y = self.dropout(y)
        y = self.linear2(y)
        x = x + y

        return x

    def apply_lora(
        self,
        target: Sequence[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
    ) -> nn.Module:
        if LoRATarget.ATTENTION in target:
            raise NotImplementedError("Attention not implemented")

        if LoRATarget.FEEDFORWARD in target:
            self.linear1 = apply_lora(cast(nn.Linear, self.linear1), rank, alpha, dropout, use_bias, quantize_base)
            self.linear2 = apply_lora(cast(nn.Linear, self.linear2), rank, alpha, dropout, use_bias, quantize_base)
            if self.gate is not None:
                self.gate[0] = apply_lora(cast(nn.Linear, self.gate[0]), rank, alpha, dropout, use_bias, quantize_base)

        # Freeze all non-LoRA matrices/weights
        freeze_non_lora(self)
        return self


class TransformerDecoderLayer(nn.Module, SupportsLoRA):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_kv: int | None = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = ReLU2(),
        gate_activation: nn.Module | None = None,
        layer_norm_eps: float = 1e-5,
        alibi_lower: int | None = None,
        alibi_upper: int | None = None,
    ):
        super().__init__()
        d_kv = d_kv or d_model
        self.nhead = nhead

        # Configure ALiBi if bounds are provided
        if alibi_lower is None and alibi_upper is None:
            self.alibi = None
        elif alibi_lower is not None and alibi_upper is not None:
            self.register_buffer("alibi", init_alibi(alibi_lower, alibi_upper, nhead))
        else:
            raise ValueError("Either both `alibi_lower` and `alibi_upper` must be provided, or neither")

        self.cross_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True, kdim=d_kv, vdim=d_kv)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        if gate_activation is not None:
            self.gate = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                deepcopy(gate_activation),
            )
        else:
            self.gate = None

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = deepcopy(activation)

    @torch.no_grad()
    def init_alibi(self, lower: int, upper: int) -> Tensor:
        return torch.logspace(lower, upper, self.nhead, base=2).reciprocal_().neg_()

    def forward(
        self,
        q: Tensor,
        kv: Tensor,
        pos_q: Tensor | None = None,
        pos_k: Tensor | None = None,
        full_precision: bool = True,
        mask_threshold: float | None = None,
    ) -> Tensor:
        # Self attention
        x = q
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        slopes: Tensor | None = None
        if self.alibi is not None:
            slopes = self.alibi.view(1, H).expand(B, -1).contiguous()
            y = self.self_attn(y, y, y, pos_q, pos_q, slopes, mask_threshold=mask_threshold)
        else:
            y = self.self_attn(y, y, y)
        x = x + y

        # Cross attention
        y = self.norm2(x)
        if self.alibi is not None:
            assert slopes is not None
            y = self.cross_attn(y, kv, kv, pos_q, pos_k, slopes, mask_threshold=mask_threshold)
        else:
            y = self.cross_attn(y, kv, kv)
        x = x + y

        # MLP
        y = self.norm3(x)
        if self.gate is not None:
            y = self.activation(self.linear1(y)) * self.gate(y)
        else:
            y = self.activation(self.linear1(y))
        y = self.dropout(y)
        y = self.linear2(y)
        x = x + y

        return x

    def apply_lora(
        self,
        target: Sequence[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
    ) -> nn.Module:
        if LoRATarget.ATTENTION in target:
            raise NotImplementedError("Attention not implemented")

        if LoRATarget.FEEDFORWARD in target:
            self.linear1 = apply_lora(cast(nn.Linear, self.linear1), rank, alpha, dropout, use_bias, quantize_base)
            self.linear2 = apply_lora(cast(nn.Linear, self.linear2), rank, alpha, dropout, use_bias, quantize_base)
            if self.gate is not None:
                self.gate[0] = apply_lora(cast(nn.Linear, self.gate[0]), rank, alpha, dropout, use_bias, quantize_base)

        # Freeze all non-LoRA matrices/weights
        freeze_non_lora(self)
        return self
