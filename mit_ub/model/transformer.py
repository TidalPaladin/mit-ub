from copy import deepcopy
from functools import partial
from typing import Sequence, Type, cast

import torch.nn as nn
from torch import Tensor

from .gqa import MultiHeadAttention
from .kernels.relu2 import ReLU2
from .lora import LoRATarget, SupportsLoRA, apply_lora, freeze_non_lora


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
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads or nhead

        head_dim = d_model // nhead
        kv_dim = head_dim * self.num_kv_heads

        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(d_model, d_model, bias=False),
            k_proj=nn.Linear(d_model, kv_dim, bias=False),
            v_proj=nn.Linear(d_model, kv_dim, bias=False),
            output_proj=nn.Linear(d_model, d_model, bias=False),
            is_causal=False,
            attn_dropout=dropout,
            q_norm=norm_layer(head_dim, eps=layer_norm_eps) if qk_norm else None,
            k_norm=norm_layer(head_dim, eps=layer_norm_eps) if qk_norm else None,
        )

        # MLP up-project is a GLU-variant -> F.silu(W1x + b1) * F.sigmoid(W2x + b2).
        # Improves probe accuracy, convergence rate, and reduces feature variance
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        if gate_activation is not None:
            self.gate = nn.Sequential(
                nn.Linear(d_model, dim_feedforward, bias=False),
                deepcopy(gate_activation),
            )
        else:
            self.gate = None

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = deepcopy(activation)

    def forward(self, x: Tensor) -> Tensor:
        # Self attention
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        y = self.self_attn(y, y)
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
        _apply_lora = partial(
            apply_lora, rank=rank, alpha=alpha, dropout=dropout, use_bias=use_bias, quantize_base=quantize_base
        )

        if LoRATarget.ATTENTION in target:
            self.self_attn.q_proj = _apply_lora(cast(nn.Linear, self.self_attn.q_proj))
            self.self_attn.k_proj = _apply_lora(cast(nn.Linear, self.self_attn.k_proj))
            self.self_attn.v_proj = _apply_lora(cast(nn.Linear, self.self_attn.v_proj))
            self.self_attn.output_proj = _apply_lora(cast(nn.Linear, self.self_attn.output_proj))

        if LoRATarget.FEEDFORWARD in target:
            self.linear1 = _apply_lora(cast(nn.Linear, self.linear1))
            self.linear2 = _apply_lora(cast(nn.Linear, self.linear2))
            if self.gate is not None:
                self.gate[0] = _apply_lora(cast(nn.Linear, self.gate[0]))

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
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        d_kv = d_kv or d_model
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads or nhead

        head_dim = d_model // nhead
        kv_dim = head_dim * self.num_kv_heads
        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(d_model, d_model, bias=False),
            k_proj=nn.Linear(d_model, kv_dim, bias=False),
            v_proj=nn.Linear(d_model, kv_dim, bias=False),
            output_proj=nn.Linear(d_model, d_model, bias=False),
            is_causal=False,
            attn_dropout=dropout,
            q_norm=norm_layer(head_dim, eps=layer_norm_eps) if qk_norm else None,
            k_norm=norm_layer(head_dim, eps=layer_norm_eps) if qk_norm else None,
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_dim,
            q_proj=nn.Linear(d_model, d_model, bias=False),
            k_proj=nn.Linear(d_kv, kv_dim, bias=False),
            v_proj=nn.Linear(d_kv, kv_dim, bias=False),
            output_proj=nn.Linear(d_model, d_model, bias=False),
            is_causal=False,
            attn_dropout=dropout,
            q_norm=norm_layer(head_dim, eps=layer_norm_eps) if qk_norm else None,
            k_norm=norm_layer(head_dim, eps=layer_norm_eps) if qk_norm else None,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False)
        if gate_activation is not None:
            self.gate = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                deepcopy(gate_activation),
            )
        else:
            self.gate = None

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=False)

        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm3 = norm_layer(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = deepcopy(activation)

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Self attention
        x = q
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        y = self.self_attn(y, y)
        x = x + y

        # Cross attention
        y = self.norm2(x)
        y = self.cross_attn(y, kv)
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
        _apply_lora = partial(
            apply_lora, rank=rank, alpha=alpha, dropout=dropout, use_bias=use_bias, quantize_base=quantize_base
        )

        if LoRATarget.ATTENTION in target:
            for layer in (self.self_attn, self.cross_attn):
                layer.q_proj = _apply_lora(cast(nn.Linear, layer.q_proj))
                layer.k_proj = _apply_lora(cast(nn.Linear, layer.k_proj))
                layer.v_proj = _apply_lora(cast(nn.Linear, layer.v_proj))
                layer.output_proj = _apply_lora(cast(nn.Linear, layer.output_proj))

        if LoRATarget.FEEDFORWARD in target:
            self.linear1 = _apply_lora(cast(nn.Linear, self.linear1))
            self.linear2 = _apply_lora(cast(nn.Linear, self.linear2))
            if self.gate is not None:
                self.gate[0] = _apply_lora(cast(nn.Linear, self.gate[0]))

        # Freeze all non-LoRA matrices/weights
        freeze_non_lora(self)
        return self
