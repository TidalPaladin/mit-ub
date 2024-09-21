from functools import partial
from typing import Sequence, Type, cast

import torch.nn as nn
from torch import Tensor

from .gqa import MultiHeadAttention
from .layer_scale import LayerScale
from .lora import LoRATarget, SupportsLoRA, apply_lora, freeze_non_lora
from .mlp import MLP, ReLU2


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
        layer_scale: float | None = None,
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

        self.mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            dropout,
            activation,
            gate_activation,
            bias=False,
        )

        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)

        self.layer_scale_attn = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()
        self.layer_scale_mlp = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()

    def reset_parameters(self):
        # Self attn
        for proj in (self.self_attn.q_proj, self.self_attn.k_proj, self.self_attn.v_proj, self.self_attn.output_proj):
            proj.reset_parameters()
        if self.self_attn.q_norm is not None and hasattr(self.self_attn.q_norm, "reset_parameters"):
            self.self_attn.q_norm.reset_parameters()
        if self.self_attn.k_norm is not None and hasattr(self.self_attn.k_norm, "reset_parameters"):
            self.self_attn.k_norm.reset_parameters()

        # MLP
        self.mlp.reset_parameters()

        # Layer scale
        for scale in (self.layer_scale_attn, self.layer_scale_mlp):
            if hasattr(scale, "reset_parameters"):
                scale.reset_parameters()

        # Norm
        for norm in (self.norm1, self.norm2):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # Self attention
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        y = self.self_attn(y, y)
        x = x + self.layer_scale_attn(y)

        # MLP
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.layer_scale_mlp(y)

        return x

    def apply_lora(
        self,
        target: Sequence[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        quantize_base: bool = False,
    ) -> nn.Module:
        _apply_lora = partial(apply_lora, rank=rank, alpha=alpha, dropout=dropout, quantize_base=quantize_base)

        if LoRATarget.ATTENTION in target:
            self.self_attn.q_proj = _apply_lora(cast(nn.Linear, self.self_attn.q_proj))
            self.self_attn.k_proj = _apply_lora(cast(nn.Linear, self.self_attn.k_proj))
            self.self_attn.v_proj = _apply_lora(cast(nn.Linear, self.self_attn.v_proj))
            self.self_attn.output_proj = _apply_lora(cast(nn.Linear, self.self_attn.output_proj))

        if LoRATarget.FEEDFORWARD in target:
            self.mlp.apply_lora(target, rank, alpha, dropout, quantize_base)

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
        layer_scale: float | None = None,
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

        self.mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            dropout,
            activation,
            gate_activation,
            bias=False,
        )

        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm3 = norm_layer(d_model, eps=layer_norm_eps)

        self.layer_scale_attn = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()
        self.layer_scale_cross = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()
        self.layer_scale_mlp = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()

    def reset_parameters(self):
        # Attention
        for attn in (self.self_attn, self.cross_attn):
            for proj in (attn.q_proj, attn.k_proj, attn.v_proj, attn.output_proj):
                proj.reset_parameters()
            if attn.q_norm is not None and hasattr(attn.q_norm, "reset_parameters"):
                attn.q_norm.reset_parameters()
            if attn.k_norm is not None and hasattr(attn.k_norm, "reset_parameters"):
                attn.k_norm.reset_parameters()

        # MLP
        self.mlp.reset_parameters()

        # Layer scale
        for scale in (self.layer_scale_attn, self.layer_scale_mlp):
            if hasattr(scale, "reset_parameters"):
                scale.reset_parameters()

        # Norm
        for norm in (self.norm1, self.norm2):
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Self attention
        x = q
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        y = self.self_attn(y, y)
        x = x + self.layer_scale_attn(y)

        # Cross attention
        y = self.norm2(x)
        y = self.cross_attn(y, kv)
        x = x + self.layer_scale_cross(y)

        # MLP
        y = self.norm3(x)
        y = self.mlp(y)
        x = x + self.layer_scale_mlp(y)

        return x

    def apply_lora(
        self,
        target: Sequence[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        quantize_base: bool = False,
    ) -> nn.Module:
        _apply_lora = partial(apply_lora, rank=rank, alpha=alpha, dropout=dropout, quantize_base=quantize_base)

        if LoRATarget.ATTENTION in target:
            for layer in (self.self_attn, self.cross_attn):
                layer.q_proj = _apply_lora(cast(nn.Linear, layer.q_proj))
                layer.k_proj = _apply_lora(cast(nn.Linear, layer.k_proj))
                layer.v_proj = _apply_lora(cast(nn.Linear, layer.v_proj))
                layer.output_proj = _apply_lora(cast(nn.Linear, layer.output_proj))

        if LoRATarget.FEEDFORWARD in target:
            self.mlp.apply_lora(target, rank, alpha, dropout, quantize_base)

        # Freeze all non-LoRA matrices/weights
        freeze_non_lora(self)
        return self
