from typing import Callable, Type

import torch.nn as nn
from torch import Tensor
from torchvision.ops import StochasticDepth

from .attention import MultiHeadAttention
from .layer_scale import LayerScale
from .mlp import MLP, relu2
from .soft_moe import SoftMoE


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        layer_norm_eps: float = 1e-5,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        layer_scale: float | None = None,
        num_experts: int | None = None,
        num_slots: int | None = None,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads or nhead

        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
        )

        mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            dropout,
            activation,
            gate_activation,
            bias=False,
        )
        if (num_experts is None) != (num_slots is None):
            raise ValueError("num_experts and num_slots must be both set or both None")
        elif num_experts is not None and num_slots is not None:
            self.mlp = SoftMoE(mlp, d_model, num_experts, num_slots, nhead, dropout=dropout)
        else:
            self.mlp = mlp

        self.stochastic_depth = StochasticDepth(stochastic_depth, mode="row")
        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)

        self.layer_scale_attn = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()
        self.layer_scale_mlp = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # Self attention
        y = self.norm1(x)
        y = self.self_attn(y, y, y)
        x = x + self.stochastic_depth(self.layer_scale_attn(y))

        # MLP
        y = self.norm2(x)
        y = self.mlp(y)
        x = x + self.stochastic_depth(self.layer_scale_mlp(y))

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_kv: int | None = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        layer_norm_eps: float = 1e-5,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        layer_scale: float | None = None,
        num_experts: int | None = None,
        num_slots: int | None = None,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        d_kv = d_kv or d_model
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads or nhead

        self.self_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            kdim=d_kv,
            vdim=d_kv,
        )

        mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            dropout,
            activation,
            gate_activation,
            bias=False,
        )
        if (num_experts is None) != (num_slots is None):
            raise ValueError("num_experts and num_slots must be both set or both None")
        elif num_experts is not None and num_slots is not None:
            self.mlp = SoftMoE(mlp, d_model, num_experts, num_slots, nhead, dropout=dropout)
        else:
            self.mlp = mlp

        self.stochastic_depth = StochasticDepth(stochastic_depth, mode="row")
        self.norm1 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm2 = norm_layer(d_model, eps=layer_norm_eps)
        self.norm3 = norm_layer(d_model, eps=layer_norm_eps)

        self.layer_scale_attn = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()
        self.layer_scale_cross = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()
        self.layer_scale_mlp = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Self attention
        x = q
        y = self.norm1(x)
        y = self.self_attn(y, y, y)
        x = x + self.stochastic_depth(self.layer_scale_attn(y))

        # Cross attention
        y = self.norm2(x)
        y = self.cross_attn(y, kv, kv)
        x = x + self.stochastic_depth(self.layer_scale_cross(y))

        # MLP
        y = self.norm3(x)
        y = self.mlp(y)
        x = x + self.stochastic_depth(self.layer_scale_mlp(y))

        return x
