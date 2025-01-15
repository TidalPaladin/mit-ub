from typing import Tuple

import torch.nn as nn
from torch import Tensor

from ..activations import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, Activation
from ..helpers import Dims2D
from .attention import MultiHeadAttention
from .convnext import ConvNextBlock
from .mlp import MLP, NormType
from .soft_moe import SoftMoE


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        layer_scale: float | None = None,
        num_experts: int | None = None,
        num_slots: int | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = True,
        norm_type: NormType = NormType.LAYER_NORM,
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
            bias=bias,
            norm=True,
            norm_type=norm_type,
            layer_scale=layer_scale,
        )

        if num_experts is not None and num_slots is not None:
            self.mlp = SoftMoE(
                d_model,
                dim_feedforward,
                num_experts,
                num_slots,
                nhead,
                dropout,
                activation,
                gate_activation,
                bias=bias,
                norm=True,
                norm_type=norm_type,
                layer_scale=layer_scale,
            )
        elif num_experts is None and num_slots is None:
            self.mlp = MLP(
                d_model,
                dim_feedforward,
                d_model,
                dropout,
                activation,
                gate_activation,
                bias=bias,
                norm=True,
                norm_type=norm_type,
                layer_scale=layer_scale,
                stochastic_depth=stochastic_depth,
            )
        else:
            raise ValueError("num_experts and num_slots must be both set or both None")

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        # Self attention
        y = self.self_attn(x, x, x)
        x = x + y

        # MLP
        y = self.mlp(x)
        x = x + y

        return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_kv: int | None = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        layer_scale: float | None = None,
        num_experts: int | None = None,
        num_slots: int | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = True,
        self_attn: bool = True,
        kv_norm: bool = False,
        norm_type: NormType = NormType.LAYER_NORM,
    ):
        super().__init__()
        d_kv = d_kv or d_model
        self.nhead = nhead
        self.num_kv_heads = num_kv_heads or nhead

        if self_attn:
            self.self_attn = MultiHeadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                num_kv_heads=self.num_kv_heads,
                dropout=dropout,
                qk_norm=qk_norm,
                bias=bias,
                norm=True,
                norm_type=norm_type,
                layer_scale=layer_scale,
                stochastic_depth=stochastic_depth,
            )
        else:
            self.register_module("self_attn", None)

        self.cross_attn = MultiHeadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            num_kv_heads=self.num_kv_heads,
            dropout=dropout,
            qk_norm=qk_norm,
            kdim=d_kv,
            vdim=d_kv,
            bias=bias,
            norm=True,
            kv_norm=kv_norm,
            norm_type=norm_type,
            layer_scale=layer_scale,
            stochastic_depth=stochastic_depth,
        )

        if num_experts is not None and num_slots is not None:
            self.mlp = SoftMoE(
                d_model,
                dim_feedforward,
                num_experts,
                num_slots,
                nhead,
                dropout,
                activation,
                gate_activation,
                bias=bias,
                norm=True,
                norm_type=norm_type,
                layer_scale=layer_scale,
            )
        elif num_experts is None and num_slots is None:
            self.mlp = MLP(
                d_model,
                dim_feedforward,
                d_model,
                dropout,
                activation,
                gate_activation,
                bias=bias,
                norm=True,
                norm_type=norm_type,
                layer_scale=layer_scale,
                stochastic_depth=stochastic_depth,
            )
        else:
            raise ValueError("num_experts and num_slots must be both set or both None")

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, q: Tensor, kv: Tensor) -> Tensor:
        # Self attention
        x = q
        if self.self_attn is not None:
            y = self.self_attn(x, x, x)
            x = x + y

        # Cross attention
        y = self.cross_attn(x, kv, kv)
        x = x + y

        # MLP
        y = self.mlp(x)
        x = x + y

        return x


class TransformerConvEncoderLayer(TransformerEncoderLayer):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        layer_scale: float | None = None,
        num_experts: int | None = None,
        num_slots: int | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = True,
        norm_type: NormType = NormType.LAYER_NORM,
        kernel_size: int | Dims2D = 7,
    ):
        super().__init__(
            d_model,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            gate_activation,
            num_kv_heads,
            qk_norm,
            layer_scale,
            num_experts,
            num_slots,
            stochastic_depth,
            bias,
            norm_type,
        )
        self.conv = ConvNextBlock(
            d_model,
            dim_feedforward,
            kernel_size=kernel_size,
            activation=activation,
            gate_activation=gate_activation,
            dropout=dropout,
            bias=bias,
            layer_scale=layer_scale,
            stochastic_depth=stochastic_depth,
        )

    def forward(self, x: Tensor, size: Tuple[int, ...]) -> Tensor:
        x = self.conv(x, size)
        return super().forward(x)


class TransformerConvDecoderLayer(TransformerDecoderLayer):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        d_kv: int | None = None,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        layer_scale: float | None = None,
        num_experts: int | None = None,
        num_slots: int | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = True,
        self_attn: bool = True,
        kv_norm: bool = False,
        norm_type: NormType = NormType.LAYER_NORM,
        kernel_size: int | Dims2D = 7,
    ):
        super().__init__(
            d_model,
            nhead,
            d_kv,
            dim_feedforward,
            dropout,
            activation,
            gate_activation,
            num_kv_heads,
            qk_norm,
            layer_scale,
            num_experts,
            num_slots,
            stochastic_depth,
            bias,
            self_attn,
            kv_norm,
            norm_type,
        )
        self.conv = ConvNextBlock(
            d_model,
            dim_feedforward,
            kernel_size=kernel_size,
            activation=activation,
            gate_activation=gate_activation,
            dropout=dropout,
            bias=bias,
            layer_scale=layer_scale,
            stochastic_depth=stochastic_depth,
        )

    def forward(self, q: Tensor, kv: Tensor, size: Tuple[int, ...]) -> Tensor:
        q = self.conv(q, size)
        return super().forward(q, kv)
