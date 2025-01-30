from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..activations import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, Activation
from ..helpers import Dims2D, grid_to_tokens, to_tuple, tokens_to_grid
from ..layers.layer_scale import LayerScale
from ..layers.mlp import MLP, NormType, mlp_forward
from ..layers.stochastic_depth import apply_stochastic_depth, stochastic_depth_indices, unapply_stochastic_depth


# NOTE: For some reason the DW conv portion of this won't torch.compile.
def convnext_block_forward_2d(
    x: Tensor,
    size: Dims2D,
    conv_w: Tensor,
    w1: Tensor,
    w2: Tensor,
    conv_b: Tensor | None,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
    w_gate: Tensor | None = None,
    b_gate: Tensor | None = None,
    dropout: float = 0.0,
    activation: Activation = DEFAULT_MLP_ACTIVATION,
    gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
    w_norm: Tensor | None = None,
    b_norm: Tensor | None = None,
    training: bool = False,
    norm_type: NormType = NormType.LAYER_NORM,
    w_layer_scale: Tensor | None = None,
    stride: Tuple[int, ...] = (1,),
) -> Tensor:
    # Depthwise convolution
    y = tokens_to_grid(x, size)
    y = F.conv2d(y, conv_w, conv_b, stride=stride, padding=conv_w.shape[-1] // 2, groups=y.shape[1])
    y = grid_to_tokens(y)

    # LayerNorm, MLP
    y = mlp_forward(
        y,
        w1,
        w2,
        b1,
        b2,
        w_gate,
        b_gate,
        dropout,
        activation,
        gate_activation,
        w_norm,
        b_norm,
        training,
        norm_type,
    )

    if w_layer_scale is not None:
        y = y * w_layer_scale

    return y


class ConvNextBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        dim_feedforward: int | None = None,
        kernel_size: int | Dims2D = 7,
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        dropout: float = 0,
        bias: bool = True,
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
        norm_type: NormType = NormType.LAYER_NORM,
        stride: int = 1,
    ):
        super().__init__()
        _kernel_size = to_tuple(kernel_size, 2)
        padding = cast(Tuple[int, int], tuple((k - stride) // 2 for k in _kernel_size))
        self._dim = dim
        self._dim_feedforward = dim_feedforward or int(dim * 4)
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=_kernel_size, stride=stride, padding=padding, groups=dim)
        self.mlp = MLP(
            dim,
            self.dim_feedforward,
            dim,
            dropout=dropout,
            activation=activation,
            gate_activation=gate_activation,
            bias=bias,
            norm=True,
            norm_type=norm_type,
        )
        if layer_scale is not None:
            self.layer_scale = LayerScale(dim, gamma=layer_scale)
        else:
            self.register_module("layer_scale", None)
        self.stochastic_depth = stochastic_depth
        self.checkpoint = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward

    @property
    def w_layer_scale(self) -> Tensor | None:
        return self.layer_scale.gamma if self.layer_scale is not None else None

    def forward(self, x: Tensor, size: Dims2D) -> Tensor:
        x_orig = x
        B = x.shape[0]
        if self.stochastic_depth > 0.0 and self.training:
            indices = stochastic_depth_indices(x, self.stochastic_depth)
            x = apply_stochastic_depth(x, indices)
        else:
            indices = None

        if self.training and self.checkpoint:
            # Workaround for checkpointing with compile + DDP
            fn = (
                torch.compiler.disable(convnext_block_forward_2d)
                if torch.distributed.is_initialized()
                else convnext_block_forward_2d
            )
            y = checkpoint(
                fn,
                x,
                size,
                self.conv_dw.weight,
                self.mlp.w_in,
                self.mlp.w_out,
                self.conv_dw.bias,
                self.mlp.b_in,
                self.mlp.b_out,
                self.mlp.w_gate,
                self.mlp.b_gate,
                self.mlp.dropout,
                self.mlp.activation,
                self.mlp.gate_activation,
                self.mlp.w_norm,
                self.mlp.b_norm,
                self.training,
                self.mlp.norm_type,
                self.w_layer_scale,
                stride=self.conv_dw.stride,
                use_reentrant=False,
            )
            assert isinstance(y, Tensor)
        else:
            y = convnext_block_forward_2d(
                x,
                size,
                self.conv_dw.weight,
                self.mlp.w_in,
                self.mlp.w_out,
                self.conv_dw.bias,
                self.mlp.b_in,
                self.mlp.b_out,
                self.mlp.w_gate,
                self.mlp.b_gate,
                self.mlp.dropout,
                self.mlp.activation,
                self.mlp.gate_activation,
                self.mlp.w_norm,
                self.mlp.b_norm,
                self.training,
                self.mlp.norm_type,
                self.w_layer_scale,
                self.conv_dw.stride,
            )

        if indices is not None:
            y = unapply_stochastic_depth(y, indices, B, training=self.training)

        return x_orig + y
