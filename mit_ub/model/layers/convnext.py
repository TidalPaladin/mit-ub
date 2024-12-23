from typing import Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import StochasticDepth

from ..activations import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, Activation
from ..helpers import Dims2D, compile_backend, compile_is_disabled, grid_to_tokens, to_tuple, tokens_to_grid
from ..layers.layer_scale import LayerScale
from ..layers.mlp import MLP, mlp_forward


@torch.compile(
    fullgraph=True,
    backend=compile_backend(),
    options={
        "max_autotune": True,
        "epilogue_fusion": True,
        "shape_padding": True,
        "triton.cudagraph_trees": True,
    },
    disable=compile_is_disabled(),
)
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
    eps: float = 1e-5,
    training: bool = False,
) -> Tensor:
    # Depthwise convolution
    y = tokens_to_grid(x, size)
    y = F.conv2d(y, conv_w, conv_b, stride=1, padding=conv_w.shape[-1] // 2, groups=y.shape[1])
    y = grid_to_tokens(y)

    # LayerNorm, MLP
    y = mlp_forward(
        y, w1, w2, b1, b2, w_gate, b_gate, dropout, activation, gate_activation, w_norm, b_norm, eps, training
    )
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
    ):
        super().__init__()
        _kernel_size = to_tuple(kernel_size, 2)
        padding = cast(Tuple[int, int], tuple(k // 2 for k in _kernel_size))
        self._dim = dim
        self._dim_feedforward = dim_feedforward or int(dim * 4)
        self.conv_dw = nn.Conv2d(dim, dim, kernel_size=_kernel_size, stride=1, padding=padding, groups=dim)
        self.mlp = MLP(
            dim,
            self.dim_feedforward,
            dim,
            dropout=dropout,
            activation=activation,
            gate_activation=gate_activation,
            bias=bias,
            norm=True,
        )
        self.layer_scale = LayerScale(dim, layer_scale) if layer_scale else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth, mode="row")
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

    def forward(self, x: Tensor, size: Dims2D) -> Tensor:
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
            training=self.training,
        )
        return x + self.stochastic_depth(self.layer_scale(y))
