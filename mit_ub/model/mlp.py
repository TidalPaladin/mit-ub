from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .compile import compile_is_disabled


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def relu2(x: Tensor) -> Tensor:
    r"""Computes squared ReLU of an input."""
    # NOTE: This is roughly as fast as the custom triton kernel
    y = F.relu(x)
    return y * y


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "epilogue_fusion": True,
        "shape_padding": True,
    },
    disable=compile_is_disabled(),
)
def mlp_forward(
    x: Tensor,
    w1: Tensor,
    w2: Tensor,
    b1: Tensor | None = None,
    b2: Tensor | None = None,
    w_gate: Tensor | None = None,
    b_gate: Tensor | None = None,
    dropout: float = 0.0,
    activation: Callable[[Tensor], Tensor] = relu2,
    gate_activation: Callable[[Tensor], Tensor] | None = None,
    output_dropout: bool = False,
) -> Tensor:
    y = F.linear(x, w1, b1)
    y = activation(y)

    if w_gate is not None:
        gate = F.linear(x, w_gate, b_gate)
        if gate_activation is not None:
            gate = gate_activation(gate)
        y = y * gate

    y = F.dropout(y, p=dropout, training=dropout > 0.0)
    y = F.linear(y, w2, b2)
    y = F.dropout(y, p=dropout, training=dropout > 0.0 and output_dropout)
    return y


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) module with optional gating mechanism and dropout.

    Args:
        in_features: Number of input features.
        hidden_features: Number of hidden features.
        out_features: Number of output features.
        dropout: Dropout probability.
        activation: Activation function to apply after the first linear layer.
        gate_activation: Activation function for the gating mechanism, or ``None`` for no gating.
        bias: Whether to use bias in the linear layers.
        output_dropout: Whether to apply dropout to the output layer.

    Basic MLP:
        >>> mlp = MLP(10, 20, 10))

    Gated Linear Unit (GLU):
        >>> mlp = MLP(10, 20, 10, activation=lambda x: x, gate_activation=torch.sigmoid)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        dropout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        bias: bool = True,
        output_dropout: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.output_dropout = output_dropout
        self.activation = activation
        self.gate_activation = gate_activation

        self.w_in = nn.Parameter(torch.empty(hidden_features, in_features))
        self.w_out = nn.Parameter(torch.empty(out_features, hidden_features))

        if bias:
            self.b_in = nn.Parameter(torch.empty(hidden_features))
            self.b_out = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("b_in", None)
            self.register_parameter("b_out", None)

        if gate_activation is not None and bias:
            self.w_gate = nn.Parameter(torch.empty(hidden_features, in_features))
            self.b_gate = nn.Parameter(torch.empty(hidden_features))
        elif gate_activation is not None:
            self.w_gate = nn.Parameter(torch.empty(hidden_features, in_features))
            self.register_parameter("b_gate", None)
        else:
            self.register_parameter("w_gate", None)
            self.register_parameter("b_gate", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.w_in, std=0.02)
        nn.init.trunc_normal_(self.w_out, std=0.02)
        if self.b_in is not None:
            nn.init.zeros_(self.b_in)
        if self.b_out is not None:
            nn.init.zeros_(self.b_out)
        if self.w_gate is not None:
            nn.init.trunc_normal_(self.w_gate, std=0.02)
        if self.b_gate is not None:
            nn.init.zeros_(self.b_gate)

    @property
    def in_features(self) -> int:
        return self.w_in.shape[-1]

    @property
    def out_features(self) -> int:
        return self.w_out.shape[-1]

    @property
    def hidden_features(self) -> int:
        return self.w_in.shape[-2]

    def forward(self, x: Tensor) -> Tensor:
        return mlp_forward(
            x,
            self.w_in,
            self.w_out,
            self.b_in,
            self.b_out,
            self.w_gate if self.w_gate is not None else None,
            self.b_gate if self.b_gate is not None else None,
            self.dropout if self.training else 0.0,
            self.activation,
            self.gate_activation,
            self.output_dropout,
        )
