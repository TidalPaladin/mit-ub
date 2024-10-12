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
    training: bool = True,
    output_dropout: bool = False,
) -> Tensor:
    y = F.linear(x, w1, b1)
    y = activation(y)

    if w_gate is not None:
        gate = F.linear(x, w_gate, b_gate)
        if gate_activation is not None:
            gate = gate_activation(gate)
        y = y * gate

    y = F.dropout(y, dropout, training)
    y = F.linear(y, w2, b2)
    if output_dropout:
        y = F.dropout(y, dropout, training)
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
        >>> mlp = MLP(10, 20, 10, activation=nn.ReLU())

    Gated Linear Unit (GLU):
        >>> mlp = MLP(10, 20, 10, activation=nn.Identity(), gate_activation=nn.Sigmoid())
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
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout) if output_dropout else nn.Identity()
        self.activation = activation
        self.gate = nn.Linear(in_features, hidden_features, bias=bias) if gate_activation is not None else None
        self.gate_activation = gate_activation

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.gate is not None:
            self.gate.reset_parameters()

    @property
    def in_features(self) -> int:
        return self.fc1.in_features

    @property
    def out_features(self) -> int:
        return self.fc2.out_features

    @property
    def hidden_features(self) -> int:
        return self.fc1.out_features

    def forward(self, x: Tensor) -> Tensor:
        return mlp_forward(
            x,
            self.fc1.weight,
            self.fc2.weight,
            self.fc1.bias,
            self.fc2.bias,
            self.gate.weight if self.gate is not None else None,
            self.gate.bias if self.gate is not None else None,
            self.dropout.p,
            self.activation,
            self.gate_activation,
            self.training,
            isinstance(self.output_dropout, nn.Dropout),
        )
