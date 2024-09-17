from copy import deepcopy
from functools import partial
from typing import Sequence, cast

import torch.nn as nn
from torch import Tensor

from .kernels.relu2 import ReLU2
from .lora import LoRATarget, SupportsLoRA, apply_lora, freeze_non_lora


class MLP(nn.Module, SupportsLoRA):
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
        activation: nn.Module = ReLU2(),
        gate_activation: nn.Module | None = None,
        bias: bool = True,
        output_dropout: bool = False,
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout) if output_dropout else nn.Identity()
        self.activation = deepcopy(activation)

        if gate_activation is not None:
            self.gate = nn.Sequential(
                nn.Linear(in_features, hidden_features, bias=bias),
                deepcopy(gate_activation),
            )
        else:
            self.gate = None

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.gate is not None:
            self.gate[0].reset_parameters()

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
        y = self.activation(self.fc1(x))

        if self.gate is not None:
            y = y * self.gate(x)

        y = self.dropout(y)
        y = self.fc2(y)
        y = self.output_dropout(y)
        return y

    def apply_lora(
        self,
        target: Sequence[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        quantize_base: bool = False,
    ) -> nn.Module:
        _apply_lora = partial(apply_lora, rank=rank, alpha=alpha, dropout=dropout, quantize_base=quantize_base)

        if LoRATarget.FEEDFORWARD in target:
            self.fc1 = _apply_lora(cast(nn.Linear, self.fc1))
            self.fc2 = _apply_lora(cast(nn.Linear, self.fc2))
            if self.gate is not None:
                self.gate[0] = _apply_lora(cast(nn.Linear, self.gate[0]))

        # Freeze all non-LoRA matrices/weights
        freeze_non_lora(self)
        return self
