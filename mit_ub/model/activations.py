from typing import Callable, Dict, Final

import torch
import torch.nn.functional as F
from torch import Tensor

from .helpers import compile_backend, compile_is_disabled


Activation = Callable[[Tensor], Tensor]


@torch.compile(fullgraph=True, backend=compile_backend(), disable=compile_is_disabled())
def relu2(x: Tensor) -> Tensor:
    r"""Computes squared ReLU of an input."""
    # NOTE: This is roughly as fast as the custom triton kernel
    y = F.relu(x)
    return y * y


@torch.compile(fullgraph=True, backend=compile_backend(), disable=compile_is_disabled())
def silu2(x: Tensor) -> Tensor:
    r"""Computes squared SiLU of an input, accounting for the sign of the input."""
    y = F.silu(x)
    return y.pow(2) * y.sign()


def identity(x: Tensor) -> Tensor:
    return x


DEFAULT_MLP_ACTIVATION: Final = relu2
DEFAULT_MLP_ACTIVATION_STR: Final = "relu2"
DEFAULT_MLP_GATE_ACTIVATION: Final = None
DEFAULT_MLP_GATE_ACTIVATION_STR: Final = None
DEFAULT_POS_ENC_ACTIVATION: Final = relu2
DEFAULT_POS_ENC_ACTIVATION_STR: Final = "relu2"


ACTIVATIONS: Dict[str, Activation] = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "mish": F.mish,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "identity": identity,
    "relu2": relu2,
    "silu2": silu2,
}


def get_activation(activation: str | Activation) -> Activation:
    if isinstance(activation, str):
        if activation not in ACTIVATIONS:
            raise KeyError(f"Activation {activation} not found")
        return ACTIVATIONS[activation]
    return activation


__all__ = [
    "ACTIVATIONS",
    "identity",
    "relu2",
    "silu2",
    "DEFAULT_POS_ENC_ACTIVATION_STR",
    "DEFAULT_MLP_ACTIVATION_STR",
    "DEFAULT_MLP_GATE_ACTIVATION_STR",
    "DEFAULT_POS_ENC_ACTIVATION",
    "DEFAULT_MLP_ACTIVATION",
    "DEFAULT_MLP_GATE_ACTIVATION",
    "Activation",
    "get_activation",
]
