from enum import StrEnum
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..activations import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, Activation
from ..helpers import Checkpointable, compile_backend, compile_is_disabled, max_autotune
from .layer_scale import LayerScale
from .stochastic_depth import apply_stochastic_depth, stochastic_depth_indices, unapply_stochastic_depth


NORM_EPS: Final = 1e-5


class NormType(StrEnum):
    LAYER_NORM = "layernorm"
    RMS_NORM = "rmsnorm"


@torch.compile(
    fullgraph=True,
    backend=compile_backend(),
    options={
        "max_autotune": max_autotune(),
        "epilogue_fusion": True,
        "shape_padding": True,
        "triton.cudagraph_trees": max_autotune(),
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
    activation: Activation = DEFAULT_MLP_ACTIVATION,
    gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
    w_norm: Tensor | None = None,
    b_norm: Tensor | None = None,
    eps: float = NORM_EPS,
    training: bool = False,
    norm_type: NormType = NormType.LAYER_NORM,
    w_layer_scale: Tensor | None = None,
    stochastic_depth: float = 0.0,
) -> Tensor:
    # Apply stochastic depth
    B = x.shape[0]
    if stochastic_depth > 0.0 and training:
        indices = stochastic_depth_indices(x, stochastic_depth)
        x = apply_stochastic_depth(x, indices)
    else:
        indices = None

    # Pre-normalization
    if w_norm is not None:
        if norm_type == NormType.LAYER_NORM:
            x = F.layer_norm(x, x.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)
        elif norm_type == NormType.RMS_NORM:
            x = F.rms_norm(x, x.shape[-1:], weight=w_norm, eps=eps)
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    # First projection
    y = F.linear(x, w1, b1)
    y = activation(y)

    # Gate projection (GLU variants)
    if w_gate is not None:
        gate = F.linear(x, w_gate, b_gate)
        if gate_activation is not None:
            gate = gate_activation(gate)
        y = y * gate

    # Dropout, second projection
    y = F.dropout(y, p=dropout, training=training, inplace=True)
    y = F.linear(y, w2, b2)
    y = F.dropout(y, p=dropout, training=training, inplace=True)

    # Layer scale
    if w_layer_scale is not None:
        y = y * w_layer_scale

    # Unapply stochastic depth
    if stochastic_depth > 0.0 and training:
        assert indices is not None
        y = unapply_stochastic_depth(y, indices, B)

    return y


class MLP(nn.Module, Checkpointable):
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
        norm: Whether to apply normalization to the input.
        norm_type: Type of pre-normalization to apply.
        layer_scale: Layer scale factor.
        stochastic_depth: Dropout probability for stochastic depth.

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
        activation: Activation = DEFAULT_MLP_ACTIVATION,
        gate_activation: Activation | None = DEFAULT_MLP_GATE_ACTIVATION,
        bias: bool = True,
        norm: bool = False,
        norm_type: NormType = NormType.LAYER_NORM,
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.activation = activation
        self.gate_activation = gate_activation
        self.norm_type = norm_type
        self.checkpoint = False
        self.stochastic_depth = stochastic_depth

        # Register optional parameters
        for prefix in ("w_", "b_"):
            for suffix in ("norm", "in", "out", "gate"):
                self.register_parameter(f"{prefix}{suffix}", None)

        self.w_in = nn.Parameter(torch.empty(hidden_features, in_features))
        self.w_out = nn.Parameter(torch.empty(out_features, hidden_features))

        if norm:
            self.w_norm = nn.Parameter(torch.empty(in_features))
            if norm_type == NormType.RMS_NORM:
                self.b_norm = nn.Parameter(torch.empty(in_features))
            else:
                self.register_parameter("b_norm", None)

        if bias:
            self.b_in = nn.Parameter(torch.empty(hidden_features))
            self.b_out = nn.Parameter(torch.empty(out_features))

        if gate_activation is not None:
            self.w_gate = nn.Parameter(torch.empty(hidden_features, in_features))
            self.b_gate = nn.Parameter(torch.empty(hidden_features)) if bias else None

        if layer_scale is not None:
            self.layer_scale = LayerScale(out_features, gamma=layer_scale)
        else:
            self.register_module("layer_scale", None)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name.startswith("b_"):
                nn.init.zeros_(param)
            elif "norm" in name:
                nn.init.ones_(param)
            elif name.startswith("w_"):
                nn.init.xavier_uniform_(param)
            elif name.startswith("layer_scale"):
                pass
            else:
                raise ValueError(f"Unsure how to initialize {name}")

        if self.layer_scale is not None:
            self.layer_scale.reset_parameters()

    @property
    def in_features(self) -> int:
        return self.w_in.shape[-1]

    @property
    def out_features(self) -> int:
        return self.w_out.shape[-2]

    @property
    def hidden_features(self) -> int:
        return self.w_in.shape[-2]

    @property
    def norm(self) -> bool:
        return self.w_norm is not None

    @property
    def bias(self) -> bool:
        return self.b_in is not None

    @property
    def w_layer_scale(self) -> Tensor | None:
        return self.layer_scale.gamma if self.layer_scale is not None else None

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, "
            f"hidden={self.hidden_features}, "
            f"out={self.out_features}, "
            f"dropout={self.dropout}, "
            f"act={self.activation.__name__}, "
            f"gate_act={self.gate_activation.__name__ if self.gate_activation is not None else None}, "
            f"bias={self.bias}, "
            f"norm={self.norm}, "
            f"norm_type={self.norm_type}, "
            f"stochastic_depth={self.stochastic_depth}"
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training and self.checkpoint:
            result = checkpoint(
                mlp_forward,
                x,
                self.w_in,
                self.w_out,
                self.b_in,
                self.b_out,
                self.w_gate,
                self.b_gate,
                self.dropout,
                self.activation,
                self.gate_activation,
                self.w_norm,
                self.b_norm,
                NORM_EPS,
                self.training,
                self.norm_type,
                self.w_layer_scale,
                self.stochastic_depth,
                use_reentrant=False,
            )
            assert isinstance(result, Tensor)
            return result
        else:
            return mlp_forward(
                x,
                self.w_in,
                self.w_out,
                self.b_in,
                self.b_out,
                self.w_gate,
                self.b_gate,
                self.dropout,
                self.activation,
                self.gate_activation,
                self.w_norm,
                self.b_norm,
                NORM_EPS,
                self.training,
                self.norm_type,
                self.w_layer_scale,
                self.stochastic_depth,
            )
