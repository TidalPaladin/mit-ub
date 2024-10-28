from typing import Any, Callable, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_helpers.helpers import to_tuple
from einops import rearrange
from torch import Tensor
from torchvision.ops import StochasticDepth

from .helpers import compile_backend, compile_is_disabled
from .layer_scale import LayerScale
from .mlp import MLP, relu2


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
def conv_2d(
    x: Tensor,
    size: Tuple[int, int],
    w: Tensor,
    b: Tensor | None = None,
    stride: int | Tuple[int, int] = 1,
    padding: int | Tuple[int, int] = 0,
    dilation: int | Tuple[int, int] = 1,
    groups: int = 1,
) -> Tensor:
    H, W = size
    x = rearrange(x, "b (h w) d -> b d h w", h=H, w=W)
    x = F.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    x = rearrange(x, "b d h w -> b (h w) d")
    return x


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
def conv_3d(
    x: Tensor,
    size: Tuple[int, int, int],
    w: Tensor,
    b: Tensor | None = None,
    stride: int | Tuple[int, int, int] = 1,
    padding: int | Tuple[int, int, int] = 0,
    dilation: int | Tuple[int, int, int] = 1,
    groups: int = 1,
) -> Tensor:
    D, H, W = size
    x = rearrange(x, "b (D h w) d -> b d D h w", D=D, h=H, w=W)
    x = F.conv3d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    x = rearrange(x, "b d D h w -> b (D h w) d")
    return x


class ConvEncoderLayer2d(nn.Module):

    def __init__(
        self,
        d_model: int,
        kernel_size: int | Tuple[int, int] = 7,
        stride: int | Tuple[int, int] = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = to_tuple(kernel_size, 2)
        stride = to_tuple(stride, 2)
        padding = cast(Tuple[int, int], tuple(k // 2 for k in cast(Tuple[int, int], kernel_size)))
        self.conv = nn.Conv2d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=d_model,
            bias=bias,
        )
        self.mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            dropout,
            activation,
            gate_activation,
            bias=bias,
            norm=True,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth, mode="row")
        self.layer_scale = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor, size: Tuple[int, int]) -> Tensor:
        H, W = size
        y = conv_2d(
            x,
            (H, W),
            self.conv.weight,
            self.conv.bias,
            stride=cast(Any, self.conv.stride),
            padding=cast(Any, self.conv.padding),
            groups=cast(Any, self.conv.groups),
        )
        y = self.mlp(y)
        return x + self.stochastic_depth(self.layer_scale(y))


class ConvEncoderLayer3d(nn.Module):

    def __init__(
        self,
        d_model: int,
        kernel_size: int | Tuple[int, int, int] = 7,
        stride: int | Tuple[int, int, int] = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = True,
    ):
        super().__init__()
        kernel_size = to_tuple(kernel_size, 3)
        stride = to_tuple(stride, 3)
        padding = cast(Tuple[int, int, int], tuple(k // 2 for k in cast(Tuple[int, int, int], kernel_size)))
        self.conv = nn.Conv3d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=d_model,
            bias=bias,
        )
        self.mlp = MLP(
            d_model,
            dim_feedforward,
            d_model,
            dropout,
            activation,
            gate_activation,
            bias=bias,
            norm=True,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth, mode="row")
        self.layer_scale = LayerScale(d_model, layer_scale) if layer_scale is not None else nn.Identity()

    def reset_parameters(self):
        for module in self.children():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(self, x: Tensor, size: Tuple[int, int, int]) -> Tensor:
        D, H, W = size
        y = conv_3d(
            x,
            (D, H, W),
            self.conv.weight,
            self.conv.bias,
            stride=cast(Any, self.conv.stride),
            padding=cast(Any, self.conv.padding),
            groups=cast(Any, self.conv.groups),
        )
        y = self.mlp(y)
        return x + self.stochastic_depth(self.layer_scale(y))
