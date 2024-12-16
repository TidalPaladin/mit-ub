import math
from typing import Callable, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..helpers import Dims2D, compile_backend, compile_is_disabled, to_tuple
from ..layer_scale import LayerScale
from ..mlp import DEFAULT_MLP_ACTIVATION, DEFAULT_MLP_GATE_ACTIVATION, MLP, mlp_forward, relu2


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def tokens_to_grid(x: Tensor, size: Sequence[int]) -> Tensor:
    r"""Convert a channel-last flat token sequence to a channel-first spatial grid.

    Args:
        x: The token sequence to convert to a grid.
        size: The size of the grid to convert to.

    Returns:
        The grid of tokens.

    Raises:
        ValueError: If the token length does not match the grid size.
    """
    _, L, _ = x.shape
    if L != math.prod(size):
        raise ValueError(f"Token length {L} does not match grid size {size}")

    if len(size) == 1:
        return rearrange(x, "b l c -> b c l")
    elif len(size) == 2:
        return rearrange(x, "b (h w) c -> b c h w", h=size[0], w=size[1])
    elif len(size) == 3:
        return rearrange(x, "b (d h w) c -> b c d h w", d=size[0], h=size[1], w=size[2])
    else:
        raise ValueError(f"Invalid size: {size}")


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def grid_to_tokens(x: Tensor) -> Tensor:
    r"""Convert a channel-first spatial grid to a channel-last flat token sequence.

    Args:
        x: The grid to convert to a token sequence.

    Returns:
        The token sequence.
    """
    return rearrange(x, "b c ... -> b (...) c")


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
    activation: Callable[[Tensor], Tensor] = DEFAULT_MLP_ACTIVATION,
    gate_activation: Callable[[Tensor], Tensor] | None = DEFAULT_MLP_GATE_ACTIVATION,
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
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0,
        bias: bool = True,
        layer_scale: float | None = None,
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
        return x + self.layer_scale(y)


class ConvNextDownStage(nn.Module):

    def __init__(
        self,
        dim: int,
        depth: int,
        out_dim: int,
        dim_feedforward: int | None = None,
        kernel_size: int = 7,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0,
        bias: bool = True,
        layer_scale: float | None = None,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    dim,
                    dim_feedforward=dim_feedforward,
                    kernel_size=kernel_size,
                    activation=activation,
                    gate_activation=gate_activation,
                    dropout=dropout,
                    bias=bias,
                    layer_scale=layer_scale,
                )
                for _ in range(depth)
            ]
        )
        self.downsample = nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)

    @property
    def dim(self) -> int:
        return self._dim

    def forward(self, x: Tensor, size: Dims2D) -> Tuple[Tensor, Dims2D]:
        # Run same-level blocks
        for block in self.blocks:
            x = block(x, size)

        # Downsample and verify new size
        x = tokens_to_grid(x, size)
        x = self.downsample(x)
        new_size = cast(Dims2D, tuple(s // 2 for s in size))
        assert new_size == x.shape[2:], f"Expected size {new_size}, got {x.shape[2:]}"

        # Permute back to token sequence, return updated size
        x = grid_to_tokens(x)
        return x, cast(Dims2D, tuple(s // 2 for s in size))


class ConvNext(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        kernel_size: int = 7,
        patch_size: int = 4,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        dropout: float = 0,
        bias: bool = True,
        layer_scale: float | None = None,
    ):
        super().__init__()
        self.dims = dims
        self.patch_size = patch_size

        self.stem = nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(dims[0])

        self.blocks = nn.ModuleList(
            [
                ConvNextDownStage(
                    dims[i],
                    depths[i],
                    dims[i + 1],
                    None,
                    kernel_size=kernel_size,
                    activation=activation,
                    gate_activation=gate_activation,
                    dropout=dropout,
                    bias=bias,
                    layer_scale=layer_scale,
                )
                for i in range(len(depths) - 1)
            ]
        )

    @property
    def dim(self) -> int:
        return self.dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        # Patch embed stem
        x = self.stem(x)
        size = cast(Dims2D, x.shape[2:])

        # Convert grid to token sequence and apply norm
        x = grid_to_tokens(x)
        x = self.norm(x)

        # Run blocks
        for block in self.blocks:
            x, size = block(x, size)

        # Convert back to grid
        x = tokens_to_grid(x, size)
        return x
