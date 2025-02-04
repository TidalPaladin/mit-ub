from typing import Tuple, cast

import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from ..helpers import Dims2D, grid_to_tokens, to_tuple, tokens_to_grid
from .cpu import LayerNormMLP as LayerNormMLPCPU


# NOTE: For some reason the DW conv portion of this won't torch.compile.
def dw_conv_forward_2d(
    x: Tensor,
    size: Tuple[int, int],
    conv_w: Tensor,
    conv_b: Tensor | None,
    stride: Tuple[int, int],
) -> Tensor:
    y = tokens_to_grid(x, size)
    y = F.conv2d(y, conv_w, conv_b, stride=stride, padding=conv_w.shape[-1] // 2, groups=y.shape[1])
    y = grid_to_tokens(y)
    return y


class ConvNextBlock2d(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        kernel_size: int | Tuple[int, int] = 7,
        activation: str = "srelu",
        normalization: str = "LayerNorm",
        stride: int | Tuple[int, int] = 1,
        checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.stride = cast(Tuple[int, int], to_tuple(stride, 2))

        # Depthwise convolution
        _kernel_size = to_tuple(kernel_size, 2)
        padding = cast(Tuple[int, int], tuple((k - s) // 2 for k, s in zip(_kernel_size, self.stride)))
        self.conv_dw = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=_kernel_size, stride=self.stride, padding=padding, groups=hidden_size
        )

        # MLP
        self.mlp = te.LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            activation=activation,
            normalization=normalization,
            **kwargs,
        )

    def reset_parameters(self) -> None:
        self.conv_dw.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, size: Dims2D) -> Tensor:
        if self.training and self.checkpoint:
            y = checkpoint(
                dw_conv_forward_2d,
                x,
                size,
                self.conv_dw.weight,
                self.conv_dw.bias,
                self.conv_dw.stride,
                use_reentrant=False,
            )
            y = self.mlp(y)
        else:
            y = dw_conv_forward_2d(x, size, self.conv_dw.weight, self.conv_dw.bias, self.stride)
            y = self.mlp(y)
        return x + y


class ConvNextBlock2dCPU(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        kernel_size: int | Tuple[int, int] = 7,
        activation: str = "srelu",
        normalization: str = "LayerNorm",
        stride: int | Tuple[int, int] = 1,
        checkpoint: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.stride = cast(Tuple[int, int], to_tuple(stride, 2))

        # Depthwise convolution
        _kernel_size = to_tuple(kernel_size, 2)
        padding = cast(Tuple[int, int], tuple((k - s) // 2 for k, s in zip(_kernel_size, self.stride)))
        self.conv_dw = nn.Conv2d(
            hidden_size, hidden_size, kernel_size=_kernel_size, stride=self.stride, padding=padding, groups=hidden_size
        )

        # MLP
        self.mlp = LayerNormMLPCPU(
            hidden_size,
            ffn_hidden_size,
            activation=activation,
            normalization=normalization,
            **kwargs,
        )

    def reset_parameters(self) -> None:
        self.conv_dw.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, x: Tensor, size: Dims2D) -> Tensor:
        if self.training and self.checkpoint:
            y = checkpoint(
                dw_conv_forward_2d,
                x,
                size,
                self.conv_dw.weight,
                self.conv_dw.bias,
                self.conv_dw.stride,
                use_reentrant=False,
            )
            y = self.mlp(y)
        else:
            y = dw_conv_forward_2d(x, size, self.conv_dw.weight, self.conv_dw.bias, self.stride)
            y = self.mlp(y)
        return x + y
