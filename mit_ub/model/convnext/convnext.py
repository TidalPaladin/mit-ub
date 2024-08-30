from copy import deepcopy
from typing import Tuple, Type

import torch.nn as nn
from timm.layers import LayerNorm2d
from torch import Tensor


class ConvNextBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        depth: int = 3,
        act_layer: nn.Module = nn.SiLU(),
        norm_layer: Type[nn.Module] = LayerNorm2d,
        mlp_ratio: float = 4,
        dropout: float = 0,
    ):
        super().__init__()
        self.conv_dw = nn.Conv2d(
            channels, channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, groups=channels
        )
        self.norm = norm_layer(channels)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, int(channels * mlp_ratio), kernel_size=1, stride=1, padding=0),
            deepcopy(act_layer),
            nn.Dropout(dropout),
            nn.Conv2d(int(channels * mlp_ratio), channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: Tensor) -> Tensor:
        y = self.conv_dw(x)
        y = self.norm(y)
        y = self.mlp(y)
        return x + y


class ConvNextDownStage(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        depth: int = 3,
        act_layer: nn.Module = nn.SiLU(),
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        mlp_ratio: float = 4,
        dropout: float = 0,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                ConvNextBlock(
                    in_channels,
                    kernel_size=kernel_size,
                    depth=depth,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        x = self.downsample(x)
        return x


class ConvNext(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        kernel_size: int = 7,
        patch_size: int = 4,
        act: nn.Module = nn.SiLU(),
        dropout: float = 0,
        norm_layer: Type[nn.Module] = LayerNorm2d,
    ):
        super().__init__()
        self.dims = dims
        self.patch_size = patch_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size),
            norm_layer(dims[0]),
        )

        self.blocks = nn.ModuleList(
            [
                ConvNextDownStage(
                    dims[i],
                    dims[i + 1],
                    kernel_size=kernel_size,
                    depth=depths[i],
                    act_layer=act,
                    norm_layer=norm_layer,
                    dropout=dropout,
                )
                for i in range(len(depths) - 1)
            ]
        )

    @property
    def dim(self) -> int:
        return self.dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        return x
