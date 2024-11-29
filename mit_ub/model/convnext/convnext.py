from typing import Callable, List, Tuple

import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor

from ..conv import ConvEncoderLayer2d, conv_2d
from ..mlp import relu2


class ConvNext(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        depths: Tuple[int, ...] = (3, 3, 9, 3),
        dims: Tuple[int, ...] = (96, 192, 384, 768),
        dims_feedforward: Tuple[int, ...] = (384, 768, 1536, 3072),
        patch_size: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.dims = dims
        self.patch_size = patch_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=patch_size, stride=patch_size),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(dims[0]),
        )

        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvEncoderLayer2d(
                            dims[i],
                            kernel_size,
                            1,
                            dims_feedforward[i],
                            dropout,
                            activation,
                            gate_activation,
                            layer_scale,
                            stochastic_depth,
                            bias,
                        )
                        for _ in range(depths[i])
                    ]
                )
                for i in range(len(depths))
            ]
        )
        self.pools = nn.ModuleList(
            [nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2) for i in range(len(depths) - 1)]
        )

    @property
    def dim(self) -> int:
        return self.dims[-1]

    def forward(self, x: Tensor, reshape: bool = True) -> Tensor:
        return self.forward_features(x, reshape)[-1]

    def forward_features(self, x: Tensor, reshape: bool = True) -> List[Tensor]:
        N, C, H, W = x.shape
        H = H // self.patch_size
        W = W // self.patch_size
        x = self.stem(x)

        result: List[Tensor] = []
        for i, level in enumerate(self.blocks):
            assert isinstance(level, nn.ModuleList)
            for block in level:
                x = block(x, (H, W))

            r = rearrange(x, "b (h w) c -> b c h w", h=H, w=W) if reshape else x
            result.append(r)

            if i < len(self.pools):
                pool = self.pools[i]
                x = conv_2d(
                    x,
                    (H, W),
                    pool.weight,
                    pool.bias,
                    stride=pool.stride,
                    padding=pool.padding,
                    dilation=pool.dilation,
                    groups=pool.groups,
                )
                H = H // pool.stride[0]
                W = W // pool.stride[1]

        return result
