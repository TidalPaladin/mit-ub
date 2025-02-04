from dataclasses import dataclass, field
from typing import ClassVar, List, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from torch import Tensor

from ..config import ModelConfig, SupportsSafeTensors
from ..helpers import Dims2D, grid_to_tokens, tokens_to_grid
from ..layers.convnext import ConvNextBlock2d
from ..layers.pool import get_global_pooling_layer


@dataclass(frozen=True)
class ConvNextConfig(ModelConfig):
    # Inputs
    in_channels: int
    patch_size: Sequence[int]

    # ConvNext Blocks
    kernel_size: Sequence[int]
    depths: Sequence[int]
    hidden_sizes: Sequence[int]
    ffn_hidden_sizes: Sequence[int]
    parallel_attention_mlp: bool = False
    zero_centered_gamma: bool = False
    normalization: str = "RMSNorm"
    bias: bool = True
    activation: str = "srelu"
    drop_path_rate: float = 0.0

    # Optimizations
    fuse_qkv_params: bool = False
    fuse_wgrad_accumulation: bool = False

    # Other
    checkpoint: bool = False

    # Optional U-Net style upsampling
    up_depths: Sequence[int] = field(default_factory=lambda: [])

    def instantiate(self) -> "ConvNext2d":
        return ConvNext2d(self)

    @property
    def isotropic_output_dim(self) -> int:
        if self.up_depths:
            return list(reversed(self.hidden_sizes))[len(self.up_depths) - 1]
        return self.hidden_sizes[-1]

    @property
    def output_ffn_hidden_size(self) -> int:
        r"""Determines the output FFN dimension, accounting for potential upsampling"""
        if self.up_depths:
            return list(reversed(self.ffn_hidden_sizes))[len(self.up_depths) - 1]
        return self.ffn_hidden_sizes[-1]


class ConvNext2d(nn.Module, SupportsSafeTensors):
    CONFIG_TYPE: ClassVar[Type[ConvNextConfig]] = ConvNextConfig

    def __init__(self, config: ConvNextConfig):
        super().__init__()
        self.config = config

        # Patch embedding stem
        self.stem = nn.Conv2d(
            self.config.in_channels,
            self.config.hidden_sizes[0],
            kernel_size=cast(Tuple[int, int], tuple(self.config.patch_size)),
            stride=cast(Tuple[int, int], tuple(self.config.patch_size)),
        )
        self.norm = self.create_norm(dim=self.config.hidden_sizes[0])

        # Down stages
        self.down_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNextBlock2d(
                            self.config.hidden_sizes[i],
                            self.config.ffn_hidden_sizes[i],
                            kernel_size=cast(Tuple[int, int], tuple(self.config.kernel_size)),
                            activation=self.config.activation,
                            normalization=self.config.normalization,
                            checkpoint=self.config.checkpoint,
                        )
                        for _ in range(self.config.depths[i])
                    ]
                )
                for i in range(len(self.config.depths))
            ]
        )
        # Downsampling blocks at end of each stage
        self.downsample = nn.ModuleList(
            [
                nn.Conv2d(self.config.hidden_sizes[i], self.config.hidden_sizes[i + 1], kernel_size=2, stride=2)
                for i in range(len(self.config.depths) - 1)
            ]
        )

        # Up stages
        up_dims = list(reversed(self.config.hidden_sizes))
        up_dims_feedforward = list(reversed(self.config.ffn_hidden_sizes))
        self.up_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNextBlock2d(
                            up_dims[i],
                            up_dims_feedforward[i],
                            kernel_size=cast(Tuple[int, int], tuple(self.config.kernel_size)),
                            activation=self.config.activation,
                            normalization=self.config.normalization,
                            checkpoint=self.config.checkpoint,
                        )
                        for _ in range(self.config.up_depths[i])
                    ]
                )
                for i in range(len(self.config.up_depths) - 1)
            ]
        )
        # Upsampling blocks at end of each stage
        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose2d(up_dims[i], up_dims[i + 1], kernel_size=2, stride=2)
                for i in range(len(self.config.up_depths) - 1)
            ]
        )

        self.embedding_norm = self.create_norm()

    def create_norm(self, dim: int | None = None, **kwargs) -> nn.Module:
        r"""Creates a normalization layer.

        Args:
            dim: Dimension of the normalization layer. By default this will be the model dimension.
        """
        dim = dim or self.config.isotropic_output_dim
        return te.LayerNorm(dim, **kwargs) if self.config.normalization == "LayerNorm" else te.RMSNorm(dim, **kwargs)

    def create_head(
        self,
        out_dim: int,
        pool_type: str | None = None,
        use_mlp: bool = False,
        **kwargs,
    ) -> nn.Module:
        r"""Creates a head for the model.

        Args:
            out_dim: Dimension of the output.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.
            use_mlp: Whether to use an MLP after pooling.

        Keyword Args:
            Overrides for the MLP
        """
        if pool_type is None:
            if use_mlp:
                layer = nn.Sequential()
                mlp = te.LayerNormMLP(
                    self.config.isotropic_output_dim,
                    self.config.output_ffn_hidden_size,
                    activation=self.config.activation,
                    normalization=self.config.normalization,
                )
                layer.add_module("mlp", mlp)
                linear = te.Linear(self.config.isotropic_output_dim, out_dim, **kwargs)
                layer.add_module("out", linear)
                return layer
            else:
                return te.LayerNormLinear(
                    self.config.isotropic_output_dim,
                    out_dim,
                    normalization=self.config.normalization,
                )

        else:
            layer = nn.Sequential()
            pool = get_global_pooling_layer(pool_type)
            layer.add_module("pool", pool)

            if use_mlp:
                mlp = te.LayerNormMLP(
                    self.config.isotropic_output_dim,
                    self.config.output_ffn_hidden_size,
                    activation=self.config.activation,
                    normalization=self.config.normalization,
                )
                layer.add_module("mlp", mlp)
                linear = te.Linear(self.config.isotropic_output_dim, out_dim)
                layer.add_module("out", linear)
                return layer
            else:
                linear = te.LayerNormLinear(
                    self.config.isotropic_output_dim,
                    out_dim,
                    normalization=self.config.normalization,
                )
                layer.add_module("out", linear)
                return layer

    def forward(self, x: Tensor, reshape: bool = True) -> Tensor:
        # Patch embed stem
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            mm_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")
            x = self.stem(x)
            torch.set_float32_matmul_precision(mm_precision)
        size = cast(Dims2D, x.shape[2:])

        # Convert grid to token sequence and apply norm
        x = grid_to_tokens(x)
        x = self.norm(x)

        # Run down blocks
        levels: List[Tensor] = []
        for i, stage in enumerate(self.down_stages):
            for block in cast(nn.ModuleList, stage):
                x = block(x, size)
            levels.append(x)

            # Downsample and verify new size
            if i < len(self.downsample):
                x = tokens_to_grid(x, size)
                x = self.downsample[i](x)
                size = cast(Dims2D, tuple(s // 2 for s in size))
                assert size == x.shape[2:], f"Expected size {size}, got {x.shape[2:]}"

                # Permute back to token sequence
                x = grid_to_tokens(x)

        levels.pop(-1)

        # Run up blocks
        for i, stage in enumerate(self.up_stages):
            # Process current level
            for block in cast(nn.ModuleList, stage):
                x = block(x, size)

            # Upsample and verify new size
            if i < len(self.upsample):
                x = tokens_to_grid(x, size)
                x = self.upsample[i](x)
                size = cast(Dims2D, tuple(s * 2 for s in size))
                assert size == x.shape[2:], f"Expected size {size}, got {x.shape[2:]}"

                # Permute back to token sequence
                x = grid_to_tokens(x)

            # Add with previous level
            y = levels[-(i + 1)]
            x = x + y
            levels[-(i + 1)] = x

        x = self.embedding_norm(x)

        # Convert back to grid
        if reshape:
            x = tokens_to_grid(x, size)
        return x

    @classmethod
    def from_args(cls, config: ConvNextConfig) -> "ConvNext2d":
        return cls(config)
