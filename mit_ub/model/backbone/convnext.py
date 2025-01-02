from dataclasses import dataclass, field
from typing import List, Sequence, cast

import torch.nn as nn
from torch import Tensor

from ..activations import DEFAULT_MLP_ACTIVATION_STR, DEFAULT_MLP_GATE_ACTIVATION_STR, Activation
from ..config import ModelConfig
from ..helpers import Dims2D, grid_to_tokens, set_checkpointing, tokens_to_grid
from ..layers.convnext import ConvNextBlock
from ..layers.mlp import MLP, NormType
from ..layers.pool import PoolType, get_global_pooling_layer


@dataclass(frozen=True)
class ConvNextConfig(ModelConfig):
    in_channels: int
    depths: Sequence[int]
    dims: Sequence[int]
    dims_feedforward: Sequence[int]
    kernel_size: int = 7
    patch_size: int = 4
    activation: str | Activation = DEFAULT_MLP_ACTIVATION_STR
    gate_activation: str | Activation | None = DEFAULT_MLP_GATE_ACTIVATION_STR
    dropout: float = 0
    bias: bool = True
    layer_scale: float | None = None
    stochastic_depth: float = 0.0
    up_depths: Sequence[int] = field(default_factory=lambda: [])
    norm_type: NormType = cast(NormType, "layernorm")
    checkpoint: bool = False

    def instantiate(self) -> "ConvNext":
        return ConvNext(self)

    @property
    def dim(self) -> int:
        if self.up_depths:
            return list(reversed(self.dims))[len(self.up_depths) - 1]
        return self.dims[-1]

    @property
    def dim_feedforward(self) -> int:
        if self.up_depths:
            return list(reversed(self.dims_feedforward))[len(self.up_depths) - 1]
        return self.dims[-1]

    @property
    def nhead(self) -> int:
        # Largest multiple of 16 that divides dim
        return max(n for n in range(16, self.dim + 1, 16) if self.dim % n == 0)


class ConvNext(nn.Module):
    def __init__(self, config: ConvNextConfig):
        super().__init__()
        self.config = config

        self.stem = nn.Conv2d(
            self.config.in_channels,
            self.config.dims[0],
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.norm = nn.LayerNorm(self.config.dims[0])

        self.down_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            self.config.dims[i],
                            cast(Sequence[int], self.config.dims_feedforward)[i],
                            kernel_size=self.config.kernel_size,
                            activation=cast(Activation, self.config.activation),
                            gate_activation=cast(Activation | None, self.config.gate_activation),
                            dropout=self.config.dropout,
                            bias=self.config.bias,
                            layer_scale=self.config.layer_scale,
                            stochastic_depth=self.config.stochastic_depth,
                            norm_type=self.config.norm_type,
                        )
                        for _ in range(self.config.depths[i])
                    ]
                )
                for i in range(len(self.config.depths))
            ]
        )
        self.downsample = nn.ModuleList(
            [
                nn.Conv2d(self.config.dims[i], self.config.dims[i + 1], kernel_size=2, stride=2)
                for i in range(len(self.config.depths) - 1)
            ]
        )

        up_dims = list(reversed(self.config.dims))
        up_dims_feedforward = list(reversed(cast(Sequence[int], self.config.dims_feedforward)))
        self.up_stages = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvNextBlock(
                            up_dims[i],
                            up_dims_feedforward[i],
                            kernel_size=self.config.kernel_size,
                            activation=cast(Activation, self.config.activation),
                            gate_activation=cast(Activation | None, self.config.gate_activation),
                            dropout=self.config.dropout,
                            bias=self.config.bias,
                            layer_scale=self.config.layer_scale,
                            stochastic_depth=self.config.stochastic_depth,
                            norm_type=self.config.norm_type,
                        )
                        for _ in range(self.config.up_depths[i])
                    ]
                )
                for i in range(len(self.config.up_depths) - 1)
            ]
        )
        self.upsample = nn.ModuleList(
            [
                nn.ConvTranspose2d(up_dims[i], up_dims[i + 1], kernel_size=2, stride=2)
                for i in range(len(self.config.up_depths) - 1)
            ]
        )

        self.embedding_norm = (
            nn.LayerNorm(self.config.dim)
            if self.config.norm_type == NormType.LAYER_NORM
            else nn.RMSNorm(self.config.dim)
        )

        if config.checkpoint:
            set_checkpointing(self, config.checkpoint)

    def create_head(
        self,
        out_dim: int,
        pool_type: PoolType | None = None,
        input_norm: bool = True,
        use_mlp: bool = True,
        **kwargs,
    ) -> nn.Module:
        r"""Creates a head for the model.

        The head has the following structure:
            - ``nn.LayerNorm`` if ``input_norm`` is ``True`` and either ``pool_type`` is not ``None`` or ``use_mlp`` is ``True``.
            - Pooling layer if ``pool_type`` is not ``None``
            - MLP if ``use_mlp`` is ``True``. MLP is pre-normalized if ``pool_type`` is not ``None``.
            - ``nn.LayerNorm``
            - ``nn.Linear`` to project to ``out_dim``

        The following MLP options differ from the backbone defaults:
            * ``bias`` is set to ``True``
            * ``stochastic_depth`` is set to ``0.0``
            * ``layer_scale`` is set to ``None``

        Args:
            out_dim: Dimension of the output.
            input_norm: Whether to apply input normalization.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.

        Keyword Args:
            Overrides for the MLP
        """
        layer = nn.Sequential()

        # Input norm
        if input_norm and (pool_type is not None or use_mlp):
            layer.add_module("input_norm", nn.LayerNorm(self.config.dim))

        # Pooling layer
        if pool_type is not None:
            pool = get_global_pooling_layer(
                cast(PoolType, kwargs.get("pool_type", pool_type)),
                self.config.dim,
                num_heads=kwargs.get("nhead", self.config.nhead),
                dropout=kwargs.get("dropout", self.config.dropout),
            )
            layer.add_module("pool", pool)

        # MLP
        if use_mlp:
            _kwargs = {
                "in_features": self.config.dim,
                "hidden_features": self.config.dim_feedforward,
                "out_features": self.config.dim,
                "dropout": kwargs.get("dropout", self.config.dropout),
                "activation": kwargs.get("activation", self.config.activation),
                "gate_activation": kwargs.get("gate_activation", self.config.gate_activation),
                "bias": kwargs.get("bias", True),
                "norm": kwargs.get("norm", pool_type is not None),
                "norm_type": kwargs.get("norm_type", self.config.norm_type),
                "layer_scale": kwargs.get("layer_scale", None),
                "stochastic_depth": kwargs.get("stochastic_depth", 0.0),
            }
            mlp = MLP(**_kwargs)
            layer.add_module("mlp", mlp)

        # Output norm
        layer.add_module("output_norm", nn.LayerNorm(self.config.dim))

        # Output linear
        linear = nn.Linear(self.config.dim, out_dim)
        nn.init.xavier_normal_(linear.weight)
        nn.init.zeros_(linear.bias)
        layer.add_module("out", linear)
        return layer

    def forward(self, x: Tensor, reshape: bool = True) -> Tensor:
        # Patch embed stem
        x = self.stem(x)
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
    def from_args(cls, config: ConvNextConfig) -> "ConvNext":
        return cls(config)
