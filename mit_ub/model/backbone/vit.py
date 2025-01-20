from dataclasses import dataclass, field
from typing import Any, Dict, Self, Sequence, Tuple, cast

import torch
import torch.nn as nn
from torch import Tensor

from ...tokens import apply_mask, create_mask
from ..activations import DEFAULT_MLP_ACTIVATION_STR, DEFAULT_MLP_GATE_ACTIVATION_STR, Activation
from ..config import ModelConfig
from ..helpers import set_checkpointing
from ..layers.mlp import MLP, NormType
from ..layers.pool import PoolType, get_global_pooling_layer
from ..layers.transformer import TransformerDecoderLayer, TransformerEncoderLayer
from ..stem import PatchEmbed2d, PatchEmbed3d
from .convnext import tokens_to_grid


@dataclass(frozen=True)
class ViTConfig(ModelConfig):
    in_channels: int
    dim: int
    patch_size: Sequence[int]
    depth: int
    nhead: int
    dim_feedforward: int
    dropout: float = 0.1
    stochastic_depth: float = 0.0
    bias: bool = False
    activation: str | Activation = DEFAULT_MLP_ACTIVATION_STR
    gate_activation: str | Activation | None = DEFAULT_MLP_GATE_ACTIVATION_STR
    num_kv_heads: int | None = None
    qk_norm: bool = False
    layer_scale: float | None = None
    norm_type: NormType = cast(NormType, "layernorm")
    checkpoint: bool = False

    moe_layers: Sequence[int] = field(default_factory=list)
    num_experts: int | None = None
    num_slots: int | None = None

    def instantiate(self) -> "ViT":
        return ViT(self)


class ViT(nn.Module):
    stem: PatchEmbed2d | PatchEmbed3d

    def __init__(self, config: ViTConfig):
        super().__init__()
        self._config = config

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.dim))

        # Stem tokenizer
        stem_type = PatchEmbed2d if isinstance(config.patch_size, int) or len(config.patch_size) == 2 else PatchEmbed3d
        self.stem = stem_type(config.in_channels, config.dim, cast(Any, config.patch_size), dropout=config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(config.depth)])
        self.embedding_norm = (
            nn.LayerNorm(config.dim) if config.norm_type == NormType.LAYER_NORM else nn.RMSNorm(config.dim)
        )
        if config.checkpoint:
            set_checkpointing(self, config.checkpoint)

    @property
    def config(self) -> ViTConfig:
        return self._config

    def create_encoder_layer(self, i: int = 0, **kwargs) -> TransformerEncoderLayer:
        """
        Creates a Transformer encoder layer.

        This method initializes a Transformer encoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        _kwargs: Dict[str, Any] = dict(
            d_model=self.config.dim,
            nhead=self.config.nhead,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            gate_activation=self.config.gate_activation,
            num_kv_heads=self.config.num_kv_heads,
            qk_norm=self.config.qk_norm,
            num_experts=self.config.num_experts if i in self.config.moe_layers else None,
            num_slots=self.config.num_slots if i in self.config.moe_layers else None,
            layer_scale=self.config.layer_scale,
            stochastic_depth=self.config.stochastic_depth,
            bias=self.config.bias,
            norm_type=self.config.norm_type,
        )
        _kwargs.update(kwargs)
        layer = TransformerEncoderLayer(**_kwargs)
        if self.config.checkpoint:
            set_checkpointing(layer, self.config.checkpoint)
        return layer

    def create_decoder_layer(self, i: int = 0, d_kv: int | None = None, **kwargs) -> TransformerDecoderLayer:
        """
        Creates a Transformer decoder layer.

        This method initializes a Transformer decoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.
            d_kv: Dimension of the key and value vectors. By default this will be the same as the model dimension.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        d_kv = d_kv or self.config.dim
        _kwargs: Dict[str, Any] = dict(
            d_model=self.config.dim,
            nhead=self.config.nhead,
            d_kv=d_kv,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation=self.config.activation,
            gate_activation=self.config.gate_activation,
            num_kv_heads=self.config.num_kv_heads,
            qk_norm=self.config.qk_norm,
            num_experts=self.config.num_experts if i in self.config.moe_layers else None,
            num_slots=self.config.num_slots if i in self.config.moe_layers else None,
            layer_scale=self.config.layer_scale,
            stochastic_depth=self.config.stochastic_depth,
            bias=self.config.bias,
            norm_type=self.config.norm_type,
        )
        _kwargs.update(kwargs)
        layer = TransformerDecoderLayer(**_kwargs)
        if self.config.checkpoint:
            set_checkpointing(layer, self.config.checkpoint)
        return layer

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

    def on_load_checkpoint(self, state_dict: Dict[str, Any], *args, **kwargs) -> None:
        r"""Called after loading a checkpoint to perform any necessary post-processing."""

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        r"""Creates a token mask for the input.

        Args:
            input: Input tensor from which to infer mask properties.
                Should be a raw input prior to tokenization.
            unmasked_ratio: Proportion of tokens to leave unmasked.
            scale: Scale of the mask.

        Shapes:
            - input: :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
            - output: :math:`(B, L)`

        Returns:
            Token mask.
        """
        batch_size = input.shape[0]
        device = input.device
        original_size = input.shape[2:]
        tokenized_size = self.stem.tokenized_size(cast(Any, original_size))
        mask = create_mask(
            tokenized_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size,
            scale=scale,
            device=device,
        )

        return mask

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Tokenize and apply mask
        x = self.stem(x)
        if mask is not None:
            x = apply_mask(mask, x, fill_value=mask_fill_value)

        # Add CLS token
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # Transformer blocks and output norm
        for block in self.blocks:
            x = block(x)
        x = self.embedding_norm(x)

        # Extract CLS token
        cls_token = x[:, 0, :].contiguous()
        x = x[:, 1:, :].contiguous()

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            x = tokens_to_grid(x, tokenized_size)

        return x, cls_token

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = ViTConfig(*args, **kwargs)
        return cls(config)
