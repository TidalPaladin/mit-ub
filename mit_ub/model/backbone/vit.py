from dataclasses import dataclass, field
from typing import Any, Dict, Self, Sequence, cast

import torch.nn as nn
from torch import Tensor

from ...tokens import apply_mask, create_mask
from ..activations import DEFAULT_MLP_ACTIVATION_STR, DEFAULT_MLP_GATE_ACTIVATION_STR, Activation
from ..config import ModelConfig
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

        # Stem tokenizer
        stem_type = PatchEmbed2d if isinstance(config.patch_size, int) or len(config.patch_size) == 2 else PatchEmbed3d
        self.stem = stem_type(config.in_channels, config.dim, cast(Any, config.patch_size), dropout=config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(config.depth)])
        self.embedding_norm = nn.LayerNorm(config.dim)

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
        )
        _kwargs.update(kwargs)
        return TransformerEncoderLayer(**_kwargs)

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
        )
        _kwargs.update(kwargs)
        layer = TransformerDecoderLayer(**_kwargs)
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
    ) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Tokenize and apply mask
        x = self.stem(x)
        if mask is not None:
            x = apply_mask(mask, x, fill_value=mask_fill_value)

        # Transformer blocks and output norm
        for block in self.blocks:
            x = block(x)
        x = self.embedding_norm(x)

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            x = tokens_to_grid(x, tokenized_size)

        return x

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = ViTConfig(*args, **kwargs)
        return cls(config)