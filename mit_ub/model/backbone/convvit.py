from dataclasses import dataclass
from typing import Any, Dict, Self, cast

import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ..helpers import Dims2D
from ..layers.layer_scale import LayerScale
from ..layers.transformer import TransformerConvDecoderLayer, TransformerDecoderLayer
from .adaptive_vit import AdaptiveViT, AdaptiveViTConfig
from .convnext import grid_to_tokens, tokens_to_grid


@dataclass(frozen=True)
class ConvViTConfig(AdaptiveViTConfig):
    kernel_size: int | Dims2D = 7

    def instantiate(self) -> "ConvViT":
        return ConvViT(self)


class ConvViT(AdaptiveViT):
    config: ConvViTConfig

    def __init__(self, config: ConvViTConfig):
        super().__init__(config)

        # Update dynamic blocks to include the ConvNext mixer
        self.dynamic_blocks = nn.ModuleList(
            [
                self.create_conv_decoder_layer(i + len(self.blocks), self_attn=False, kv_norm=True)
                for i in range(config.depth)
            ]
        )
        if config.share_layers:
            self.set_shared_layers()

        # Override the layer scale of the ConvNext mixer
        for block in self.dynamic_blocks:
            block.conv.layer_scale = (
                LayerScale(self.config.dim, config.layer_scale_adaptive)
                if config.layer_scale_adaptive is not None
                else nn.Identity()
            )

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        raise NotImplementedError("ConvViT does not support masks")

    def create_conv_decoder_layer(self, i: int = 0, d_kv: int | None = None, **kwargs) -> TransformerDecoderLayer:
        """
        Creates a Transformer decoder layer with ConvNext mixing on the queries.

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
            kernel_size=self.config.kernel_size,
            self_attn=False,
            norm_type=self.config.norm_type,
        )
        _kwargs.update(kwargs)
        layer = TransformerConvDecoderLayer(**_kwargs)
        return layer

    def forward(self, x: Tensor, reshape: bool = True) -> Tensor:
        B, C, *original_size = x.shape
        dynamic_tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))
        fixed_tokenized_size = self.stem.target_tokenized_shape

        # Tokenize to fixed and dynamic tokens
        fixed_tokens, dynamic_tokens = self.stem(x)

        # Run the backbone
        for block, dynamic_block in zip(self.blocks, self.dynamic_blocks):
            fixed_tokens = block(fixed_tokens, dynamic_tokens)
            dynamic_tokens = dynamic_block(dynamic_tokens, fixed_tokens, size=dynamic_tokenized_size)

        # Upsample fixed tokens and add to dynamic tokens
        fixed_tokens = tokens_to_grid(fixed_tokens, fixed_tokenized_size)
        fixed_tokens = F.interpolate(fixed_tokens, size=dynamic_tokenized_size, mode="nearest")
        fixed_tokens = grid_to_tokens(fixed_tokens)
        assert fixed_tokens.shape == dynamic_tokens.shape
        dynamic_tokens = self.dynamic_output_scale(dynamic_tokens) + fixed_tokens

        # Output norm
        dynamic_tokens = self.embedding_norm(dynamic_tokens)

        # Reshape to original grid if requested
        if reshape:
            dynamic_tokens = tokens_to_grid(dynamic_tokens, dynamic_tokenized_size)

        return dynamic_tokens

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = ConvViTConfig(*args, **kwargs)
        return cls(config)
