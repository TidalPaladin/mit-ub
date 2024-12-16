from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from ..tokens import apply_mask
from .helpers import compile_is_disabled
from .mlp import relu2
from .stem import PatchEmbed2d, PatchEmbed3d
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


@torch.compile(disable=compile_is_disabled())
def resize_mask(
    size: Tuple[int, int] | Tuple[int, int, int],
    target_size: Tuple[int, int] | Tuple[int, int, int],
    mask: Tensor,
) -> Tensor:
    r"""Resizes a mask to a target size.

    Args:
        size: Size of the input mask.
        target_size: Target size to resize the mask to.
        mask: Mask to resize.

    Returns:
        Resized mask.
    """
    B = mask.shape[0]
    mask = mask.view(B, 1, *size)
    mask = F.interpolate(mask.float(), size=target_size, mode="nearest").to(mask.dtype)
    mask = mask.view(B, -1)
    return mask


def reshape_tokens(
    x: Tensor,
    tokenized_size: Tuple[int, int] | Tuple[int, int, int],
) -> Tensor:
    r"""Reshapes a sequence of tokens to a channel-first grid.

    Args:
        x: Tokens to reshape.
        tokenized_size: Size of the tokenized input.

    Returns:
        Reshaped tokens.
    """
    if len(tokenized_size) == 3:
        D, H, W = tokenized_size
        x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
    elif len(tokenized_size) == 2:
        H, W = tokenized_size
        x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
    else:
        raise ValueError(f"Invalid tokenized size: {tokenized_size}")
    return x


class ViT(nn.Module):
    stem: PatchEmbed2d | PatchEmbed3d

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int | Tuple[int, int] | Tuple[int, int, int],
        depth: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        num_experts: int | None = None,
        num_slots: int | None = None,
        moe_layers: List[int] = [],
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self._dim = dim
        self._nhead = nhead if nhead is not None else self.dim // 32
        self._in_channels = in_channels
        self._dim_feedforward = dim_feedforward = dim_feedforward or 4 * dim
        self._num_kv_heads = num_kv_heads
        self._qk_norm = qk_norm
        self._num_experts = num_experts
        self._num_slots = num_slots
        self._moe_layers = moe_layers
        self._layer_scale = layer_scale
        self._stochastic_depth = stochastic_depth
        self._activation = activation
        self._gate_activation = gate_activation
        self._bias = bias
        self._dropout = dropout

        # Stem tokenizer
        stem_type = PatchEmbed2d if isinstance(patch_size, int) or len(patch_size) == 2 else PatchEmbed3d
        self.stem = stem_type(in_channels, dim, cast(Any, patch_size), dropout=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(depth)])
        self.embedding_norm = nn.LayerNorm(dim)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def in_channels(self) -> int:
        return self._in_channels

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
            d_model=self._dim,
            nhead=self._nhead,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
            activation=self._activation,
            gate_activation=self._gate_activation,
            num_kv_heads=self._num_kv_heads,
            qk_norm=self._qk_norm,
            num_experts=self._num_experts if i in self._moe_layers else None,
            num_slots=self._num_slots if i in self._moe_layers else None,
            layer_scale=self._layer_scale,
            stochastic_depth=self._stochastic_depth,
            bias=self._bias,
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
        d_kv = d_kv or self._dim
        _kwargs: Dict[str, Any] = dict(
            d_model=self._dim,
            nhead=self._nhead,
            d_kv=d_kv,
            dim_feedforward=self._dim_feedforward,
            dropout=self._dropout,
            activation=self._activation,
            gate_activation=self._gate_activation,
            num_kv_heads=self._num_kv_heads,
            qk_norm=self._qk_norm,
            num_experts=self._num_experts if i in self._moe_layers else None,
            num_slots=self._num_slots if i in self._moe_layers else None,
            layer_scale=self._layer_scale,
            stochastic_depth=self._stochastic_depth,
            bias=self._bias,
        )
        _kwargs.update(kwargs)
        layer = TransformerDecoderLayer(**_kwargs)
        return layer

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
            x = reshape_tokens(x, tokenized_size)

        return x


class AdaptiveViT(ViT):

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int | Tuple[int, int] | Tuple[int, int, int],
        target_shape: Tuple[int, int] | Tuple[int, int, int],
        depth: int,
        high_res_depth: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = relu2,
        gate_activation: Callable[[Tensor], Tensor] | None = None,
        num_kv_heads: int | None = None,
        qk_norm: bool = False,
        stochastic_depth: float = 0.0,
        bias: bool = False,
        num_experts: int | None = None,
        num_slots: int | None = None,
        moe_layers: List[int] = [],
        layer_scale: float | None = None,
    ):
        super().__init__(
            in_channels,
            dim,
            patch_size,
            depth,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            gate_activation,
            num_kv_heads,
            qk_norm,
            num_experts,
            num_slots,
            moe_layers,
            layer_scale,
            stochastic_depth,
            bias,
        )
        self.resize_to_fixed = nn.Upsample(
            size=self.stem.original_size(cast(Any, target_shape)),
            mode="bilinear",
            align_corners=False,
        )
        self.target_shape = cast(Any, target_shape)

        # Blocks updating fixed (coarse) tokens with cross attention to high res (dynamic) tokens
        self.fixed_blocks = nn.ModuleList(
            [self.create_decoder_layer(i + len(self.blocks)) for i in range(high_res_depth)]
        )
        # Blocks updating high res (dynamic) tokens with cross attention to fixed (coarse) tokens
        self.dynamic_blocks = nn.ModuleList(
            [self.create_decoder_layer(i + len(self.blocks), self_attn=False) for i in range(high_res_depth)]
        )

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Tokenize to fixed and dynamic tokens
        dynamic_tokens = self.stem(x)
        fixed_tokens = self.stem(self.resize_to_fixed(x))

        # Mask tokens if given, storing the resized mask
        if mask is not None:
            size = self.stem.tokenized_size(cast(Any, original_size))
            fixed_mask = resize_mask(size, self.target_shape, mask)
            fixed_tokens = apply_mask(fixed_mask, fixed_tokens, fill_value=mask_fill_value)
            dynamic_tokens = apply_mask(mask, dynamic_tokens, fill_value=mask_fill_value)
        else:
            fixed_mask = size = None

        # Self attention encoder blocks to fixed tokens
        for block in cast(Any, self.blocks):
            fixed_tokens = block(fixed_tokens)

        # Alternating cross attention blocks between fixed and dynamic tokens
        for fixed_block, dynamic_block in zip(self.fixed_blocks, self.dynamic_blocks):
            dynamic_tokens = dynamic_block(dynamic_tokens, fixed_tokens)
            fixed_tokens = fixed_block(fixed_tokens, dynamic_tokens)

        # Output norm
        dynamic_tokens = self.embedding_norm(dynamic_tokens)

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            dynamic_tokens = reshape_tokens(dynamic_tokens, tokenized_size)

        return dynamic_tokens
