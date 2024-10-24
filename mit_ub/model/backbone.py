from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import torch.nn as nn
from deep_helpers.tokens import apply_mask
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from .mlp import relu2
from .stem import AdaptiveTokenizer2d, AdaptiveTokenizer3d, PatchEmbed2d, PatchEmbed3d
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


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
        return TransformerDecoderLayer(**_kwargs)

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
        elif reshape and len(tokenized_size) == 3:
            D, H, W = tokenized_size
            x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        elif reshape:
            assert len(tokenized_size) == 2
            H, W = tokenized_size
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x


class AdaptiveViT(ViT):
    stem: AdaptiveTokenizer2d | AdaptiveTokenizer3d

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
        high_res_layer_scale: float | None = 1e-5,
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

        # Adaptive stem tokenizer
        stem_type = AdaptiveTokenizer2d if isinstance(patch_size, int) or len(patch_size) == 2 else AdaptiveTokenizer3d
        self.stem = stem_type(
            in_channels,
            dim,
            cast(Any, patch_size),
            cast(Any, target_shape),
            dropout=dropout,
        )

        # Cross attention to high res tokens
        self.high_res_blocks = nn.ModuleList(
            [
                self.create_decoder_layer(i + len(self.blocks), layer_scale=high_res_layer_scale)
                for i in range(high_res_depth)
            ]
        )

    def convert_mask_to_kv_mask(
        self,
        input_size: Tuple[int, int] | Tuple[int, int, int],
        mask: Tensor,
    ) -> Tensor:
        size = self.stem.tokenized_size(cast(Any, input_size))
        kv_size = self.stem.kv_size(cast(Any, input_size))
        B = mask.shape[0]
        mask = mask.view(B, 1, *size)
        mask = F.interpolate(mask.float(), size=kv_size, mode="nearest").to(mask.dtype)
        mask = mask.view(B, -1)
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

        # Tokenize to q (pooled tokens) and kv (high-res tokens)
        q, kv = self.stem(x)

        # Apply token mask if given
        if mask is not None:
            kv_mask = self.convert_mask_to_kv_mask(cast(Any, tuple(original_size)), mask)
            q = apply_mask(mask, q, fill_value=mask_fill_value)
            kv = apply_mask(kv_mask, kv, fill_value=mask_fill_value)
        else:
            kv_mask = None

        # Self attention encoder blocks to coarse tokens
        if self.blocks is not None:
            for block in cast(Any, self.blocks):
                q = block(q)

        # Cross attention blocks between fixed backbone tokens and high res input tokens
        for block in self.high_res_blocks:
            q = block(q, kv)
        x = self.embedding_norm(q)

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape and len(tokenized_size) == 3:
            D, H, W = tokenized_size
            x = rearrange(x, "b (d h w) c -> b c d h w", d=D, h=H, w=W)
        elif reshape:
            assert len(tokenized_size) == 2
            H, W = tokenized_size
            x = rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x
