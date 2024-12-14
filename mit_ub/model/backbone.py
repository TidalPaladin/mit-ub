from typing import Any, Callable, Dict, List, Optional, Tuple, cast, Self

import torch
import math
import torch.nn as nn
from deep_helpers.tokens import apply_mask
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F
from copy import deepcopy

from .helpers import compile_is_disabled
from .mlp import relu2
from .layer_scale import LayerScale
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


def add_masked(
    base: Tensor,
    add: Tensor,
    base_mask: Tensor,
    add_mask: Tensor,
    base_shape: Tuple[int, int] | Tuple[int, int, int],
    add_shape: Tuple[int, int] | Tuple[int, int, int],
) -> Tensor:
    r"""Adds two sets of masked tokens with different resolutions together.

    Args:
        base: Tensor of high resolution tokens.
        add: Tensor of low resolution tokens to be added to the high resolution tokens.
        base_mask: Mask for the high resolution tokens.
        add_mask: Mask for the low resolution tokens.
        base_shape: Shape of the high resolution tokens (:math:`(H, W)` or :math:`(D, H, W)`)
        add_shape: Shape of the low resolution tokens (:math:`(H', W')` or :math:`(D', H', W')`)

    Shapes:
        base: :math:`(B, L, D)`
        add: :math:`(B, L', D)`
        base_mask: :math:`(B, L)`
        add_mask: :math:`(B, L')`

    Returns:
        Tensor of high resolution tokens with the low resolution tokens added.
    """
    # Create grids
    B, _, D = base.shape
    base_grid = base.new_zeros(B, math.prod(base_shape), D)
    base_grid[base_mask] = base.reshape(-1, D)
    base_grid = rearrange(base_grid, "b (h w) d -> b d h w", h=base_shape[0], w=base_shape[1])

    add_grid = add.new_zeros(B, math.prod(add_shape), D)
    add_grid[add_mask] = add.reshape(-1, D)
    add_grid = rearrange(add_grid, "b (h w) d -> b d h w", h=add_shape[0], w=add_shape[1])

    # Reshape masks to match grid shapes
    if len(base_shape) == 3:
        D, H, W = base_shape
        base_mask_grid = rearrange(base_mask, "b (d h w) -> b () d h w", d=D, h=H, w=W)
    else:
        H, W = base_shape
        base_mask_grid = rearrange(base_mask, "b (h w) -> b () h w", h=H, w=W)
        
    if len(add_shape) == 3:
        D, H, W = add_shape
        add_mask_grid = rearrange(add_mask, "b (d h w) -> b () d h w", d=D, h=H, w=W)
    else:
        H, W = add_shape
        add_mask_grid = rearrange(add_mask, "b (h w) -> b () h w", h=H, w=W)
        
    # Resize add grid and mask to base size
    add_grid = F.interpolate(add_grid, size=base_shape, mode='nearest')
    add_mask_grid = F.interpolate(add_mask_grid.float(), size=base_shape, mode='nearest').bool()
    
    # Add grids where base is masked and add is unmasked
    combined_grid = base_grid.clone()
    combined_mask = (base_mask_grid & add_mask_grid).expand_as(combined_grid)
    combined_grid[combined_mask] = combined_grid[combined_mask] + add_grid[combined_mask]
    
    # Reshape back to sequence
    if len(base_shape) == 3:
        D, H, W = base_shape
        combined = rearrange(combined_grid, "b c d h w -> b (d h w) c")
    else:
        H, W = base_shape
        combined = rearrange(combined_grid, "b c h w -> b (h w) c")

    combined = apply_mask(base_mask, combined)
    return combined


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
        latent_dim: int | None = None,
        dynamic_layer_scale: float | None = None,
    ):
        self._latent_dim = latent_dim or dim
        if self._latent_dim != dim:
            raise NotImplementedError("Latent dimension must match model dimension")
        super().__init__(
            in_channels,
            self._latent_dim,
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
        self._dim = dim
        self._target_shape = cast(Any, target_shape)
        self._no_grad_vit = False
        self.stem.original_size(cast(Any, self.target_shape))
        self.dynamic_layer_scale = dynamic_layer_scale

        # Resize to fixed input size
        self.resize_to_fixed = nn.Upsample(
            size=self.fixed_input_size,
            mode="bilinear",
            align_corners=False,
        )

        # Stem for dynamic tokens
        stem_type = type(self.stem)
        #self.dynamic_stem = stem_type(in_channels, self._dim, cast(Any, patch_size), dropout=dropout)
        self.stem_layer_scale = LayerScale(self._dim, self.dynamic_layer_scale) if self.dynamic_layer_scale is not None else nn.Identity()

        # Blocks updating fixed (coarse) tokens with cross attention to high res (dynamic) tokens
        self.fixed_blocks = nn.ModuleList(
            [
                self.create_decoder_layer(
                    i + len(self.blocks), 
                    layer_scale=self.dynamic_layer_scale,
                    d_model=self._latent_dim,
                    d_kv=self._dim,
                ) 
                for i in range(high_res_depth)
            ]
        )
        # Blocks updating high res (dynamic) tokens with cross attention to fixed (coarse) tokens
        self.dynamic_blocks = nn.ModuleList(
            [
                self.create_decoder_layer(
                    i + len(self.blocks), 
                    self_attn=False, 
                    layer_scale=self.dynamic_layer_scale,
                    d_model=self._dim,
                    d_kv=self._latent_dim,
                ) 
                for i in range(high_res_depth)
            ]
        )

        # Output norm for dynamic tokens
        self.dynamic_norm = nn.LayerNorm(self._dim)

    @property
    def target_shape(self) -> Tuple[int, int] | Tuple[int, int, int]:
        return self._target_shape

    @property
    def fixed_input_size(self) -> Tuple[int, int] | Tuple[int, int, int]:
        r"""Size of the input (in pixels) to the fixed (coarse) tokens.

        This is the size of the input which, after patch embedding, matches the size of the fixed token grid.
        """
        return self.stem.original_size(cast(Any, self.target_shape))

    @property
    def fixed_tokenized_size(self) -> Tuple[int, int] | Tuple[int, int, int]:
        r"""Size of the input fixed token grid. Matches `target_shape`."""
        return self._target_shape

    def freeze_vit_weights(self: Self, no_grad: bool = True) -> Self:
        r"""Freeze weights present in the ViT backbone which are not part of the adaptive pathway.

        This includes the fixed size stem, the transformer blocks, and the embedding norm. The
        cross attention pathway remains unfrozen.

        Args:
            no_grad: Whether to automatically skip gradient computation for the frozen parameters.
        """
        for param in self.stem.parameters():
            param.requires_grad = False
        for param in self.blocks.parameters():
            param.requires_grad = False
        for param in self.embedding_norm.parameters():
            param.requires_grad = False
        if no_grad:
            self._no_grad_vit = True
        return self

    def unfreeze_vit_weights(self: Self) -> Self:
        r"""Unfreeze weights present in the ViT backbone which are not part of the adaptive pathway.

        This is the inverse of `freeze_vit_weights`.
        """
        for param in self.stem.parameters():
            param.requires_grad = True
        for param in self.blocks.parameters():
            param.requires_grad = True
        for param in self.embedding_norm.parameters():
            param.requires_grad = True
        self._no_grad_vit = False
        return self

    @torch.no_grad()
    def resize_mask_to_fixed(self, mask: Tensor, tokenized_size: Tuple[int, int] | Tuple[int, int, int]) -> Tensor:
        return resize_mask(tokenized_size, self.target_shape, mask) 

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, original_size))
        L_dynamic = math.prod(tokenized_size)
        L_fixed = math.prod(self.target_shape)

        # If mask is provided, resize it to the fixed token grid size
        fixed_mask = resize_mask(
            tokenized_size,
            self.target_shape, 
            mask,
        ) if mask is not None else None
        assert fixed_mask is None or fixed_mask.shape == (B, L_fixed)
        assert mask is None or mask.shape == (B, L_dynamic)

        # Run vanilla ViT on fixed resolution token grid
        with torch.set_grad_enabled(not self._no_grad_vit):
            fixed_tokens = super().forward(
                self.resize_to_fixed(x),
                reshape=False,
                mask=fixed_mask,
                mask_fill_value=mask_fill_value,
            )

        # Tokenize dynamic tokens
        dynamic_tokens = self.stem_layer_scale(self.stem(x))
        assert dynamic_tokens.shape == (B, L_dynamic, self._dim)

        # Mask dynamic tokens if given
        if mask is not None:
            dynamic_tokens = apply_mask(mask, dynamic_tokens, fill_value=mask_fill_value)

        # Add fixed tokens to dynamic tokens
        if mask is not None:
            assert fixed_mask is not None
            dynamic_tokens = add_masked(dynamic_tokens, fixed_tokens, mask, fixed_mask, tokenized_size, self.target_shape)
        else:
            _fixed_mask = fixed_tokens.new_ones(B, math.prod(self.target_shape), dtype=torch.bool)
            _mask = fixed_tokens.new_ones(B, math.prod(tokenized_size), dtype=torch.bool)
            dynamic_tokens = add_masked(
                dynamic_tokens, 
                fixed_tokens, 
                _mask, 
                _fixed_mask, 
                tokenized_size, 
                self.target_shape,
            )

        # Alternating cross attention blocks between fixed and dynamic tokens
        for fixed_block, dynamic_block in zip(self.fixed_blocks, self.dynamic_blocks):
            fixed_tokens = fixed_block(fixed_tokens, dynamic_tokens)
            dynamic_tokens = dynamic_block(dynamic_tokens, fixed_tokens)

        # Output norm
        dynamic_tokens = self.dynamic_norm(dynamic_tokens)

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            dynamic_tokens = reshape_tokens(dynamic_tokens, tokenized_size)

        return dynamic_tokens
