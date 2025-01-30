from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from ...tokens import apply_mask, create_mask, mask_is_ragged, unapply_mask
from ..activations import DEFAULT_POS_ENC_ACTIVATION, get_activation
from ..helpers import compile_is_disabled, set_checkpointing
from ..layers.layer_scale import LayerScale
from ..stem import AdaptiveTokenizer2d, AdaptiveTokenizer3d, PoolType
from .convnext import grid_to_tokens, tokens_to_grid
from .vit import ViT, ViTConfig


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


@dataclass(frozen=True)
class AdaptiveViTConfig(ViTConfig):
    target_shape: Sequence[int] | None = None
    share_layers: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.target_shape is None:
            raise ValueError("`target_shape` must be provided")

    def instantiate(self) -> "AdaptiveViT":
        return AdaptiveViT(self)


class AdaptiveViT(ViT):
    stem: AdaptiveTokenizer2d | AdaptiveTokenizer3d
    config: AdaptiveViTConfig
    CONFIG_TYPE: ClassVar[Type[AdaptiveViTConfig]] = AdaptiveViTConfig

    def __init__(self, config: AdaptiveViTConfig):
        super().__init__(config)
        stem_act = (
            get_activation(config.activation)
            if config.activation != "identity"
            else (
                get_activation(config.gate_activation)
                if config.gate_activation is not None and config.gate_activation != "identity"
                else DEFAULT_POS_ENC_ACTIVATION
            )
        )
        stem_type = (
            AdaptiveTokenizer2d
            if isinstance(config.patch_size, int) or len(config.patch_size) == 2
            else AdaptiveTokenizer3d
        )
        self.stem = stem_type(
            config.in_channels,
            config.dim,
            cast(Any, config.patch_size),
            cast(Any, config.target_shape),
            dropout=config.dropout,
            pool_type=PoolType.AVG,
            activation=stem_act,
            norm_type=config.norm_type,
        )

        # Convert ViT blocks into decoder layers that cross-attend to the dynamic tokens.
        # We also override the layer scale of the cross attention so that the initialization condition
        # is close to identity.
        self.blocks = nn.ModuleList([self.create_decoder_layer(i, kv_norm=True) for i in range(config.depth)])

        # Blocks updating high res (dynamic) tokens with cross attention to fixed (coarse) tokens
        self.dynamic_blocks = nn.ModuleList(
            [
                self.create_decoder_layer(i + len(self.blocks), self_attn=False, kv_norm=True)
                for i in range(config.depth)
            ]
        )
        # Since we are updating the KV tokens iteratively we must ensure that they are always normalized.
        # Failure do to do will result in numerical instability that is hard to debug.
        assert all(
            block.cross_attn.kv_norm for block in self.dynamic_blocks
        ), "Dynamic blocks must use KV normalization"
        assert all(block.cross_attn.kv_norm for block in self.blocks), "Fixed blocks must use KV normalization"

        if config.share_layers:
            self.set_shared_layers()

        # Initialize the dynamic pathway to have low contribution
        self.dynamic_output_scale = (
            LayerScale(config.dim, config.layer_scale) if config.layer_scale is not None else nn.Identity()
        )

        if config.checkpoint:
            set_checkpointing(self, config.checkpoint)

    @property
    def is_sharing_layers(self) -> bool:
        return self.blocks[0].mlp is self.dynamic_blocks[0].mlp

    def on_load_checkpoint(self, state_dict: Dict[str, Any], *args, **kwargs) -> None:
        # Only initialize the dynamic blocks with fixed block weights if we aren't weight sharing
        # and the checkpoint doesn't already contain dynamic blocks
        if not self.is_sharing_layers and not any(k.startswith("dynamic_blocks") for k in state_dict.keys()):
            self.init_dynamic_from_fixed()

    @torch.no_grad()
    def set_shared_layers(self) -> None:
        r"""Sets the layers of dynamic blocks that participate in sharing.

        The following layers are shared:
            - ``block.cross_attn`` -> ``dynamic_block.cross_attn``
            - ``block.mlp`` -> ``dynamic_block.mlp``

        Since dynamic blocks do not contain a self-attention layer, this layer is not shared.
        """
        shared = ("cross_attn", "mlp")
        for block, dynamic_block in zip(self.blocks, self.dynamic_blocks):
            for name in shared:
                child = getattr(block, name)
                setattr(dynamic_block, name, child)

    @torch.no_grad()
    def init_dynamic_from_fixed(self) -> None:
        r"""Initializes weights of the dynamic blocks by copying the weights of the fixed blocks.

        Note that this is distinct from layer sharing, which is done by the ``set_shared_layers`` method.
        This initializer retains distinct parameters for the dynamic blocks, which may diverge after training.
        :meth:`set_shared_layers` links the respective layers, ensuring that the weights remain the same across
        a linkage.

        The following initialization is used:
            - ``block.mlp`` -> ``dynamic_block.mlp``
            - ``block.self_attn.`` -> ``dynamic_block.cross_attn``
        """
        for block, dynamic_block in zip(self.blocks, self.dynamic_blocks):
            # Copy the parameters of the self-attention layer to the cross-attention layer
            dynamic_block.cross_attn.copy_parameters(block.self_attn)

            # Copy the parameters of the MLP layer to the dynamic MLP layer
            for name, param in block.mlp.named_parameters():
                if hasattr(dynamic_block.mlp, name):
                    setattr(dynamic_block.mlp, name, nn.Parameter(param.clone()))

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        batch_size = input.shape[0]
        device = input.device

        # Create the mask based on the fixed tokenized size
        fixed_size = self.stem.tokenized_size(cast(Any, self.config.target_shape))
        mask = create_mask(
            fixed_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size,
            scale=scale,
            device=device,
        )

        # Resize the mask to the dynamic tokenized size to ensure non-ragged mask
        dynamic_size = self.stem.tokenized_size(cast(Any, input.shape[2:]))
        mask = resize_mask(fixed_size, dynamic_size, mask)
        assert not mask_is_ragged(mask), "Mask is ragged"

        return mask

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        dynamic_tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))
        fixed_tokenized_size = self.stem.target_tokenized_shape
        fixed_mask = resize_mask(dynamic_tokenized_size, fixed_tokenized_size, mask) if mask is not None else None

        # Tokenize to fixed and dynamic tokens
        fixed_tokens, dynamic_tokens = self.stem(x)

        # Mask tokens if given, storing the resized mask
        if mask is not None and fixed_mask is not None:
            fixed_tokens = apply_mask(fixed_mask, fixed_tokens, fill_value=mask_fill_value)
            dynamic_tokens = apply_mask(mask, dynamic_tokens, fill_value=mask_fill_value)

        # Add CLS token (fixed pathway only)
        fixed_tokens = torch.cat([self.cls_token.view(1, 1, -1).expand(B, -1, -1), fixed_tokens], dim=1)

        # Run the backbone
        for block, dynamic_block in zip(self.blocks, self.dynamic_blocks):
            fixed_tokens = block(fixed_tokens, dynamic_tokens)
            dynamic_tokens = dynamic_block(dynamic_tokens, fixed_tokens)

        # Extract CLS token (fixed pathway only)
        cls_token = fixed_tokens[:, 0, :].contiguous()
        fixed_tokens = fixed_tokens[:, 1:, :].contiguous()

        # Upsample fixed tokens and add to dynamic tokens
        if mask is not None and fixed_mask is not None:
            fixed_tokens = unapply_mask(fixed_mask, fixed_tokens)
        fixed_tokens = tokens_to_grid(fixed_tokens, fixed_tokenized_size)
        fixed_tokens = F.interpolate(fixed_tokens, size=dynamic_tokenized_size, mode="nearest")
        fixed_tokens = grid_to_tokens(fixed_tokens)
        if mask is not None and fixed_mask is not None:
            fixed_tokens = apply_mask(mask, fixed_tokens, fill_value=mask_fill_value)
        assert fixed_tokens.shape == dynamic_tokens.shape
        dynamic_tokens = self.dynamic_output_scale(dynamic_tokens) + fixed_tokens

        # Output norm
        dynamic_tokens = self.embedding_norm(dynamic_tokens)
        cls_token = self.embedding_norm(cls_token)

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            dynamic_tokens = tokens_to_grid(dynamic_tokens, dynamic_tokenized_size)

        return dynamic_tokens, cls_token

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = AdaptiveViTConfig(*args, **kwargs)
        return cls(config)
