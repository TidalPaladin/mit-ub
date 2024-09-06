from typing import Any, Callable, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
from deep_helpers.helpers import to_tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from ssl_tasks.tokens import TokenMask
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .adaptive_tokenizer import AdaptiveTokenizer2d
from .pos_enc import RelativeFactorizedPosition
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


class ViT(nn.Module):

    def __init__(
        self,
        in_channels: int,
        dim: int,
        patch_size: int | Sequence[int],
        depth: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: nn.Module = nn.SiLU(),
        gate_activation: nn.Module | None = None,
        position_noise: bool = False,
        output_norm: bool = True,
        num_kv_heads: int | None = None,
    ):
        super().__init__()
        self._dim = dim
        self._nhead = nhead if nhead is not None else self.dim // 32
        self._in_channels = in_channels
        self._dim_feedforward = dim_feedforward = dim_feedforward or 4 * dim
        self._position_noise = position_noise

        # Make patch size length 3 (D H W) for 2D or 3D inputs
        if isinstance(patch_size, int):
            patch_size = to_tuple(patch_size, 3)
        elif len(patch_size) == 2:
            patch_size = (1, *patch_size)
        self._patch_size = patch_size
        assert len(self.patch_size) == 3

        # Patch embeddings
        self.patch_embed_2d = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=self.patch_size_2d, stride=self.patch_size_2d),
            Rearrange("b c h w -> b (h w) c"),
            nn.LayerNorm(dim),
        )
        self.pos_enc_2d = RelativeFactorizedPosition(2, dim)

        # TODO: When patch size is (1, H, W) we can share weight / bias with Conv2d
        # Disabled for now because in DDP this triggers unused params
        # self.patch_embed_3d = nn.Sequential(
        #    nn.Conv3d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size),
        #    Rearrange("b c d h w -> b (d h w) c"),
        #    nn.LayerNorm(dim),
        # )
        # self.pos_enc_3d = RelativeFactorizedPosition(3, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    dim,
                    nhead,
                    dim_feedforward,
                    dropout,
                    activation,
                    gate_activation,
                    num_kv_heads=num_kv_heads,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim) if output_norm else nn.Identity()

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

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return cast(Tuple[int, int, int], self._patch_size)

    @property
    def patch_size_2d(self) -> Tuple[int, int]:
        return self.patch_size[-2:]

    def tokenized_size(self, *size: int) -> Sequence[int]:
        patch_size = self.patch_size[-len(size) :]
        return tuple(s // p for s, p in zip(size, patch_size))

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: TokenMask | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.tokenized_size(*original_size)

        # Patch embedding and positional encoding.
        # Since medical inputs have high dynamic range, we use float32 for the patch embedding
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            dtype = x.dtype
            x = x.to(torch.float32)
            if is_3d := x.ndim == 5:
                raise NotImplementedError("3D input not supported")
            else:
                x = self.patch_embed_2d(x)
                x += self.pos_enc_2d.from_grid(
                    tokenized_size,
                    B,
                    proto=x,
                    normalize=True,
                    add_noise=self.pos_enc_2d.training and self._position_noise,
                )
            x = x.to(dtype)

        if mask is not None:
            x = mask.apply_to_tokens(x, fill_value=mask_fill_value)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )

        if reshape and is_3d:
            raise NotImplementedError("3D reshape not implemented")
        elif reshape:
            x = rearrange(x, "b (h w) c -> b c h w", h=tokenized_size[0], w=tokenized_size[1])

        return x

    def register_mask_hook(self, func: Callable, *args, **kwargs) -> RemovableHandle:
        r"""Register a token masking hook to be applied after the patch embedding step.

        Args:
            func: Callable token making hook with signature given in :func:`register_forward_hook`

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``.
        """
        return self.patch_embed_2d.register_forward_hook(func, *args, **kwargs)


class AdaptiveViT(ViT):

    def __init__(
        self,
        in_channels: int,
        dim: int,
        kv_dim: int,
        patch_size: int | Sequence[int],
        target_shape: Tuple[int, int],
        encoder_depth: int,
        decoder_depth: int,
        nhead: int,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation: nn.Module = nn.SiLU(),
        gate_activation: nn.Module | None = None,
        position_noise: bool = False,
        output_norm: bool = True,
        num_kv_heads: int | None = None,
    ):
        # NOTE: The naming of transformer layers follows PyTorch. However, our "encoder" is a TransformerDecoderLayer
        # that cross-attends to high-res input tokens and our "decoder" is a TransformerEncoderLayer that attends only to
        # the backbone tokens.
        super().__init__(
            in_channels,
            dim,
            patch_size,
            decoder_depth,
            nhead,
            dim_feedforward,
            dropout,
            activation,
            gate_activation,
            position_noise,
            output_norm,
            num_kv_heads,
        )
        # These are all provided by the adaptive tokenizer
        delattr(self, "patch_embed_2d")
        # delattr(self, "patch_embed_3d")
        delattr(self, "pos_enc_2d")
        # delattr(self, "pos_enc_3d")
        if not decoder_depth:
            self.blocks = None

        # TODO: For now only a 2D tokenizer is provided. 3D tokenization support is ready via AdaptiveTokenizer3d
        # but there isn't an immediate need for it. Having unused parameters complicates DDP training, so we omit
        # the 3D tokenizer for now.
        self.tokenizer = AdaptiveTokenizer2d(in_channels, dim, kv_dim, self.patch_size_2d, target_shape)
        self.encoder_blocks = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    dim,
                    nhead,
                    kv_dim,
                    self.dim_feedforward,
                    dropout,
                    activation,
                    gate_activation,
                    num_kv_heads=num_kv_heads,
                )
                for _ in range(encoder_depth)
            ]
        )

    def tokenized_size(self, *size: int) -> Sequence[int]:
        return self.tokenizer.target_shape

    def equivalent_size_2d(self, *size: int) -> Sequence[int]:
        return tuple(s * p for s, p in zip(self.tokenized_size(*size), self.patch_size_2d))

    @property
    def pos_enc_2d(self) -> nn.Module:
        return self.tokenizer.pos_enc_q

    @property
    def pos_enc_3d(self) -> nn.Module:
        return self.tokenizer.pos_enc_q

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: TokenMask | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.tokenized_size(*original_size)
        is_3d = x.ndim == 5
        assert not is_3d, "3D input not supported for AdaptiveViT"

        # Tokenize (includes position encoding)
        # Since medical inputs have high dynamic range, we use float32 for the patch embedding
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            q, kv = self.tokenizer(x.to(torch.float32))
            q = q.to(x.dtype)
            kv = kv.to(x.dtype)

        # Determine the grid size of the key/value tokens
        kv_tokenized_size = self.tokenizer.kv_size(x.shape[2:])
        # Determine the effective input size of the key/value tokens by scaling up by patch size
        effective_kv_size = tuple(t * p for t, p in zip(kv_tokenized_size, self.patch_size_2d))

        # Apply token mask if given
        if mask is not None:
            kv_mask = mask.resize(effective_kv_size)
            q = mask.apply_to_tokens(q, fill_value=mask_fill_value)
            kv = kv_mask.apply_to_tokens(kv, fill_value=mask_fill_value)
        else:
            kv_mask = None

        # Cross attention blocks between fixed backbone tokens and high res input tokens
        for block in self.encoder_blocks:
            q = block(q, kv)
        x = q

        # Transformer blocks
        if self.blocks is not None:
            for block in cast(Any, self.blocks):
                x = block(x)

        x = self.norm(x)

        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )

        if reshape and is_3d:
            raise NotImplementedError("3D reshape not implemented")
            # x = rearrange(x, "b (d h w) c -> b c d h w", d=tokenized_size[0], h=tokenized_size[1], w=tokenized_size[2])
        elif reshape:
            x = rearrange(x, "b (h w) c -> b c h w", h=tokenized_size[0], w=tokenized_size[1])

        return x
