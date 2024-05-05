from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from deep_helpers.helpers import to_tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .kernels.attention import MultiheadAttention
from .pos_enc import RelativeFactorizedPosition


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.GELU(),
        layer_norm_eps: float = 1e-5,
        alibi_upper: int = 8,
    ):
        super().__init__()
        self.nhead = nhead

        # Cross attention 1
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.register_buffer("alibi", self.init_alibi(alibi_upper))

    def init_alibi(self, upper: int) -> Tensor:
        return torch.logspace(0, upper, self.nhead, base=2).reciprocal_().neg_()

    def forward(
        self, x: Tensor, pos: Tensor | None = None, full_precision: bool = True, mask_threshold: float | None = None
    ) -> Tensor:
        # Self attention
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        slopes = self.alibi.view(1, H).expand(B, -1).contiguous()
        y = self.self_attn(y, y, y, pos, pos, slopes, full_precision=full_precision, mask_threshold=mask_threshold)
        x += y

        # MLP
        y = self.norm2(x)
        y = self.linear1(y)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear2(y)
        x += y

        return x


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
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__()
        self._dim = dim
        self._nhead = nhead if nhead is not None else self.dim // 32
        self._in_channels = in_channels
        dim_feedforward = dim_feedforward or 4 * dim

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
        )
        # TODO: When patch size is (1, H, W) we can share weight / bias with Conv2d
        self.patch_embed_3d = nn.Sequential(
            nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            Rearrange("b c d h w -> b (d h w) c"),
        )

        # Positional encoding
        self.pos_enc_2d = RelativeFactorizedPosition(2, dim)
        self.pos_enc_3d = RelativeFactorizedPosition(3, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, nhead, dim_feedforward, dropout, activation, alibi_upper=i) for i in range(depth)]
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def nhead(self) -> int:
        return self._nhead

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    @property
    def patch_size_2d(self) -> Tuple[int, int]:
        return self.patch_size[-2:]

    def tokenized_size(self, *size: int) -> Sequence[int]:
        patch_size = self.patch_size[-len(size) :]
        return tuple(s // p for s, p in zip(size, patch_size))

    def forward(self, x: Tensor, reshape: bool = True) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.tokenized_size(*original_size)

        # Patch embedding and positional encoding
        if is_3d := x.ndim == 5:
            x = self.patch_embed_3d(x)
            x += self.pos_enc_3d.from_grid(tokenized_size, B, proto=x, normalize=True)
        else:
            x = self.patch_embed_2d(x)
            x += self.pos_enc_2d.from_grid(tokenized_size, B, proto=x, normalize=True)

        # Position values for ALiBi
        # Either pos_enc_3d or pos_enc_2d can be used here
        position = self.pos_enc_3d.create_grid(
            tokenized_size,
            proto=x,
            normalize=False,
            requires_grad=False,
        )
        position = position.contiguous().view(1, 1, -1, len(tokenized_size)).expand(B, self.nhead, -1, -1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, position)

        if reshape and is_3d:
            x = rearrange(x, "b (d h w) c -> b c d h w", d=tokenized_size[0], h=tokenized_size[1], w=tokenized_size[2])
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
