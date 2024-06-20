from typing import Callable, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
from deep_helpers.helpers import to_tuple
from einops import rearrange
from einops.layers.torch import Rearrange
from ssl_tasks.tokens import TokenMask
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from .gqa import GroupedQueryAttention
from .kernels.attention import MultiheadAttention
from .pos_enc import RelativeFactorizedPosition


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: nn.Module = nn.SiLU(),
        layer_norm_eps: float = 1e-5,
        alibi_lower: int | None = None,
        alibi_upper: int | None = None,
    ):
        super().__init__()
        self.nhead = nhead

        # Configure ALiBi if bounds are provided
        if alibi_lower is None and alibi_upper is None:
            self.alibi = None
        elif alibi_lower is not None and alibi_upper is not None:
            self.register_buffer("alibi", self.init_alibi(alibi_lower, alibi_upper))
        else:
            raise ValueError("Either both `alibi_lower` and `alibi_upper` must be provided, or neither")

        self.self_attn = (
            MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            if self.alibi is not None
            else GroupedQueryAttention(d_model, nhead, nhead // 2, dropout=dropout)
        )

        # MLP up-project is a GLU-variant -> F.silu(W1x + b1) * F.sigmoid(W2x + b2).
        # This was hallucinated by GPT and empirically works better than standard SwiGLU.
        # Maybe call this SigmoidGatedSiLU or something? Has it been done in literature?
        # Seems similar to squeeze/excite.
        #
        # Improves probe accuracy, convergence rate, and reduces feature variance
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.gate = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Sigmoid(),
        )

        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

    @torch.no_grad()
    def init_alibi(self, lower: int, upper: int) -> Tensor:
        return torch.logspace(lower, upper, self.nhead, base=2).reciprocal_().neg_()

    def forward(
        self,
        x: Tensor,
        pos: Tensor | None = None,
        full_precision: bool = True,
        mask_threshold: float | None = None,
    ) -> Tensor:
        # Self attention
        y = self.norm1(x)
        B, H = x.shape[0], self.nhead
        if self.alibi is not None:
            slopes = self.alibi.view(1, H).expand(B, -1).contiguous()
            y = self.self_attn(y, y, y, pos, pos, slopes, mask_threshold=mask_threshold)
        else:
            y = self.self_attn(y)
        x = x + y

        # MLP
        y = self.norm2(x)
        y = self.activation(self.linear1(y)) * self.gate(y)
        y = self.dropout(y)
        y = self.linear2(y)
        x = x + y

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
        activation: nn.Module = nn.SiLU(),
        alibi: bool = True,
    ):
        super().__init__()
        self._dim = dim
        self._nhead = nhead if nhead is not None else self.dim // 32
        self._in_channels = in_channels
        self._alibi = alibi
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
            nn.Conv3d(in_channels, dim, kernel_size=self.patch_size, stride=self.patch_size),
            Rearrange("b c d h w -> b (d h w) c"),
        )

        # Positional encoding
        self.pos_enc_2d = RelativeFactorizedPosition(2, dim)
        self.pos_enc_3d = RelativeFactorizedPosition(3, dim)

        # Transformer blocks
        alibi_bounds = [self.get_alibi_bounds(i) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(dim, nhead, dim_feedforward, dropout, activation, alibi_lower=lower, alibi_upper=upper)
                for lower, upper in alibi_bounds
            ]
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
        return cast(Tuple[int, int, int], self._patch_size)

    @property
    def patch_size_2d(self) -> Tuple[int, int]:
        return self.patch_size[-2:]

    def tokenized_size(self, *size: int) -> Sequence[int]:
        patch_size = self.patch_size[-len(size) :]
        return tuple(s // p for s, p in zip(size, patch_size))

    def get_alibi_bounds(self, depth: int) -> Tuple[int, int] | Tuple[None, None]:
        r"""Get the ALiBi bounds for a given depth

        Args:
            depth: Depth of the transformer block

        Returns:
            Lower and upper bounds for the alibi slopes, or ``None`` if alibi is not used
        """
        if self._alibi:
            return 0, depth
        else:
            return None, None

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: TokenMask | None = None,
        mask_fill_value: float | Tensor | None = None,
        pos_mask_threshold: float | None = None,
        full_precision: bool = True,
    ) -> Tensor:
        B, C, *original_size = x.shape
        tokenized_size = self.tokenized_size(*original_size)

        # Patch embedding and positional encoding
        if is_3d := x.ndim == 5:
            x = self.patch_embed_3d(x)
            x += self.pos_enc_3d.from_grid(tokenized_size, B, proto=x, normalize=True)
        else:
            x = self.patch_embed_2d(x)
            x += self.pos_enc_2d.from_grid(tokenized_size, B, proto=x, normalize=True)
        if mask is not None:
            x = mask.apply_to_tokens(x, fill_value=mask_fill_value)

        # ALiBi positions are unnormalized, i.e. in pixel coordinate units
        position = self.create_alibi_positions(x, tokenized_size, mask, mask_fill_value, normalize=False)
        position = position.expand(-1, self.nhead, -1, -1)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, position, mask_threshold=pos_mask_threshold, full_precision=full_precision)

        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )

        if reshape and is_3d:
            x = rearrange(x, "b (d h w) c -> b c d h w", d=tokenized_size[0], h=tokenized_size[1], w=tokenized_size[2])
        elif reshape:
            x = rearrange(x, "b (h w) c -> b c h w", h=tokenized_size[0], w=tokenized_size[1])

        return x

    @torch.no_grad()
    def create_alibi_positions(
        self,
        tokens: Tensor,
        tokenized_size: Sequence[int],
        mask: TokenMask | None = None,
        mask_fill_value: float | Tensor | None = None,
        normalize: bool = False,
    ) -> Tensor:
        r"""Create position values for ALiBi

        Args:
            tokens: Input tokens. Used only as a proto to extract device info from
            tokenized_size: Tokenized size of the input
            mask: Optional token mask
            mask_fill_value: Fill value for the mask, or ``None`` to drop masked tokens.
            normalize: Whether to normalize the position values to the range :math:`[-1, 1]`

        Shapes:
            - Output: :math:`(B, 1, L, C)` where
                - B: Batch size
                - L: Number of tokens
                - C: Number of spatial dimensions

        Returns:
            ALiBi position values
        """
        # Position values for ALiBi
        # Either pos_enc_3d or pos_enc_2d can be used here
        B = tokens.shape[0]
        position = self.pos_enc_3d.create_grid(
            tokenized_size,
            proto=tokens,
            normalize=normalize,
            requires_grad=False,
        )
        if mask is not None:
            position = position.view(1, -1, len(tokenized_size)).expand(B, -1, -1)
            position = mask.apply_to_tokens(position, fill_value=mask_fill_value)
            position = position.contiguous().view(B, 1, -1, len(tokenized_size))
        else:
            position = position.contiguous().view(1, 1, -1, len(tokenized_size)).expand(B, -1, -1, -1)

        return position

    def register_mask_hook(self, func: Callable, *args, **kwargs) -> RemovableHandle:
        r"""Register a token masking hook to be applied after the patch embedding step.

        Args:
            func: Callable token making hook with signature given in :func:`register_forward_hook`

        Returns:
            A handle that can be used to remove the added hook by calling ``handle.remove()``.
        """
        return self.patch_embed_2d.register_forward_hook(func, *args, **kwargs)
