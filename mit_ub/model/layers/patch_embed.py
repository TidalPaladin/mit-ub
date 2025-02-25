from typing import Tuple

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from einops import rearrange
from torch import Tensor

from ..helpers import to_tuple
from .pos_enc import RelativeFactorizedPosition


class PatchEmbed2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int],
        normalization: str = "LayerNorm",
    ):
        super().__init__()
        self._patch_size = to_tuple(patch_size, 2)
        self.patch = nn.Conv2d(in_channels, embed_dim, self.patch_size, stride=self.patch_size)
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim)
        self.norm = te.LayerNorm(embed_dim) if normalization == "LayerNorm" else te.RMSNorm(embed_dim)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def original_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return ht, wt

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            mm_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")
            y = self.patch(x)
            y = rearrange(y, "b c h w -> b (h w) c")
            torch.set_float32_matmul_precision(mm_precision)

        H, W = x.shape[2:]
        dims = self.tokenized_size((H, W))
        pos = self.pos_enc(dims)
        return self.norm(y + pos)
