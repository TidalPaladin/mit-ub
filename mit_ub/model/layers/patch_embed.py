from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from einops import rearrange
from torch import Tensor
from torch.utils.cpp_extension import load

from ..helpers import to_tuple
from .pos_enc import RelativeFactorizedPosition


try:
    import navit

    _navit = navit
except ImportError:
    if torch.cuda.is_available():
        _navit = load(
            name="navit",
            sources=[str(Path(__file__).parents[3] / "csrc" / "patch.cpp")],
            extra_cuda_cflags=["-O3"],
        )
    else:
        _navit = None


def tokenized_size(input_size: Tuple[int, ...], patch_size: Tuple[int, ...]) -> Tuple[int, ...]:
    return _navit.tokenized_size(input_size, patch_size)


def token_count(input_size: Tuple[int, ...], patch_size: Tuple[int, ...]) -> int:
    return _navit.token_count(input_size, patch_size)


def tokenized_size_foreach(
    input_sizes: List[Tuple[int, ...]], patch_sizes: List[Tuple[int, ...]]
) -> List[Tuple[int, ...]]:
    return _navit.tokenized_size_foreach(input_sizes, patch_sizes)


def token_count_foreach(input_sizes: List[Tuple[int, ...]], patch_sizes: List[Tuple[int, ...]]) -> List[int]:
    return _navit.token_count_foreach(input_sizes, patch_sizes)


def size_at_scale(input_size: Tuple[int, ...], scale: float) -> Tuple[int, ...]:
    return _navit.size_at_scale(input_size, scale)


def tokenized_size_at_scale(input_size: Tuple[int, ...], patch_size: Tuple[int, ...], scale: float) -> Tuple[int, ...]:
    return _navit.tokenized_size_at_scale(input_size, patch_size, scale)


def token_count_at_scale(input_size: Tuple[int, ...], patch_size: Tuple[int, ...], scale: float) -> int:
    return _navit.token_count_at_scale(input_size, patch_size, scale)


def pack(tensors: List[Tensor]) -> Tuple[Tensor, Tensor, int]:
    return _navit.pack(tensors)


def unpack(packed: Tensor, cu_seq_lens: Tensor) -> List[Tensor]:
    return _navit.unpack(packed, cu_seq_lens)


def binary_search_for_scale(
    current_size: Tuple[int, ...], patch_size: Tuple[int, ...], drop_rate: float, budget: int
) -> Tuple[int, ...]:
    return _navit.binary_search_for_scale(current_size, patch_size, drop_rate, budget)


def calculate_sizes_for_budget(
    input_sizes: List[Tuple[int, ...]], patch_sizes: List[Tuple[int, ...]], drop_rates: List[float], budget: int
) -> List[Tuple[int, ...]]:
    return _navit.calculate_sizes_for_budget(input_sizes, patch_sizes, drop_rates, budget)


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
        ht, wt = tokenized_size(size, self.patch_size)
        return ht, wt

    def original_size(self, size: Tuple[int, int], patch_size: Tuple[int, int] | None = None) -> Tuple[int, int]:
        patch_size = patch_size if patch_size is not None else self.patch_size
        ht, wt = tuple(s * p for s, p in zip(size, patch_size))
        return ht, wt

    def forward(self, x: Tensor, patch_size: Tuple[int, int] | None = None) -> Tensor:
        weight = self.resize_patch_weights(patch_size) if patch_size is not None else self.patch.weight
        bias = self.patch.bias
        stride = weight.shape[2:]
        y = F.conv2d(x, weight, bias, stride=stride)
        pos = self.pos_enc(y.shape[2:])
        y = rearrange(y, "b c h w -> b (h w) c")
        return self.norm(y + pos)
