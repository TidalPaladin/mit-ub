import math
from functools import partial
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def resize_patch_weights(self, target: Tuple[int, int]) -> Tensor:
        # Based on https://github.com/bwconrad/flexivit/blob/main/flexivit_pytorch/patch_embed.py
        current_shape = self.patch.weight.shape[2:]
        if len(target) != len(current_shape):
            raise ValueError(f"Target ndim {target} must match input ndim {current_shape}")
        if current_shape == target:
            return self.patch.weight

        current_numel = math.prod(current_shape)
        resize = partial(F.interpolate, mode="bicubic", antialias=True)

        unravel_idx = torch.arange(current_numel, device=self.patch.weight.device)
        basis_vecs = self.patch.weight.new_zeros(current_numel, *current_shape)
        basis_vecs[unravel_idx, *torch.unravel_index(unravel_idx, current_shape)] = 1.0
        basis_vecs = resize(basis_vecs.unsqueeze(1), target).squeeze(1)
        resize_matrix_pinv = torch.linalg.pinv(basis_vecs.view(current_numel, -1))

        def resample_patch_embed(patch_embed: Tensor):
            h, w = target
            resampled_kernel = resize_matrix_pinv @ patch_embed.reshape(-1)
            return rearrange(resampled_kernel, "(h w) -> h w", h=h, w=w)

        v_resample_patch_embed = torch.vmap(torch.vmap(resample_patch_embed, 0, 0), 1, 1)

        return v_resample_patch_embed(self.patch.weight)

    def tokenized_size(self, size: Tuple[int, int], patch_size: Tuple[int, int] | None = None) -> Tuple[int, int]:
        patch_size = patch_size if patch_size is not None else self.patch_size
        ht, wt = tuple(s // p for s, p in zip(size, patch_size))
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

    def pack(self, x: List[Tensor], patch_size: List[Tuple[int, int]] | None = None) -> Tuple[Tensor, Tensor]:
        r"""Runs patch embedding and packs the sequences into a single tensor with minimal padding.
        This follows NaViT's method.
        """
        if not len(x):
            raise ValueError("x must not be empty")
        if patch_size is None:
            patch_size = [self.patch_size] * len(x)
        if len(patch_size) != len(x):
            raise ValueError(f"patch_size must be the same length as x, got {len(patch_size)} and {len(x)}")
        if not all(xi.ndim == 3 for xi in x):
            raise ValueError(f"x must be a list of 3D tensors, got {[xi.shape for xi in x]}")

        # Run patch embedding, extract sequence lengths, pack into single sequence
        packed = [self.forward(xi[None], pi) for xi, pi in zip(x, patch_size)]
        seq_lens = (
            torch.tensor([0] + [t.shape[1] for t in packed], device=packed[0].device).cumsum(dim=0).to(torch.int32)
        )
        packed = torch.cat(packed, dim=1).squeeze(0)

        return packed, seq_lens

    def unpack(self, packed: Tensor, cu_seq_lens: Tensor) -> List[Tensor]:
        if packed.ndim != 2:
            raise ValueError(f"packed must be 2D, got shape {packed.shape}")
        if cu_seq_lens.ndim != 1:
            raise ValueError(f"cu_seq_lens must be 1D, got shape {cu_seq_lens.shape}")

        # Split packed sequence into individual sequences using cumulative sequence lengths
        seqs = []
        for i in range(len(cu_seq_lens) - 1):
            start = cu_seq_lens[i]
            end = cu_seq_lens[i + 1]
            seq = packed[start:end].unsqueeze(0)
            seqs.append(seq)
        return seqs
