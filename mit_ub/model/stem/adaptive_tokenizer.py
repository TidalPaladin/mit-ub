import math
from enum import StrEnum
from typing import Any, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..activations import Activation
from ..helpers import compile_backend, compile_is_disabled, max_autotune, to_tuple
from ..layers.pos_enc import (
    DEFAULT_POS_ENC_ACTIVATION,
    RelativeFactorizedPosition,
    relative_factorized_position_forward,
)
from .patch_embed import PatchEmbed, _init_patch_embed


class PoolType(StrEnum):
    MAX = "max"
    AVG = "avg"


@torch.compile(
    fullgraph=True,
    backend=compile_backend(),
    disable=compile_is_disabled(),
    options={
        "max_autotune": max_autotune(),
        "triton.cudagraph_trees": max_autotune(),
        "shape_padding": True,
    },
)
def adaptive_patch_embed_forward(
    x: Tensor,
    target_size: Sequence[int],
    pool_type: PoolType,
    w_patch: Tensor,
    b_patch: Tensor | None,
    stride: Sequence[int],
    w_norm: Tensor | None,
    b_norm: Tensor | None,
    w1_pos: Tensor,
    b1_pos: Tensor | None,
    w2_pos: Tensor,
    b2_pos: Tensor | None,
    w_pos_norm: Tensor | None,
    b_pos_norm: Tensor | None,
    dropout: float = 0.0,
    activation: Activation = DEFAULT_POS_ENC_ACTIVATION,
    eps: float = 1e-5,
    training: bool = False,
    high_precision: bool = True,
) -> Tuple[Tensor, Tensor]:
    # Rearrange into patches
    dims = tuple(dim_size // dim_stride for dim_size, dim_stride in zip(x.shape[2:], stride))
    if x.ndim == 4:
        x = rearrange(x, "b c (ht hp) (wt wp) -> b ht wt (hp wp c)", hp=stride[0], wp=stride[1])
    elif x.ndim == 5:
        x = rearrange(
            x, "b c (dt dp) (ht hp) (wt wp) -> b dt ht wt (dp hp wp c)", dp=stride[0], hp=stride[1], wp=stride[2]
        )
    else:
        raise ValueError(f"Invalid input dimension: {x.ndim}")

    # Project and patch
    with torch.autocast(device_type=x.device.type, dtype=torch.float32, enabled=high_precision):
        x = F.linear(x, w_patch, b_patch)

    # Add position encoding
    pos = relative_factorized_position_forward(
        dims, w1_pos, b1_pos, w2_pos, b2_pos, w_pos_norm, b_pos_norm, activation, dropout=dropout, training=training
    )
    x = x + pos.expand(x.shape[0], -1, -1).view_as(x)

    # Pool to fixed size
    pooled = rearrange(x, "b ... d -> b d ...")
    match (x.ndim, pool_type):
        case (4, PoolType.MAX):
            pooled = F.adaptive_max_pool2d(pooled, cast(Tuple[int, int], target_size))
        case (4, PoolType.AVG):
            pooled = F.adaptive_avg_pool2d(pooled, cast(Tuple[int, int], target_size))
        case (5, PoolType.MAX):
            pooled = F.adaptive_max_pool3d(pooled, cast(Tuple[int, int, int], target_size))
        case (5, PoolType.AVG):
            pooled = F.adaptive_avg_pool3d(pooled, cast(Tuple[int, int, int], target_size))
        case _:
            raise ValueError(f"Invalid input dimension / pool type: {x.ndim} / {pool_type}")

    x = rearrange(x, "b ... d -> b (...) d")
    pooled = rearrange(pooled, "b d ... -> b (...) d")

    x = F.layer_norm(x, x.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)
    pooled = F.layer_norm(pooled, pooled.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)
    return pooled, x


class AdaptiveTokenizer2d(nn.Module, PatchEmbed[Tuple[int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int],
        target_shape: Tuple[int, int],
        dropout: float = 0.0,
        activation: Activation = DEFAULT_POS_ENC_ACTIVATION,
        pool_type: PoolType = PoolType.MAX,
    ):
        super().__init__()
        self._target_shape = to_tuple(target_shape, 2)
        self._patch_size = to_tuple(patch_size, 2)
        self._pool_type = pool_type

        d_in = math.prod(self.patch_size) * in_channels
        self.w_in = nn.Parameter(torch.empty(embed_dim, d_in))
        self.b_in = nn.Parameter(torch.empty(embed_dim))
        self.w_norm = nn.Parameter(torch.empty(embed_dim))
        self.b_norm = nn.Parameter(torch.empty(embed_dim))
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim, dropout=dropout, activation=activation, norm=False)
        self.reset_parameters()

    def reset_parameters(self):
        _init_patch_embed(cast(Any, self))

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    @property
    def target_shape(self) -> Tuple[int, int]:
        return self._target_shape

    @property
    def target_tokenized_shape(self) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(self.target_shape, self.patch_size))
        return ht, wt

    @property
    def in_channels(self) -> int:
        return self.w_in.shape[1] // math.prod(self.patch_size)

    @property
    def embed_dim(self) -> int:
        return self.w_in.shape[0]

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def original_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return ht, wt

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, "
            f"embed={self.embed_dim}, "
            f"patch_size={self.patch_size}, "
            f"target_shape={self._target_shape}"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mm_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        result = adaptive_patch_embed_forward(
            x,
            self.target_tokenized_shape,
            self._pool_type,
            self.w_in,
            self.b_in,
            self.patch_size,
            self.w_norm,
            self.b_norm,
            self.pos_enc.w_in,
            self.pos_enc.b_in,
            self.pos_enc.w_out,
            self.pos_enc.b_out,
            self.pos_enc.w_norm,
            self.pos_enc.b_norm,
            dropout=self.pos_enc.dropout,
            activation=self.pos_enc.activation,
            training=self.training,
        )
        torch.set_float32_matmul_precision(mm_precision)
        return result


class AdaptiveTokenizer3d(nn.Module, PatchEmbed[Tuple[int, int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
        dropout: float = 0.0,
        activation: Activation = DEFAULT_POS_ENC_ACTIVATION,
        pool_type: PoolType = PoolType.MAX,
    ):
        super().__init__()
        self._target_shape = to_tuple(target_shape, 3)
        self._patch_size = to_tuple(patch_size, 3)
        self._pool_type = pool_type
        d_in = math.prod(self.patch_size) * in_channels
        self.w_in = nn.Parameter(torch.empty(embed_dim, d_in))
        self.b_in = nn.Parameter(torch.empty(embed_dim))
        self.w_norm = nn.Parameter(torch.empty(embed_dim))
        self.b_norm = nn.Parameter(torch.empty(embed_dim))
        self.pos_enc = RelativeFactorizedPosition(3, embed_dim, dropout=dropout, activation=activation, norm=False)
        self.reset_parameters()

    def reset_parameters(self):
        _init_patch_embed(cast(Any, self))

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    @property
    def target_shape(self) -> Tuple[int, int, int]:
        return self._target_shape

    @property
    def target_tokenized_shape(self) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(self.target_shape, self.patch_size))
        return dt, ht, wt

    @property
    def in_channels(self) -> int:
        return self.w_in.shape[1] // math.prod(self.patch_size)

    @property
    def embed_dim(self) -> int:
        return self.w_in.shape[0]

    def tokenized_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def original_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def extra_repr(self) -> str:
        return (
            f"in={self.in_channels}, "
            f"embed={self.embed_dim}, "
            f"patch_size={self.patch_size}, "
            f"target_shape={self._target_shape}"
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        mm_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("high")
        result = adaptive_patch_embed_forward(
            x,
            self.target_tokenized_shape,
            self._pool_type,
            self.w_in,
            self.b_in,
            self.patch_size,
            self.w_norm,
            self.b_norm,
            self.pos_enc.w_in,
            self.pos_enc.b_in,
            self.pos_enc.w_out,
            self.pos_enc.b_out,
            self.pos_enc.w_norm,
            self.pos_enc.b_norm,
            dropout=self.pos_enc.dropout,
            activation=self.pos_enc.activation,
            training=self.training,
        )
        torch.set_float32_matmul_precision(mm_precision)
        return result
