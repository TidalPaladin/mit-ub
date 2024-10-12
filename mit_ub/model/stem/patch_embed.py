import math
from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, Tuple, Type, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_helpers.helpers import to_tuple
from einops import rearrange
from torch import Tensor

from ..compile import compile_is_disabled
from ..mlp import relu2
from ..pos_enc import RelativeFactorizedPosition, relative_factorized_position_forward


T = TypeVar("T", bound=Tuple[int, ...])


class PatchEmbed(ABC, Generic[T]):

    @property
    @abstractmethod
    def patch_size(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def tokenized_size(self, size: T) -> T:
        raise NotImplementedError


@torch.compile(
    fullgraph=True,
    disable=compile_is_disabled(),
    options={
        "max_autotune": True,
        "shape_padding": True,
    },
)
def patch_embed_forward(
    x: Tensor,
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
    activation: Callable[[Tensor], Tensor] = relu2,
    training: bool = True,
) -> Tensor:
    dims = tuple(dim_size // dim_stride for dim_size, dim_stride in zip(x.shape[2:], stride))
    if x.ndim == 4:
        x = rearrange(x, "b c (ht hp) (wt wp) -> b (ht wt) (hp wp c)", hp=stride[0], wp=stride[1])
    elif x.ndim == 5:
        x = rearrange(
            x, "b c (dt dp) (ht hp) (wt wp) -> b (dt ht wt) (dp hp wp c)", dp=stride[0], hp=stride[1], wp=stride[2]
        )
    else:
        raise ValueError(f"Invalid input dimension: {x.ndim}")

    x = F.linear(x, w_patch, b_patch)
    x = F.layer_norm(x, x.shape[-1:], weight=w_norm, bias=b_norm)
    pos = relative_factorized_position_forward(
        dims, w1_pos, b1_pos, w2_pos, b2_pos, w_pos_norm, b_pos_norm, activation, dropout=dropout, training=training
    )
    x += pos
    return x


class PatchEmbed2d(nn.Module, PatchEmbed[Tuple[int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int],
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        dropout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = relu2,
    ):
        super().__init__()
        self._patch_size = to_tuple(patch_size, 2)
        d_in = math.prod(self.patch_size) * in_channels
        self.patch = nn.Linear(d_in, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim, dropout=dropout, activation=activation)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed_forward(
            x,
            self.patch.weight,
            self.patch.bias,
            self.patch_size,
            self.norm.weight,
            self.norm.bias,
            self.pos_enc.fc1.weight,
            self.pos_enc.fc1.bias,
            self.pos_enc.fc2.weight,
            self.pos_enc.fc2.bias,
            self.pos_enc.norm.weight,
            self.pos_enc.norm.bias,
            dropout=self.pos_enc.dropout.p,
            training=self.training,
        )


class PatchEmbed3d(nn.Module, PatchEmbed[Tuple[int, int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int, int],
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        dropout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = relu2,
    ):
        super().__init__()
        self._patch_size = to_tuple(patch_size, 3)
        d_in = math.prod(self.patch_size) * in_channels
        self.patch = nn.Linear(d_in, embed_dim)
        self.norm = norm_layer(embed_dim)
        self.pos_enc = RelativeFactorizedPosition(3, embed_dim, dropout=dropout, activation=activation)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def tokenized_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed_forward(
            x,
            self.patch.weight,
            self.patch.bias,
            self.patch_size,
            self.norm.weight,
            self.norm.bias,
            self.pos_enc.fc1.weight,
            self.pos_enc.fc1.bias,
            self.pos_enc.fc2.weight,
            self.pos_enc.fc2.bias,
            self.pos_enc.norm.weight,
            self.pos_enc.norm.bias,
            dropout=self.pos_enc.dropout.p,
            training=self.training,
        )
