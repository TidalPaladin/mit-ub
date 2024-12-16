import math
from abc import ABC, abstractmethod
from typing import Callable, Generic, Sequence, Tuple, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_helpers.helpers import to_tuple
from einops import rearrange
from torch import Tensor

from ..helpers import Dims2D, Dims3D, compile_backend, compile_is_disabled
from ..pos_enc import DEFAULT_POS_ENC_ACTIVATION, RelativeFactorizedPosition, relative_factorized_position_forward


T = TypeVar("T", bound=Tuple[int, ...])


class PatchEmbed(ABC, Generic[T]):

    @property
    @abstractmethod
    def patch_size(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def tokenized_size(self, size: T) -> T:
        r"""Computes the tokenized size of an input image.
        This is the size of the of the visual token grid accounting for the patch size.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def in_channels(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def original_size(self, size: T) -> T:
        r"""Computes the original size of a tokenized image.
        This is the size of the of the visual token grid upscaling by the patch size.
        """
        raise NotImplementedError


@torch.compile(
    fullgraph=True,
    backend=compile_backend(),
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
    activation: Callable[[Tensor], Tensor] = DEFAULT_POS_ENC_ACTIVATION,
    eps: float = 1e-5,
    training: bool = False,
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
    x = F.layer_norm(x, x.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)
    pos = relative_factorized_position_forward(
        dims, w1_pos, b1_pos, w2_pos, b2_pos, w_pos_norm, b_pos_norm, activation, dropout=dropout, training=training
    )
    x += pos
    return x


def _init_patch_embed(layer: nn.Module) -> None:
    layer.pos_enc.reset_parameters()
    nn.init.ones_(layer.w_norm)
    nn.init.zeros_(layer.b_norm)
    nn.init.xavier_uniform_(layer.w_in)
    nn.init.zeros_(layer.b_in)


class PatchEmbed2d(nn.Module, PatchEmbed[Dims2D]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Dims2D,
        dropout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = DEFAULT_POS_ENC_ACTIVATION,
    ):
        super().__init__()
        self._patch_size = to_tuple(patch_size, 2)
        d_in = math.prod(self.patch_size) * in_channels
        self.w_in = nn.Parameter(torch.empty(embed_dim, d_in))
        self.b_in = nn.Parameter(torch.empty(embed_dim))
        self.w_norm = nn.Parameter(torch.empty(embed_dim))
        self.b_norm = nn.Parameter(torch.empty(embed_dim))
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim, dropout=dropout, activation=activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_patch_embed(self)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

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
        return f"in={self.in_channels}, " f"embed={self.embed_dim}, " f"patch_size={self.patch_size}"

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed_forward(
            x,
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
            activation=self.pos_enc.activation,
            dropout=self.pos_enc.dropout,
            training=self.training,
        )


class PatchEmbed3d(nn.Module, PatchEmbed[Dims3D]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Dims3D,
        dropout: float = 0.0,
        activation: Callable[[Tensor], Tensor] = DEFAULT_POS_ENC_ACTIVATION,
    ):
        super().__init__()
        self._patch_size = to_tuple(patch_size, 3)
        d_in = math.prod(self.patch_size) * in_channels
        self.w_in = nn.Parameter(torch.empty(embed_dim, d_in))
        self.b_in = nn.Parameter(torch.empty(embed_dim))
        self.w_norm = nn.Parameter(torch.empty(embed_dim))
        self.b_norm = nn.Parameter(torch.empty(embed_dim))
        self.pos_enc = RelativeFactorizedPosition(3, embed_dim, dropout=dropout, activation=activation)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        _init_patch_embed(self)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    @property
    def in_channels(self) -> int:
        return self.w_in.shape[1] // math.prod(self.patch_size)

    @property
    def embed_dim(self) -> int:
        return self.w_in.shape[0]

    def tokenized_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r"""Computes the tokenized size of an input image.
        This is the size of the of the visual token grid accounting for the patch size.
        """
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def original_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        r"""Computes the original size of a tokenized image.
        This is the size of the of the visual token grid upscaling by the patch size.
        """
        dt, ht, wt = tuple(s * p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def extra_repr(self) -> str:
        return f"in={self.in_channels}, " f"embed={self.embed_dim}, " f"patch_size={self.patch_size}"

    def forward(self, x: Tensor) -> Tensor:
        return patch_embed_forward(
            x,
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
