from enum import StrEnum
from typing import Tuple, Type, cast

import torch.nn as nn
from deep_helpers.helpers import to_tuple
from einops.layers.torch import Rearrange
from ssl_tasks.helpers import divide_tuple
from torch import Tensor

from ..pos_enc import RelativeFactorizedPosition
from .patch_embed import PatchEmbed


class PoolType(StrEnum):
    MAX = "max"
    AVG = "avg"


class AdaptiveTokenizer2d(nn.Module, PatchEmbed[Tuple[int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kv_dim: int,
        patch_size: int | Tuple[int, int],
        target_shape: Tuple[int, int],
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        position_noise: bool = False,
        autocast: bool = False,
        pool_type: PoolType = PoolType.MAX,
    ):
        super().__init__()
        self._target_shape = to_tuple(target_shape, 2)
        self._patch_size = to_tuple(patch_size, 2)
        self._kv_dim = kv_dim
        self.position_noise = position_noise
        self.autocast = autocast

        pool_cls = nn.AdaptiveMaxPool2d if pool_type == PoolType.MAX else nn.AdaptiveAvgPool2d
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            pool_cls(target_shape),
            Rearrange("b c h w -> b (h w) c"),
            norm_layer(embed_dim),
        )
        self.kv = nn.Sequential(
            nn.Conv2d(in_channels, kv_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Rearrange("b c h w -> b (h w) c"),
            norm_layer(kv_dim),
        )
        self.pos_enc_q = RelativeFactorizedPosition(2, embed_dim)
        self.pos_enc_kv = RelativeFactorizedPosition(2, kv_dim)

    @property
    def patch_size(self) -> Tuple[int, int]:
        return self._patch_size

    def tokenized_size(self, _: Tuple[int, int]) -> Tuple[int, int]:
        return self._target_shape

    def kv_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        return cast(Tuple[int, int], divide_tuple(input_size, self.patch_size))

    def equivalent_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = tuple(s * p for s, p in zip(self.tokenized_size(input_size), self.patch_size))
        return h, w

    def equivalent_size_kv(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        h, w = tuple(s * p for s, p in zip(self.kv_size(input_size), self.patch_size))
        return h, w

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, _, H, W = x.shape

        q = self.query(x)
        q = q + self.pos_enc_q.from_grid(
            self.tokenized_size((H, W)),
            B,
            proto=q,
            normalize=True,
            add_noise=self.training and self.position_noise,
        )

        kv = self.kv(x)
        kv = kv + self.pos_enc_kv.from_grid(
            self.kv_size((H, W)), B, proto=kv, normalize=True, add_noise=self.training and self.position_noise
        )

        return q, kv


class AdaptiveTokenizer3d(nn.Module, PatchEmbed[Tuple[int, int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        kv_dim: int,
        patch_size: int | Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        position_noise: bool = False,
        autocast: bool = False,
        pool_type: PoolType = PoolType.MAX,
    ):
        super().__init__()
        self._target_shape = to_tuple(target_shape, 3)
        self._patch_size = to_tuple(patch_size, 3)
        self._kv_dim = kv_dim
        self.position_noise = position_noise
        self.autocast = autocast

        pool_cls = nn.AdaptiveMaxPool3d if pool_type == PoolType.MAX else nn.AdaptiveAvgPool3d
        self.query = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            pool_cls(target_shape),
            Rearrange("b c d h w -> b (h w d) c"),
            norm_layer(embed_dim),
        )
        self.kv = nn.Sequential(
            nn.Conv3d(in_channels, kv_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange("b c d h w -> b (h w d) c"),
            norm_layer(kv_dim),
        )
        self.pos_enc_q = RelativeFactorizedPosition(3, embed_dim)
        self.pos_enc_kv = RelativeFactorizedPosition(3, kv_dim)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size

    def tokenized_size(self, _: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return self._target_shape

    def kv_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return cast(Tuple[int, int, int], divide_tuple(input_size, self.patch_size))

    def equivalent_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        d, h, w = tuple(s * p for s, p in zip(self.tokenized_size(input_size), self.patch_size))
        return d, h, w

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        B, _, D, H, W = x.shape

        q = self.query(x)
        q = q + self.pos_enc_q.from_grid(
            self.tokenized_size((D, H, W)),
            B,
            proto=q,
            normalize=True,
            add_noise=self.training and self.position_noise,
        )

        kv = self.kv(x)
        kv = kv + self.pos_enc_kv.from_grid(
            self.kv_size((D, H, W)), B, proto=kv, normalize=True, add_noise=self.training and self.position_noise
        )

        return q, kv
