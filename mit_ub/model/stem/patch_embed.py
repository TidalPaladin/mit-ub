from abc import ABC, abstractmethod
from typing import Generic, Tuple, Type, TypeVar, cast

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor

from ..pos_enc import RelativeFactorizedPosition


T = TypeVar("T", bound=Tuple[int, ...])


class PatchEmbed(ABC, Generic[T]):

    @property
    @abstractmethod
    def patch_size(self) -> T:
        raise NotImplementedError

    @abstractmethod
    def tokenized_size(self, size: T) -> T:
        raise NotImplementedError


class PatchEmbed2d(nn.Module, PatchEmbed[Tuple[int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int | Tuple[int, int],
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        position_noise: bool = False,
        autocast: bool = False,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
            Rearrange("b c h w -> b (h w) c", c=embed_dim),
        )
        self.norm = norm_layer(embed_dim)
        self.pos_enc = RelativeFactorizedPosition(2, embed_dim)
        self.position_noise = position_noise
        self.autocast = autocast

    @property
    def patch_size(self) -> Tuple[int, int]:
        conv = cast(nn.Conv2d, self.patch[0])
        hp, wp = conv.kernel_size
        return hp, wp

    def tokenized_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return ht, wt

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=self.autocast):
            B, C, H, W = x.shape
            x = self.patch(x)
            x = self.norm(x)
            x += self.pos_enc.from_grid(
                self.tokenized_size((H, W)),
                B,
                proto=x,
                normalize=True,
                add_noise=self.training and self.position_noise,
            )
            return x


class PatchEmbed3d(nn.Module, PatchEmbed[Tuple[int, int, int]]):

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: Tuple[int, int, int],
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        position_noise: bool = False,
        autocast: bool = False,
    ):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv3d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange("b c d h w -> b (h w d) c", c=embed_dim),
        )
        self.norm = norm_layer(embed_dim)
        self.pos_enc = RelativeFactorizedPosition(3, embed_dim)
        self.position_noise = position_noise
        self.autocast = autocast

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        conv = cast(nn.Conv3d, self.patch[0])
        dp, hp, wp = conv.kernel_size
        return dp, hp, wp

    def tokenized_size(self, size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        dt, ht, wt = tuple(s // p for s, p in zip(size, self.patch_size))
        return dt, ht, wt

    def forward(self, x: Tensor) -> Tensor:
        with torch.autocast(device_type=x.device.type, enabled=self.autocast):
            B, C, D, H, W = x.shape
            x = self.patch(x)
            x = self.norm(x)
            x += self.pos_enc.from_grid(
                self.tokenized_size((D, H, W)),
                B,
                proto=x,
                normalize=True,
                add_noise=self.training and self.position_noise,
            )
            return x
