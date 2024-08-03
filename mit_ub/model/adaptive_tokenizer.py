from typing import Tuple

import torch.nn as nn
from einops.layers.torch import Rearrange
from ssl_tasks.helpers import divide_tuple
from torch import Tensor

from .pos_enc import RelativeFactorizedPosition


def _adaptive_tokenizer_forward(
    query: nn.Module,
    key_value: nn.Module,
    pos_enc_q: RelativeFactorizedPosition,
    pos_enc_kv: RelativeFactorizedPosition,
    to_seq: Rearrange,
    x: Tensor,
    position_noise: bool = False,
) -> Tuple[Tensor, Tensor]:
    # Tokenize q and kv
    q = query(x)
    kv = key_value(x)

    # Add position encoding to q
    B = x.shape[0]
    q_tokenized_size = q.shape[2:]
    q = to_seq(q)
    q += pos_enc_q.from_grid(
        q_tokenized_size, B, proto=q, normalize=True, add_noise=pos_enc_q.training and position_noise
    )

    # Add position encoding to kv
    kv_tokenized_size = kv.shape[2:]
    kv = to_seq(kv)
    kv += pos_enc_kv.from_grid(
        kv_tokenized_size, B, proto=kv, normalize=True, add_noise=pos_enc_kv.training and position_noise
    )

    return q, kv


class AdaptiveTokenizer2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kv_dim: int,
        patch_size: Tuple[int, int],
        target_shape: Tuple[int, int],
        position_noise: bool = False,
    ):
        super().__init__()
        self.target_shape = tuple(target_shape)
        self.patch_size = tuple(patch_size)
        self.d_model = d_model
        self.kv_dim = kv_dim
        self.position_noise = position_noise

        self.query = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
            nn.AdaptiveAvgPool2d(target_shape),
        )
        self.kv = nn.Conv2d(in_channels, kv_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc_q = RelativeFactorizedPosition(2, d_model)
        self.pos_enc_kv = RelativeFactorizedPosition(2, kv_dim)

        self.to_seq = Rearrange("b c h w -> b (h w) c")

    def kv_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        return divide_tuple(input_size, self.patch_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return _adaptive_tokenizer_forward(
            self.query,
            self.kv,
            self.pos_enc_q,
            self.pos_enc_kv,
            self.to_seq,
            x,
            self.position_noise,
        )


class AdaptiveTokenizer3d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kv_dim: int,
        patch_size: Tuple[int, int, int],
        target_shape: Tuple[int, int, int],
        position_noise: bool = False,
    ):
        super().__init__()
        self.target_shape = tuple(target_shape)
        self.patch_size = tuple(patch_size)
        self.d_model = d_model
        self.kv_dim = kv_dim
        self.position_noise = position_noise

        self.query = nn.Sequential(
            nn.Conv3d(in_channels, d_model, kernel_size=patch_size, stride=patch_size),
            nn.AdaptiveAvgPool3d(target_shape),
        )
        self.kv = nn.Conv3d(in_channels, kv_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_enc_q = RelativeFactorizedPosition(3, d_model)
        self.pos_enc_kv = RelativeFactorizedPosition(3, kv_dim)

        self.to_seq = Rearrange("b c d h w -> b (d h w) c")

    def kv_size(self, input_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return divide_tuple(input_size, self.patch_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return _adaptive_tokenizer_forward(
            self.query,
            self.kv,
            self.pos_enc_q,
            self.pos_enc_kv,
            self.to_seq,
            x,
            self.position_noise,
        )
