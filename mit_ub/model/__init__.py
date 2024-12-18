from typing import Final

import torch
import torch.nn.functional as F
from registry import Registry

from .backbone import AdaptiveViT, ConvViT, ViT
from .convnext import ConvNext
from .helpers import compile_is_disabled
from .mlp import identity
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


torch._dynamo.config.cache_size_limit = 128  # type: ignore

BACKBONES = Registry("backbones")

CIFAR10_DIM: Final = 384
CIFAR10_HEAD_DIM: Final = 32
CIFAR10_ADAPTIVE_DIM: Final = 384

BACKBONES(
    ViT,
    name="vit-cifar10",
    in_channels=3,
    dim=CIFAR10_DIM,
    patch_size=4,
    depth=12,
    nhead=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    num_kv_heads=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    dropout=0.1,
    stochastic_depth=0.1,
    bias=False,
    qk_norm=True,
    activation=identity,
    gate_activation=F.silu,
)
BACKBONES(
    ViT,
    name="vit-cifar10-moe",
    in_channels=3,
    dim=CIFAR10_DIM,
    patch_size=4,
    depth=12,
    nhead=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    num_kv_heads=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    dropout=0.1,
    stochastic_depth=0.1,
    bias=False,
    qk_norm=True,
    activation=identity,
    gate_activation=F.silu,
    num_experts=8,
    num_slots=32,
    moe_layers=[11],
)
BACKBONES(
    AdaptiveViT,
    name="vit-cifar10-adaptive",
    in_channels=3,
    dim=CIFAR10_DIM,
    patch_size=4,
    target_shape=(16, 16),
    depth=12,
    nhead=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    num_kv_heads=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    dropout=0.1,
    stochastic_depth=0.1,
    bias=False,
    qk_norm=True,
    activation=identity,
    gate_activation=F.silu,
    # layer_scale=0.01,
    layer_scale_adaptive=0.001,
    share_weights=True,
)
BACKBONES(
    ConvViT,
    name="convvit-cifar10",
    in_channels=3,
    dim=CIFAR10_DIM,
    patch_size=4,
    target_shape=(16, 16),
    depth=12,
    nhead=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    num_kv_heads=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    dropout=0.1,
    stochastic_depth=0.1,
    bias=False,
    qk_norm=True,
    activation=identity,
    gate_activation=F.silu,
    layer_scale_adaptive=1.0,
)


BACKBONES(
    ConvNext,
    name="convnext-cifar10",
    in_channels=3,
    depths=(3, 5, 7),
    dims=(128, 256, 512),
    patch_size=2,
    dropout=0.1,
    kernel_size=3,
)


__all__ = [
    "BACKBONES",
    "ViT",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "AdaptiveViT",
    "ConvNext",
    "compile_is_disabled",
    "identity",
]
