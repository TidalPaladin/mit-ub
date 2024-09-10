from typing import Final

import torch.nn as nn
from registry import Registry

from .backbone import AdaptiveViT, ViT
from .convnext import ConvNext
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


BACKBONES = Registry("backbones")
QUERY_GROUPS: Final = 2
HEAD_DIM: Final = 64

# Small
dim, kv_dim = 512, 256
BACKBONES(
    ViT,
    name="vit-i1-p16-d512",
    in_channels=1,
    dim=dim,
    patch_size=16,
    depth=12,
    nhead=dim // HEAD_DIM,
    num_kv_heads=dim // HEAD_DIM // QUERY_GROUPS,
    dropout=0.1,
)
BACKBONES(
    AdaptiveViT,
    name="vit-i1-p16-d512-a16_12",
    in_channels=1,
    dim=dim,
    kv_dim=kv_dim,
    patch_size=16,
    target_shape=(16, 12),
    depth=12,
    high_res_depth=4,
    nhead=dim // HEAD_DIM,
    num_kv_heads=dim // HEAD_DIM // QUERY_GROUPS,
    dropout=0.1,
)
BACKBONES(
    AdaptiveViT,
    name="vit-i1-p16-d512-a32_24",
    in_channels=1,
    dim=dim,
    kv_dim=kv_dim,
    patch_size=16,
    target_shape=(32, 24),
    depth=12,
    high_res_depth=4,
    nhead=dim // HEAD_DIM,
    num_kv_heads=dim // HEAD_DIM // QUERY_GROUPS,
    dropout=0.1,
)


# Large
BACKBONES(
    ViT,
    name="vit-i1-p16-d768",
    in_channels=1,
    dim=768,
    patch_size=16,
    depth=24,
    nhead=768 // HEAD_DIM,
    num_kv_heads=768 // HEAD_DIM // QUERY_GROUPS,
    dropout=0.1,
)
BACKBONES(
    AdaptiveViT,
    name="vit-i1-p16-d768-a16_12",
    in_channels=1,
    dim=dim,
    kv_dim=kv_dim,
    patch_size=16,
    target_shape=(16, 12),
    depth=24,
    high_res_depth=6,
    nhead=dim // HEAD_DIM,
    num_kv_heads=dim // HEAD_DIM // QUERY_GROUPS,
    dropout=0.1,
)
BACKBONES(
    AdaptiveViT,
    name="vit-i1-p16-d768-a32_24",
    in_channels=1,
    dim=dim,
    kv_dim=kv_dim,
    patch_size=16,
    target_shape=(32, 24),
    depth=24,
    high_res_depth=6,
    nhead=dim // HEAD_DIM,
    num_kv_heads=dim // HEAD_DIM // QUERY_GROUPS,
    dropout=0.1,
)


CIFAR10_DIM: Final = 384
CIFAR10_HEAD_DIM: Final = 32
CIFAR10_ADAPTIVE_DIM: Final = 64

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
)

BACKBONES(
    AdaptiveViT,
    name="vit-cifar10-adaptive",
    in_channels=3,
    dim=CIFAR10_DIM,
    kv_dim=CIFAR10_ADAPTIVE_DIM,
    patch_size=4,
    target_shape=(4, 4),
    depth=12,
    high_res_depth=4,
    nhead=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    num_kv_heads=CIFAR10_DIM // CIFAR10_HEAD_DIM,
    dropout=0.1,
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

BACKBONES(
    ConvNext,
    name="convnext-mammo",
    in_channels=1,
    depths=(3, 5, 7, 15, 3),
    dims=(128, 256, 512, 1024, 2048),
    patch_size=8,
    dropout=0.1,
    kernel_size=7,
    norm_layer=nn.BatchNorm2d,
)

__all__ = ["BACKBONES", "ViT", "TransformerEncoderLayer", "TransformerDecoderLayer", "AdaptiveViT", "ConvNext"]
