from registry import Registry

from .backbone import AdaptiveViT, ViT
from .transformer import TransformerDecoderLayer, TransformerEncoderLayer


BACKBONES = Registry("backbones")

BACKBONES(
    ViT,
    name="vit-i1-p16-d512",
    in_channels=1,
    dim=512,
    patch_size=16,
    depth=12,
    nhead=512 // 32,
    dropout=0.1,
)
BACKBONES(
    AdaptiveViT,
    name="vit-i1-p16-d512-adaptive",
    in_channels=1,
    dim=512,
    kv_dim=64,
    patch_size=16,
    target_shape=(16, 12),
    depth=12,
    nhead=512 // 32,
    dropout=0.1,
)


BACKBONES(
    ViT,
    name="vit-cifar10",
    in_channels=3,
    dim=256,
    patch_size=4,
    depth=12,
    nhead=256 // 32,
    dropout=0.1,
)

BACKBONES(
    AdaptiveViT,
    name="vit-cifar10-adaptive",
    in_channels=3,
    dim=256,
    kv_dim=32,
    patch_size=4,
    target_shape=(4, 4),
    tokenizer_depth=3,
    depth=12,
    nhead=256 // 32,
    dropout=0.1,
)

__all__ = ["BACKBONES", "ViT", "TransformerEncoderLayer", "TransformerDecoderLayer", "AdaptiveViT"]
