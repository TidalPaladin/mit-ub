from registry import Registry

from .backbone import ViT
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

__all__ = ["BACKBONES", "ViT", "TransformerEncoderLayer", "TransformerDecoderLayer"]
