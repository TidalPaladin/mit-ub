from .convnext import ConvNext2d, ConvNextConfig
from .navit import NaViT, NaViTConfig
from .vit import ViT, ViTConfig


AnyModelConfig = ViTConfig | ConvNextConfig | NaViTConfig
AnyViTConfig = ViTConfig | NaViTConfig

__all__ = [
    "ViT",
    "ViTConfig",
    "ConvNext2d",
    "ConvNextConfig",
    "AnyModelConfig",
    "AnyViTConfig",
    "NaViT",
    "NaViTConfig",
]
