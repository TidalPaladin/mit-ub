from .convnext import ConvNext2d, ConvNextConfig
from .vit import ViT, ViTConfig


AnyModelConfig = ViTConfig | ConvNextConfig

__all__ = [
    "ViT",
    "ViTConfig",
    "ConvNext2d",
    "ConvNextConfig",
    "AnyModelConfig",
]
