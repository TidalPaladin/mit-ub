from .convnext import ConvNext, ConvNextConfig
from .vit import ViT, ViTConfig
from .vit_te import ViTTE, ViTTEConfig


AnyModelConfig = ViTConfig | ConvNextConfig

__all__ = [
    "ViT",
    "ViTConfig",
    "ConvNext",
    "ConvNextConfig",
    "AnyModelConfig",
    "ViTTE",
    "ViTTEConfig",
]
