from .adaptive_vit import AdaptiveViT, AdaptiveViTConfig
from .convnext import ConvNext, ConvNextConfig
from .convvit import ConvViT, ConvViTConfig
from .vit import ViT, ViTConfig


AnyModelConfig = ViTConfig | AdaptiveViTConfig | ConvNextConfig | ConvViTConfig

__all__ = [
    "ViT",
    "ViTConfig",
    "AdaptiveViT",
    "AdaptiveViTConfig",
    "ConvViT",
    "ConvViTConfig",
    "ConvNext",
    "ConvNextConfig",
    "AnyModelConfig",
]
