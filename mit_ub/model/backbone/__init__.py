from .convnext import ConvNext2d, ConvNextConfig
from .two_stage import TwoStageViT, TwoStageViTConfig
from .vit import ViT, ViTConfig


AnyModelConfig = ViTConfig | ConvNextConfig | TwoStageViTConfig

__all__ = [
    "ViT",
    "ViTConfig",
    "ConvNext2d",
    "ConvNextConfig",
    "TwoStageViT",
    "TwoStageViTConfig",
    "AnyModelConfig",
]
