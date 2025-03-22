from .convnext import ConvNext2d, ConvNextConfig
from .two_stage import TwoStageViT, TwoStageViTConfig, WindowedViT, WindowedViTConfig
from .vit import ViT, ViTConfig


AnyModelConfig = ViTConfig | ConvNextConfig | TwoStageViTConfig | WindowedViTConfig
AnyViTConfig = ViTConfig | TwoStageViTConfig | WindowedViTConfig

__all__ = [
    "ViT",
    "ViTConfig",
    "ConvNext2d",
    "ConvNextConfig",
    "TwoStageViT",
    "TwoStageViTConfig",
    "WindowedViT",
    "WindowedViTConfig",
    "AnyModelConfig",
    "AnyViTConfig",
]
