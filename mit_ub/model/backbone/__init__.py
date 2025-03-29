from .convnext import ConvNext2d, ConvNextConfig
from .s3 import S3, S3Config
from .vit import ViT, ViTConfig


AnyModelConfig = ViTConfig | ConvNextConfig | S3Config

__all__ = [
    "ViT",
    "ViTConfig",
    "ConvNext2d",
    "ConvNextConfig",
    "AnyModelConfig",
    "S3",
    "S3Config",
]
