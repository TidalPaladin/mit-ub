import torch

from .activations import ACTIVATIONS
from .backbone import (
    AdaptiveViT,
    AdaptiveViTConfig,
    AnyModelConfig,
    ConvNext,
    ConvNextConfig,
    ConvViT,
    ConvViTConfig,
    ViT,
    ViTConfig,
)
from .helpers import compile_is_disabled
from .layers.transformer import TransformerDecoderLayer, TransformerEncoderLayer


torch._dynamo.config.cache_size_limit = 128  # type: ignore


__all__ = [
    "ACTIVATIONS",
    "ViT",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "AdaptiveViT",
    "ConvNext",
    "compile_is_disabled",
    "ViTConfig",
    "AdaptiveViTConfig",
    "ConvViTConfig",
    "ConvViT",
    "ConvNextConfig",
    "AnyModelConfig",
]
