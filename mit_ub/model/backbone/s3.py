import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformer_engine.pytorch as te
from einops import rearrange
from torch import Tensor

from ...tokens import apply_mask
from ..config import ModelConfig, convert_sequences
from ..layers.pos_enc import RelativeFactorizedPosition
from .convnext import ConvNext2d, ConvNextConfig
from .vit import ViT, ViTConfig


@dataclass(frozen=True)
class S3Config(ModelConfig):
    vit_config: ViTConfig
    convnext_config: ConvNextConfig

    convnext_resolution: Sequence[int]
    resolutions: Sequence[Sequence[int]]
    token_fracs: float | Sequence[float]

    def __post_init__(self) -> None:
        if isinstance(self.token_fracs, float):
            object.__setattr__(self, "token_fracs", [self.token_fracs] * (len(self.resolutions) - 1) + [1.0])
        assert isinstance(self.token_fracs, Sequence)
        if len(self.resolutions) != len(self.token_fracs):
            raise ValueError("`resolutions` and `num_tokens` must have the same length")
        convert_sequences(self, tuple)
        object.__setattr__(self, "resolutions", tuple(tuple(r) for r in self.resolutions))

        # Verify resolutions are sorted in descending order by area
        areas = [math.prod(res) for res in self.resolutions]
        if not all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1)):
            raise ValueError("Resolutions must be sorted in descending order by area")

        # Verify last token_frac is 1.0
        if self.token_fracs[-1] != 1.0:
            raise ValueError("Last token_frac must be 1.0")

    @property
    def vit_resolution(self) -> Sequence[int]:
        return self.resolutions[-1]

    @property
    def num_scales(self) -> int:
        return len(self.resolutions)

    @property
    def isotropic_output_dim(self) -> int:
        return self.vit_config.hidden_size

    @property
    def checkpoint(self) -> bool:
        return self.vit_config.checkpoint

    def instantiate(self) -> "S3":
        return S3(self)

    @property
    def transformer_kwargs(self) -> Dict[str, Any]:
        return self.vit_config.transformer_kwargs


class S3(ViT):
    CONFIG_TYPE: ClassVar[Type[S3Config]] = S3Config

    def __init__(self, config: S3Config):
        super().__init__(config.vit_config)
        self._config = config

        # Saliency predictors
        self.convnext = ConvNext2d(config.convnext_config)
        self.vit_saliency_head = te.LayerNormLinear(self.config.vit_config.isotropic_output_dim, 1)
        self.convnext_saliency_head = te.LayerNormLinear(self.config.convnext_config.isotropic_output_dim, 1)

        # Scale-specific position encodings
        self.scale_pos_encs = nn.ModuleList()
        for _ in range(config.num_scales):
            layer = RelativeFactorizedPosition(2, self.config.vit_config.hidden_size)
            self.scale_pos_encs.append(layer)

    @property
    def config(self) -> S3Config:
        return self._config

    def forward_saliency(self, x: Tensor) -> Tensor:
        r"""Computes a saliency map for the input image.

        The computation uses both a ViT model and a shallow ConvNext model. It is expected that the ViT model
        operates at a lower resolution than the ConvNext model. ViT saliency is upsampled to the ConvNext
        resolution, and the two saliency maps are averaged as logits.

        Args:
            x: The input image.

        Shapes:
            - x: :math:`(B, C, H, W)` or :math:`(B, D, C, H, W)`
            - saliency: :math:`(B*D, 1, H', W')` where :math:`H'` and :math:`W'` are the height and width
                of the ConvNext output.

        Returns:
            A saliency map (as logits) for the input image.
        """
        if x.ndim == 5:
            x = rearrange(x, "b d c h w -> (b d) c h w")

        # Resize image to match ViT's low resolution input
        x_vit = F.interpolate(x, size=self.config.vit_resolution, mode="bilinear", align_corners=False)
        features_vit, cls_token = super().forward(x_vit, reshape=True)
        saliency_vit = self.vit_saliency_head(features_vit.movedim(1, -1)).movedim(-1, 1)

        # Resize image to match ConvNext's high resolution input
        x_convnext = F.interpolate(x, size=self.config.convnext_resolution, mode="bilinear", align_corners=False)
        features_convnext = self.convnext(x_convnext)
        saliency_convnext = self.convnext_saliency_head(features_convnext.movedim(1, -1)).movedim(-1, 1)

        # Upsample ViT saliency to match ConvNext's resolution
        saliency_vit = F.interpolate(
            saliency_vit, size=saliency_convnext.shape[2:], mode="bilinear", align_corners=False
        )

        # Average logits
        saliency = (saliency_vit + saliency_convnext) / 2

        return saliency

    def select_tokens(self, x: Tensor, saliency: Tensor) -> List[Tensor]:
        r"""Selects top-k tokens from the input based on the saliency map.

        Args:
            x: The input tensor.
            saliency: The saliency map.

        Shapes:
            - x: :math:`(B, C, H, W)` or :math:`(B, D, C, H, W)`
            - saliency: :math:`(B*D, 1, H', W')`

        Returns:
            A list of tensors, each containing the selected tokens for a given resolution.
        """
        # For 3D inputs, fold depth into the batch dimension
        B = x.shape[0]
        if is_3d := x.ndim == 5:
            x = rearrange(x, "b d c h w -> (b d) c h w")
            assert x.shape[0] == saliency.shape[0]

        stages: List[Tensor] = []
        assert isinstance(self.config.token_fracs, Sequence)
        for resolution, token_frac, pos_enc in zip(
            self.config.resolutions, self.config.token_fracs, self.scale_pos_encs
        ):
            # Resize the input to match the current resolution
            x_resized = F.interpolate(x, size=resolution, mode="bilinear", align_corners=False)

            # Tokenize
            _, _, *original_size = x_resized.shape
            tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))
            x_resized = self.stem(x_resized)
            if is_3d:
                x_resized = rearrange(x_resized, "(b d) l ... -> b (d l) ...", b=B)

            # Resize the saliency map to match the tokenized size and select top-k tokens
            saliency_resized = F.interpolate(saliency, size=tokenized_size, mode="bilinear", align_corners=False)
            if is_3d:
                saliency_resized = rearrange(saliency_resized, "(b d) ... -> b (d ...)", b=B)
            else:
                saliency_resized = rearrange(saliency_resized, "b ... -> b (...)", b=B)

            k = int(token_frac * math.prod(tokenized_size))
            _, indices = torch.topk(saliency_resized, k, dim=-1)

            # Extract patch embeddings for the top-k tokens
            D = self.config.vit_config.hidden_size
            gather_indices = indices.view(B, -1, 1).expand(-1, -1, D)

            x_resized = torch.gather(x_resized, dim=1, index=gather_indices)

            # Apply stage-specific position encoding
            pos = pos_enc(tokenized_size).expand(saliency.shape[0], -1, -1).reshape(B, -1, D)
            pos = torch.gather(pos, dim=1, index=gather_indices)
            x_resized = x_resized + pos

            stages.append(x_resized)

        return stages

    def forward(
        self,
        x: Tensor,
        reshape: bool = False,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
        saliency: Tensor | None = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if reshape:
            raise NotImplementedError("Reshaping is not supported for S3")

        # Compute saliency map
        B = x.shape[0]
        saliency = self.forward_saliency(x) if saliency is None else saliency

        # Select input tokens based on saliency
        x = torch.cat(self.select_tokens(x, saliency), dim=1)
        if mask is not None:
            x = apply_mask(mask, x, fill_value=mask_fill_value)

        # Add CLS token
        x = torch.cat([self.cls_token.view(1, 1, -1).expand(B, -1, -1), x], dim=1)

        # Transformer blocks and output norm
        for block in self.blocks:
            x = block(x, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token
        cls_token = x[:, 0, :].contiguous()
        x = x[:, 1:, :].contiguous()

        return x, cls_token, saliency

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = S3Config(*args, **kwargs)
        return cls(config)
