from dataclasses import dataclass
from typing import Any, ClassVar, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ...tokens import create_mask
from ..layers import RelativeFactorizedPosition
from .convnext import tokens_to_grid
from .vit import ViT, ViTConfig


@dataclass(frozen=True)
class TwoStageViTConfig(ViTConfig):
    first_stage_size: Sequence[int] = (256, 256)
    second_stage_depth: int | None = None
    second_stage_cross_attention: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.second_stage_depth is None:
            object.__setattr__(self, "second_stage_depth", self.depth)

    def instantiate(self) -> "TwoStageViT":
        return TwoStageViT(self)


class TwoStageViT(ViT):
    CONFIG_TYPE: ClassVar[Type[TwoStageViTConfig]] = TwoStageViTConfig

    def __init__(self, config: TwoStageViTConfig):
        super().__init__(config)
        assert config.second_stage_depth is not None
        for param in self.parameters():
            param.requires_grad = False

        # Stage two
        self.stage_two_pos_enc = RelativeFactorizedPosition(2, self.config.hidden_size)
        self.stage_two_blocks = nn.ModuleList(
            [
                (
                    self.create_encoder_layer(i)
                    if not self.config.second_stage_cross_attention
                    else self.create_decoder_layer(i)
                )
                for i in range(config.second_stage_depth)
            ]
        )
        self.stage_one_proj = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    @property
    def config(self) -> TwoStageViTConfig:
        return cast(TwoStageViTConfig, self._config)

    def stage_two_tokenized_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        Hp = original_size[0] // self.config.first_stage_size[0]
        Wp = original_size[1] // self.config.first_stage_size[1]
        return Hp, Wp

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        batch_size = input.shape[0]
        Ht, Wt = self.stage_two_tokenized_size(input.shape[-2:])
        device = input.device
        original_size = self.config.first_stage_size
        tokenized_size = self.stem.tokenized_size(cast(Any, original_size))

        # Create a mask that has the same number of tokens masked in each stage one patch
        mask = create_mask(
            tokenized_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size * Ht * Wt,
            scale=scale,
            device=device,
        )

        mask = rearrange(mask, "(b ht wt) l -> b (ht wt l)", ht=Ht, wt=Wt)
        return mask

    @torch.inference_mode()
    def _forward_stage_one(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        Hp, Wp = self.config.first_stage_size
        Ht, Wt = self.stage_two_tokenized_size(x.shape[-2:])
        x = rearrange(x, "b c (ht hp) (wt wp) -> (b ht wt) c hp wp", hp=Hp, wp=Wp, ht=Ht, wt=Wt)
        x, cls_token = super().forward(x, reshape=False, mask=mask, mask_fill_value=mask_fill_value)
        x = rearrange(x, "(b ht wt) l d -> b (l ht wt) d", ht=Ht, wt=Wt)
        cls_token = rearrange(cls_token, "(b ht wt) d -> b (ht wt) d", ht=Ht, wt=Wt)
        return x, cls_token

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Stage one
        if self.training:
            x, cls_tokens = self._forward_stage_one(x, mask, mask_fill_value)
            # Must clone for autograd because of inference_mode()
            x = x.clone()
            cls_tokens = cls_tokens.clone()
        else:
            x, cls_tokens = self._forward_stage_one(x, mask, mask_fill_value)

        # Create stage two CLS token as average of stage one CLS tokens
        stage_two_cls_token = cls_tokens.mean(dim=1, keepdim=True)

        # Create query of stage two CLS token and stage one CLS tokens
        Ht, Wt = self.stage_two_tokenized_size(cast(Any, original_size))
        pos_enc = self.stage_two_pos_enc((Ht, Wt))
        cls_tokens = cls_tokens + pos_enc
        query = torch.cat([stage_two_cls_token, cls_tokens], dim=1)

        # Run stage two transformer blocks with optional cross attention to stage one features
        encoder_output = x if self.config.second_stage_cross_attention else None
        for block in self.stage_two_blocks:
            query = block(query, encoder_output=encoder_output, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token
        stage_two_cls_token = query[:, 0, :].contiguous()
        stage_two_features = query[:, 1:, :].contiguous()

        # Combine stage one features with stage two features.
        # NOTE: We assume identical number of tokens are masked in each stage one patch.
        # Under this assumption we know the L dimension of x is tokens_per_patch * num_patches
        x = self.stage_one_proj(x)
        x = rearrange(x, "b (ht wt l) d -> b (ht wt) l d", ht=Ht, wt=Wt)
        stage_two_features = rearrange(stage_two_features, "b l d -> b l () d")
        x = x + stage_two_features
        x = rearrange(x, "b (ht wt) l d -> b (ht wt l) d", ht=Ht, wt=Wt)

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            x = tokens_to_grid(x, tokenized_size)

        return x, stage_two_cls_token

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = TwoStageViTConfig(*args, **kwargs)
        return cls(config)
