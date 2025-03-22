import math
from dataclasses import dataclass
from typing import Any, ClassVar, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ...tokens import apply_mask, create_mask, mask_is_ragged
from ..layers import RelativeFactorizedPosition
from .convnext import tokens_to_grid
from .vit import ViT, ViTConfig


@dataclass(frozen=True)
class TwoStageViTConfig(ViTConfig):
    first_stage_size: Sequence[int] = (256, 256)
    second_stage_depth: int | None = None
    second_stage_cross_attention: bool = False
    freeze_first_stage: bool = True

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
            param.requires_grad = not config.freeze_first_stage

        # Stage two
        self.stage_two_cls_token = nn.Parameter(torch.empty(self.config.hidden_size))
        nn.init.trunc_normal_(self.stage_two_cls_token, std=0.02)
        self.stage_two_pos_enc = RelativeFactorizedPosition(2, self.config.hidden_size)
        self.stage_two_blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(config.second_stage_depth)])

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
        tokenized_size = self.stage_two_tokenized_size(input.shape[-2:])
        device = input.device

        # Create a mask that has the same number of tokens masked in each stage one patch
        mask = create_mask(
            tokenized_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size,
            scale=scale,
            device=device,
        )
        return mask

    def tile_image(self, x: Tensor) -> Tensor:
        Ht, Wt = self.config.first_stage_size
        y = rearrange(x, "b c (hl ht) (wl wt) -> (b hl wl) c ht wt", ht=Ht, wt=Wt)
        assert y.shape[-2:] == (Ht, Wt)
        return y

    def untile_sequence(self, x: Tensor, batch_size: int) -> Tensor:
        return rearrange(x, "(b l) ... d -> b (l ...) d", b=batch_size)

    def forward_stage_one(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        if mask is not None:
            mask = self.tile_mask(x, mask)
        x = self.tile_image(x)
        x, cls_token = super().forward(x, reshape=False, mask=mask, mask_fill_value=mask_fill_value)
        x = self.untile_sequence(x, B).contiguous()
        cls_token = self.untile_sequence(cls_token, B).contiguous()
        return x, cls_token

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Stage one
        if self.training:
            with torch.inference_mode(mode=not self.config.freeze_first_stage):
                x, cls_tokens = self.forward_stage_one(x)
            # Must clone for autograd because of inference_mode()
            if self.config.freeze_first_stage:
                x = x.clone()
                cls_tokens = cls_tokens.clone()
        else:
            x, cls_tokens = self.forward_stage_one(x)

        if mask is not None:
            cls_tokens = apply_mask(mask, cls_tokens)

        # Create query of stage two CLS token and stage one CLS tokens
        stage_two_cls_token = self.stage_two_cls_token.view(1, 1, -1).expand(B, -1, -1)
        Ht, Wt = self.stage_two_tokenized_size(cast(Any, original_size))
        pos_enc_cls = self.stage_two_pos_enc((Ht, Wt))
        if mask is not None:
            pos_enc_cls = apply_mask(mask, pos_enc_cls.expand(B, -1, -1))
        cls_tokens = cls_tokens + pos_enc_cls
        query = torch.cat([stage_two_cls_token, cls_tokens], dim=1)

        # Run stage two transformer blocks with optional cross attention to stage one features
        for block in self.stage_two_blocks:
            query = block(query, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token
        stage_two_cls_token = query[:, 0, :].contiguous()
        stage_two_features = query[:, 1:, :].contiguous()

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            stage_two_features = tokens_to_grid(stage_two_features, (Ht, Wt))

        return stage_two_features, stage_two_cls_token

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = TwoStageViTConfig(*args, **kwargs)
        return cls(config)


@dataclass(frozen=True)
class WindowedViTConfig(ViTConfig):
    window_size: Sequence[int] = (256, 256)
    global_depth: int = 4
    freeze_first_stage: bool = True

    def instantiate(self) -> "WindowedViT":
        return WindowedViT(self)


class WindowedViT(ViT):
    CONFIG_TYPE: ClassVar[Type[WindowedViTConfig]] = WindowedViTConfig

    def __init__(self, config: WindowedViTConfig):
        super().__init__(config)
        for param in self.parameters():
            param.requires_grad = not config.freeze_first_stage
        self.global_pos_enc = RelativeFactorizedPosition(2, self.config.hidden_size)
        self.cls_token_pos_enc = RelativeFactorizedPosition(2, self.config.hidden_size)
        self.global_blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(config.global_depth)])

    @property
    def config(self) -> WindowedViTConfig:
        return cast(WindowedViTConfig, self._config)

    def window_grid_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        return (original_size[0] // self.config.window_size[0], original_size[1] // self.config.window_size[1])

    def num_windows(self, original_size: Tuple[int, int]) -> int:
        return math.prod(self.window_grid_size(original_size))

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        batch_size = input.shape[0]
        device = input.device

        # Create a mask that has the same number of tokens masked in each stage one patch
        window_size = self.config.window_size
        tokenized_size_per_window = self.stem.tokenized_size(window_size)
        num_windows = self.num_windows(input.shape[-2:])
        mask = create_mask(
            tokenized_size_per_window,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size * num_windows,
            scale=scale,
            device=device,
        )
        mask = rearrange(mask, "(b w) t -> b (w t)", w=num_windows)
        assert not mask_is_ragged(mask)
        return mask

    def tile_image(self, x: Tensor) -> Tensor:
        Ht, Wt = self.config.window_size
        y = rearrange(x, "b c (hl ht) (wl wt) -> (b hl wl) c ht wt", ht=Ht, wt=Wt)
        assert y.shape[-2:] == (Ht, Wt)
        return y

    def tile_mask(self, mask: Tensor) -> Tensor:
        tokens_per_window = math.prod(self.stem.tokenized_size(self.config.window_size))
        mask = rearrange(mask, "b (w t) -> (b w) t", t=tokens_per_window)
        assert not mask_is_ragged(mask)
        return mask

    def untile_sequence(self, x: Tensor, batch_size: int) -> Tensor:
        return rearrange(x, "(b l) ... d -> b (l ...) d", b=batch_size)

    def forward_windowed(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B = x.shape[0]
        if mask is not None:
            mask = self.tile_mask(mask)
        x = self.tile_image(x)
        x, cls_tokens = super().forward(x, reshape=False, mask=mask, mask_fill_value=mask_fill_value)
        x = self.untile_sequence(x, B).contiguous()
        cls_tokens = self.untile_sequence(cls_tokens, B).contiguous()
        return x, cls_tokens

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Run on windows
        x, cls_tokens = self.forward_windowed(x, mask=mask, mask_fill_value=mask_fill_value)

        # Apply global positional encoding to CLS tokens
        pos_enc = self.cls_token_pos_enc(self.window_grid_size(original_size))
        cls_tokens = cls_tokens + pos_enc

        # Apply global positional encoding to features
        pos_enc = self.global_pos_enc(tokenized_size)
        if mask is not None:
            pos_enc = apply_mask(mask, pos_enc.expand(B, -1, -1))
        x = x + pos_enc

        # Combine CLS token and features
        x = torch.cat([cls_tokens, x], dim=1)

        # Run global transformer blocks with optional cross attention to stage one features
        for block in self.global_blocks:
            x = block(x, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token
        num_cls_tokens = cls_tokens.shape[1]
        cls_token = x[:, :num_cls_tokens, :].mean(dim=1)
        x = x[:, num_cls_tokens:, :].contiguous()

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            x = tokens_to_grid(x, tokenized_size)

        return x, cls_token

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = WindowedViTConfig(*args, **kwargs)
        return cls(config)
