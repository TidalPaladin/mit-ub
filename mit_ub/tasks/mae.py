from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch.nn as nn
from einops import rearrange
from ssl_tasks.mae.task import MAE as MAEBase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from ..model import BACKBONES


def mask_fn(module: nn.Module, args: Any, output: Tensor, mask: TokenMask) -> Tensor:
    output = mask.apply_to_tokens(output, fill_value=0)
    return output


class MAE(MAEBase):
    r"""Implements MAE style pretraining.

    Args:
        backbone: Backbone architecture for the model.
        mask_ratio: Ratio of tokens to mask. Defaults to 0.4.
        mask_scale: Scale of the mask. Increasing this will mask tokens in larger groups. Defaults to 2.
        optimizer_init: Initial configuration for the optimizer.
        lr_scheduler_init: Initial configuration for the learning rate scheduler.
        lr_interval: Interval for learning rate update. Defaults to "epoch".
        lr_monitor: Metric to monitor for learning rate scheduler. Defaults to "train/total_loss_epoch".
        checkpoint: Path to the checkpoint file.
        strict_checkpoint: If True, loading checkpoint is strict.
        log_train_metrics_interval: Interval for logging training metrics.
        log_train_metrics_on_epoch: If True, logs training metrics on epoch end.
        weight_decay_exemptions: Set of exemptions for weight decay.

    """

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_head(self) -> nn.Module:
        dim = cast(Any, self.backbone).dim
        out_dim = cast(Any, self.backbone).in_channels
        patch_h, patch_w = cast(Any, self.backbone).patch_size_2d
        outputs_per_token = out_dim * patch_h * patch_w
        return nn.Conv2d(dim, outputs_per_token, kernel_size=1)

    def create_token_mask(self, x: Tensor) -> TokenMask:
        size = x.shape[2:]
        batch_size = x.shape[0]
        device = x.device
        return TokenMask.create(
            size,
            self.backbone.patch_size[-len(size) :],
            batch_size,
            device=device,
            mask_ratio=self.mask_ratio,
            scale=self.mask_scale,
        )

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        y = self.backbone(x)
        y = self.mae_head(y).contiguous()

        Hp, Wp = self.backbone.patch_size_2d
        H, W = y.shape[-2:]
        y = rearrange(
            y,
            "b (hp wp c) h w -> b c (h hp) (w wp)",
            hp=Hp,
            wp=Wp,
            h=H,
            w=W,
        )

        if mask_hook is not None:
            mask_hook.remove()
        return {"mae": y}
