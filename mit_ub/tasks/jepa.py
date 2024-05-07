from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set
from copy import deepcopy

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor
from functools import partial
from ..model import BACKBONES

from ssl_tasks.tokens import TokenMask


def mask_fn(module: nn.Module, args: Any, output: Tensor, mask: TokenMask) -> Tensor:
    output = mask.apply_to_tokens(output, fill_value=None)
    return output


class JEPA(Task):
    def __init__(
        self,
        backbone: str,
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        ema_alpha: float = 0.999,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        weight_decay_exemptions: Set[str] = set(),
    ):
        super().__init__(
            optimizer_init,
            lr_scheduler_init,
            lr_interval,
            lr_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            weight_decay_exemptions,
        )

        self.context_ratio = context_ratio
        self.context_scale = context_scale
        self.target_ratio = target_ratio
        self.target_scale = target_scale
        assert self.context_ratio > 0
        assert self.target_ratio > 0
        self.ema_alpha = ema_alpha

        self.backbone = self.prepare_backbone(backbone)
        self.ema_backbone = deepcopy(self.backbone)
        self.jepa_loss = nn.MSELoss()
        self.jepa_query = nn.Parameter(torch.randn(1, 1, self.backbone.dim))

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_context_mask(self, x: Tensor) -> TokenMask:
        size = x.shape[2:]
        batch_size = x.shape[0]
        device = x.device
        return TokenMask.create(
            size,
            self.backbone.patch_size[-len(size) :],
            batch_size,
            device=device,
            # Flip this so we get context_mask unmasked
            mask_ratio=1 - self.context_ratio,
            scale=self.context_scale,
        )

    def create_target_mask(self, x: Tensor) -> TokenMask:
        size = x.shape[2:]
        batch_size = x.shape[0]
        device = x.device
        return TokenMask.create(
            size,
            self.backbone.patch_size[-len(size) :],
            batch_size,
            device=device,
            # Flip this so we get target_mask unmasked
            mask_ratio=1 - self.target_ratio,
            scale=self.target_scale,
        )

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection({})

    def forward(
        self,
        x: Tensor,
        mask: Optional[TokenMask] = None,
    ) -> Dict[str, Tensor]:
        x = self.backbone(x, mask=mask, mask_fill_value=None, reshape=False)
        return {"jepa": x}

    @torch.no_grad()
    def update_ema(self):
        alpha = min(1 - 1 / (self.global_step + 1), self.alpha)
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        x: Tensor = batch["img"]

        # generate context and target masks
        context_mask = self.create_context_mask(x)
        target_mask = self.create_target_mask(x)

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            target: Tensor = self.ema_backbone(x, reshape=False)
            target = target_mask.apply_to_tokens(target, fill_value=None)

        predictor_in = self.prepare_predictor_input(x, context_mask, target_mask)


        loss = self.jepa_loss(...)




        output = {
            "masked": masked_img,
            "jepa_pred": jepa_pred,
            "jepa_true": jepa_true,
            "log": {
                "loss_jepa": loss,
            },
        }

        return output

    def prepare_predictor_input(self, x: Tensor, context_mask: TokenMask, target_mask: TokenMask) -> Tensor:
        # Run encoder on context
        context: Tensor = self(x, context_mask)["jepa"]

        # Create positional embeddings and apply target mask
        B, _, D = context.shape
        tokenized_size = self.backbone.tokenized_size(*x.shape[2:])
        if is_3d := x.ndim == 5:
            pos_emb = self.backbone.pos_enc_3d.from_grid(tokenized_size, B, proto=context, normalize=True)
        else:
            pos_emb = self.backbone.pos_enc_2d.from_grid(tokenized_size, B, proto=context, normalize=True)
        query = target_mask.apply_to_tokens(pos_emb, fill_value=None) + self.jepa_query.type_as(pos_emb)

        # Update queries that intersect the context with the encoded context
        xor_mask = (context_mask.mask ^ target_mask.mask).unsqueeze_(-1)
        xor_mask = context_mask.apply_to_tokens(xor_mask, fill_value=None).squeeze_(-1)
        import pdb; pdb.set_trace()
        query[intersection_mask_target] = context[intersection_mask_context]

        # Combine queries with the difference of the target and context
        diff_mask = (context_mask.mask & (~target_mask.mask))
        diff_mask = context_mask.apply_to_tokens(diff_mask.unsqueeze_(-1), fill_value=None).squeeze_(-1)
        diff_mask = TokenMask(diff_mask, context_mask.size, context_mask.patch_size)
        query[diff_mask] = target[diff_mask]

        return x

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "jepa": pred["jepa"],
        }