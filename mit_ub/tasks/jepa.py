from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, cast, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor
from functools import partial
from ..model import BACKBONES, TransformerBlock, ViT

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

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        self.ema_backbone = deepcopy(self.backbone)
        self.jepa_loss = nn.MSELoss()
        self.jepa_query = nn.Parameter(torch.randn(1, 1, self.backbone.dim))

        predictor_depth = 4
        predictor_dim_ff = self.backbone.dim
        self.jepa_predictor = nn.ModuleList(
            [
                TransformerBlock(
                    self.backbone.dim, 
                    self.backbone.nhead, 
                    predictor_dim_ff, 
                    dropout=0.1, 
                    activation=nn.GELU(), 
                    alibi_upper=i,
                ) 
                for i in range(predictor_depth)
            ]
        )

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

    def forward(self, x: Tensor, context_mask: TokenMask, target_mask: TokenMask) -> Dict[str, Tensor]:
        # Run encoder on context and broadcast back to full size with 0 padding
        context: Tensor = self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)
        context = context_mask.restore_tokens(context, 0)

        # Create empty queries w/ position encoding that forms the initial predictor input
        B, _, D = context.shape
        tokenized_size = self.backbone.tokenized_size(*x.shape[2:])
        if is_3d := x.ndim == 5:
            query = self.backbone.pos_enc_3d.from_grid(tokenized_size, B, proto=context, normalize=True, requires_grad=False)
        else:
            query = self.backbone.pos_enc_2d.from_grid(tokenized_size, B, proto=context, normalize=True, requires_grad=False)
        query = query.contiguous()
        query += self.jepa_query.type_as(query)

        # Generate full size ALiBi position encodings for the queries
        positions = self.backbone.create_alibi_positions(query, tokenized_size, normalize=False).view(B, -1, len(tokenized_size))

        # Use xor mask to inject encoder context into queries that aren't part of the target mask.
        # Query now contains context only at locations that are not part of the target.
        with torch.no_grad():
            xor_mask = (context_mask.mask ^ target_mask.mask).unsqueeze_(-1)
        query = torch.where(xor_mask, query, context)

        # Create a context or target mask. 
        # Since context and target may overlap, we may end up with an inconsistent number of tokens 
        # for each example in the batch. To resolve this we will pad to match the largest number 
        # of tokens in an example, and adjust the ALiBi positions such that these padding tokens 
        # are masked in the predictor.
        mask = (context_mask.mask | target_mask.mask)
        mask = TokenMask(mask, context_mask.size, context_mask.patch_size)
        query = mask.apply_to_tokens(query, fill_value=None)
        assert False

        # Do the same to the ALiBi position encodings
        positions = mask.apply_to_tokens(positions, fill_value=None)
        assert query.shape[:2] == positions.shape[:2]

        # Run the queries and ALiBi positions through the predictor
        B, L = query.shape[:2]
        position = positions.view(B, 1, L, -1).expand(-1, self.backbone.nhead, -1, -1)
        for block in self.jepa_predictor:
            block = cast(TransformerBlock, block)
            query = block(query, position)

        # Extract only the target queries from the full set of queries
        query = mask.restore_tokens(query, 0)
        query = target_mask.apply_to_tokens(query, fill_value=None)

        return {"jepa": query}

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
        torch.autograd.set_detect_anomaly(True)

        # generate context and target masks
        context_mask = self.create_context_mask(x)
        target_mask = self.create_target_mask(x)

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            target: Tensor = self.ema_backbone(x, reshape=False)
            target = target_mask.apply_to_tokens(target, fill_value=None)

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred: Tensor = self(x, context_mask, target_mask)["jepa"]

        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss = self.jepa_loss(pred, target)

        output = {
            "log": {
                "loss_jepa": loss,
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "jepa": pred["jepa"],
        }