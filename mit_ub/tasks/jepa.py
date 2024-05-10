from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Set, cast, Tuple
from copy import deepcopy

import torch
import numpy as np
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor
from functools import partial
from ..model import BACKBONES, TransformerBlock, ViT

from ssl_tasks.tokens import TokenMask
from ssl_tasks.contrastive.loss import PointwiseContrastiveEmbeddingLoss


class JEPA(Task):
    def __init__(
        self,
        backbone: str,
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        ema_alpha: float = 0.95,
        linear_probe: bool = True,
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
        for p in self.ema_backbone.parameters():
            p.requires_grad = False

        #self.context_norm = nn.LayerNorm(self.backbone.dim, elementwise_affine=False)
        self.context_norm = nn.Identity()
        self.jepa_loss = nn.MSELoss()
        self.jepa_query = nn.Parameter(torch.empty(1, 1, self.backbone.dim))
        torch.nn.init.trunc_normal_(self.jepa_query, mean=0, std=1)

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
        self.contrastive_loss = PointwiseContrastiveEmbeddingLoss()

        # linear probe
        if linear_probe:
            self.linear_probe = nn.Linear(self.backbone.dim, 1)
            self.linear_probe_loss = nn.BCEWithLogitsLoss()
        else:
            self.linear_probe = None
            self.linear_probe_loss = None

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
        return tm.MetricCollection({"probe_acc": tm.Accuracy(task="binary")})

    def forward(self, x: Tensor, context_mask: TokenMask, target_mask: TokenMask) -> Dict[str, Tensor]:
        # Run encoder on context and broadcast back to full size with 0 padding
        context: Tensor = self.context_norm(self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False))
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

        # Do the same to the ALiBi position encodings, ensuring that we set mask token positions to "inf"
        positions = mask.apply_to_tokens(positions, fill_value=None, padding_value=float("inf"))
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
        for i, (ema_param, param) in enumerate(zip(self.ema_backbone.parameters(), self.backbone.parameters())):
            ema_param.data.mul_(self.ema_alpha).add_(param.data, alpha=1 - self.ema_alpha)
            assert not ema_param.requires_grad
        self.synchronize_ema_weights()

    @torch.no_grad()
    def synchronize_ema_weights(self):
        if self.trainer.world_size > 1:
            for ema_param in self.ema_backbone.parameters():
                torch.distributed.all_reduce(ema_param.data, op=torch.distributed.ReduceOp.SUM)
                ema_param.data /= self.trainer.world_size

    @torch.no_grad()
    def weight_histogram(self, module: nn.Module, bins: int = 100) -> Tuple[Tensor, Tensor]:
        r"""Create a histogram of weights in a given module."""
        weights = torch.cat([p.detach().float().ravel() for p in module.parameters() if p.requires_grad])
        return tuple(t.cpu().numpy() for t in torch.histogram(weights.cpu(), bins=bins))

    @torch.no_grad()
    def tensor_histogram(self, tensor: Tensor, bins: int = 100) -> Tuple[Tensor, Tensor]:
        r"""Create a histogram of weights in a given module."""
        tensor = tensor.detach().float().ravel()
        return tuple(t.cpu().numpy() for t in torch.histogram(tensor.cpu(), bins=bins))

    @torch.no_grad()
    def create_linprobe_gt(self, batch: Dict[str, Any]) -> Tensor:
        view = [e.get("ViewPosition", "unknown") for e in batch["manifest"]]
        view_int = [
            0 if v.startswith("ml") or v.startswith("lm")
            else 1 if "cc" in v else -1
            for v in view
        ]
        return batch["img"].new_tensor(view_int, dtype=torch.long)

    def forward_linear_probe(self, batch: Dict[str, Any], target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # linear probe forward
        N = target.shape[0]
        linprobe_pred = self.linear_probe(target.mean(1).view(N, -1)).view(N)
        assert linprobe_pred.requires_grad or not self.training

        # Build ground truth and compute loss
        linprobe_gt = self.create_linprobe_gt(batch).view(N)
        mask = linprobe_gt != -1
        linprobe_loss = self.linear_probe_loss(linprobe_pred[mask], linprobe_gt[mask].float())
        return linprobe_pred[mask], linprobe_gt[mask], linprobe_loss

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        x: Tensor = batch["img"]

        # ema update from previous step
        self.update_ema()

        # generate context and target masks
        context_mask = self.create_context_mask(x)
        target_mask = self.create_target_mask(x)

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            self.ema_backbone.eval()
            target: Tensor = self.context_norm(self.ema_backbone(x, reshape=False))
            target = target_mask.apply_to_tokens(target, fill_value=None)

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred: Tensor = self(x, context_mask, target_mask)["jepa"]

        # compute loss between target and predictor encoded latents
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss = self.jepa_loss(pred, target)

        # linear probe
        if self.linear_probe is not None:
            linprobe_pred, linprobe_gt, linprobe_loss = self.forward_linear_probe(batch, target)
            with torch.no_grad():
                for name, metric in (metrics or {}).items():
                    if "probe" in name:
                        metric.update(linprobe_pred, linprobe_gt)

        # collapse mitigation
        pred_pool = pred.mean(1)
        #target_pool = target.mean(1)
        #loss_contrastive = (
        #    self.contrastive_loss(pred_pool, pred_pool)
        #    + self.contrastive_loss(pred_pool, target_pool)
        #).sum() / 2
        loss_contrastive = self.contrastive_loss(pred_pool, pred_pool).sum()

        output = {
            "log": {
                "loss_jepa": loss,
                "loss_contrastive": loss_contrastive,
                "loss_linprobe": linprobe_loss,
            },
        }

        if self.trainer.global_step % 100 == 0:
            with torch.no_grad():
                target_std = target.std(dim=0)
                histograms = {
                    "target_hist": self.tensor_histogram(target),
                    "pred_hist": self.tensor_histogram(pred),
                    "std_hist": self.tensor_histogram(target_std),
                }
                output.update(histograms)

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "jepa": pred["jepa"],
        }