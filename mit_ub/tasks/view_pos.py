from copy import copy
from typing import Any, Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import State
from torch import Tensor

from .jepa import JEPAWithProbe


class JEPAWithViewPosition(JEPAWithProbe):
    def __init__(
        self,
        backbone: str,
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        ema_alpha: float = 0.95,
        activation_clip: float | None = None,
        margin: float | None = 0.5,
        loss_fn: str = "cosine",
        distribution_loss: bool = False,
        predictor_depth: int = 4,
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
            backbone,
            context_ratio,
            context_scale,
            target_ratio,
            target_scale,
            ema_alpha,
            activation_clip,
            margin,
            loss_fn,
            distribution_loss,
            predictor_depth,
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

    def create_probe_head(self) -> nn.Module:
        return nn.Linear(self.backbone.dim, 1)

    def create_metrics(self, state: State) -> tm.MetricCollection:
        return tm.MetricCollection({"view_pos_acc": tm.Accuracy(task="binary")})

    @torch.no_grad()
    def create_view_pos_gt(self, batch: Dict[str, Any]) -> Tensor:
        view = [e.get("ViewPosition", "unknown") for e in batch["manifest"]]
        view_int = [0 if v.startswith("ml") or v.startswith("lm") else 1 if "cc" in v else -1 for v in view]
        return batch["img"].new_tensor(view_int, dtype=torch.long)

    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        # Forward pass of linear probe using target features
        features: Tensor = output["target"]
        assert self.linear_probe is not None
        N = features.shape[0]
        linprobe_logits = self.linear_probe(features.mean(1).view(N, -1)).view(N)
        assert linprobe_logits.requires_grad or not self.training

        # Build ground truth and compute loss
        linprobe_gt = self.create_view_pos_gt(batch).view(N)
        mask = linprobe_gt != -1
        linprobe_loss = F.binary_cross_entropy_with_logits(linprobe_logits[mask], linprobe_gt[mask].float())
        if not mask.any():
            linprobe_loss = torch.zeros_like(linprobe_loss)

        # Logits -> probs
        with torch.no_grad():
            linprobe_probs = torch.sigmoid(linprobe_logits)

        # Compute metrics
        with torch.no_grad():
            if mask.any():
                for name, metric in (metrics or {}).items():
                    if "view_pos" in name and linprobe_gt.numel():
                        metric.update(linprobe_probs[mask], linprobe_gt[mask])

        output = copy(output)
        output["log"]["loss_linprobe"] = linprobe_loss
        return output