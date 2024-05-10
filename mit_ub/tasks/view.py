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


class ViewPrediction(Task):
    def __init__(
        self,
        backbone: str,
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

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        self.head = nn.Linear(self.backbone.dim, 1)
        self.criterion = nn.BCEWithLogitsLoss()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection({"view_acc": tm.Accuracy(task="binary")})

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        N = x.shape[0]
        x = self.backbone(x, reshape=False)
        x = self.head(x.mean(1).view(N, -1)).view(N, 1)
        return {"view": x}

    @torch.no_grad()
    def create_target(self, batch: Dict[str, Any]) -> Tensor:
        view = [e.get("ViewPosition", "unknown") for e in batch["manifest"]]
        view_int = [
            0 if v.startswith("ml") or v.startswith("lm")
            else 1 if "cc" in v else -1
            for v in view
        ]
        return batch["img"].new_tensor(view_int, dtype=torch.long)

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        x: Tensor = batch["img"]
        N = x.shape[0]

        pred: Tensor = self(x)["view"]
        pred = pred.view(N)

        # Build ground truth and compute loss
        target = self.create_target(batch).view(N)
        mask = target != -1
        loss = self.criterion(pred[mask], target[mask].float())

        with torch.no_grad():
            if metrics is not None:
                metrics.update(pred[mask], target[mask])

        output = {
            "log": {
                "loss_view": loss,
            },
        }

        return output