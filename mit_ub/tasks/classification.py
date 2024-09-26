from copy import copy
from typing import Any, Dict, Optional, Set, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from einops.layers.torch import Rearrange
from torch import Tensor

from ..model import BACKBONES, ViT
from .jepa import JEPAWithProbe


class ClassificationTask(Task):
    """
    Implements a generic image classification task.

    Args:
        backbone: Name of the backbone to use for the task.
        num_classes: Number of classes
        optimizer_init: Initial configuration for the optimizer.
        lr_scheduler_init: Initial configuration for the learning rate scheduler.
        lr_interval: Frequency of learning rate update. Can be 'step' or 'epoch'.
        lr_monitor: Quantity to monitor for learning rate scheduler.
        named_datasets: If True, datasets are named, else they are indexed by integers.
        checkpoint: Path to the checkpoint file to initialize the model.
        strict_checkpoint: If True, the model must exactly match the checkpoint.
        log_train_metrics_interval: Interval (in steps) at which to log training metrics.
        log_train_metrics_on_epoch: If True, log training metrics at the end of each epoch.
        weight_decay_exemptions: Set of parameter names to exempt from weight decay.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
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
        self.num_classes = num_classes

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        dim = self.backbone.dim
        self.classification_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b c () () -> b c"),
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        return tm.MetricCollection(
            {
                "acc": tm.Accuracy(task="multiclass", num_classes=self.num_classes),
            }
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        cls = self.classification_head(x)
        return {"pred": cls.view(-1, 1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        x = batch["img"]
        y = batch["label"].long()
        N = y.shape[0]

        # forward pass
        pred_logits = self(x)["pred"].view(N, self.num_classes)

        # compute loss
        loss = cast(Tensor, (self.criterion(pred_logits, y)))

        with torch.no_grad():
            pred = pred_logits.argmax(dim=1)

        # log metrics
        with torch.no_grad():
            if metrics is not None:
                metrics.update(pred, y)

        output = {
            "log": {
                "loss_classification": loss,
            },
        }

        return output


class JEPAWithClassification(JEPAWithProbe):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        ema_alpha: float = 0.95,
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
        self.num_classes = num_classes
        super().__init__(
            backbone,
            context_ratio,
            context_scale,
            target_ratio,
            target_scale,
            ema_alpha,
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
        return nn.Sequential(
            nn.LayerNorm(self.backbone.dim),
            nn.Linear(self.backbone.dim, self.num_classes),
        )

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        metrics = super().create_metrics(*args, **kwargs)
        metrics.add_metrics({"acc": tm.Accuracy(task="multiclass", num_classes=self.num_classes)})
        return metrics

    @torch.no_grad()
    def create_gt(self, batch: Dict[str, Any]) -> Tensor:
        y = batch["label"].long()
        return y

    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        # Forward pass of linear probe using target features
        features = self.get_probe_features_from_output(output)

        assert self.linear_probe is not None
        N = features.shape[0]
        linprobe_logits = self.linear_probe(features.mean(1).view(N, -1)).view(N, -1)
        assert linprobe_logits.requires_grad or not self.training

        # Build ground truth and compute loss
        linprobe_gt = self.create_gt(batch).view(N)
        mask = linprobe_gt != -1
        linprobe_loss = F.cross_entropy(linprobe_logits[mask], linprobe_gt[mask].long())
        if not mask.any():
            linprobe_loss = torch.zeros_like(linprobe_loss)

        # Logits -> probs
        with torch.no_grad():
            linprobe_probs = torch.sigmoid(linprobe_logits)

        # Compute metrics
        with torch.no_grad():
            if mask.any():
                for name, metric in (metrics or {}).items():
                    if "acc" in name and linprobe_gt.numel():
                        metric.update(linprobe_probs[mask], linprobe_gt[mask])

        output = copy(output)
        output["log"]["loss_linprobe"] = linprobe_loss
        return output
