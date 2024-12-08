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
from torchvision.ops import sigmoid_focal_loss

from ..model import BACKBONES, ViT
from .jepa import JEPAWithProbe


class JEPAChexpert(JEPAWithProbe):

    def create_probe_head(self) -> nn.Module:
        return nn.Sequential(nn.LayerNorm(self.backbone.dim), nn.Dropout(0.1), nn.Linear(self.backbone.dim, 1))

    def create_metrics(self, state: State) -> tm.MetricCollection:
        metrics = super().create_metrics(state)
        metrics.add_metrics({"acc": tm.Accuracy(task="binary")})
        metrics.add_metrics({"auroc": tm.AUROC(task="binary")})
        return metrics

    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        # Forward pass of linear probe using target features
        features = self.get_probe_features_from_output(output)
        assert self.linear_probe is not None
        N = features.shape[0]
        linprobe_logits = self.linear_probe(features.mean(1).view(N, -1)).view(N)
        assert linprobe_logits.requires_grad or not self.training

        # Build ground truth and compute loss
        linprobe_gt = batch["finding"]
        mask = linprobe_gt != -1
        if mask.any():
            linprobe_loss = F.binary_cross_entropy_with_logits(linprobe_logits[mask], linprobe_gt[mask].float())
        else:
            linprobe_loss = linprobe_gt.new_tensor(0.0, requires_grad=True)

        # Logits -> probs
        with torch.no_grad():
            linprobe_probs = torch.sigmoid(linprobe_logits)

        # Compute metrics
        with torch.no_grad():
            if mask.any():
                for name, metric in (metrics or {}).items():
                    metric.update(linprobe_probs[mask], linprobe_gt[mask])

        output = copy(output)
        output["log"]["loss_linprobe"] = linprobe_loss
        return output


class ChexpertTask(Task):
    """

    Args:
        backbone: Name of the backbone to use for the task.
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
        focal_loss: bool = False,
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
        dim = self.backbone.dim
        self.finding_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b c () () -> b c"),
            nn.LayerNorm(dim),
            nn.Dropout(0.1),
            nn.Linear(dim, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none") if not focal_loss else sigmoid_focal_loss
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, state: State, **kwargs) -> tm.MetricCollection:
        return tm.MetricCollection(
            {
                "auroc": tm.AUROC(task="binary"),
                "acc": tm.Accuracy(task="binary"),
            }
        )

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.backbone(x)
        cls = self.finding_head(x)
        return {"finding": cls.view(-1, 1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        x = batch["img"]
        y = batch["finding"]

        # forward pass
        result = self(x)

        # compute loss
        pred_logits = cast(Tensor, result["finding"].flatten())

        # Build ground truth and compute loss
        mask = y != -1
        if mask.any():
            loss = F.binary_cross_entropy_with_logits(pred_logits[mask], y[mask].float())
        else:
            loss = y.new_tensor(0.0, requires_grad=True)

        with torch.no_grad():
            pred = pred_logits.sigmoid()

        # log metrics
        with torch.no_grad():
            for metric in (metrics or {}).values():
                _pred = pred[mask]
                _label = y[mask].long()
                metric.update(_pred, _label)

        output = {
            "finding_score": pred.detach(),
            "log": {
                "loss_finding": loss,
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        result = self(batch["img"])
        pred_logits = cast(Tensor, result["finding"].flatten())
        return {
            "finding_score": pred_logits.sigmoid(),
        }
