from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor

from ..data import is_mixed, is_mixed_with_unknown, mixup, mixup_dense_label, sample_mixup_parameters
from ..model import AdaptiveViTConfig, AnyModelConfig, ViTConfig
from ..model.layers.pool import PoolType
from .distillation import DistillationConfig, DistillationWithProbe
from .jepa import JEPAConfig, JEPAWithProbe


def is_valid_categorical_label(label: Tensor) -> Tensor:
    return label >= 0


def is_valid_binary_label(label: Tensor) -> Tensor:
    return (label == 0).logical_or_(label == 1)


def categorical_loss(
    logits: Tensor,
    label: Tensor,
    num_classes: int,
    mixup_weight: Tensor | None = None,
) -> Tensor:
    # Filter valid labels
    mask = is_valid_categorical_label(label)

    # Apply mixup to labels if needed
    if mixup_weight is not None:
        label = mixup_dense_label(label, mixup_weight, num_classes=num_classes)
        mask = mask & ~is_mixed_with_unknown(mixup_weight, mask)

    if not mask.any():
        return logits.new_zeros(1)

    loss = F.cross_entropy(logits[mask], label[mask])
    assert loss >= 0.0, f"Loss is negative: {loss}"
    return loss


def binary_loss(
    logits: Tensor,
    label: Tensor,
    mixup_weight: Tensor | None = None,
) -> Tensor:
    # Filter valid labels
    mask = is_valid_binary_label(label)

    # Apply mixup to labels if needed
    if mixup_weight is not None:
        label = mixup(label, mixup_weight)
        mask = mask & ~is_mixed_with_unknown(mixup_weight, mask)

    if not mask.any():
        return logits.new_zeros(1)

    loss = F.binary_cross_entropy_with_logits(logits[mask].flatten(), label[mask].flatten().float())
    assert loss >= 0.0, f"Loss is negative: {loss}"
    return loss


@torch.no_grad()
def update_metrics(
    metrics: tm.MetricCollection,
    pred: Tensor,
    label: Tensor,
    loss: Tensor,
    mixup_weight: Tensor | None,
    is_binary: bool,
    metric_names: Sequence[str] = ("acc", "auroc"),
) -> None:
    # Filter valid labels without mixup for metric computation
    mask = is_valid_binary_label(label) if is_binary else is_valid_categorical_label(label)
    mask = mask & ~is_mixed(mixup_weight) if mixup_weight is not None else mask
    if not mask.any():
        return

    _pred = pred[mask]
    _y = label[mask]
    for name, metric in metrics.items():
        if name.endswith("_loss") and name != "jepa_loss":
            metric.update(loss)
        elif name in metric_names:
            metric.update(_pred.view_as(_y), _y)


def step_classification_from_features(
    features: Tensor,
    label: Tensor,
    probe: nn.Module,
    config: "ClassificationConfig",
    mixup_weight: Tensor | None = None,
    metrics: tm.MetricCollection | None = None,
    metric_names: Sequence[str] = ("acc", "macro_acc", "auroc"),
) -> Dict[str, Tensor]:
    # Forward pass
    N = features.shape[0]
    pred_logits = probe(features).view(N, -1)
    assert pred_logits.requires_grad or not probe.training

    # compute loss
    if config.is_binary:
        loss = binary_loss(pred_logits, label, mixup_weight)
    else:
        loss = categorical_loss(pred_logits, label, config.num_classes, mixup_weight)
    assert loss >= 0.0, f"Loss is negative: {loss}"

    # logits -> predictions
    with torch.no_grad():
        if config.is_binary:
            pred = pred_logits.sigmoid()
        else:
            pred = pred_logits.argmax(dim=1)

    # log metrics
    if metrics is not None:
        update_metrics(metrics, pred, label, loss, mixup_weight, config.is_binary, metric_names)

    return {
        "loss": loss,
        "pred_logits": pred_logits,
        "pred": pred,
    }


def create_metrics(config: "ClassificationConfig") -> tm.MetricCollection:
    if config.is_binary:
        metrics = tm.MetricCollection(
            {
                "acc": tm.Accuracy(task="binary"),
                "macro_acc": tm.Accuracy(task="binary", average="macro"),
                "auroc": tm.AUROC(task="binary"),
                "bce_loss": tm.MeanMetric(),
            }
        )
    else:
        metrics = tm.MetricCollection(
            {
                "acc": tm.Accuracy(task="multiclass", num_classes=config.num_classes),
                "macro_acc": tm.Accuracy(task="multiclass", average="macro", num_classes=config.num_classes),
                "ce_loss": tm.MeanMetric(),
            }
        )
    return metrics


@dataclass
class ClassificationConfig:
    """
    Configuration for classification related hyperparameters.

    Args:
        num_classes: Number of classes.
        mixup_alpha: Alpha parameter for the Beta distribution used to sample the mixup weight.
        mixup_prob: Probability of applying mixup to the input and target.
        freeze_backbone: If True, the backbone is frozen during training.
        pool_type: Type of pooling to use.
        label_key: Key in the batch dictionary that contains the label.
        mlp_tower: If True, use a MLP tower instead of a simple linear layer.
        tower_input_norm: If True, apply input normalization to the tower.
            Input normalization should not be necessary for backbones that already have an output normalization layer.
    """

    num_classes: int
    mixup_alpha: float = 1.0
    mixup_prob: float = 0.2
    freeze_backbone: bool = False
    # TODO: jsonargparse can't handle the strenum it seems
    pool_type: str | PoolType | None = None
    label_key: str = "label"
    mlp_tower: bool = False
    tower_input_norm: bool = False

    def __post_init__(self) -> None:
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not 0 < self.mixup_alpha:
            raise ValueError("mixup_alpha must be positive")
        if not 0 <= self.mixup_prob <= 1:
            raise ValueError("mixup_prob must be in the range [0, 1]")
        if isinstance(self.pool_type, str):
            self.pool_type = PoolType(self.pool_type)

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2


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
        backbone_config: AnyModelConfig,
        classification_config: ClassificationConfig,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        parameter_groups: List[Dict[str, Any]] = [],
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
            parameter_groups,
        )
        self.config = classification_config

        self.backbone = backbone_config.instantiate()
        if self.config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.config.dim

        self.classification_head = self.backbone.create_head(
            out_dim=self.config.num_classes if not self.config.is_binary else 1,
            pool_type=cast(PoolType | None, self.config.pool_type),
            use_mlp=self.config.mlp_tower,
            input_norm=self.config.tower_input_norm,
        )
        self.save_hyperparameters()

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        return create_metrics(self.config)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        with torch.set_grad_enabled(not self.config.freeze_backbone):
            x = self.backbone(x, reshape=False)
        cls = self.classification_head(x)
        return {"pred": cls.view(-1, 1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        # get inputs
        x = batch["img"]
        y = batch[self.config.label_key].long()
        N = y.shape[0]

        # mixup input
        if self.training and self.config.mixup_prob > 0:
            mixup_weight = sample_mixup_parameters(N, self.config.mixup_prob, self.config.mixup_alpha, device=x.device)
            x = mixup(x, mixup_weight)
        else:
            mixup_weight = None

        # forward backbone
        with torch.set_grad_enabled(not self.config.freeze_backbone and self.training):
            features, cls_token = self.backbone(x)

        # step from features
        output = step_classification_from_features(
            cls_token,
            y,
            self.classification_head,
            self.config,
            mixup_weight,
            metrics,
        )

        output = {
            "log": {
                "loss_classification": output["loss"],
            },
            "pred_logits": output["pred_logits"],
            "pred": output["pred"],
            "mixup_weight": mixup_weight,
        }

        return output


class JEPAWithClassification(JEPAWithProbe):
    def __init__(
        self,
        backbone_config: ViTConfig | AdaptiveViTConfig,
        classification_config: ClassificationConfig,
        jepa_config: JEPAConfig = JEPAConfig(),
        probe_key: str = "target_cls_token",
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        parameter_groups: List[Dict[str, Any]] = [],
    ):
        self.classification_config = classification_config
        super().__init__(
            backbone_config,
            jepa_config,
            probe_key,
            optimizer_init,
            lr_scheduler_init,
            lr_interval,
            lr_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            parameter_groups,
        )

    def create_probe_head(self) -> nn.Module:
        return self.backbone.create_head(
            out_dim=self.classification_config.num_classes if not self.classification_config.is_binary else 1,
            pool_type=cast(PoolType | None, self.classification_config.pool_type),
            use_mlp=self.classification_config.mlp_tower,
            input_norm=self.classification_config.tower_input_norm,
        )

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        metrics = super().create_metrics(*args, **kwargs)
        metrics.add_metrics(dict(create_metrics(self.classification_config)))  # type: ignore
        return metrics

    @torch.no_grad()
    def create_gt(self, batch: Dict[str, Any]) -> Tensor:
        y = batch[self.classification_config.label_key].long()
        return y

    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        # Get inputs
        features = self.get_probe_features_from_output(output)
        mixup_weight = output.get("mixup_weight", None)
        N = features.shape[0]
        y = self.create_gt(batch).view(N)

        # step from features
        probe_output = step_classification_from_features(
            features,
            y,
            self.linear_probe,
            self.classification_config,
            mixup_weight,
            metrics,
        )

        output = copy(output)
        output["log"]["loss_linprobe"] = probe_output["loss"]
        return output


class DistillationWithClassification(DistillationWithProbe):
    def __init__(
        self,
        backbone_config: AnyModelConfig,
        teacher_config: AnyModelConfig,
        teacher_checkpoint: Path,
        classification_config: ClassificationConfig,
        distillation_config: DistillationConfig = DistillationConfig(),
        probe_key: str = "pred",
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        parameter_groups: List[Dict[str, Any]] = [],
    ):
        self.classification_config = classification_config
        super().__init__(
            backbone_config,
            teacher_config,
            teacher_checkpoint,
            distillation_config,
            probe_key,
            optimizer_init,
            lr_scheduler_init,
            lr_interval,
            lr_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            parameter_groups,
        )

    def create_probe_head(self) -> nn.Module:
        return self.backbone.create_head(
            out_dim=self.classification_config.num_classes if not self.classification_config.is_binary else 1,
            pool_type=cast(PoolType | None, self.classification_config.pool_type),
            use_mlp=self.classification_config.mlp_tower,
            input_norm=self.classification_config.tower_input_norm,
        )

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        metrics = super().create_metrics(*args, **kwargs)
        metrics.add_metrics(dict(create_metrics(self.classification_config)))  # type: ignore
        return metrics

    @torch.no_grad()
    def create_gt(self, batch: Dict[str, Any]) -> Tensor:
        y = batch[self.classification_config.label_key].long()
        return y

    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        # Get inputs
        features = self.get_probe_features_from_output(output)
        mixup_weight = output.get("mixup_weight", None)
        N = features.shape[0]
        y = self.create_gt(batch).view(N)

        # step from features
        probe_output = step_classification_from_features(
            features,
            y,
            self.linear_probe,
            self.classification_config,
            mixup_weight,
            metrics,
        )

        output = copy(output)
        output["log"]["loss_linprobe"] = probe_output["loss"]
        return output
