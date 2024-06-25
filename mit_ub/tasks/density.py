import math
from typing import Any, Dict, Final, Optional, Set, cast, TypeVar, Union, Tuple
import numpy as np

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


UNKNOWN_INT: Final = -1

T = TypeVar("T", bound=Union[float, Tensor, np.ndarray])
LETTER_TO_INT: Final = {
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
}
INT_TO_LETTER: Final = {v: k for k, v in LETTER_TO_INT.items()}
UNKNOWN: Final = -1

def get_bool_from_dict(d: Dict[str, Any], key: str) -> bool:
    return str(d.get(key, "false")).strip().lower() == "true"


@torch.no_grad()
def score_to_density(score: T) -> T:
    if isinstance(score, float):
        return cast(T, round((score * 3) + 1))
    else:
        result = (score * 3) + 1
        assert isinstance(result, (Tensor, np.ndarray))
        result = result.round()
        return cast(T, result)


class Density(Task):
    """
    Implements a generic density task for images.

    Args:
        backbone: Name of the backbone to use for the task.
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
        detach: bool = False,
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
        self.detach = detach

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        dim = self.backbone.dim
        self.density_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange("b c () () -> b c"),
            nn.Linear(dim, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.save_hyperparameters()

        if detach:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        return tm.MetricCollection(
            {
                "auroc": tm.AUROC(task="binary"),
                "bin-acc": tm.Accuracy(task="binary"),
                "macro-acc": tm.Accuracy(task="multiclass", num_classes=4, average="macro"),
            }
        )

    @classmethod
    def is_high_quality_view(cls, manifest: Dict[str, Any]) -> bool:
        r"""Checks if the mammogram is of high quality based on the manifest.

        High quality views are loosely defined as views in which the breast density
        should be accurately assessable.
        A view is considered high quality if the following conditions are met:
            * Not a for-processing view
            * Not a spot compression or magnification view
            * Not a specimen or biopsy view
            * If the patient has implants, the view must be implant displaced

        Returns:
            True if the view is of high quality, otherwise False.
        """
        is_disqualified_view = any(
            (
                get_bool_from_dict(manifest, "spot_compression"),
                get_bool_from_dict(manifest, "magnification"),
                get_bool_from_dict(manifest, "specimen"),
                get_bool_from_dict(manifest, "stereo"),
                (get_bool_from_dict(manifest, "implant") and not get_bool_from_dict(manifest, "implant_displaced")),
                #get_bool_from_dict(manifest, "for_processing"),
            )
        )
        return not is_disqualified_view

    @torch.no_grad()
    def create_target(self, y: Tensor) -> Tensor:
        return y.float().sub(1).div(3)

    @torch.no_grad()
    def postprocess(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        p = x.sigmoid()
        density = score_to_density(p).long()
        return density, p

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self.detach:
            with torch.no_grad():
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        cls = self.density_head(x)
        return {"density": cls.view(-1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        output: Dict[str, Any] = {"log": {}}
        x = batch["img"]

        # Read inputs
        y = torch.tensor(
            [LETTER_TO_INT.get(ann.get("density", "unknown").lower(), UNKNOWN) for ann in batch["annotation"]],
            device=self.device,
        )
        is_high_quality_view = torch.tensor(
            [self.is_high_quality_view(manifest) for manifest in batch["manifest"]],
            device=self.device,
        )

        # Prepare target and loss weight
        target = self.create_target(y)
        target_is_valid = (y != UNKNOWN) & is_high_quality_view

        # forward pass
        pred_logits: Tensor = self(x)["density"]

        # compute loss
        loss = F.binary_cross_entropy_with_logits(pred_logits, target, weight=target_is_valid.float())
        output["log"]["loss_density"] = loss

        # Apply post-processing and log predictions
        with torch.no_grad():
            # Raw logit and target
            output["logit"] = pred_logits
            output["target"] = target

            # Post-processed predictions and original target
            pred, p = self.postprocess(pred_logits)
            output["pred"] = p
            output["pred_cls"] = pred
            output["true"] = y

        if metrics is not None and target_is_valid.any():
            # we predict 1-4, but the metrics expect 0-3
            categorical_pred = pred.sub(1)[target_is_valid]
            categorical_true = y.sub(1)[target_is_valid]
            cast(tm.Metric, metrics["macro-acc"]).update(categorical_pred, categorical_true)

            # binary metrics
            binary_pred = output["logit"].sigmoid()[target_is_valid]
            binary_true = output["target"].round()[target_is_valid]
            assert 0 <= binary_true.min() and binary_true.max() <= 1, f"invalid binary_true: {binary_true}"
            assert binary_pred.shape == binary_true.shape == (target_is_valid.sum(),)
            cast(tm.Metric, metrics["bin-acc"]).update(binary_pred, binary_true)
            cast(tm.Metric, metrics["auroc"]).update(binary_pred, binary_true)

        return output