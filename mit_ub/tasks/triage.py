import math
from typing import Any, Dict, Final, Optional, Set, cast

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from gpvit import GPViT
from gpvit.layers import MLPMixerPooling
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss

# from gpvit.train import BACKBONES
from ..model import BACKBONES


UNKNOWN_INT: Final = -1


def _check_weights_sum_to_one(weights: Tensor, tol: float = 0.01) -> None:
    total = weights.sum().item()
    assert math.isclose(total, 1, abs_tol=tol), f"Weights should sum to 1, got {total}"


def _str_or_bool(value: Any) -> bool:
    return value.lower() == "true" if isinstance(value, str) else bool(value)


class TriageTask(Task):
    """
    Implements a generic triage task for images.

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

        self.backbone = cast(GPViT, self.prepare_backbone(backbone))
        dim = self.backbone.dim
        group_tokens = self.backbone.num_group_tokens
        self.triage_head = nn.Sequential(
            MLPMixerPooling(dim, group_tokens, group_tokens, dim, dropout=0.1),
            nn.Linear(dim, 1),
        )
        self.criterion = nn.BCEWithLogitsLoss(reduction="none") if not focal_loss else sigmoid_focal_loss

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, *args, **kwargs) -> tm.MetricCollection:
        return tm.MetricCollection(
            {
                "auroc": tm.AUROC(task="binary"),
                "acc": tm.Accuracy(task="binary"),
                "pos-acc": tm.Accuracy(task="binary", ignore_index=0),
            }
        )

    @torch.no_grad()
    def get_crop_mask(self, batch: Dict[str, Any]) -> Tensor:
        """
        Determines which examples in the batch are crops.

        Args:
            batch: The batch of examples.

        Returns:
            Mask of shape :math:`(B,)`, where :math:`B` is the batch size.
        """
        return torch.tensor(
            ["_crop" in str(path) for path in batch["path"]],
            device=self.device,
        )

    @torch.no_grad()
    def get_malignant_trace_mask(self, batch: Dict[str, Any]) -> Tensor:
        """
        Determines which examples in the batch have malignant traces.

        Args:
            batch: The batch of examples.

        Returns:
            Mask of shape :math:`(B,)`, where :math:`B` is the batch size.
        """
        return torch.tensor(
            [
                any(trait == "malignant" for trait in (example_boxes or {}).get("trait", []))
                for example_boxes in batch["bounding_boxes"]
            ],
            device=self.device,
        )

    @torch.no_grad()
    def sanitize_boxes(self, batch: Dict[str, Any], min_size: float = 16.0) -> Dict[str, Any]:
        for example in batch["bounding_boxes"]:
            box_metadata = example or {}
            boxes = box_metadata.get("boxes", None)
            trait = box_metadata.get("trait", None)
            types = box_metadata.get("types", None)
            if boxes is not None:
                assert trait is not None
                assert types is not None

                # Determine the area of the boxes
                x1, y1, x2, y2 = boxes.unbind(dim=-1)
                width = x2 - x1
                height = y2 - y1
                valid_size = (width >= min_size) & (height >= min_size)

                # Filter out boxes that are too small
                boxes = boxes[valid_size]
                trait = [t for i, t in enumerate(trait) if valid_size[i]]
                types = [t for i, t in enumerate(types) if valid_size[i]]

                if boxes.numel():
                    box_metadata["boxes"] = boxes
                    box_metadata["trait"] = trait
                    box_metadata["types"] = types
                else:
                    box_metadata.pop("boxes", None)
                    box_metadata.pop("trait", None)
                    box_metadata.pop("types", None)

        return batch

    @torch.no_grad()
    def get_tensor_label(self, batch: Dict[str, Any], crop_adjust: bool = True) -> Tensor:
        """
        Extracts the label from the batch and converts it into a tensor.

        The label is extracted from the "malignant" field of each example in the batch.
        If the field is not present or the example is None, the label is considered unknown.

        The labels are then converted into a tensor of float32, with 1 representing a positive label,
        -1 representing an unknown label, and 0 representing a negative label.

        Args:
            batch: The batch of examples.
            crop_adjust: If True, adjust the label for crops.

        Returns:
            The tensor of labels of shape :math:`(B,)`, where :math:`B` is the batch size.
        """
        # Get the original label
        label = torch.tensor(
            [_str_or_bool((example or {}).get("malignant", None)) for example in batch["annotation"]],
            dtype=torch.float32,
            device=self.device,
        )
        label[self.get_unknown_label_mask(batch, crop_adjust=False)] = UNKNOWN_INT

        # Apply adjustment to crops
        if crop_adjust:
            # Determine which examples are crops and which examples have malignant traces
            is_crop = self.get_crop_mask(batch)
            has_malignant_trace = self.get_malignant_trace_mask(batch)
            assert is_crop.shape == has_malignant_trace.shape == label.shape

            # For examples that are crops we need to update the label.
            # The following update is applied:
            # - If the original label is positive and no malignant bounding box is present, the crop label is updated to unknown.
            # Other cases are left unchanged.
            label[is_crop & (label == 1) & (~has_malignant_trace)] = UNKNOWN_INT
            assert not (
                is_crop & (label == 1) & (~has_malignant_trace)
            ).any(), "Crops with no malignant trace should have unknown label"

        return label.float()

    @torch.no_grad()
    def get_unknown_label_mask(self, batch: Dict[str, Any], crop_adjust: bool = True) -> Tensor:
        r"""Returns a boolean mask indicating which examples have an unknown label.

        Args:
            batch: The batch of examples.
            crop_adjust: If True, adjust the mask for crops.

        Returns:
            The boolean mask indicating which examples have an unknown label, with shape :math:`(B,)`,
            where :math:`B` is the batch size.
        """
        mask = torch.tensor(
            [(example or {}).get("malignant", None) is None for example in batch["annotation"]],
            device=self.device,
        )

        if crop_adjust:
            label = self.get_tensor_label(batch)
            mask = torch.where(label == UNKNOWN_INT, label, mask)

        return mask.bool()

    @torch.no_grad()
    def compute_loss_weight(self, batch: Dict[str, Any]) -> Tensor:
        """
        Compute the loss weight for each example in the batch.

        The loss weight is computed as the inverse of the number of known labels in the batch.
        For examples with unknown labels, the loss weight is set to 0.

        Args:
            batch: The batch of examples.

        Returns:
            The loss weight for each example in the batch.
        """
        unknown_label = self.get_unknown_label_mask(batch)
        num_known = (~unknown_label).sum().item()
        weight = torch.full_like(unknown_label, fill_value=1 / num_known, dtype=torch.float32)
        weight[unknown_label] = 0
        _check_weights_sum_to_one(weight)
        return weight

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x, groups = self.backbone(x)
        cls = self.triage_head(groups)
        return {"triage": cls.view(-1, 1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        batch = self.sanitize_boxes(batch)
        x = batch["img"]
        y = self.get_tensor_label(batch)

        # forward pass
        result = self(x)

        # compute loss
        pred_logits = cast(Tensor, result["triage"].flatten())
        weight = self.compute_loss_weight(batch).flatten()
        assert (weight[y == UNKNOWN_INT] == 0).all(), "Unknown labels should have weight 0"
        loss = cast(Tensor, (self.criterion(pred_logits, y) * weight).sum())

        with torch.no_grad():
            pred = pred_logits.sigmoid()

        # log metrics
        with torch.no_grad():
            if metrics is not None:
                metrics.update(pred[weight > 0], y[weight > 0])

        output = {
            "triage_score": pred.detach(),
            "train_label": [self.int_label_to_str(int(label)) for label in y],
            "log": {
                "loss_triage": loss,
            },
        }

        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        result = self(batch["img"])
        pred_logits = cast(Tensor, result["triage"].flatten())
        return {
            "triage_score": pred_logits.sigmoid(),
        }

    @classmethod
    def int_label_to_str(cls, label: int) -> str:
        if label == 1:
            return "malignant"
        elif label == 0:
            return "benign"
        else:
            return "unknown"


class BreastTriageTask(TriageTask):
    def __init__(
        self,
        backbone: str,
        focal_loss: bool = False,
        pos_weight: float = 1.0,
        standard_view_weight: float = 1.0,
        implant_weight: float = 0.5,
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
            focal_loss,
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
        self.pos_weight = pos_weight
        self.standard_view_weight = standard_view_weight
        self.implant_weight = implant_weight

    @torch.no_grad()
    def get_standard_view_mask(self, batch: Dict[str, Any]) -> Tensor:
        r"""Returns a boolean mask of shape :math:`(B,)` indicating which examples are standard mammogram views."""
        return torch.tensor(
            [bool((example or {}).get("standard_view", None)) for example in batch["manifest"]],
            device=self.device,
        )

    @torch.no_grad()
    def get_implant_mask(self, batch: Dict[str, Any]) -> Tensor:
        r"""Returns a boolean mask of shape :math:`(B,)` indicating which examples implant views."""
        implant = torch.tensor(
            [bool((example or {}).get("implant", None)) for example in batch["manifest"]],
            device=self.device,
        )
        implant_displaced = torch.tensor(
            [bool((example or {}).get("implant_displaced", None)) for example in batch["manifest"]],
            device=self.device,
        )
        return implant.logical_and_(implant_displaced.logical_not_())

    @torch.no_grad()
    def compute_loss_weight(self, batch: Dict[str, Any]) -> Tensor:
        """
        Compute the loss weight for each example in the batch.

        The baseline loss weight is computed as the inverse of the number of known labels in the batch.
        For examples with unknown labels, the loss weight is set to 0. The loss is then scaled by the

        Args:
            batch: The batch of examples.

        Returns:
            The loss weight for each example in the batch.
        """
        baseline_weight = super().compute_loss_weight(batch)

        # Apply standard view weight
        standard_view_mask = self.get_standard_view_mask(batch)
        baseline_weight[standard_view_mask] *= self.standard_view_weight

        # Apply implant weight
        implant_mask = self.get_implant_mask(batch)
        baseline_weight[implant_mask] *= self.implant_weight

        # Apply pos weight
        pos_mask = self.get_tensor_label(batch) == 1
        baseline_weight[pos_mask] *= self.pos_weight

        # Normalize weights
        baseline_weight /= baseline_weight.sum()
        _check_weights_sum_to_one(baseline_weight)

        return baseline_weight
