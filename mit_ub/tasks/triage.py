import hashlib
import math
import sys
from pathlib import Path
from typing import Any, Dict, Final, Iterable, List, Optional, Set, Tuple, cast

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from dicom_utils.container.collection import iterate_input_path
from dicom_utils.dicom import has_dicm_prefix
from dicom_utils.volume import ReduceVolume
from einops.layers.torch import Rearrange
from jsonargparse import ActionConfigFile, ArgumentParser, Namespace
from torch import Tensor
from torch_dicom.inference.lightning import LightningInferencePipeline
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC
from torchmetrics.classification import BinarySensitivityAtSpecificity as BinarySensitivityAtSpecificityBase
from torchmetrics.classification import BinarySpecificityAtSensitivity as BinarySpecificityAtSensitivityBase
from torchmetrics.functional.classification.accuracy import binary_accuracy
from torchmetrics.functional.classification.auroc import _binary_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat
from torchvision.ops import sigmoid_focal_loss
from tqdm import tqdm

from ..model import BACKBONES, ViT


UNKNOWN_INT: Final = -1
MAX_HASH: Final = torch.iinfo(torch.long).max


class BinarySensitivityAtSpecificity(BinarySensitivityAtSpecificityBase):

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs)[0]


class BinarySpecificityAtSensitivity(BinarySpecificityAtSensitivityBase):

    def compute(self, *args, **kwargs):
        return super().compute(*args, **kwargs)[0]


def _check_weights_sum_to_one(weights: Tensor, tol: float = 0.01) -> None:
    total = weights.sum().item()
    assert math.isclose(total, 1, abs_tol=tol), f"Weights should sum to 1, got {total}"


def _str_or_bool(value: Any) -> bool:
    return value.lower() == "true" if isinstance(value, str) else bool(value)


def _get_case_metric_state(studies: Tensor, preds: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
    # group by study using max reduction
    seen_studies, study_indices = torch.unique(studies, return_inverse=True)
    study_preds = preds.new_zeros(len(seen_studies))
    study_target = target.new_zeros(len(seen_studies))
    study_preds.scatter_reduce_(0, study_indices, preds, reduce="max")
    study_target.scatter_reduce_(0, study_indices, target, reduce="max")
    return (study_preds, study_target)


def _hash_study_uids(study_uids: Iterable[str], proto: Optional[Tensor] = None) -> Tensor:
    # We don't use the builtin hash, as it seems non-deterministic across processes
    # and will give incorrect results in multi-GPU mode.
    # We also take care to ensure we don't overflow torch.long.
    hashes = [int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % MAX_HASH for s in study_uids]
    return proto.new_tensor(hashes, dtype=torch.long) if proto is not None else torch.tensor(hashes, dtype=torch.long)


class CaseAUROC(BinaryAUROC):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("studies", default=[], dist_reduce_fx="cat")

    def update(self, study_uids: List[str], preds: Tensor, target: Tensor) -> None:
        assert len(preds) == len(target) == len(study_uids)
        cast(List, self.preds).append(preds)
        cast(List, self.target).append(target)

        # NOTE: we need to store states as tensors, so we hash the study uids.
        study_uid_hashes = _hash_study_uids(study_uids, proto=target)
        cast(List, self.studies).append(study_uid_hashes)

    def compute(self) -> Tensor:
        # join all the batches
        preds = dim_zero_cat(cast(List, self.preds))
        target = dim_zero_cat(cast(List, self.target))
        studies = dim_zero_cat(cast(List, self.studies))
        assert len(preds) == len(target) == len(studies)

        # group by study using max reduction and compute
        state = _get_case_metric_state(studies, preds, target)
        result = _binary_auroc_compute(state, self.thresholds, self.max_fpr)
        assert isinstance(result, Tensor)
        return result


class CaseAccuracy(BinaryAccuracy):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("studies", default=[], dist_reduce_fx="cat")

    def update(self, study_uids: List[str], preds: Tensor, target: Tensor) -> None:
        assert len(preds) == len(target) == len(study_uids)
        cast(List, self.preds).append(preds)
        cast(List, self.target).append(target)
        study_uid_hashes = _hash_study_uids(study_uids, proto=target)
        cast(List, self.studies).append(study_uid_hashes)

    def compute(self) -> Tensor:
        # join all the batches
        preds = dim_zero_cat(cast(List, self.preds))
        target = dim_zero_cat(cast(List, self.target))
        studies = dim_zero_cat(cast(List, self.studies))
        assert len(preds) == len(target) == len(studies)

        # group by study using max reduction and compute
        preds, target = _get_case_metric_state(studies, preds, target)
        return binary_accuracy(
            preds, target, self.threshold, cast(Any, self.multidim_average), self.ignore_index, self.validate_args
        )


class TriageTask(Task):
    """
    Implements a generic triage task for images.

    Args:
        backbone: Name of the backbone to use for the task.
        crop_adjust: If True, adjust the global label for crops using the presence of malignant traces.
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
        crop_adjust: bool = False,
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
        self.crop_adjust = crop_adjust

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        dim = self.backbone.dim
        self.triage_head = nn.Sequential(
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
                "spec-at-sens=86_9": BinarySpecificityAtSensitivity(min_sensitivity=0.869),
                "sens-at-spec=88_9": BinarySensitivityAtSpecificity(min_specificity=0.889),
                "auroc_case": CaseAUROC(dist_sync_on_step=state.mode != Mode.TRAIN),
                "acc_case": CaseAccuracy(dist_sync_on_step=state.mode != Mode.TRAIN),
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
        x = self.backbone(x)
        cls = self.triage_head(x)
        return {"triage": cls.view(-1, 1)}

    def step(
        self, batch: Any, batch_idx: int, state: State, metrics: Optional[tm.MetricCollection] = None
    ) -> Dict[str, Any]:
        batch = self.sanitize_boxes(batch)
        x = batch["img"]
        y = self.get_tensor_label(batch, crop_adjust=self.crop_adjust)

        # forward pass
        result = self(x)

        # compute loss
        [x.stem for x in batch["path"]]
        pred_logits = cast(Tensor, result["triage"].flatten())
        weight = self.compute_loss_weight(batch).flatten()
        assert (weight[y == UNKNOWN_INT] == 0).all(), "Unknown labels should have weight 0"
        loss = cast(Tensor, (self.criterion(pred_logits, y) * weight).sum())

        with torch.no_grad():
            pred = pred_logits.sigmoid()

        # log metrics
        with torch.no_grad():
            for metric in (metrics or {}).values():
                is_valid = weight > 0
                _pred = pred[is_valid]
                _label = y[is_valid].long()
                if isinstance(metric, (CaseAccuracy, CaseAUROC)):
                    study_uids = [
                        example["StudyInstanceUID"] for i, example in enumerate(batch["manifest"]) if is_valid[i]
                    ]
                    metric.update(study_uids, _pred, _label)
                else:
                    metric.update(_pred, _label)

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


class BreastTriage(TriageTask):
    def __init__(
        self,
        backbone: str,
        crop_adjust: bool = False,
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
            crop_adjust,
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


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="mit-ub-breast-triage", description="Triage mammograms for breast cancer.")
    parser.add_argument("--config", action=ActionConfigFile)

    parser.add_argument(
        "target",
        type=Path,
        help="Target DICOMs or images to process. Can be a file, directory, or text file containing paths.",
    )
    BreastTriage.add_args_to_parser(parser, skip={"weights"}, subclass=True)

    parser.add_argument(
        "-e",
        "--enumerate",
        default=False,
        action="store_true",
        help="Enumerate the input files. May take time but enables a progress bar.",
    )
    parser.add_argument(
        "-i",
        "--image-size",
        type=int,
        nargs=2,
        help="Image size for inference, or omit to infer from checkopint.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=1, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading.")
    parser.add_argument(
        "--uid-from-stem",
        default=False,
        action="store_true",
        help="Use the file stem as the SOPInstanceUID when it is not available in the metadata.",
    )

    cfg = parser.parse_args()
    cfg = parser.instantiate_classes(cfg)
    cfg = BreastTriage.on_after_parse(cfg)
    return cfg


def main(args: Namespace) -> None:
    # Prepare model
    # TODO: This is coupled to breast triage for now, but should support arbitrary triage
    model: BreastTriage = args.model
    model = model.to(args.device)
    checkpoint: Path = Path(args.checkpoint)
    model.checkpoint = checkpoint.absolute()
    model.setup()

    def is_dicom(path: Path) -> bool:
        # Try to avoid slow opening the file if possible by checking the suffix
        if path.suffix.lower() == ".dcm":
            return True
        return has_dicm_prefix(path)

    # Filter paths for processing
    dicom_paths = filter(is_dicom, iterate_input_path(args.target, ignore_missing=True))
    image_paths = filter(
        lambda p: p.suffix.lower() in (".png", ".tiff"), iterate_input_path(args.target, ignore_missing=True)
    )

    # Determine image size for inference
    if hasattr(model, "img_size"):
        img_size = model.img_size
    elif args.image_size:
        img_size = args.image_size
    else:
        raise ValueError("Image size for inference not specified in CLI or checkpoint.")

    pipeline = LightningInferencePipeline(
        dicom_paths=dicom_paths,
        image_paths=image_paths,
        device=args.device,
        batch_size=args.batch_size,
        dataloader_kwargs={"num_workers": args.num_workers},
        volume_handler=ReduceVolume(skip_edge_frames=5),
        models=[model],
        transform=LightningInferencePipeline.create_default_transform(img_size=img_size),
        enumerate_inputs=args.enumerate,
    )

    header = None
    for example, pred in pipeline:
        # Example keys are different for DICOM and PNG. We also support falling back to the path for the SOPInstanceUID
        path = example["record"].path if "record" in example else example["path"]
        sop = example["record"].SOPInstanceUID if "record" in example else path.stem if args.uid_from_stem else ""
        result = {
            "path": path,
            "sop_instance_uid": sop,
            "triage_score": float(pred["triage_score"].item()),
        }
        with tqdm.external_write_mode():
            if header is None:
                header = ",".join(result.keys())
                print(header)
            print(",".join(str(v) for v in result.values()))
            sys.stdout.flush()
            sys.stderr.flush()


def entrypoint():
    main(parse_args())


if __name__ == "__main__":
    entrypoint()
