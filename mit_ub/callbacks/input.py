from typing import Any, ClassVar, Dict, Iterator, List, Optional, Tuple, Type, TypedDict, Union, cast

import pytorch_lightning as pl
import torch
import wandb
from deep_helpers.callbacks import LoggerIntegration, QueuedLoggingCallback
from deep_helpers.tasks import Task
from einops import rearrange, repeat
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor


class TargetDict(TypedDict):
    input: Tensor
    patient: Optional[str]
    label: Optional[str]
    bounding_boxes: Optional[Dict[str, Any]]


def is_volume(x: Tensor) -> bool:
    return x.shape[-3] > 1


def iterate_dict(d: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    length = len(next(iter(d.values())))
    for i in range(length):
        yield {k: v[i] for k, v in d.items()}


CLASS_LABELS = {0: "benign", 1: "malignant", -1: "unknown"}


class WandBLoggerIntegration(LoggerIntegration[WandbLogger, TargetDict]):
    logger_type: ClassVar[Type[Logger]] = WandbLogger

    def __call__(
        self,
        target: List[TargetDict],
        pl_module: Task,
        tag: str,
        step: int,
    ) -> None:
        images_to_log: List[wandb.Image] = []
        for t in target:
            x = t["input"]

            box_data: List[Dict[str, Any]] = []
            if boxes := t.get("bounding_boxes", {}):
                for data in iterate_dict(boxes):
                    foo = self.trace_to_wandb(data, x.shape[-2:])
                    box_data.append(foo)

            caption = self.build_caption(t["patient"], t["label"])
            box_dict = {
                "labels": {
                    "box_data": box_data,
                    "class_labels": CLASS_LABELS,
                },
            }
            processed = self.to_image(x, caption=caption, boxes=box_dict if box_data else None)
            images_to_log.append(processed)

        if images_to_log:
            pl_module.logger.experiment.log({tag: images_to_log}, step=step)

    def build_caption(self, patient: Optional[str], label: Optional[str]) -> str:
        return f"{patient if patient is not None else 'Unknown'}\n" f"{label if label is not None else 'Unknown'}"

    @torch.no_grad()
    def to_video(self, x: Tensor, **kwargs) -> wandb.Video:
        assert is_volume(x)
        x = rearrange(x, "d h w -> d () h w")
        x = repeat(x, "d () h w -> d c h w", c=3).contiguous().mul_(255).byte()
        return wandb.Video(x.cpu().numpy(), **kwargs)

    @torch.no_grad()
    def to_image(self, x: Tensor, **kwargs) -> wandb.Image:
        assert not is_volume(x)
        x = repeat(x, "() h w -> h w c", c=3).contiguous().mul_(255).byte()
        return wandb.Image(x.cpu().numpy(), **kwargs)

    def trace_to_wandb(self, metadata: Dict[str, Any], img_size: Tuple[int, int]) -> Dict[str, Any]:
        boxes = cast(Tensor, metadata["boxes"])
        trait = cast(str, metadata["trait"])
        types = cast(str, metadata["types"])

        # Convert bounding boxes from absolute xyxy to relative xyxy using img_size
        H, W = img_size
        boxes = boxes / boxes.new_tensor([W, H, W, H])
        boxes = boxes.flatten().tolist()

        bbox_dict = {
            "minX": max(0, boxes[0]),
            "minY": max(0, boxes[1]),
            "maxX": min(1.0, boxes[2]),
            "maxY": min(1.0, boxes[3]),
        }
        class_id = 1 if trait == "malignant" else 0 if trait == "benign" else -1
        caption = f"{trait} {'/'.join(types.split(' '))}"
        result = {
            "position": bbox_dict,
            "class_id": class_id,
            "box_caption": caption,
        }
        return result


class LogInputCallback(QueuedLoggingCallback):
    """
    Callback to log batch input images.
    """

    integrations: ClassVar[List[LoggerIntegration]] = [WandBLoggerIntegration()]

    @classmethod
    def get_priority(cls, example: Dict[str, Any], pred: Dict[str, Any]) -> Optional[Union[int, float]]:
        # Bypass this for now because we don't have a priority
        return 0

    def prepare_target(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> TargetDict:
        r"""Prepare the batch for logging."""
        label = (outputs.get("annotation", {}) or {}).get("malignant", "unknown")
        train_label = outputs.get("train_label", None)
        return {
            "input": batch["img"],
            "patient": (batch.get("manifest", {}) or {}).get("Patient", None),
            "label": train_label or label,
            "bounding_boxes": batch.get("bounding_boxes", None),
        }
