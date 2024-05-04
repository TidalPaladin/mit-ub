from typing import Any, ClassVar, Dict, List, Optional, Union

import pytorch_lightning as pl
from deep_helpers.callbacks import LoggerIntegration, QueuedLoggingCallback
from deep_helpers.tasks import Task

from .input import TargetDict, WandBLoggerIntegration


class LogMAECallback(QueuedLoggingCallback):
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
            "input": outputs["mae_pred"],
            "patient": (batch.get("manifest", {}) or {}).get("Patient", None),
            "label": train_label or label,
            "bounding_boxes": batch.get("bounding_boxes", None),
        }
