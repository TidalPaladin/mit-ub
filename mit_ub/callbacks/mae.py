from typing import Any, ClassVar, Dict, List, Optional, Type, TypedDict, Union

import pytorch_lightning as pl
import torch
import wandb
from deep_helpers.callbacks import LoggerIntegration, QueuedLoggingCallback
from deep_helpers.tasks import Task
from einops import rearrange, repeat
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger
from torch import Tensor
from .input import WandBLoggerIntegration, TargetDict


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
        x = outputs["mae_pred"]
        return {"input": x}