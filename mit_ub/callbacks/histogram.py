from typing import Any, ClassVar, Dict, List, Type

import pytorch_lightning as pl
import wandb
from deep_helpers.callbacks import LoggerIntegration, LoggingCallback
from deep_helpers.tasks import Task
from pytorch_lightning.loggers import Logger
from pytorch_lightning.loggers.wandb import WandbLogger


class WandBLoggerIntegration(LoggerIntegration):
    logger_type: ClassVar[Type[Logger]] = WandbLogger

    def __call__(
        self,
        target: Dict[str, Any],
        pl_module: Task,
        tag: str,
        step: int,
    ) -> None:
        log_dict = {f"{tag}_{k}": wandb.Histogram(np_histogram=v) for k, v in target.items()}
        pl_module.logger.experiment.log(log_dict, step=step)


class HistogramCallback(LoggingCallback):
    integrations: ClassVar[List[LoggerIntegration]] = [WandBLoggerIntegration()]

    def reset(self, *args, **kwargs):
        pass

    def register(self, *args, **kwargs):
        pass

    def prepare_target(
        self,
        trainer: pl.Trainer,
        pl_module: Task,
        outputs: Dict[str, Any],
        batch: Dict[str, Any],
        batch_idx: int,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        r"""Prepare the batch for logging."""
        return {k.replace("_hist", ""): v for k, v in outputs.items() if k.endswith("_hist")}
