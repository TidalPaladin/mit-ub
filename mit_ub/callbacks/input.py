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


class TargetDict(TypedDict):
    input: Tensor


def is_volume(x: Tensor) -> bool:
    return x.shape[-3] > 1


class WandBLoggerIntegration(LoggerIntegration[WandbLogger, TargetDict]):
    logger_type: ClassVar[Type[Logger]] = WandbLogger

    def __call__(
        self,
        target: List[TargetDict],
        pl_module: Task,
        tag: str,
        step: int,
    ) -> None:
        to_log = []
        for t in target:
            x = t["input"]
            if is_volume(x):
                processed = self.to_video(x, fps=1)
            else:
                processed = self.to_image(x)
            to_log.append(processed)
        pl_module.logger.experiment.log({tag: to_log}, step=step)

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
        x = batch["img"]
        return {"input": x}