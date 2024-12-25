from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.helpers import load_checkpoint
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from torch import Tensor

from ..data.mixup import mixup, sample_mixup_parameters
from ..data.noise import RandomNoise
from ..metrics.layer_scale import MaxLayerScale, MeanLayerScale
from ..model import AnyModelConfig
from ..model.helpers import grid_to_tokens
from ..model.layers import has_layer_scale
from .jepa import apply_noise_batched, mixup


@dataclass
class DistillationConfig:
    """
    Configuration for JEPA related hyperparameters.

    Args:
        mixup_alpha: Alpha parameter for the Beta distribution used to sample the mixup weight.
        mixup_prob: Probability of applying mixup to the input and target.
        use_noise: If True, apply noise to the input.
        noise_scale: Scale of the noise to apply to the input.
        noise_clip: If True, clip the noise to the range [0, 1].
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input.
    """

    mixup_alpha: float = 1.0
    mixup_prob: float = 0.2
    use_noise: bool = True
    noise_scale: float = 0.2
    noise_clip: bool = True
    salt_pepper_prob: float | Tuple[float, float] = (0.01, 0.05)

    def __post_init__(self) -> None:
        if not 0 < self.mixup_alpha:
            raise ValueError("mixup_alpha must be positive")
        if not 0 <= self.mixup_prob <= 1:
            raise ValueError("mixup_prob must be in the range [0, 1]")


class Distillation(Task):

    def __init__(
        self,
        backbone_config: AnyModelConfig,
        teacher_config: AnyModelConfig,
        teacher_checkpoint: Path,
        distillation_config: DistillationConfig = DistillationConfig(),
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
        self.config = distillation_config

        # Student backbone
        backbone = backbone_config.instantiate()
        self.backbone = backbone

        # Teacher backbone
        teacher_backbone = teacher_config.instantiate()
        self.teacher_backbone = teacher_backbone

        student_dim = backbone.config.dim
        teacher_dim = teacher_backbone.config.dim
        self.proj = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()

        # Load teacher checkpoint and freeze parameters
        self.teacher_backbone = self.load_teacher_checkpoint(teacher_checkpoint)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        self.random_noise = RandomNoise(
            self.config.noise_scale,
            self.config.salt_pepper_prob,
            self.config.noise_clip,
        )

        self.save_hyperparameters()

    def load_teacher_checkpoint(self, teacher_checkpoint: Path) -> nn.Module:
        state_dict = torch.load(teacher_checkpoint, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        teacher_backbone = load_checkpoint(self.teacher_backbone, state_dict, strict=True)
        return teacher_backbone

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        metrics = tm.MetricCollection({"distill_loss": tm.MeanMetric()})

        if has_layer_scale(self.backbone) and state.mode == Mode.TRAIN:
            metrics["layer_scale_max"] = MaxLayerScale()
            metrics["layer_scale_mean"] = MeanLayerScale()

        return metrics

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        pred = self.backbone(x)
        pred = grid_to_tokens(pred)
        pred_proj = self.proj(pred)
        return {"distill_pred": pred, "distill_pred_proj": pred_proj}

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        torch.compiler.cudagraph_mark_step_begin()
        x: Tensor = batch["img"]

        with torch.no_grad():
            # generate ground truth with forward pass of teacher backbone
            self.teacher_backbone.eval()
            target: Tensor = self.teacher_backbone(x, reshape=False)

            if self.training and self.config.use_noise:
                x = apply_noise_batched(self.random_noise, x)

            # apply mixup
            if self.training and self.config.mixup_prob > 0:
                mixup_weight = sample_mixup_parameters(
                    x.shape[0], self.config.mixup_prob, self.config.mixup_alpha, device=x.device
                )
                x = mixup(x, mixup_weight)
                target = mixup(target, mixup_weight)
            else:
                mixup_weight = None

        pred_dict = self(x)
        pred: Tensor = pred_dict["distill_pred"]
        pred_proj: Tensor = pred_dict["distill_pred_proj"]

        # compute loss between target and student predictions
        assert (
            pred_proj.shape == target.shape
        ), f"Prediction shape {pred_proj.shape} does not match target shape {target.shape}"
        loss = F.smooth_l1_loss(pred_proj, target)

        if metrics is not None:
            with torch.no_grad():
                metrics["distill_loss"].update(loss)

        output = {
            "log": {
                "loss_distill": loss,
            },
            "pred": pred,
            "pred_proj": pred_proj,
            "target": target,
            "mixup_weight": mixup_weight,
        }
        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return pred


class DistillationWithProbe(Distillation, ABC):
    def __init__(
        self,
        backbone_config: AnyModelConfig,
        teacher_config: AnyModelConfig,
        teacher_checkpoint: Path,
        distillation_config: DistillationConfig = DistillationConfig(),
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
            backbone_config,
            teacher_config,
            teacher_checkpoint,
            distillation_config,
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
        self.linear_probe = self.create_probe_head()

    @abstractmethod
    def create_probe_head(self) -> nn.Module:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        r"""Compute the linear probe loss and update the metrics"""
        raise NotImplementedError  # pragma: no cover

    def get_probe_features_from_output(self, output: Dict[str, Any]) -> Tensor:
        features: Tensor = output["pred"].detach()
        return features

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        output = super().step(batch, batch_idx, state, metrics)
        output = self.step_linear_probe(batch, output, metrics)
        return output
