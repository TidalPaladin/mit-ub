from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from convnext import ConvNextConfig
from convnext.block import grid_to_tokens
from deep_helpers.helpers import load_checkpoint
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from torch import Tensor
from vit import ViT, ViTConfig

from ..data.invert import invert_
from ..data.mixup import mixup
from ..data.noise import (
    DEFAULT_NOISE_PROB,
    MULTIPLICATIVE_NOISE_MAX,
    MULTIPLICATIVE_NOISE_MIN,
    SALT_PEPPER_NOISE_MAX,
    SALT_PEPPER_NOISE_MIN,
    SALT_PEPPER_NOISE_PROB,
    UNIFORM_NOISE_MAX,
    UNIFORM_NOISE_MIN,
    apply_noise_batched,
)


@dataclass
class DistillationConfig:
    """
    Configuration for JEPA related hyperparameters.

    Args:
        mixup_alpha: Alpha parameter for the Beta distribution used to sample the mixup weight.
        mixup_prob: Probability of applying mixup to the input and target.
        use_noise: If True, apply noise to the input.
        uniform_noise_scale: Scale of the uniform noise to apply to the input.
        multiplicative_noise_scale: Scale of the multiplicative noise to apply to the input.
        noise_prob: Probability of applying a given noise transform.
        noise_clip: If True, clip the noise to the range [0, 1].
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input.
        student_pool_type: Type of pooling to use for the student backbone.
        teacher_pool_type: Type of pooling to use for the teacher backbone.
        teacher_resolution: If provided, resize the teacher input to this resolution.
            The user must ensure that the teacher backbone is compatible with this resolution,
            and that the size of the student's and teacher's outputs are equal under this resolution.
        invert_prob: Probability of inverting the input.
        solarize_prob: Probability of solarizing the input.
        solarize_threshold: Threshold for solarizing the input.
    """

    mixup_alpha: float = 1.0
    mixup_prob: float = 0.2
    use_noise: bool = True
    uniform_noise_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX)
    multiplicative_noise_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX)
    noise_prob: float = DEFAULT_NOISE_PROB
    noise_clip: bool = True
    salt_pepper_prob: float = SALT_PEPPER_NOISE_PROB
    salt_pepper_pixel_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX)
    student_pool_type: str | None = None
    teacher_pool_type: str | None = None
    teacher_resolution: Sequence[int] | None = None
    invert_prob: float = 0.0
    solarize_prob: float = 0.0
    solarize_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not 0 < self.mixup_alpha:
            raise ValueError("mixup_alpha must be positive")
        if not 0 <= self.mixup_prob <= 1:
            raise ValueError("mixup_prob must be in the range [0, 1]")
        if not 0 <= self.invert_prob <= 1:
            raise ValueError("invert_prob must be in the range [0, 1]")
        if not 0 <= self.solarize_prob <= 1:
            raise ValueError("solarize_prob must be in the range [0, 1]")


class Distillation(Task):

    def __init__(
        self,
        backbone_config: ViTConfig | ConvNextConfig,
        teacher_config: ViTConfig | ConvNextConfig,
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
        if not isinstance(self.teacher_backbone, ViT):
            raise ValueError("Teacher backbone must be a ViT")

        student_dim = backbone.config.isotropic_output_dim
        teacher_dim = teacher_backbone.config.isotropic_output_dim
        self.proj = nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else nn.Identity()
        self.student_pool = self.backbone.create_head(
            out_dim=teacher_dim,
            pool_type=self.config.student_pool_type,
        )
        self.teacher_pool = self.teacher_backbone.create_head(
            out_dim=teacher_dim,
            pool_type=self.config.teacher_pool_type,
        )

        # Resize teacher input if necessary
        self.teacher_resize = (
            nn.UpsamplingBilinear2d(cast(Any, self.config.teacher_resolution))
            if self.config.teacher_resolution
            else nn.Identity()
        )

        # Load teacher checkpoint and freeze parameters
        self.teacher_backbone = self.load_teacher_checkpoint(teacher_checkpoint)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        self.save_hyperparameters()

    def load_teacher_checkpoint(self, teacher_checkpoint: Path) -> nn.Module:
        state_dict = torch.load(teacher_checkpoint, weights_only=False)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        teacher_backbone = load_checkpoint(self.teacher_backbone, state_dict, strict=True)
        return teacher_backbone

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        metrics = tm.MetricCollection(
            {
                "distill_loss": tm.MeanMetric(),
                "distill_loss_cls": tm.MeanMetric(),
            }
        )
        return metrics

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # ViTs
        if isinstance(self.backbone, ViT):
            pred, pred_cls_token = self.backbone(x, reshape=False)
            pred_cls_token = self.student_pool(pred_cls_token)
        # CNNs
        else:
            pred = self.backbone(x)
            pred = grid_to_tokens(pred)
            pred_cls_token = self.student_pool(pred)

        pred_proj = self.proj(pred)
        return {"distill_pred": pred, "distill_pred_proj": pred_proj, "distill_pred_cls_token": pred_cls_token}

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        torch.compiler.cudagraph_mark_step_begin()
        x: Tensor = batch["img"]

        with torch.inference_mode():
            # generate ground truth with forward pass of teacher backbone
            self.teacher_backbone.eval()
            target, target_cls_token = cast(
                Tuple[Tensor, Tensor],
                self.teacher_backbone(self.teacher_resize(x)),
            )

            if self.training and self.config.use_noise:
                x = apply_noise_batched(
                    x,
                    prob=self.config.noise_prob,
                    uniform_scale=self.config.uniform_noise_scale,
                    multiplicative_scale=self.config.multiplicative_noise_scale,
                    salt_pepper_prob=self.config.salt_pepper_prob,
                    salt_pepper_pixel_prob=self.config.salt_pepper_pixel_prob,
                    clip=self.config.noise_clip,
                )

            # apply mixup
            if self.training and self.config.mixup_prob > 0:
                mixup_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                x = mixup(x, self.config.mixup_prob, self.config.mixup_alpha, mixup_seed)
                target = mixup(target, self.config.mixup_prob, self.config.mixup_alpha, mixup_seed)
                target_cls_token = mixup(target_cls_token, self.config.mixup_prob, self.config.mixup_alpha, mixup_seed)
            else:
                mixup_seed = None

            # invert input
            if self.training and self.config.invert_prob > 0:
                torch.cuda.nvtx.range_push("invert")
                invert_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
                invert_(
                    x, self.config.invert_prob, self.config.solarize_prob, self.config.solarize_threshold, invert_seed
                )
                torch.cuda.nvtx.range_pop()

        if self.training:
            x = x.clone()
            target = target.clone()
            target_cls_token = target_cls_token.clone()

        pred_dict = self(x)
        pred: Tensor = pred_dict["distill_pred"]
        pred_proj: Tensor = pred_dict["distill_pred_proj"]
        pred_cls_token: Tensor = pred_dict["distill_pred_cls_token"]

        # compute loss between target and student grid predictions
        assert (
            pred_proj.shape == target.shape
        ), f"Prediction shape {pred_proj.shape} does not match target shape {target.shape}"
        loss = F.smooth_l1_loss(pred_proj, target)

        # compute loss between target and student cls token predictions
        target_cls_token = self.teacher_pool(target_cls_token)
        assert (
            pred_cls_token.shape == target_cls_token.shape
        ), f"Prediction shape {pred_cls_token.shape} does not match target shape {target_cls_token.shape}"
        loss_cls_token = F.smooth_l1_loss(pred_cls_token, target_cls_token)

        if metrics is not None:
            with torch.inference_mode():
                metrics["distill_loss"].update(loss)
                metrics["distill_loss_cls"].update(loss_cls_token)

        output = {
            "log": {
                "loss_distill": loss,
                "loss_distill_cls_token": loss_cls_token,
            },
            "pred": pred,
            "pred_proj": pred_proj,
            "target": target,
            "pred_cls_token": pred_cls_token,
            "target_cls_token": target_cls_token,
            "mixup_seed": mixup_seed,
        }
        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return pred


class DistillationWithProbe(Distillation, ABC):
    def __init__(
        self,
        backbone_config: ViTConfig | ConvNextConfig,
        teacher_config: ViTConfig | ConvNextConfig,
        teacher_checkpoint: Path,
        distillation_config: DistillationConfig = DistillationConfig(),
        probe_key: str = "pred_cls_token",
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
        self.classification_head = self.create_probe_head()
        self.probe_key = probe_key

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
        features: Tensor = output[self.probe_key].detach()
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
