from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.helpers import load_checkpoint, to_tuple
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from einops import rearrange
from torch import Tensor

from ..model import BACKBONES, AdaptiveViT, ConvNext, ViT
from .jepa import mixup


class Distillation(Task):
    """
    Joint Embedding Predictive Architecture (JEPA) Task.

    This class implements the JEPA task, which involves predicting target embeddings
    from context embeddings using a backbone model. The task also includes an Exponential
    Moving Average (EMA) of the backbone parameters for stable target generation.

    Args:
        backbone: Name of the backbone to use for the task.
        context_ratio: Ratio of the input to sample as context.
        context_scale: Integer scale at which to sample contiguous blocks of context tokens.
            Increasing this ensures more adjacent tokens appear together in the context.
        target_ratio: Ratio of the input to sample as a prediction target.
        target_scale: Integer scale at which to sample contiguous blocks of target tokens.
            Increasing this ensures more adjacent tokens appear together in the target.
        context_subsample_ratio: Sampling ratio for encoded context just before passing
            it to the predictor.
        ema_alpha: Smoothing factor for EMA updates.
        momentum_schedule: If True, use a momentum schedule for EMA updates.
        predictor_depth: Depth of the predictor network.
        mixup_alpha: Alpha parameter for the Beta distribution used to sample the mixup weight.
        mixup_prob: Probability of applying mixup to the input and target.
        optimizer_init: Initial configuration for the optimizer.
        lr_scheduler_init: Initial configuration for the learning rate scheduler.
        lr_interval: Frequency of learning rate update. Can be 'step' or 'epoch'.
        lr_monitor: Quantity to monitor for learning rate scheduler.
        named_datasets: If True, datasets are named, else they are indexed by integers.
        checkpoint: Path to the checkpoint file to initialize the model.
        strict_checkpoint: If True, the model must exactly match the checkpoint.
        log_train_metrics_interval: Interval (in steps) at which to log training metrics.
        log_train_metrics_on_epoch: If True, log training metrics at the end of each epoch.
        parameter_groups: Dictionary of parameter groups and their corresponding weight decay values.
        weight_decay_final: Final weight decay value. If set, the weight decay will be linearly
            annealed from the current value to this value over the course of training.
    """

    def __init__(
        self,
        backbone: str,
        teacher_backbone: str,
        teacher_checkpoint: Path,
        mixup_alpha: float = 1.0,
        mixup_prob: float = 0.2,
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
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

        # Student and teacher backbones
        self.backbone = cast(ConvNext, self.prepare_backbone(backbone))
        self.teacher_backbone = cast(ViT | AdaptiveViT, self.prepare_backbone(teacher_backbone))

        # Extract the index of the FPN level with resolution equal to the isotropic teacher model
        teacher_patch_size = self.teacher_backbone.stem.patch_size
        student_patch_size = to_tuple(self.backbone.patch_size, len(teacher_patch_size))
        target_level = tuple(t // s - 1 for t, s in zip(teacher_patch_size, student_patch_size))
        assert all(
            t >= 0 for t in target_level
        ), f"Teacher patch size {teacher_patch_size} is not divisible by student patch size {student_patch_size}"
        assert len(set(target_level)) == 1, f"Multiple target levels found: {target_level}"
        self.target_level = next(iter(target_level))

        # Build projections for each hierarchical level back to isotropic
        self.distill_projs = nn.ModuleList()
        for i in range(self.target_level, len(self.backbone.dims)):
            level_dim = self.backbone.dims[i]
            proj = nn.Sequential(
                nn.Conv2d(level_dim, self.teacher_backbone.dim, kernel_size=1),
                nn.UpsamplingNearest2d(scale_factor=2 ** (i - self.target_level)),
            )
            self.distill_projs.append(proj)

        # Load teacher checkpoint
        state_dict = torch.load(teacher_checkpoint, weights_only=True)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if any(k.startswith("backbone.") for k in state_dict):
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
        self.teacher_backbone = load_checkpoint(self.teacher_backbone, state_dict, strict=True)
        for p in self.teacher_backbone.parameters():
            p.requires_grad = False

        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection({})

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Run the student model
        pred = self.backbone.forward_features(x)
        assert len(pred) == len(self.backbone.dims)

        # Convert hierarchical features to isotropic
        base_proj = self.distill_projs[0]
        isotropic = base_proj(pred[self.target_level])
        assert isotropic.shape[2:] == pred[self.target_level].shape[2:]
        for i in range(self.target_level + 1, len(self.backbone.dims)):
            proj = self.distill_projs[i - self.target_level]
            isotropic = isotropic + proj(pred[i])

        assert isotropic.shape[2:] == pred[self.target_level].shape[2:]
        isotropic = rearrange(isotropic, "b c h w -> b (h w) c")
        return {"distill_pred": isotropic}

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

            # apply mixup
            if self.training and self.mixup_prob > 0:
                x, target = mixup(x, target, self.mixup_alpha, self.mixup_prob)

        pred_dict = self(x)
        pred: Tensor = pred_dict["distill_pred"]

        # compute loss between target and student predictions
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss = F.smooth_l1_loss(pred, target)

        output = {
            "log": {
                "loss_distill": loss,
            },
            "pred": pred,
            "target": target,
        }
        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return pred
