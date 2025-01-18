import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from torch import Tensor
from torch.optim.optimizer import Optimizer

from ..data.mixup import mixup, sample_mixup_parameters
from ..data.noise import (
    DEFAULT_NOISE_PROB,
    MULTIPLICATIVE_NOISE_MAX,
    MULTIPLICATIVE_NOISE_MIN,
    SALT_PEPPER_NOISE_MAX,
    SALT_PEPPER_NOISE_MIN,
    SALT_PEPPER_NOISE_PROB,
    UNIFORM_NOISE_MAX,
    UNIFORM_NOISE_MIN,
    RandomNoise,
)
from ..metrics.cosine_sim import AveragePairwiseCosineSimilarity, ExampleSimilarity, TokenSimilarity
from ..metrics.distance import ExampleRMSDistance, RMSPairwiseDistance, TokenRMSDistance
from ..metrics.layer_scale import MaxLayerScale, MeanLayerScale
from ..model import AdaptiveViT, AdaptiveViTConfig, ViT, ViTConfig
from ..model.layers.layer_scale import has_layer_scale
from ..model.layers.pool import PoolType
from ..model.layers.pos_enc import RelativeFactorizedPosition
from ..model.layers.transformer import TransformerDecoderLayer
from ..tokens import apply_mask, create_mask, generate_non_overlapping_mask, mask_is_ragged
from .student_teacher import EMAConfig, get_ema_momentum, synchronize_teacher, update_teacher


@dataclass
class JEPAConfig:
    """
    Configuration for JEPA related hyperparameters.

    Args:
        context_ratio: Ratio of the input to sample as context.
        target_ratio: Ratio of the input to sample as a prediction target.
        scale: Integer scale at which to sample contiguous blocks of context tokens.
            Increasing this ensures more adjacent tokens appear together in the context.
        context_subsample_ratio: Sampling ratio for encoded context just before passing
            it to the predictor.
        ema_config: Configuration for EMA updates.
        predictor_depth: Depth of the predictor network.
        mixup_alpha: Alpha parameter for the Beta distribution used to sample the mixup weight.
        mixup_prob: Probability of applying mixup to the input and target.
        use_noise: If True, apply noise to the input.
        uniform_noise_scale: Scale of the uniform noise to apply to the input.
        multiplicative_noise_scale: Scale of the multiplicative noise to apply to the input.
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input.
        noise_prob: Probability of applying a given noise transform.
        noise_clip: If True, clip the noise to the range [0, 1].
        weight_decay_final: Final weight decay value. If set, the weight decay will be linearly
            annealed from the current value to this value over the course of training.
        self_attn: If True, use self-attention in the predictor.
        mlp_tower: If True, use a MLP tower on the JEPA predictor output instead of a simple linear layer.
            Empirically it seems better to not use a MLP tower.
        tower_input_norm: If True, apply input normalization to the tower.
            Input normalization should not be necessary for backbones that already have an output normalization layer.
            Only has an effect if ``mlp_tower`` is ``True``.
        contrastive_weight: Weight of the contrastive loss.
        contrastive_margin: Margin for the contrastive loss.
    """

    context_ratio: float = 0.5
    target_ratio: float = 0.25
    scale: int = 4
    context_subsample_ratio: float = 0.5
    ema_config: EMAConfig = field(default_factory=lambda: EMAConfig())
    predictor_depth: int = 4
    mixup_alpha: float = 1.0
    mixup_prob: float = 0.2
    use_noise: bool = True
    uniform_noise_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX)
    multiplicative_noise_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX)
    salt_pepper_prob: float = SALT_PEPPER_NOISE_PROB
    salt_pepper_pixel_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX)
    noise_prob: float = DEFAULT_NOISE_PROB
    noise_clip: bool = True
    weight_decay_final: float | None = None
    self_attn: bool = False
    mlp_tower: bool = False
    tower_input_norm: bool = False
    siglip_weight: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        if not 0 < self.context_subsample_ratio <= 1:
            raise ValueError("context_subsample_ratio must be in the range (0, 1]")
        if not 0 < self.mixup_alpha:
            raise ValueError("mixup_alpha must be positive")
        if not 0 <= self.mixup_prob <= 1:
            raise ValueError("mixup_prob must be in the range [0, 1]")
        if self.predictor_depth < 1:
            raise ValueError("predictor_depth must be at least 1")
        if self.weight_decay_final is not None and not 0 <= self.weight_decay_final:
            raise ValueError("weight_decay_final must be non-negative")
        if self.siglip_weight < 0:
            raise ValueError("siglip_weight must be non-negative")


class JEPA(Task):
    """
    Joint Embedding Predictive Architecture (JEPA) Task.

    This class implements the JEPA task, which involves predicting target embeddings
    from context embeddings using a backbone model. The task also includes an Exponential
    Moving Average (EMA) of the backbone parameters for stable target generation.

    Args:
        backbone: Name of the backbone to use for the task.
        jepa_config: Configuration for JEPA related hyperparameters.
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
    """

    backbone: ViT | AdaptiveViT

    def __init__(
        self,
        backbone_config: ViTConfig | AdaptiveViTConfig,
        jepa_config: JEPAConfig = JEPAConfig(),
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
        self.jepa_config = jepa_config

        # Backbone and EMA weights
        backbone = backbone_config.instantiate()
        assert isinstance(backbone, (ViT, AdaptiveViT))
        self.backbone = backbone
        self.teacher_backbone = deepcopy(self.backbone)
        self.teacher_backbone.requires_grad_(False)
        self.teacher_backbone.eval()

        # Position encoding / initialization for prediction queries.
        self.pos_enc = RelativeFactorizedPosition(
            len(self.backbone.stem.patch_size),
            self.backbone.config.dim,
            dropout=0.1,
        )

        # JEPA predictor
        self.jepa_predictor = nn.ModuleList(
            [
                self.backbone.create_decoder_layer(i, self_attn=self.jepa_config.self_attn)
                for i in range(self.jepa_config.predictor_depth)
            ]
        )
        self.jepa_head = self.backbone.create_head(
            self.backbone.config.dim,
            pool_type=None,
            use_mlp=self.jepa_config.mlp_tower,
            input_norm=self.jepa_config.tower_input_norm,
        )

        # SigLIP pooling layer
        self.pool = self.backbone.create_head(
            self.backbone.config.dim,
            pool_type=PoolType.SIMPLE_ATTENTION,
            use_mlp=self.jepa_config.mlp_tower,
            input_norm=self.jepa_config.tower_input_norm,
        )
        self.siglip_t = nn.Parameter(torch.empty(1))
        self.siglip_b = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.siglip_t, math.log(10))
        nn.init.constant_(self.siglip_b, -10)

        self.save_hyperparameters()

        # Random noise
        self.random_noise = RandomNoise(
            self.jepa_config.noise_prob,
            self.jepa_config.uniform_noise_scale,
            self.jepa_config.multiplicative_noise_scale,
            self.jepa_config.salt_pepper_prob,
            self.jepa_config.salt_pepper_pixel_prob,
            self.jepa_config.noise_clip,
        )

    def on_task_checkpoint_loaded(self, path: Path, state_dict: Dict[str, Any]) -> None:
        self.backbone.on_load_checkpoint(state_dict)

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        metrics = tm.MetricCollection(
            {
                "example_sim": ExampleSimilarity(self.backbone.config.dim),
                "micro_token_sim": TokenSimilarity(self.backbone.config.dim),
                "macro_token_sim": AveragePairwiseCosineSimilarity(self.backbone.config.dim),
                "example_rms": ExampleRMSDistance(self.backbone.config.dim),
                "micro_token_rms": TokenRMSDistance(self.backbone.config.dim),
                "macro_token_rms": RMSPairwiseDistance(self.backbone.config.dim),
                "jepa_loss": tm.MeanMetric(),
                "siglip_loss": tm.MeanMetric(),
            }
        )
        if has_layer_scale(self.backbone) and state.mode == Mode.TRAIN:
            metrics["layer_scale_max"] = MaxLayerScale()
            metrics["layer_scale_mean"] = MeanLayerScale()
        if state.mode == Mode.TRAIN:
            metrics["ema_momentum"] = tm.MeanMetric()

        return metrics

    def forward(self, x: Tensor, context_mask: Tensor, target_mask: Tensor) -> Dict[str, Tensor]:
        # Run encoder on context
        torch.cuda.nvtx.range_push("context_backbone")
        context: Tensor = self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)
        torch.cuda.nvtx.range_pop()
        B, L, _ = context.shape

        # Sample a subset of the context as input to the predictor and project
        if self.jepa_config.context_subsample_ratio < 1.0:
            subsample_mask = create_mask(
                (L,),
                mask_ratio=1 - self.jepa_config.context_subsample_ratio,
                batch_size=B,
                device=context.device,
            )
            context = apply_mask(subsample_mask, context, fill_value=None)

        # Pooling layer. When siglip_weight is 0, a stop gradient is applied to the context.
        torch.cuda.nvtx.range_push("pool")
        pooled = self.pool(context if self.jepa_config.siglip_weight else context.detach())
        torch.cuda.nvtx.range_pop()

        # Prepare positional encoding for target queries
        tokenized_size = self.backbone.stem.tokenized_size(cast(Any, x.shape[2:]))
        query = self.pos_enc(tokenized_size).expand(B, -1, -1)
        query = apply_mask(target_mask, query, fill_value=None)

        # Run query and context through predictor
        torch.cuda.nvtx.range_push("jepa_predictor")
        for block in self.jepa_predictor:
            block = cast(TransformerDecoderLayer, block)
            query = block(query, context)
        torch.cuda.nvtx.range_pop()

        pred = self.jepa_head(query)
        return {"jepa": pred, "jepa_context": context, "pooled": pooled}

    @property
    def fraction_complete(self) -> float | None:
        r"""The fraction of training complete as a float between 0 and 1, or None if we can't determine it."""
        if self.trainer.max_steps:
            current = self.trainer.global_step
            total = self.trainer.max_steps
        elif self.trainer.max_epochs:
            current = self.trainer.current_epoch
            total = self.trainer.max_epochs
        else:
            return None
        return current / total

    def get_ema_momentum(self) -> float:
        r"""Get the momentum for the EMA update based on the current step or epoch."""
        if not (max_steps := self.trainer.max_steps):
            raise RuntimeError("Cannot determine EMA momentum without trainer.max_steps")
        if (global_step := self.trainer.global_step) == 0 and not getattr(self.trainer, "fast_dev_run", False):
            self.jepa_config.ema_config.validate_schedule(max_steps)

        return get_ema_momentum(max_steps, global_step, self.jepa_config.ema_config)

    @torch.no_grad()
    def update_ema(self, batch_idx: int) -> None:
        """Update the Exponential Moving Average (EMA) of the backbone parameters."""
        momentum = self.get_ema_momentum()

        # NOTE: This also handles synchronization of EMA weights across processes
        torch.cuda.nvtx.range_push("ema_update")
        update_teacher(
            self.backbone,
            self.teacher_backbone,
            momentum,
            batch_idx,
            self.trainer.global_step,
            self.trainer.accumulate_grad_batches,
            self.trainer.world_size,
            self.jepa_config.ema_config.sync_interval,
        )
        torch.cuda.nvtx.range_pop()

    @torch.no_grad()
    def synchronize_ema_weights(self) -> None:
        """Synchronize the EMA weights across all processes using parameter buckets."""
        synchronize_teacher(self.teacher_backbone, self.trainer.world_size)

    def update_weight_decay(self) -> List[float | None]:
        """Update the weight decay for each parameter group based on the current training progress."""
        assert self.jepa_config.weight_decay_final is not None

        # Get optimizer and wrap as a list if necessary
        optimizers = self.optimizers()
        optimizers = [optimizers] if isinstance(optimizers, Optimizer) else optimizers
        optimizers = cast(List[Optimizer], optimizers)

        # Check where we are in the training loop, abort if we can't determine
        fraction_complete = self.fraction_complete
        if fraction_complete is None:
            return [pg.get("weight_decay", None) for opt in optimizers for pg in opt.param_groups]

        def update_weight_decay(pg: Dict[str, Any]) -> float | None:
            current_wd = pg["weight_decay"]
            # Some layers may be exempt from weight decay
            if current_wd == 0:
                return

            initial_wd = pg.setdefault("initial_weight_decay", current_wd)
            final_wd = self.jepa_config.weight_decay_final
            new_wd = max(current_wd, initial_wd + (final_wd - initial_wd) * fraction_complete)
            pg["weight_decay"] = new_wd
            return new_wd

        # Update weight decay for each optimizer
        updates = map(
            update_weight_decay,
            (pg for optimizer in optimizers for pg in optimizer.param_groups if "weight_decay" in pg),
        )
        return list(updates)

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        torch.compiler.cudagraph_mark_step_begin()
        x: Tensor = batch["img"]

        # ema update from previous step when training
        if state.mode == Mode.TRAIN:
            self.update_ema(batch_idx)
        # update weight decay
        if self.jepa_config.weight_decay_final is not None:
            self.update_weight_decay()

        # generate context mask - will always be non-ragged
        torch.cuda.nvtx.range_push("context_mask")
        context_mask = self.backbone.create_mask(x, self.jepa_config.context_ratio, self.jepa_config.scale)
        assert context_mask.device == x.device, "Context mask device does not match input device"
        assert not mask_is_ragged(context_mask), "Context mask is ragged"
        torch.cuda.nvtx.range_pop()

        # generate target mask - select non-ragged target mask from locations not in context mask
        torch.cuda.nvtx.range_push("target_mask")
        target_mask = generate_non_overlapping_mask(
            context_mask, self.jepa_config.context_ratio, self.jepa_config.target_ratio
        )
        assert target_mask.device == x.device, "Target mask device does not match input device"
        assert not mask_is_ragged(target_mask), "Target mask is ragged"
        assert not (context_mask & target_mask).any(), "Context and target masks overlap"
        torch.cuda.nvtx.range_pop()

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            self.teacher_backbone.eval()
            torch.cuda.nvtx.range_push("ema_backbone")
            full_target: Tensor = self.teacher_backbone(x, reshape=False)
            torch.cuda.nvtx.range_pop()
            siglip_target = torch.eye(full_target.shape[0], device=full_target.device, dtype=torch.float32)

            # apply random noise
            if self.training and self.jepa_config.use_noise:
                torch.cuda.nvtx.range_push("noise")
                x = self.random_noise.apply_batched(x)
                torch.cuda.nvtx.range_pop()

            # apply mixup, not overwriting full_target
            if self.training and self.jepa_config.mixup_prob > 0:
                torch.cuda.nvtx.range_push("mixup")
                mixup_weight = sample_mixup_parameters(
                    x.shape[0], self.jepa_config.mixup_prob, self.jepa_config.mixup_alpha, device=x.device
                )
                x = mixup(x, mixup_weight)
                full_target = mixup(full_target, mixup_weight)
                siglip_target = mixup(siglip_target, mixup_weight)
                torch.cuda.nvtx.range_pop()
            else:
                mixup_weight = None

            target = apply_mask(target_mask, full_target, fill_value=None)

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred_dict = self(x, context_mask, target_mask)
        pred: Tensor = pred_dict["jepa"]
        context: Tensor = pred_dict["jepa_context"]
        pooled: Tensor = pred_dict["pooled"]

        # compute loss between target and predictor encoded latents
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss_jepa = F.smooth_l1_loss(pred, target)

        # compute pooled similarity logits
        assert pooled.ndim == 2, f"Pooled shape {pooled.shape} does not match expected shape (B, D)"
        pred_pooled_norm = F.normalize(pooled, dim=-1)
        with torch.no_grad():
            target_pooled_norm = F.normalize(self.pool(full_target), dim=-1)

        logits = torch.matmul(pred_pooled_norm, target_pooled_norm.T) * self.siglip_t.exp() + self.siglip_b
        loss_siglip = F.binary_cross_entropy_with_logits(logits, siglip_target)

        # Compute metrics
        if metrics is not None:
            with torch.no_grad():
                torch.cuda.nvtx.range_push("metrics")
                metrics["jepa_loss"].update(loss_jepa)
                metrics["siglip_loss"].update(loss_siglip)
                metrics["example_sim"].update(full_target)
                metrics["example_rms"].update(full_target)
                metrics["micro_token_sim"].update(full_target)
                metrics["micro_token_rms"].update(full_target)
                metrics["macro_token_sim"].update(full_target)
                metrics["macro_token_rms"].update(full_target)

                if "layer_scale_mean" in metrics:
                    metrics["layer_scale_mean"].update(self.backbone)
                    metrics["layer_scale_max"].update(self.backbone)

                if "ema_momentum" in metrics:
                    metrics["ema_momentum"].update(self.get_ema_momentum())
                torch.cuda.nvtx.range_pop()

        output = {
            "log": {
                "loss_jepa": loss_jepa,
                "loss_siglip": loss_siglip * (self.jepa_config.siglip_weight or 1.0),
            },
            "context": context,
            "jepa_pred": pred,
            "target": target,
            "full_target": full_target,
            "mixup_weight": mixup_weight,
        }
        return output

    @torch.no_grad()
    def predict_step(self, batch: Any, *args, **kwargs) -> Dict[str, Any]:
        pred = self(batch["img"])
        return {
            "jepa": pred["jepa"],
        }


class JEPAWithProbe(JEPA, ABC):
    def __init__(
        self,
        backbone_config: ViTConfig | AdaptiveViTConfig,
        jepa_config: JEPAConfig = JEPAConfig(),
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
            jepa_config,
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
        features: Tensor = output["full_target"].detach()
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
