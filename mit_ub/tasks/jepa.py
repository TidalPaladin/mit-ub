from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Final, Iterator, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce
from torch.distributed import barrier as dist_barrier
from torch.optim.optimizer import Optimizer

from ..data.augment import RandomNoise
from ..model import BACKBONES, AdaptiveViT, ViT, compile_is_disabled
from ..model.layer_scale import LayerScale
from ..model.pos_enc import RelativeFactorizedPosition
from ..model.transformer import TransformerDecoderLayer
from ..tokens import apply_mask, create_mask


EPS: Final = 1e-8


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def average_pairwise_cosine_similarity(x: Tensor, pairwise_dim: int, embed_dim: int, eps: float = EPS) -> Tensor:
    r"""Compute the average pairwise cosine similarity without manifesting the full pairwise matrix.

    To avoid quadratic memory usage we compute average cosine similarity as the squared norm of the mean vector.

    Args:
        x: The input tensor.
        pairwise_dim: The dimension over which to compute the average pairwise cosine similarity.
        embed_dim: The dimension to normalize the vectors to before computing the cosine similarity.
        eps: A small constant to avoid division by zero.
    """
    N = x.shape[pairwise_dim]
    x = F.normalize(x, dim=embed_dim, eps=eps)
    y = x.mean(pairwise_dim, keepdim=True).norm(dim=embed_dim, keepdim=True).pow(2).squeeze(embed_dim, pairwise_dim)
    y.sub(1 / N).mul(N / max(N - 1, 1))
    return y


def iterate_layer_scales(module: nn.Module) -> Iterator[Tensor]:
    r"""Iterate over all layer scales in a module."""
    assert isinstance(module, nn.Module)
    for m in module.modules():
        if isinstance(m, LayerScale):
            yield m.gamma.detach()


class MaxLayerScale(tm.MaxMetric):
    r"""Track the maximum absolute value of layer scale across all layers."""

    def update(self, module: nn.Module) -> None:
        max_scales = torch.tensor([t.abs().max() for t in iterate_layer_scales(module)])
        super().update(max_scales.max())


class MeanLayerScale(tm.MeanMetric):
    r"""Track the mean absolute value of layer scale across all layers."""

    def update(self, module: nn.Module) -> None:
        for t in iterate_layer_scales(module):
            super().update(t.abs().mean())


@dataclass
class JEPAConfig:
    """
    Configuration for JEPA related hyperparameters.

    Args:
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
        noise_scale: Scale of the noise to apply to the input.
        noise_clip: If True, clip the noise to the range [0, 1].
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input.
        weight_decay_final: Final weight decay value. If set, the weight decay will be linearly
            annealed from the current value to this value over the course of training.
    """

    context_ratio: float = 0.5
    context_scale: int = 4
    target_ratio: float = 0.25
    target_scale: int = 2
    context_subsample_ratio: float = 0.5
    ema_alpha: float = 0.95
    momentum_schedule: bool = False
    predictor_depth: int = 4
    mixup_alpha: float = 1.0
    mixup_prob: float = 0.2
    noise_scale: float = 0.2
    noise_clip: bool = True
    salt_pepper_prob: float | Tuple[float, float] = (0.01, 0.05)
    weight_decay_final: float | None = None

    def __post_init__(self) -> None:
        if not 0 < self.context_ratio <= 1:
            raise ValueError("context_ratio must be in the range (0, 1]")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in the range (0, 1]")
        if not 0 < self.context_subsample_ratio <= 1:
            raise ValueError("context_subsample_ratio must be in the range (0, 1]")
        if not 0 < self.ema_alpha < 1:
            raise ValueError("ema_alpha must be in the range (0, 1)")
        if not 0 < self.mixup_alpha:
            raise ValueError("mixup_alpha must be positive")
        if not 0 <= self.mixup_prob <= 1:
            raise ValueError("mixup_prob must be in the range [0, 1]")
        if self.predictor_depth < 1:
            raise ValueError("predictor_depth must be at least 1")
        if self.weight_decay_final is not None and not 0 <= self.weight_decay_final:
            raise ValueError("weight_decay_final must be non-negative")


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
        backbone: str,
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
        self.backbone = cast(ViT | AdaptiveViT, self.prepare_backbone(backbone))
        self.ema_backbone = deepcopy(self.backbone)
        for p in self.ema_backbone.parameters():
            p.requires_grad = False

        # Position encoding / initialization for prediction queries.
        self.pos_enc = RelativeFactorizedPosition(
            len(self.backbone.stem.patch_size),
            self.backbone.dim,
            dropout=0.1,
        )

        # Projections for the input context and output predictions
        self.jepa_norm = nn.LayerNorm(self.backbone.dim)
        self.jepa_out_proj = nn.Linear(self.backbone.dim, self.backbone.dim)
        nn.init.xavier_uniform_(self.jepa_out_proj.weight)
        nn.init.zeros_(self.jepa_out_proj.bias)

        # JEPA predictor
        self.jepa_predictor = nn.ModuleList(
            [
                self.backbone.create_decoder_layer(i, stochastic_depth=0.0)
                for i in range(self.jepa_config.predictor_depth)
            ]
        )
        self.save_hyperparameters()

        # Random noise
        self.random_noise = RandomNoise(
            self.jepa_config.noise_scale,
            self.jepa_config.salt_pepper_prob,
            self.jepa_config.noise_clip,
        )

    def on_task_checkpoint_loaded(self, path: Path, state_dict: Dict[str, Any]) -> None:
        self.backbone.on_load_checkpoint(state_dict)

    def prepare_backbone(self, name: str) -> nn.Module:
        backbone = BACKBONES.get(name).instantiate_with_metadata().fn
        assert isinstance(backbone, nn.Module)
        return backbone

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        metrics = tm.MetricCollection(
            {
                "example_sim": tm.MeanMetric(),
                "token_sim": tm.MeanMetric(),
            }
        )
        has_layer_scale = any(isinstance(layer, LayerScale) for layer in self.backbone.modules())
        if has_layer_scale and state.mode == Mode.TRAIN:
            metrics["layer_scale_max"] = MaxLayerScale()
            metrics["layer_scale_mean"] = MeanLayerScale()

        return metrics

    def forward(self, x: Tensor, context_mask: Tensor, target_mask: Tensor) -> Dict[str, Tensor]:
        # Run encoder on context
        context: Tensor = self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)
        B, L, _ = context.shape

        # Sample a subset of the context as input to the predictor and project
        if self.jepa_config.context_subsample_ratio < 1.0:
            subsample_mask = create_mask((L,), mask_ratio=1 - self.jepa_config.context_subsample_ratio, batch_size=B)
            context = apply_mask(subsample_mask, context, fill_value=None)

        # Prepare positional encoding for target queries
        tokenized_size = self.backbone.stem.tokenized_size(cast(Any, x.shape[2:]))
        query = self.pos_enc(tokenized_size).expand(B, -1, -1)
        query = apply_mask(target_mask, query, fill_value=None)

        # Run query and context through predictor
        for block in self.jepa_predictor:
            block = cast(TransformerDecoderLayer, block)
            query = block(query, context)

        pred = self.jepa_norm(query)
        pred = self.jepa_out_proj(pred)
        return {"jepa": pred, "jepa_context": context}

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
        if not self.jepa_config.momentum_schedule:
            return self.jepa_config.ema_alpha
        fraction_complete = self.fraction_complete
        fraction_complete = fraction_complete if fraction_complete is not None else 0.0
        return self.jepa_config.ema_alpha + (1 - self.jepa_config.ema_alpha) * fraction_complete

    @torch.no_grad()
    def update_ema(self) -> None:
        """Update the Exponential Moving Average (EMA) of the backbone parameters."""
        momentum = self.get_ema_momentum()
        for ema_param, param in zip(self.ema_backbone.parameters(), self.backbone.parameters()):
            ema_param.lerp_(param, 1 - momentum)
        self.synchronize_ema_weights()

    @torch.no_grad()
    def synchronize_ema_weights(self) -> None:
        """
        Synchronize the Exponential Moving Average (EMA) weights across all processes.

        This method ensures that the EMA weights are consistent across all processes
        in a distributed training setup. It uses barriers to avoid sporadic deadlocks
        in Distributed Data Parallel (DDP) training.
        """
        if self.trainer.world_size > 1:
            for ema_param in self.ema_backbone.parameters():
                # There seems to be sporadic deadlocks in DDP, so we use barriers to keep things synchronized
                dist_barrier()
                all_reduce(ema_param.data, op=ReduceOp.SUM)
                ema_param.data /= self.trainer.world_size
            dist_barrier()

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

    @torch.no_grad()
    def mixup(self, x: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Implements MixUp on inputs and teacher network outputs.

        At a high level, this involves linearly combining two inputs and their
        corresponding targets in a random manner.

        Args:
            x: The input tensor.
            target: The target tensor.

        Returns:
            A tuple containing the mixed input and target.
        """
        # Generate mixup weight
        N = x.shape[0]
        dist = torch.distributions.Beta(self.jepa_config.mixup_alpha, self.jepa_config.mixup_alpha)
        lam = dist.sample(torch.Size((N,))).to(x.device)

        # Generate mask of mixup samples
        mixup_mask = torch.rand_like(lam) < self.jepa_config.mixup_prob

        def right_broadcast(inp: Tensor, proto: Tensor) -> Tensor:
            return inp.view(-1, *(1,) * len(proto.shape[1:]))

        # Apply mixup to input and target
        x = torch.where(
            right_broadcast(mixup_mask, x),
            x.roll(1, 0).lerp_(x, right_broadcast(lam, x)),
            x,
        )
        target = torch.where(
            right_broadcast(mixup_mask, target),
            target.roll(1, 0).lerp_(target, right_broadcast(lam.type_as(target), target)),
            target,
        )
        return x, target

    @torch.no_grad()
    def apply_noise_batched(self, x: Tensor) -> Tensor:
        r"""Applies noise to a batch of images such that each image in the batch is
        independently transformed. This is an alternative to `self.random_noise` which
        applies the same noise to all images in the batch.
        """
        x = x.clone()
        for i in range(x.shape[0]):
            x[i] = self.random_noise(x[i])
        return x

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
            self.update_ema()
        # update weight decay
        if self.jepa_config.weight_decay_final is not None:
            self.update_weight_decay()

        # generate context and target masks
        target_mask = self.backbone.create_mask(x, self.jepa_config.target_ratio, self.jepa_config.target_scale)
        context_mask = self.backbone.create_mask(x, self.jepa_config.context_ratio, self.jepa_config.context_scale)

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            self.ema_backbone.eval()
            full_target: Tensor = self.ema_backbone(x, reshape=False)

            # apply random noise
            x = self.apply_noise_batched(x)

            # apply mixup, not overwriting full_target
            if self.training and self.jepa_config.mixup_prob > 0:
                x, full_target_mixed = self.mixup(x, full_target)
                target = apply_mask(target_mask, full_target_mixed, fill_value=None)
            else:
                target = apply_mask(target_mask, full_target, fill_value=None)

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred_dict = self(x, context_mask, target_mask)
        pred: Tensor = pred_dict["jepa"]
        context: Tensor = pred_dict["jepa_context"]

        # compute loss between target and predictor encoded latents
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss = F.smooth_l1_loss(pred, target)

        # Compute metrics
        if metrics is not None:
            with torch.no_grad():
                example_sim = average_pairwise_cosine_similarity(full_target.mean(1), 0, 1)
                metrics["example_sim"].update(example_sim)
                token_sim = average_pairwise_cosine_similarity(full_target, 1, 2)
                metrics["token_sim"].update(token_sim)

                if "layer_scale_mean" in metrics:
                    metrics["layer_scale_mean"].update(self.backbone)
                    metrics["layer_scale_max"].update(self.backbone)

        output = {
            "log": {
                "loss_jepa": loss,
            },
            "context": context,
            "jepa_pred": pred,
            "target": target,
            "full_target": full_target,
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
        backbone: str,
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
            backbone,
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
