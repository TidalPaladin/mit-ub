from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Final, List, Optional, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from deep_helpers.tokens import apply_mask, create_mask
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce
from torch.distributed import barrier as dist_barrier
from torch.optim.optimizer import Optimizer

from ..model import BACKBONES, AdaptiveViT, ViT, compile_is_disabled
from ..model.pos_enc import RelativeFactorizedPosition
from ..model.transformer import TransformerDecoderLayer


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
    y.sub(1 / N).mul(N / (N - 1))
    return y


class JEPA(Task):
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

    backbone: ViT | AdaptiveViT

    def __init__(
        self,
        backbone: str,
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        context_subsample_ratio: float = 0.5,
        ema_alpha: float = 0.95,
        momentum_schedule: bool = False,
        predictor_depth: int = 4,
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
        weight_decay_final: float | None = None,
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
        self.context_ratio = context_ratio
        self.context_scale = context_scale
        self.target_ratio = target_ratio
        self.target_scale = target_scale
        self.context_subsample_ratio = context_subsample_ratio
        assert self.context_ratio > 0
        assert self.target_ratio > 0
        assert self.context_subsample_ratio > 0
        self.ema_alpha = ema_alpha
        self.momentum_schedule = momentum_schedule
        self.weight_decay_final = weight_decay_final
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob

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
            [self.backbone.create_decoder_layer(i, stochastic_depth=0.0) for i in range(predictor_depth)]
        )
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_mask(self, x: Tensor, unmasked_ratio: float, scale: int) -> Tensor:
        batch_size = x.shape[0]
        device = x.device
        size = self.backbone.stem.tokenized_size(cast(Any, x.shape[2:]))
        mask = create_mask(
            size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size,
            scale=scale,
            device=device,
        )
        return mask

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection(
            {
                "example_sim": tm.MeanMetric(),
                "token_sim": tm.MeanMetric(),
            }
        )

    def forward(self, x: Tensor, context_mask: Tensor, target_mask: Tensor) -> Dict[str, Tensor]:
        # Run encoder on context
        context: Tensor = self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)
        B, L, _ = context.shape

        # Sample a subset of the context as input to the predictor and project
        if self.context_subsample_ratio < 1.0:
            subsample_mask = create_mask((L,), mask_ratio=1 - self.context_subsample_ratio, batch_size=B)
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
        if not self.momentum_schedule:
            return self.ema_alpha
        fraction_complete = self.fraction_complete
        fraction_complete = fraction_complete if fraction_complete is not None else 0.0
        return self.ema_alpha + (1 - self.ema_alpha) * fraction_complete

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
        assert self.weight_decay_final is not None

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
            final_wd = self.weight_decay_final
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
        dist = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha)
        lam = dist.sample(torch.Size((N,))).to(x.device)

        # Generate mask of mixup samples
        mixup_mask = torch.rand_like(lam) < self.mixup_prob

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
        if self.weight_decay_final is not None:
            self.update_weight_decay()

        # generate context and target masks
        target_mask = self.create_mask(x, self.target_ratio, self.target_scale)
        context_mask = self.create_mask(x, self.context_ratio, self.context_scale)

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            self.ema_backbone.eval()
            full_target: Tensor = self.ema_backbone(x, reshape=False)

            # apply mixup, not overwriting full_target
            if self.training and self.mixup_prob > 0:
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
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        context_subsample_ratio: float = 0.5,
        ema_alpha: float = 0.95,
        momentum_schedule: bool = False,
        predictor_depth: int = 4,
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
        weight_decay_final: float | None = None,
    ):
        super().__init__(
            backbone,
            context_ratio,
            context_scale,
            target_ratio,
            target_scale,
            context_subsample_ratio,
            ema_alpha,
            momentum_schedule,
            predictor_depth,
            mixup_alpha,
            mixup_prob,
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
            weight_decay_final,
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
