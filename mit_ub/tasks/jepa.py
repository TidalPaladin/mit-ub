from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Final, List, Optional, Set, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from ssl_tasks.tokens import TokenMask
from torch import Tensor
from torch.distributed import ReduceOp, all_reduce
from torch.distributed import barrier as dist_barrier
from torch.optim.optimizer import Optimizer

from ..model import BACKBONES, AdaptiveViT, TransformerEncoderLayer, ViT
from ..model.pos_enc import RelativeFactorizedPosition


EPS: Final = 1e-8


@torch.compile(fullgraph=True)
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


@torch.compile(fullgraph=True)
def cosine_similarity_loss(x: Tensor, y: Tensor, eps: float = EPS) -> Tensor:
    y = 1 - F.cosine_similarity(x, y, dim=-1, eps=eps)
    return y.mean()


@torch.compile(fullgraph=True)
def sample_tokens(x: Tensor, ratio: float) -> Tensor:
    B, L_in, _ = x.shape
    L_out = int(L_in * ratio)

    # Create array of token indices
    token_idx = torch.arange(L_in).unsqueeze_(0).expand(B, -1)

    # Create a random sort order for each example
    sort_order = torch.rand_like(token_idx, dtype=torch.float32)

    # Sort each example's tokens by it's random sort order
    indices = torch.argsort(sort_order, dim=-1)[..., :L_out]
    token_idx = torch.gather(token_idx, dim=-1, index=indices)

    # Build final result with
    batch_idx = torch.arange(B).view(B, 1).expand(-1, L_out)
    return x[batch_idx.flatten(), token_idx.flatten()].view(B, L_out, -1).contiguous()


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
        weight_decay_exemptions: Set of parameter names to exempt from weight decay.
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
        weight_decay_exemptions: Set[str] = set(),
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
            weight_decay_exemptions,
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
        self.context_proj = nn.Linear(self.backbone.dim, self.backbone.dim)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.backbone.dim),
            nn.Linear(self.backbone.dim, self.backbone.dim),
        )

        # JEPA predictor
        encoder_proto = next(filter(lambda l: isinstance(l, TransformerEncoderLayer), self.backbone.modules()), None)
        if encoder_proto is None:
            raise ValueError(
                "Could not find encoder prototype in backbone. "
                "Ensure the backbone has a TransformerEncoderLayer module."
            )
        self.jepa_predictor = nn.ModuleList([deepcopy(encoder_proto) for _ in range(predictor_depth)])
        for block in self.jepa_predictor:
            block.reset_parameters()

        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_mask(self, x: Tensor, unmasked_ratio: float, scale: int) -> TokenMask:
        size = x.shape[2:]

        # For AdaptiveViT we choose a token mask that matches the size of the fixed token grid produced
        # by the ViT.
        if isinstance(self.backbone, AdaptiveViT):
            size = self.backbone.stem.equivalent_size(cast(Any, size))

        batch_size = x.shape[0]
        device = x.device
        mask = TokenMask.create(
            size,
            self.backbone.stem.patch_size,
            batch_size,
            device=device,
            mask_ratio=1 - unmasked_ratio,
            scale=scale,
        )

        # If we get unlucky and sample a complete mask, just sample again
        if not mask.mask.any():
            return self.create_mask(x, unmasked_ratio, scale)

        assert not mask.is_ragged, "Mask should not be ragged"
        return mask

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection(
            {
                "example_sim": tm.MeanMetric(),
                "token_sim": tm.MeanMetric(),
                "normalized_jepa_loss": tm.MeanMetric(),
            }
        )

    def forward(self, x: Tensor, context_mask: TokenMask, target_mask: TokenMask) -> Dict[str, Tensor]:
        # Run encoder on context
        context: Tensor = self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)

        # Sample a subset of the context as input to the predictor and project
        context = sample_tokens(context, self.context_subsample_ratio)
        context = self.context_proj(context)

        # Prepare positional encoding for target queries
        B, _, _ = context.shape
        tokenized_size = self.backbone.stem.tokenized_size(cast(Any, x.shape[2:]))
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            query = self.pos_enc.from_grid(
                tokenized_size,
                B,
                proto=context.to(torch.float32, copy=False),
                normalize=True,
                add_noise=self.training and self.backbone._position_noise,
            ).contiguous()
        query = target_mask.apply_to_tokens(query, fill_value=None)
        L = query.shape[1]

        # Run query and context through predictor
        query = torch.cat([query, context], dim=1)
        for block in self.jepa_predictor:
            block = cast(TransformerEncoderLayer, block)
            query = block(query)

        # Separate predictions from context
        pred = self.out_proj(query[:, :L])

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

    def update_weight_decay(self) -> None:
        """Update the weight decay for each parameter group based on the current training progress."""
        assert self.weight_decay_final is not None

        # Check where we are in the training loop, abort if we can't determine
        fraction_complete = self.fraction_complete
        if fraction_complete is None:
            return

        # Get optimizer and wrap as a list if necessary
        optimizers = self.optimizers()
        optimizers = [optimizers] if isinstance(optimizers, Optimizer) else optimizers
        optimizers = cast(List[Optimizer], optimizers)

        def update_weight_decay(pg: Dict[str, Any]) -> None:
            current_wd = pg["weight_decay"]
            # Some layers may be exempt from weight decay
            if current_wd == 0:
                return

            initial_wd = pg.setdefault("initial_weight_decay", current_wd)
            final_wd = self.weight_decay_final
            new_wd = initial_wd + (final_wd - initial_wd) * fraction_complete
            pg["weight_decay"] = new_wd

        # Update weight decay for each optimizer
        map(
            update_weight_decay,
            (pg for optimizer in optimizers for pg in optimizer.param_groups if "weight_decay" in pg),
        )

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
            target.roll(1, 0).lerp_(target, right_broadcast(lam, target)),
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
                target = target_mask.apply_to_tokens(full_target_mixed, fill_value=None)
            else:
                target = target_mask.apply_to_tokens(full_target, fill_value=None)

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred_dict = self(x, context_mask, target_mask)
        pred: Tensor = pred_dict["jepa"]
        context: Tensor = pred_dict["jepa_context"]

        # compute loss between target and predictor encoded latents
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss = cosine_similarity_loss(pred, target)

        # Compute metrics
        if metrics is not None:
            with torch.no_grad():
                example_sim = average_pairwise_cosine_similarity(full_target.mean(1), 0, 1)
                metrics["example_sim"].update(example_sim)
                token_sim = average_pairwise_cosine_similarity(full_target, 1, 2)
                metrics["token_sim"].update(token_sim)
                # Normalize JEPA loss by target similarity
                target_sim = average_pairwise_cosine_similarity(target.view(-1, target.shape[-1]), 0, 1)
                normalized_jepa_loss = loss / (1 - target_sim.clip(min=0, max=1 - EPS))
                metrics["normalized_jepa_loss"].update(normalized_jepa_loss)

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
        weight_decay_exemptions: Set[str] = set(),
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
            weight_decay_exemptions,
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
