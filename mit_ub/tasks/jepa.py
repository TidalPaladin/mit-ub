import math
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from lightning_utilities.core.rank_zero import rank_zero_info
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torchvision.ops import sigmoid_focal_loss

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
from ..metrics.cosine_sim import AveragePairwiseCosineSimilarity, TokenSimilarity
from ..metrics.distance import RMSPairwiseDistance, TokenRMSDistance
from ..metrics.layer_scale import MaxLayerScale, MeanLayerScale
from ..model import AdaptiveViT, AdaptiveViTConfig, ViT, ViTConfig
from ..model.helpers import compile_backend, compile_is_disabled, max_autotune
from ..model.layers.layer_scale import has_layer_scale
from ..model.layers.pos_enc import DEFAULT_POS_ENC_ACTIVATION, RelativeFactorizedPosition
from ..model.layers.transformer import TransformerDecoderLayer
from ..tokens import apply_mask, create_mask, generate_non_overlapping_mask, mask_is_ragged
from .student_teacher import EMAConfig, get_ema_momentum, synchronize_teacher, update_teacher


sigmoid_focal_loss = torch.compile(
    sigmoid_focal_loss,
    fullgraph=True,
    backend=compile_backend(),
    disable=compile_is_disabled(),
)


@torch.no_grad()
def ring_exchange(tensor: Tensor, rank: int, world_size: int) -> Tensor:
    r"""Run one iteration of ring exchange.

    The exchange sends to the next rank, and receives from the previous rank.

    Args:
        tensor: The tensor to exchange.
        rank: The rank of the current process.
        world_size: The number of processes in the distributed group.

    Returns:
        The exchanged tensor
    """
    if world_size == 1 or not torch.distributed.is_initialized():
        return tensor
    recv_tensor = torch.zeros_like(tensor)
    send_op = dist.P2POp(dist.isend, tensor, (rank + 1) % world_size)
    recv_op = dist.P2POp(dist.irecv, recv_tensor, (rank - 1 + world_size) % world_size)
    reqs = dist.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return recv_tensor


@torch.no_grad()
def ring_exchange_all(tensor: Tensor, rank: int, world_size: int) -> Iterator[Tensor]:
    r"""Run ring exchange across all ranks.

    The final tensor received in the exchange is the local tensor (i.e. ``tensor``).

    Args:
        tensor: The tensor to exchange.
        rank: The rank of the current process.
        world_size: The number of processes in the distributed group.

    Returns:
        An iterator over the tensors exchanged at each round of the ring exchange.
    """
    for _ in range(world_size):
        tensor = ring_exchange(tensor, rank, world_size)
        yield tensor


@torch.compile(
    fullgraph=True,
    backend=compile_backend(),
    options={
        "max_autotune": max_autotune(),
        "epilogue_fusion": True,
        "shape_padding": True,
        "triton.cudagraph_trees": max_autotune(),
    },
    disable=compile_is_disabled(),
)
def compute_siglip_logits(x1: Tensor, x2: Tensor, t: Tensor, b: Tensor) -> Tensor:
    r"""Compute the logits for the SigLIP loss.

    Logits are computed as:

    .. math::
        \text{logits} = \text{x1} \cdot \text{x2}^\top \cdot \text{t} + \text{b}

    Args:
        x1: The first set of embeddings.
        x2: The second set of embeddings.
        t: The temperature parameter.
        b: The bias parameter.

    Returns:
        The logits for the SigLIP loss.
    """
    return torch.matmul(x1, x2) * t.exp() + b


def compute_siglip_loss(
    x1: Tensor,
    x2: Tensor,
    target: Tensor,
    t: Tensor,
    b: Tensor,
    rank: int,
    world_size: int,
    eps: float = 1e-12,
    gamma: float | None = 2.0,
) -> Tensor:
    r"""Compute the SigLIP loss across all ranks.

    Args:
        x1: The first set of embeddings.
        x2: The second set of embeddings.
        target: The target embeddings.
        t: The temperature parameter.
        b: The bias parameter.
        rank: The rank of the current process.
        world_size: The number of processes in the distributed group.
        eps: The epsilon value to use for normalization.
        gamma: If set, use focal loss with the provided gamma value.

    Returns:
        The SigLIP loss.
    """
    x1 = F.normalize(x1, dim=-1, eps=eps)
    x2 = F.normalize(x2, dim=-1, eps=eps)

    x2_globals = ring_exchange_all(x2, rank, world_size)
    x1_globals = ring_exchange_all(x1, rank, world_size)

    # Target is all 0 when crossing ranks
    null_target = torch.zeros_like(target)

    loss = x1.new_zeros(())
    for i, (x1_global, x2_global) in enumerate(zip(x1_globals, x2_globals)):
        # The SigLIP target will be all 0 unless we are on the last iteration of ring exchange.
        # In the last iteration, x2_global should match x2 (i.e. the local target).
        _target = target if i == world_size - 1 else null_target

        # We want to ensure gradients flow through both x1 and x2. However, gradients do not
        # flow through the ring exchange operations. To overcome this, we exchange both x1 and x2,
        # compute the loss using the local counterpart of the exchanged tensor, and then combine
        # the losses for the two permutations.
        logits_x1 = compute_siglip_logits(x1, x2_global.mT, t, b)
        logits_x2 = compute_siglip_logits(x1_global, x2.mT, t, b)

        if gamma is not None:
            loss_x1 = sigmoid_focal_loss(logits_x1, _target, reduction="sum", gamma=gamma, alpha=-1)
            loss_x2 = sigmoid_focal_loss(logits_x2, _target, reduction="sum", gamma=gamma, alpha=-1)
        else:
            loss_x1 = F.binary_cross_entropy_with_logits(logits_x1, _target, reduction="sum")
            loss_x2 = F.binary_cross_entropy_with_logits(logits_x2, _target, reduction="sum")

        loss += (loss_x1 + loss_x2) / (target.numel() * world_size * 2)

    return loss


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
        siglip_weight: Weight of the SigLIP loss. If 0, SigLIP does not contribute to the backbone gradients.
        siglip_gamma: Gamma value for the SigLIP focal loss. If None, use binary cross entropy.
        siglip_t: Temperature parameter for the SigLIP loss.
        siglip_b: Bias parameter for the SigLIP loss.
        trainable_siglip_params: If True, the SigLIP parameters are trainable.
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

    # SigLIP parameters
    siglip_weight: float = 1.0
    siglip_gamma: float | None = 2.0
    siglip_t: float = 10.0
    siglip_b: float = -10.0
    trainable_siglip_params: bool = True

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
        if self.siglip_gamma is not None and self.siglip_gamma <= 0:
            raise ValueError("siglip_gamma must be non-negative")


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
            norm=True,
            norm_type=self.backbone.config.norm_type,
            activation=self.backbone.get_external_activation(default=DEFAULT_POS_ENC_ACTIVATION),
        )

        # JEPA predictor
        self.jepa_predictor = nn.ModuleList(
            [
                self.backbone.create_decoder_layer(i, self_attn=self.jepa_config.self_attn, kv_norm=False)
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
        rank_zero_info(f"Using SigLIP weight: {self.jepa_config.siglip_weight}")
        self.siglip_head = self.backbone.create_head(
            self.backbone.config.dim,
            pool_type=None,
            use_mlp=self.jepa_config.mlp_tower,
            input_norm=self.jepa_config.tower_input_norm,
        )
        self.siglip_t = nn.Parameter(torch.empty(1))
        self.siglip_b = nn.Parameter(torch.empty(1))
        nn.init.constant_(self.siglip_t, math.log(self.jepa_config.siglip_t))
        nn.init.constant_(self.siglip_b, self.jepa_config.siglip_b)
        self.siglip_t.requires_grad = self.jepa_config.trainable_siglip_params
        self.siglip_b.requires_grad = self.jepa_config.trainable_siglip_params

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
                "example_sim": AveragePairwiseCosineSimilarity(self.backbone.config.dim),
                "micro_token_sim": TokenSimilarity(self.backbone.config.dim),
                "macro_token_sim": AveragePairwiseCosineSimilarity(self.backbone.config.dim),
                "example_rms": RMSPairwiseDistance(self.backbone.config.dim),
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
            metrics["siglip_t"] = tm.MeanMetric()
            metrics["siglip_b"] = tm.MeanMetric()

        return metrics

    def forward(self, x: Tensor, context_mask: Tensor, target_mask: Tensor) -> Dict[str, Tensor]:
        # Run encoder on context
        torch.cuda.nvtx.range_push("context_backbone")
        context, cls_token = cast(
            Tuple[Tensor, Tensor], self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)
        )
        torch.cuda.nvtx.range_pop()
        B, L, D = context.shape

        # Sample a subset of the context as input to the predictor and project
        if self.jepa_config.context_subsample_ratio < 1.0:
            subsample_mask = create_mask(
                (L,),
                mask_ratio=1 - self.jepa_config.context_subsample_ratio,
                batch_size=B,
                device=context.device,
            )
            context = apply_mask(subsample_mask, context, fill_value=None)
            L = context.shape[1]

        # Add CLS token to context
        context = torch.cat([cls_token.view(B, 1, -1).expand(B, -1, -1), context], dim=1)
        assert context.shape == (B, L + 1, D)

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
        return {"jepa": pred, "jepa_context": context, "jepa_cls_token": cls_token}

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
        with torch.inference_mode():
            self.teacher_backbone.eval()
            torch.cuda.nvtx.range_push("ema_backbone")
            full_target, target_cls_token = cast(Tuple[Tensor, Tensor], self.teacher_backbone(x, reshape=False))
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
                target_cls_token = mixup(target_cls_token, mixup_weight)
                siglip_target = mixup(siglip_target, mixup_weight)
                torch.cuda.nvtx.range_pop()
            else:
                mixup_weight = None

            target = apply_mask(target_mask, full_target, fill_value=None)

        # clone inference tensors if training for use with autograd
        if self.training:
            x = x.clone()
            target = target.clone()
            siglip_target = siglip_target.clone()
            target_cls_token = target_cls_token.clone()
            if mixup_weight is not None:
                mixup_weight = mixup_weight.clone()

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred_dict = self(x, context_mask, target_mask)
        pred: Tensor = pred_dict["jepa"]
        pred_cls_token = pred_dict["jepa_cls_token"]
        context: Tensor = pred_dict["jepa_context"]

        # compute loss between target and predictor encoded latents
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        loss_jepa = F.smooth_l1_loss(pred, target)

        # Compute siglip loss across all ranks. When siglip_weight is 0, a stop gradient is applied to the CLS token.
        _pred_cls_token = pred_cls_token.detach() if self.jepa_config.siglip_weight == 0 else pred_cls_token
        siglip_pred_token = self.siglip_head(_pred_cls_token)
        siglip_target_token = self.siglip_head(target_cls_token)
        loss_siglip = compute_siglip_loss(
            siglip_pred_token,
            siglip_target_token,
            siglip_target,
            self.siglip_t,
            self.siglip_b,
            self.trainer.global_rank,
            self.trainer.world_size,
            gamma=self.jepa_config.siglip_gamma,
        )

        # Compute metrics
        if metrics is not None:
            with torch.inference_mode():
                torch.cuda.nvtx.range_push("metrics")
                metrics["jepa_loss"].update(loss_jepa)
                metrics["siglip_loss"].update(loss_siglip)
                metrics["example_sim"].update(target_cls_token)
                metrics["example_rms"].update(target_cls_token)
                metrics["micro_token_sim"].update(full_target)
                metrics["micro_token_rms"].update(full_target)
                metrics["macro_token_sim"].update(full_target)
                metrics["macro_token_rms"].update(full_target)

                if "layer_scale_mean" in metrics:
                    metrics["layer_scale_mean"].update(self.backbone)
                    metrics["layer_scale_max"].update(self.backbone)

                if "ema_momentum" in metrics:
                    metrics["ema_momentum"].update(self.get_ema_momentum())
                if "siglip_t" in metrics:
                    metrics["siglip_t"].update(self.siglip_t)
                if "siglip_b" in metrics:
                    metrics["siglip_b"].update(self.siglip_b)
                torch.cuda.nvtx.range_pop()

        output = {
            "log": {
                "loss_jepa": loss_jepa,
                "loss_siglip": loss_siglip * (self.jepa_config.siglip_weight or 1.0),
            },
            "context": context,
            "cls_token": pred_cls_token,
            "target_cls_token": target_cls_token,
            "jepa_pred": pred,
            "target": target,
            "full_target": full_target,
            "mixup_weight": mixup_weight,
        }
        return output

    def on_after_backward(self, *args, **kwargs):
        if self.global_step == 0:
            for name, param in self.named_parameters():
                if param.grad is None and param.requires_grad:
                    raise RuntimeError(f"Gradient is None for parameter {name}")

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
        probe_key: str = "target_cls_token",
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
        self.probe_key = probe_key
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
