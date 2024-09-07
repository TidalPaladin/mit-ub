from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Set, Tuple, cast

import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import Mode, State
from deep_helpers.tasks import Task
from ssl_tasks.contrastive.loss import PointwiseContrastiveEmbeddingLoss
from ssl_tasks.tokens import TokenMask
from torch import Tensor
from torch.distributed import ReduceOp, all_gather, all_reduce
from torch.distributed import barrier as dist_barrier
from torch.distributed import is_initialized as dist_is_initialized

from ..model import BACKBONES, AdaptiveViT, TransformerEncoderLayer, ViT
from ..model.pos_enc import RelativeFactorizedPosition


class NormallyDistributed(nn.Module):
    r"""Criterion that measures if a tensor is normally distributed.

    The following factors are considered:
        - Zero-mean
        - Unit variance

    Args:
        dim: The dimension to measure the distribution of.
    """

    def __init__(self, dim: int | None = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        var, mean = torch.var_mean(x, dim=self.dim, unbiased=False, keepdim=True)
        mean_loss = mean.abs().mean()
        var_loss = (var - 1).abs().mean()
        total_loss = mean_loss + var_loss
        return total_loss


class JEPA(Task):
    def __init__(
        self,
        backbone: str,
        context_ratio: float = 0.5,
        context_scale: int = 4,
        target_ratio: float = 0.25,
        target_scale: int = 2,
        ema_alpha: float = 0.95,
        activation_clip: float | None = None,
        margin: float | None = 0.5,
        loss_fn: str = "cosine",
        distribution_loss: bool = False,
        predictor_depth: int = 4,
        dist_gather: bool = False,
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
        assert self.context_ratio > 0
        assert self.target_ratio > 0
        self.ema_alpha = ema_alpha
        self.activation_clip = activation_clip
        assert self.activation_clip is None or self.activation_clip > 0
        self.dist_gather = dist_gather

        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        self.ema_backbone = deepcopy(self.backbone)
        for p in self.ema_backbone.parameters():
            p.requires_grad = False

        if loss_fn == "cosine":
            self.jepa_loss = nn.CosineEmbeddingLoss()
        elif loss_fn == "mse":
            self.jepa_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn}, expected 'cosine' or 'mse'")

        self.normally_distributed = NormallyDistributed() if distribution_loss else None
        self.jepa_query = nn.Parameter(torch.empty(1, 1, self.backbone.dim))
        torch.nn.init.trunc_normal_(self.jepa_query, mean=0, std=1)

        self.pos_enc_2d = RelativeFactorizedPosition(2, self.backbone.dim)

        encoder_proto = next(filter(lambda l: isinstance(l, TransformerEncoderLayer), self.backbone.modules()), None)
        if encoder_proto is None:
            raise ValueError(
                "Could not find encoder prototype in backbone. "
                "Ensure the backbone has a TransformerEncoderLayer module."
            )
        self.jepa_predictor = nn.ModuleList([deepcopy(encoder_proto) for _ in range(predictor_depth)])

        self.contrastive_loss = PointwiseContrastiveEmbeddingLoss(margin=margin) if margin is not None else None
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_context_mask(self, x: Tensor) -> TokenMask:
        size = x.shape[2:]
        # For AdaptiveViT we choose a token mask that matches the size of the fixed token grid produced
        # by the ViT.
        if isinstance(self.backbone, AdaptiveViT):
            size = self.backbone.equivalent_size_2d(*size)
        batch_size = x.shape[0]
        device = x.device
        return TokenMask.create(
            size,
            self.backbone.patch_size[-len(size) :],
            batch_size,
            device=device,
            # Flip this so we get context_mask unmasked
            mask_ratio=1 - self.context_ratio,
            scale=self.context_scale,
        )

    def create_target_mask(self, x: Tensor) -> TokenMask:
        size = x.shape[2:]
        # For AdaptiveViT we choose a token mask that matches the size of the fixed token grid produced
        # by the ViT.
        if isinstance(self.backbone, AdaptiveViT):
            size = self.backbone.equivalent_size_2d(*size)
        batch_size = x.shape[0]
        device = x.device
        return TokenMask.create(
            size,
            self.backbone.patch_size[-len(size) :],
            batch_size,
            device=device,
            # Flip this so we get target_mask unmasked
            mask_ratio=1 - self.target_ratio,
            scale=self.target_scale,
        )

    def create_metrics(self, state: State) -> tm.MetricCollection:
        r"""Gets a MetricCollection for a given state"""
        return tm.MetricCollection({})

    def forward(self, x: Tensor, context_mask: TokenMask, target_mask: TokenMask) -> Dict[str, Tensor]:
        # Run encoder on context and broadcast back to full size with 0 padding
        dense_context: Tensor = self.backbone(x, mask=context_mask, mask_fill_value=None, reshape=False)
        context = context_mask.restore_tokens(dense_context, 0)

        # Create empty queries w/ position encoding that forms the initial predictor input
        B, _, D = context.shape
        tokenized_size = self.backbone.tokenized_size(*x.shape[2:])
        if is_3d := x.ndim == 5:
            raise NotImplementedError("3D not implemented")
        else:
            query = self.pos_enc_2d.from_grid(tokenized_size, B, proto=context, normalize=True, requires_grad=False)
        query = query.contiguous()
        query += self.jepa_query.type_as(query)

        # Use xor mask to inject encoder context into queries that aren't part of the target mask.
        # Query now contains context only at locations that are not part of the target.
        with torch.no_grad():
            xor_mask = (context_mask.mask ^ target_mask.mask).unsqueeze_(-1)
        query = torch.where(xor_mask, query, context)

        # Create a context or target mask.
        # Since context and target may overlap, we may end up with an inconsistent number of tokens
        # for each example in the batch. To resolve this we will pad to match the largest number
        # of tokens in an example, and adjust the ALiBi positions such that these padding tokens
        # are masked in the predictor.
        mask = context_mask.mask | target_mask.mask
        mask = TokenMask(mask, context_mask.size, context_mask.patch_size)
        query = mask.apply_to_tokens(query, fill_value=None)

        # Run the queries and ALiBi positions through the predictor
        B, L = query.shape[:2]
        for block in self.jepa_predictor:
            block = cast(TransformerEncoderLayer, block)
            query = block(query)

        # Extract only the target queries from the full set of queries
        query = mask.restore_tokens(query, 0)
        query = target_mask.apply_to_tokens(query, fill_value=None)

        return {"jepa": query, "jepa_context": dense_context}

    @torch.no_grad()
    def update_ema(self):
        for i, (ema_param, param) in enumerate(zip(self.ema_backbone.parameters(), self.backbone.parameters())):
            ema_param.data.mul_(self.ema_alpha).add_(param.data, alpha=1 - self.ema_alpha)
            assert not ema_param.requires_grad
        self.synchronize_ema_weights()

    @torch.no_grad()
    def synchronize_ema_weights(self):
        if self.trainer.world_size > 1:
            for ema_param in self.ema_backbone.parameters():
                # There seems to be sporadic deadlocks in DDP, so we use barriers to keep things synchronized
                dist_barrier()
                all_reduce(ema_param.data, op=ReduceOp.SUM)
                ema_param.data /= self.trainer.world_size
            dist_barrier()

    @torch.no_grad()
    def weight_histogram(self, module: nn.Module, bins: int = 100) -> Tuple[Tensor, Tensor]:
        r"""Create a histogram of weights in a given module."""
        weights = torch.cat([p.detach().float().ravel() for p in module.parameters() if p.requires_grad])
        result = tuple(t.cpu().numpy() for t in torch.histogram(weights.cpu(), bins=bins))
        return cast(Tuple[Tensor, Tensor], result)

    @torch.no_grad()
    def tensor_histogram(self, tensor: Tensor, bins: int = 100) -> Tuple[Tensor, Tensor]:
        r"""Create a histogram of weights in a given module."""
        tensor = tensor[~tensor.isnan()].detach().float().ravel()
        result = tuple(t.cpu().numpy() for t in torch.histogram(tensor.cpu(), bins=bins))
        return cast(Tuple[Tensor, Tensor], result)

    def clip_activations(self, x: Tensor) -> Tensor:
        if self.activation_clip is None:
            return x
        return torch.clip(x, -self.activation_clip, self.activation_clip)

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

        # generate context and target masks
        context_mask = self.create_context_mask(x)
        target_mask = self.create_target_mask(x)
        assert not context_mask.mask.all()

        # generate ground truth with forward pass of ema backbone on unmasked image
        with torch.no_grad():
            self.ema_backbone.eval()
            full_target: Tensor = self.ema_backbone(x, reshape=False)
            target = target_mask.apply_to_tokens(full_target, fill_value=None)
            target = self.clip_activations(target)

        # generate predictions by encoding the context and then running the encoded context
        # plus the positional target queries through the predictor
        pred_dict = self(x, context_mask, target_mask)
        pred: Tensor = pred_dict["jepa"]
        context: Tensor = pred_dict["jepa_context"]

        # compute loss between target and predictor encoded latents
        assert pred.shape == target.shape, f"Prediction shape {pred.shape} does not match target shape {target.shape}"
        if isinstance(self.jepa_loss, nn.CosineEmbeddingLoss):
            N = pred.shape[0]
            loss = self.jepa_loss(pred.view(N, -1), target.view(N, -1), target.new_full((N,), 1, dtype=torch.long))
        elif isinstance(self.jepa_loss, nn.MSELoss):
            loss = self.jepa_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss function: {self.jepa_loss}")

        # compute distribution loss
        loss_distribution = self.normally_distributed(pred) if self.normally_distributed is not None else None

        # compute contrastive loss for collapse mitigation
        if self.contrastive_loss is not None:
            pred_pool = pred.mean(1)

            # Gather average-pooled predictions from all GPUs if requested
            if self.dist_gather and dist_is_initialized():
                gathered_preds = [torch.zeros_like(pred_pool) for _ in range(self.trainer.world_size)]
                all_gather(gathered_preds, pred_pool)
                pred_pool = torch.cat(gathered_preds, dim=0)
            loss_contrastive = self.contrastive_loss(pred_pool, pred_pool).sum()
        else:
            loss_contrastive = None

        # calculate descriptive statistics for target
        with torch.no_grad():
            target_var, target_mean = torch.var_mean(target.view(-1, self.backbone.dim), dim=-1)
            target_var = target_var.mean()
            target_mean = target_mean.mean()

        # combine prediction and target into a single tensor that requires grad.
        # this can be used with a supervised loss to backprop through the backbone.
        combined = full_target + context_mask.restore_tokens(context, 0)

        output = {
            "log": {
                "loss_jepa": loss,
                "var": target_var,
                "mean": target_mean,
            },
            "jepa_pred": pred,
            "target": target,
            "combined": combined,
        }
        if loss_contrastive is not None:
            output["log"]["loss_contrastive"] = loss_contrastive
        if loss_distribution is not None:
            output["log"]["loss_distribution"] = loss_distribution

        if self.trainer.global_step % 100 == 0 or (not self.training and batch_idx == 0):
            with torch.no_grad():
                target_std = target.std(dim=0)
                histograms = {
                    "target_hist": self.tensor_histogram(target),
                    "pred_hist": self.tensor_histogram(pred),
                    "std_hist": self.tensor_histogram(target_std),
                }
                output.update(**histograms)

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
        ema_alpha: float = 0.95,
        activation_clip: float | None = None,
        margin: float | None = 0.5,
        loss_fn: str = "cosine",
        distribution_loss: bool = False,
        predictor_depth: int = 4,
        dist_gather: bool = False,
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
    ):
        super().__init__(
            backbone,
            context_ratio,
            context_scale,
            target_ratio,
            target_scale,
            ema_alpha,
            activation_clip,
            margin,
            loss_fn,
            distribution_loss,
            predictor_depth,
            dist_gather,
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
        self.linear_probe = self.create_probe_head()

    @abstractmethod
    def create_probe_head(self) -> nn.Module:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def create_metrics(self, state: State) -> tm.MetricCollection:
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def step_linear_probe(
        self, batch: Dict[str, Any], output: Dict[str, Any], metrics: tm.MetricCollection | None
    ) -> Dict[str, Any]:
        r"""Compute the linear probe loss and update the metrics"""
        raise NotImplementedError  # pragma: no cover

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
