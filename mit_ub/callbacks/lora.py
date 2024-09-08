from typing import List

import pytorch_lightning as pl
from lightning_utilities.core.rank_zero import rank_zero_info
from pytorch_lightning.callbacks import Callback

from ..model.backbone import AdaptiveViT, ViT
from ..model.lora import LoRATarget, SupportsLoRA


class LoRACallback(Callback):
    """LoRA (Low-Rank Adaptation) callback for PyTorch Lightning.

    This callback applies LoRA to specified targets in a model's backbone,
    allowing for efficient fine-tuning by modifying a small number of parameters.
    The target model should have a :class:`ViT` or :class:`AdaptiveViT` backbone.

    Attributes:
        targets: List of LoRA targets to apply the adaptation to.
        rank: Rank of the low-rank approximation.
        alpha: Scaling factor for the LoRA weights.
        dropout: Dropout probability for LoRA layers.
        quantize_base: Whether to quantize the base weights.
        freeze_stem: Whether to freeze the stem parameters.
        freeze_norm: Whether to freeze the norm parameters.

    Raises:
        AttributeError: If the model does not have a `backbone` attribute.
        TypeError: If the `backbone` attribute is not an instance of :class:`ViT` or :class:`AdaptiveViT`.
    """

    def __init__(
        self,
        targets: List[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        quantize_base: bool = False,
        freeze_stem: bool = True,
        freeze_norm: bool = True,
    ):
        self.targets = targets
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.quantize_base = quantize_base
        self.freeze_stem = freeze_stem
        self.freeze_norm = freeze_norm

    def on_fit_start(self, _: pl.Trainer, pl_module: pl.LightningModule):
        rank_zero_info(
            f"Applying LoRA to targets: {[target.value for target in self.targets]}, rank: {self.rank}, "
            f"alpha: {self.alpha}, dropout: {self.dropout}, quantize_base: {self.quantize_base}"
        )
        if not hasattr(pl_module, "backbone"):
            raise AttributeError("LoRA callback requires a `backbone` module")
        elif not isinstance(pl_module.backbone, (ViT, AdaptiveViT)):
            raise TypeError("LoRA callback requires a `backbone` module of type ViT or AdaptiveViT")
        self.apply_lora(pl_module.backbone)

    def apply_lora(self, model: ViT | AdaptiveViT):
        # Apply LoRA to all modules that support it
        for module in model.modules():
            if isinstance(module, SupportsLoRA):
                module.apply_lora(self.targets, self.rank, self.alpha, self.dropout, self.quantize_base)

        # Freeze/unfreeze norm layers
        for module in model.modules():
            if isinstance(module, model.norm_layer):
                for param in module.parameters():
                    param.requires_grad = not self.freeze_norm

        # Handle non-LoRA stem modules
        for name, param in model.stem.named_parameters():
            if "lora" not in name:
                param.requires_grad = not self.freeze_stem
