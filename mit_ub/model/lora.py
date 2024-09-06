from abc import abstractmethod
from enum import StrEnum
from typing import Protocol, Sequence, Tuple, TypeVar

import torch
import torch.nn as nn
from torch import Tensor
from torchtune.modules.peft import LoRALinear


T = TypeVar("T", bound=nn.Module)


class LoRATarget(StrEnum):
    ATTENTION = "attention"
    FEEDFORWARD = "feedforward"


@torch.no_grad()
def apply_lora(
    target: nn.Linear | Tuple[Tensor, Tensor | None],
    rank: int,
    alpha: float,
    dropout: float = 0.0,
    use_bias: bool = False,
    quantize_base: bool = False,
) -> nn.Module:
    """Apply LoRA (Low-Rank Adaptation) to a linear layer or weight tensor.

    Args:
        target: Either a nn.Linear module or a tuple of (weight, bias) tensors.
        rank: Rank of the low-rank approximation.
        alpha: Scaling factor for the LoRA weights.
        dropout: Dropout probability for LoRA layers.
        use_bias: Whether to use bias in the LoRA layer.
        quantize_base: Whether to quantize the base weights.

    Returns:
        A LoRALinear module with the LoRA adaptation applied.
    """
    if isinstance(target, nn.Linear):
        # Extract w and b from the original linear layer
        w = target.weight
        b = target.bias if use_bias else None
    else:
        w, b = target
        b = b if use_bias else None

    # Create the LoRA linear layer
    d_out, d_in = w.shape
    lora_linear = LoRALinear(d_in, d_out, rank, alpha, dropout, use_bias, quantize_base)

    # Move to device of input linear layer
    lora_linear = lora_linear.to(w.device)

    # Copy over the weights and biases
    lora_linear.weight.copy_(w)
    if b is not None:
        lora_linear.bias.copy_(b)

    return lora_linear


@torch.no_grad()
def freeze_non_lora(model: nn.Module) -> None:
    """Freeze all non-LoRA parameters in the model.

    This function iterates through all modules and parameters in the given model,
    setting requires_grad to False for all parameters except those belonging to
    LoRA layers (identified by names ending with "lora_a" or "lora_b").

    Args:
        model: The neural network model to freeze.

    """
    # Freeze all non-LoRA matrices/weights
    for name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            param.requires_grad = name.endswith("lora_a") or name.endswith("lora_b")


class SupportsLoRA(Protocol):
    """Protocol for classes that support LoRA."""

    @abstractmethod
    def apply_lora(
        self: T,
        target: Sequence[LoRATarget],
        rank: int,
        alpha: float,
        dropout: float = 0.0,
        use_bias: bool = False,
        quantize_base: bool = False,
    ) -> T:
        """Apply LoRA (Low-Rank Adaptation) to the specified targets in the model.

        This method applies LoRA to the specified components of the model, allowing for
        efficient fine-tuning by modifying a small number of parameters.

        Args:
            target: Sequence of LoRATarget enum values specifying which components
                    to apply LoRA to (e.g., attention, feedforward).
            rank: The rank of the LoRA decomposition.
            alpha: The alpha parameter for LoRA, controlling the scale of the update.
            dropout: Dropout probability for the LoRA layers. Defaults to 0.0.
            use_bias: Whether to use bias in the LoRA layers. Defaults to False.
            quantize_base: Whether to quantize the base model. Defaults to False.

        Returns:
            The modified model with LoRA applied.
        """
        raise NotImplementedError
