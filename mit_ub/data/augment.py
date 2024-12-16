from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import Tensor
from torchvision.transforms.v2 import Compose, RandomApply, RandomChoice, Transform

from mit_ub.model.helpers import compile_is_disabled


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def uniform_noise(x: Tensor, min: float = -0.2, max: float = 0.2, clip: bool = True) -> Tensor:
    """Apply uniform noise to tensor with same shape as input.

    Args:
        x: Input tensor to match shape
        min: Minimum value for uniform distribution
        max: Maximum value for uniform distribution
        clip: Whether to clip the result to the range :math:`[0, 1]`

    Returns:
        Input with uniform noise applied
    """
    noise = torch.rand_like(x) * (max - min) + min
    result = x + noise
    if clip:
        result = result.clip(min=0, max=1)
    return result


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def salt_pepper_noise(x: Tensor, prob: float | Tuple[float, float] = (0.01, 0.05)) -> Tensor:
    """Apply salt and pepper noise to tensor with same shape as input.

    Args:
        x: Input tensor to match shape
        prob: Probability of setting a pixel to min or max, or a tuple of the form (min, max)
            giving the range of the uniform distribution to sample from

    Returns:
        Input with salt and pepper noise applied
    """
    if isinstance(prob, Sequence):
        low, high = prob
        p = x.new_empty(1).uniform_(low, high)
    else:
        p = x.new_tensor(prob)
    mask = torch.rand_like(x) < p
    value = torch.rand_like(x).round()
    return torch.where(mask, value, x)


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def multiplicative_noise(x: Tensor, scale: float = 0.2, clip: bool = True) -> Tensor:
    """Apply multiplicative noise to tensor with same shape as input.

    Args:
        x: Input tensor to match shape
        scale: Scale factor for noise
        clip: Whether to clip the result to the range :math:`[0, 1]`

    Returns:
        Input with multiplicative noise applied
    """
    result = x * (1 + torch.randn_like(x) * scale)
    if clip:
        result = torch.clamp(result, min=0, max=1)
    return result


class UniformNoise(Transform):

    def __init__(self, min: float = -0.2, max: float = 0.2, clip: bool = True):
        super().__init__()
        self.min = min
        self.max = max
        self.clip = clip

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return uniform_noise(
            inpt,
            min=self.min,
            max=self.max,
            clip=self.clip,
        )


class SaltPepperNoise(Transform):

    def __init__(self, prob: float | Tuple[float, float] = (0.01, 0.05)):
        super().__init__()
        self.prob = prob

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return salt_pepper_noise(
            inpt,
            prob=self.prob,
        )


class MultiplicativeNoise(Transform):

    def __init__(self, scale: float = 0.2, clip: bool = True):
        super().__init__()
        self.scale = scale
        self.clip = clip

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return multiplicative_noise(
            inpt,
            scale=self.scale,
            clip=self.clip,
        )


class RandomNoise(Compose):

    def __init__(
        self,
        scale: float = 0.2,
        salt_pepper_prob: float | Tuple[float, float] = (0.01, 0.05),
        clip: bool = True,
    ):
        primary_noise = RandomApply(
            [
                RandomChoice(
                    [
                        UniformNoise(min=-scale, max=scale, clip=clip),
                        MultiplicativeNoise(scale=scale, clip=clip),
                    ]
                )
            ],
            p=0.5,
        )
        salt_pepper_noise = RandomApply([SaltPepperNoise(prob=salt_pepper_prob)], p=0.5)
        super().__init__([primary_noise, salt_pepper_noise])
