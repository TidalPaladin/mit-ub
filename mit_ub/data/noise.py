from typing import Any, Dict, Final, List, Sequence, Tuple, cast

import torch
from torch import Tensor
from torchvision.transforms.v2 import Compose, RandomApply, Transform


UNIFORM_NOISE_MIN: Final = -0.2
UNIFORM_NOISE_MAX: Final = 0.2
MULTIPLICATIVE_NOISE_MIN: Final = 0.02
MULTIPLICATIVE_NOISE_MAX: Final = 0.2
SALT_PEPPER_NOISE_MIN: Final = 0.01
SALT_PEPPER_NOISE_MAX: Final = 0.05
DEFAULT_NOISE_PROB: Final = 0.25


def to_tuple(x: float | Sequence[float]) -> Tuple[float, float]:
    if isinstance(x, Sequence):
        assert len(x) == 2, "Expected a sequence of length 2"
        return cast(Tuple[float, float], tuple(x))
    else:
        return (x, x)


@torch.no_grad()
def uniform_noise(
    x: Tensor, min: float = UNIFORM_NOISE_MIN, max: float = UNIFORM_NOISE_MAX, clip: bool = True
) -> Tensor:
    """Apply uniform noise to tensor with same shape as input.

    Args:
        x: Input tensor to match shape
        min: Minimum value for uniform distribution
        max: Maximum value for uniform distribution
        clip: Whether to clip the result to the range :math:`[0, 1]`

    Returns:
        Input with uniform noise applied
    """
    noise = torch.empty_like(x).uniform_(min, max)
    result = noise.add_(x)
    if clip:
        result.clip_(min=0, max=1)
    return result


@torch.no_grad()
def salt_pepper_noise(x: Tensor, min: float = SALT_PEPPER_NOISE_MIN, max: float = SALT_PEPPER_NOISE_MAX) -> Tensor:
    """Apply salt and pepper noise to tensor with same shape as input.

    Args:
        x: Input tensor to match shape
        min: Minimum probability of setting a pixel to min or max
        max: Maximum probability of setting a pixel to min or max

    Returns:
        Input with salt and pepper noise applied
    """
    p = x.new_empty(1).uniform_(min, max)
    mask = torch.rand_like(x).lt_(p).bool()
    value = torch.randint_like(x, 0, 2)
    return torch.where(mask, value, x)


@torch.no_grad()
def multiplicative_noise(
    x: Tensor, min: float = MULTIPLICATIVE_NOISE_MIN, max: float = MULTIPLICATIVE_NOISE_MAX, clip: bool = True
) -> Tensor:
    """Apply multiplicative noise to tensor with same shape as input.

    Args:
        x: Input tensor to match shape
        min: Minimum scale factor for noise
        max: Maximum scale factor for noise
        clip: Whether to clip the result to the range :math:`[0, 1]`

    Returns:
        Input with multiplicative noise applied
    """
    scale = torch.empty(1).uniform_(min, max).item()
    result = torch.empty_like(x).normal_(mean=1.0, std=scale).clip_(min=0, max=2)
    result = result.mul_(x)
    if clip:
        result.clamp_(min=0, max=1)
    return result


class UniformNoise(Transform):

    def __init__(self, min: float = UNIFORM_NOISE_MIN, max: float = UNIFORM_NOISE_MAX, clip: bool = True):
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

    def __init__(self, prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX)):
        super().__init__()
        self.min, self.max = to_tuple(prob)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return salt_pepper_noise(
            inpt,
            min=self.min,
            max=self.max,
        )


class MultiplicativeNoise(Transform):

    def __init__(self, min: float = MULTIPLICATIVE_NOISE_MIN, max: float = MULTIPLICATIVE_NOISE_MAX, clip: bool = True):
        super().__init__()
        self.min = min
        self.max = max
        self.clip = clip

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return multiplicative_noise(
            inpt,
            min=self.min,
            max=self.max,
            clip=self.clip,
        )


class RandomNoise(Compose):

    def __init__(
        self,
        prob: float = DEFAULT_NOISE_PROB,
        uniform_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX),
        multiplicative_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX),
        salt_pepper_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
        clip: bool = True,
    ):
        self.uniform_scale = to_tuple(uniform_scale)
        self.multiplicative_scale = to_tuple(multiplicative_scale)
        self.salt_pepper_prob = to_tuple(salt_pepper_prob)
        self.prob = prob
        self.clip = clip
        uniform_noise = RandomApply(
            [UniformNoise(min=self.uniform_scale[0], max=self.uniform_scale[1], clip=self.clip)], p=prob
        )
        multiplicative_noise = RandomApply(
            [MultiplicativeNoise(min=self.multiplicative_scale[0], max=self.multiplicative_scale[1], clip=self.clip)],
            p=prob,
        )
        salt_pepper_noise = RandomApply([SaltPepperNoise(prob=self.salt_pepper_prob)], p=prob)
        super().__init__([uniform_noise, multiplicative_noise, salt_pepper_noise])

    def apply_batched(self, x: Tensor) -> Tensor:
        return apply_noise_batched(
            x,
            prob=self.prob,
            uniform_scale=self.uniform_scale,
            multiplicative_scale=self.multiplicative_scale,
            salt_pepper_prob=self.salt_pepper_prob,
            clip=self.clip,
        )


@torch.no_grad()
def apply_noise_batched(
    x: Tensor,
    prob: float = DEFAULT_NOISE_PROB,
    uniform_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX),
    multiplicative_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX),
    salt_pepper_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
    clip: bool = True,
) -> Tensor:
    r"""Applies noise to a batch of images such that each image in the batch is
    independently transformed. This applies all of the noise types in sequence.

    Args:
        x: Input tensor
        prob: Probability of applying noise to the input
        uniform_scale: Scale of the uniform noise to apply to the input
        multiplicative_scale: Scale of the multiplicative noise to apply to the input
        salt_pepper_prob: Proportion of salt and pepper noise to apply to the input
        clip: Whether to clip the result to the range :math:`[0, 1]`

    Shape:
        - Input: :math:`(N, ...)`
        - Output: Same shape as input

    Returns:
        Input with noise applied
    """
    N = x.shape[0]
    x = x.clone()

    # Uniform noise
    mask = torch.rand(N, device=x.device).lt_(prob).bool()
    min, max = to_tuple(uniform_scale)
    uniform_noised_x = uniform_noise(x, min=min, max=max, clip=clip)
    x[mask] = uniform_noised_x[mask]

    # Multiplicative noise
    mask = torch.rand(N, device=x.device).lt_(prob).bool()
    min, max = to_tuple(multiplicative_scale)
    multiplicative_noised_x = multiplicative_noise(x, min=min, max=max, clip=clip)
    x[mask] = multiplicative_noised_x[mask]

    # Salt and pepper noise
    mask = torch.rand(N, device=x.device).lt_(prob).bool()
    min, max = to_tuple(salt_pepper_prob)
    salt_pepper_noised_x = salt_pepper_noise(x, min=min, max=max)
    x[mask] = salt_pepper_noised_x[mask]

    return x
