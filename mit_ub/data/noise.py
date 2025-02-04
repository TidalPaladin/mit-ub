from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Final, List, Sequence, Tuple, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import Compose, RandomApply, Transform
from torchvision.utils import make_grid


UNIFORM_NOISE_MIN: Final = -0.25
UNIFORM_NOISE_MAX: Final = 0.25
MULTIPLICATIVE_NOISE_MIN: Final = 0.5
MULTIPLICATIVE_NOISE_MAX: Final = 1.5
SALT_PEPPER_NOISE_PROB: Final = 0.1
SALT_PEPPER_NOISE_MIN: Final = 0.01
SALT_PEPPER_NOISE_MAX: Final = 0.05
DEFAULT_NOISE_PROB: Final = 0.25


def to_tuple(x: float | Sequence[float]) -> Tuple[float, float]:
    if isinstance(x, Sequence):
        assert len(x) == 2, "Expected a sequence of length 2"
        return cast(Tuple[float, float], tuple(x))
    else:
        return (x, x)


def batched_uniform_noise(x: Tensor, min: float, max: float) -> Tensor:
    center = (min + max) / 2
    min_ranges = x.new_empty(x.shape[0]).uniform_(min, center).view(-1, *([1] * (x.ndim - 1)))
    max_ranges = x.new_empty(x.shape[0]).uniform_(center, max).view(-1, *([1] * (x.ndim - 1)))
    return torch.rand_like(x).mul_(max_ranges - min_ranges).add_(min_ranges)


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
    noise = batched_uniform_noise(x, min, max)
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
    noise = batched_uniform_noise(x, min, max)
    result = noise.mul_(x)
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

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return uniform_noise(
            inpt,
            min=self.min,
            max=self.max,
            clip=self.clip,
        )


class SaltPepperNoise(Transform):

    def __init__(self, scale: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX)):
        super().__init__()
        self.min, self.max = to_tuple(scale)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return dict()

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
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

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
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
        salt_pepper_prob: float = SALT_PEPPER_NOISE_PROB,
        salt_pepper_pixel_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
        clip: bool = True,
    ):
        self.uniform_scale = to_tuple(uniform_scale)
        self.multiplicative_scale = to_tuple(multiplicative_scale)
        self.salt_pepper_prob = salt_pepper_prob
        self.salt_pepper_pixel_prob = to_tuple(salt_pepper_pixel_prob)
        self.prob = prob
        self.clip = clip
        uniform_noise = RandomApply(
            [UniformNoise(min=self.uniform_scale[0], max=self.uniform_scale[1], clip=self.clip)], p=prob
        )
        multiplicative_noise = RandomApply(
            [MultiplicativeNoise(min=self.multiplicative_scale[0], max=self.multiplicative_scale[1], clip=self.clip)],
            p=prob,
        )
        salt_pepper_noise = RandomApply([SaltPepperNoise(scale=self.salt_pepper_pixel_prob)], p=self.salt_pepper_prob)
        super().__init__([uniform_noise, multiplicative_noise, salt_pepper_noise])

    def apply_batched(self, x: Tensor) -> Tensor:
        return apply_noise_batched(
            x,
            prob=self.prob,
            uniform_scale=self.uniform_scale,
            multiplicative_scale=self.multiplicative_scale,
            salt_pepper_prob=self.salt_pepper_prob,
            salt_pepper_pixel_prob=self.salt_pepper_pixel_prob,
            clip=self.clip,
        )


@torch.no_grad()
def apply_noise_batched(
    x: Tensor,
    prob: float = DEFAULT_NOISE_PROB,
    uniform_scale: float | Tuple[float, float] = (UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX),
    multiplicative_scale: float | Tuple[float, float] = (MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX),
    salt_pepper_prob: float = SALT_PEPPER_NOISE_PROB,
    salt_pepper_pixel_prob: float | Tuple[float, float] = (SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
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
        salt_pepper_pixel_prob: Proportion of salt and pepper noise to apply to each pixel
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
    mask = torch.rand(N, device=x.device).lt_(salt_pepper_prob).bool()
    min, max = to_tuple(salt_pepper_pixel_prob)
    salt_pepper_noised_x = salt_pepper_noise(x, min=min, max=max)
    x[mask] = salt_pepper_noised_x[mask]

    return x


def parse_args() -> Namespace:
    parser = ArgumentParser(prog="noise", description="Preview noise applied to an image")
    parser.add_argument("image", type=Path, help="Path to the image to apply noise to")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("-p", "--prob", type=float, default=0.5, help="Probability of applying noise")
    parser.add_argument(
        "-u",
        "--uniform-scale",
        type=float,
        nargs=2,
        default=(UNIFORM_NOISE_MIN, UNIFORM_NOISE_MAX),
        help="Scale of the uniform noise",
    )
    parser.add_argument(
        "-m",
        "--multiplicative-scale",
        type=float,
        nargs=2,
        default=(MULTIPLICATIVE_NOISE_MIN, MULTIPLICATIVE_NOISE_MAX),
        help="Scale of the multiplicative noise",
    )
    parser.add_argument(
        "-s",
        "--salt-pepper-prob",
        type=float,
        default=SALT_PEPPER_NOISE_PROB,
        help="Probability of applying salt and pepper noise",
    )
    parser.add_argument(
        "-sp",
        "--salt-pepper-pixel-prob",
        type=float,
        nargs=2,
        default=(SALT_PEPPER_NOISE_MIN, SALT_PEPPER_NOISE_MAX),
        help="Probability of applying salt and pepper noise to a given pixel",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for random number generator")
    return parser.parse_args()


def main(args: Namespace):
    image = Image.open(args.image)
    image = np.array(image)
    image = torch.from_numpy(image)
    image = image.to(torch.float32) / torch.iinfo(image.dtype).max
    if image.ndim == 2:
        image.unsqueeze_(-1)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze_(0).expand(args.batch_size, -1, -1, -1)

    torch.manual_seed(args.seed)
    noise = RandomNoise(
        prob=args.prob,
        uniform_scale=args.uniform_scale,
        multiplicative_scale=args.multiplicative_scale,
        salt_pepper_prob=args.salt_pepper_prob,
    ).apply_batched(image)

    noise = noise.mul_(255).clamp_(0, 255).to(torch.uint8)
    grid = make_grid(noise)
    grid = grid.permute(1, 2, 0)
    grid = Image.fromarray(grid.numpy())
    grid.save("noise_preview.png")


if __name__ == "__main__":
    main(parse_args())
