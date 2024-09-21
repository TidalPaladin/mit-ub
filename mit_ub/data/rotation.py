from typing import Sequence

from torchvision.transforms.v2 import RandomRotation as TVRandomRotation


class RandomRotation(TVRandomRotation):
    r"""Wrapper around torchvision's RandomRotation with a type signature that works with jsonargparse."""

    def __init__(self, degrees: Sequence[float] | float, expand: bool = False):
        super().__init__(degrees, expand=expand)
