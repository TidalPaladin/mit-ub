from typing import Sequence

from torchvision.transforms.v2 import RandomRotation as TVRandomRotation


class RandomRotation(TVRandomRotation):
    def __init__(self, degrees: Sequence[float] | float, expand: bool = False):
        super().__init__(degrees, expand=expand)
