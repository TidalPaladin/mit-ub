from typing import Final

from torch import Tensor
from torchvision.transforms.v2 import Transform
from torchvision.tv_tensors import Image, Video


# Most mammo machines have 14 bits of dynamic range. Here we will compress the entire
# image to 4x this range.
DEFAULT_MAX_VAL: Final = 4 / 2**14


class CompressDynamicRange(Transform):
    r"""Compress the dynamic range of the input image.

    Mostly used for testing a model's ability to handle a wide dynamic range.
    """

    def __init__(self, max_val: float = DEFAULT_MAX_VAL):
        super().__init__()
        self.max_val = max_val

    def forward(self, x: Tensor | Image | Video) -> Tensor | Image | Video:
        x_min, x_max = x.aminmax()
        delta = (x_max - x_min).clip_(min=1e-9)
        y = (x - x_min).div_(delta).mul_(self.max_val)

        if isinstance(x, Image):
            return Image(y)
        elif isinstance(x, Video):
            return Video(y)
        elif isinstance(x, Tensor):
            return y
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
