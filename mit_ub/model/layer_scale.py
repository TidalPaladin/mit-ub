import torch
import torch.nn as nn
from torch import Tensor

from .helpers import compile_is_disabled


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def layer_scale(x: Tensor, gamma: Tensor, inplace: bool = False) -> Tensor:
    return x.mul_(gamma) if inplace else x * gamma


class LayerScale(nn.Module):

    def __init__(self, dim: int, gamma: float = 1e-5, inplace: bool = False):
        super().__init__()
        self._gamma = gamma
        self.gamma = nn.Parameter(torch.full((dim,), gamma))
        self.inplace = inplace

    def reset_parameters(self):
        self.gamma.data.fill_(self._gamma)

    def extra_repr(self) -> str:
        return f"gamma={self._gamma}, inplace={self.inplace}"

    def forward(self, x: Tensor) -> Tensor:
        return layer_scale(x, self.gamma, self.inplace)
