import torch
import torch.nn as nn


class LayerScale(nn.Module):

    def __init__(self, dim: int, gamma: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), gamma))
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
