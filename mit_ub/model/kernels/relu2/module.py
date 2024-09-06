import torch.nn as nn
from torch import Tensor

from .kernel import relu2


class ReLU2(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return relu2(x)
