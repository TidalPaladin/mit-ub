import torch.nn as nn
from torch import Tensor


class AveragePool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)


class MaxPool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.amax(dim=1)


def get_global_pooling_layer(pool_type: str) -> nn.Module:
    match pool_type:
        case "avg":
            return AveragePool()
        case "max":
            return MaxPool()
        case _:
            raise ValueError(f"Unknown pooling type: {pool_type}")
