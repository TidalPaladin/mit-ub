from typing import Final

import torch
import torchmetrics as tm
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import Tensor

from ..model import compile_is_disabled


EPS: Final = 1e-8


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def rms_pairwise_distance(x: Tensor, pairwise_dim: int, embed_dim: int, p: float = 2) -> Tensor:
    """Compute the average pairwise Lp distance without manifesting the full pairwise matrix.

    Uses the fact that mean pairwise distance can be computed as:
    mean_dist = mean(||x_i - x_j||^p) = 2 * (mean(||x_i||^p) - ||mean(x_i)||^p)

    Args:
        x: The input tensor
        pairwise_dim: The dimension over which to compute average pairwise distance
        embed_dim: The embedding dimension
        p: The norm degree (default: 2)
    """
    mean_norm2 = x.pow(2).sum(dim=embed_dim, keepdim=True).mean(dim=pairwise_dim, keepdim=True)
    mean_x = x.mean(dim=pairwise_dim, keepdim=True)
    mean_x_norm2 = mean_x.pow(2).sum(dim=embed_dim, keepdim=True)
    return (2 * (mean_norm2 - mean_x_norm2)).sqrt().squeeze(embed_dim, pairwise_dim)


class RMSPairwiseDistance(tm.Metric):
    """Tracks the average pairwise RMS distance of embeddings.

    Args:
        dim: The dimension of the embeddings

    Shapes:
        - x: :math:`(..., D)`
        - output: Scalar
    """

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.add_state("sum_x", default=torch.zeros(dim, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("sum_x_sq", default=torch.zeros(dim, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        x = x.view(-1, self.dim)
        self.sum_x += x.sum(0)
        self.sum_x_sq += x.pow(2).sum(0)
        self.count += x.shape[0]

    def compute(self) -> Tensor:
        count = int(self.count.item())
        if count == 0:
            rank_zero_warn(f"compute() method of {self.__class__.__name__} called with no examples")
            return self.sum_x.new_tensor(0.0)

        mean_norm2 = self.sum_x_sq.sum(dim=0) / count
        mean_x = self.sum_x / count
        mean_x_norm2 = mean_x.pow(2).sum(dim=0)
        return (2 * (mean_norm2 - mean_x_norm2)).sqrt()


class ExampleRMSDistance(RMSPairwiseDistance):
    """Tracks the average pairwise RMS distance of token averages across examples.

    This is useful for tracking the distance between embeddings of different examples.
    Inputs are averaged over the sequence dimension before computing the RMS distance.

    Shapes:
        - x: :math:`(..., L, D)`
        - output: Scalar
    """

    def update(self, x: Tensor) -> None:
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dimension of input to be {self.dim}, got {x.shape[-1]}")
        if x.ndim < 2:
            raise ValueError(f"Expected input to be of shape (..., L, D), got {x.shape}")

        x = x.mean(1)
        super().update(x)


class TokenRMSDistance(RMSPairwiseDistance):
    """Tracks the average pairwise RMS distance of embeddings within each example.

    Shapes:
        - x: :math:`(..., L, D)`
        - output: Scalar
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_state("sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        if x.ndim < 2:
            raise ValueError(f"Expected input to be of shape (..., L, D), got {x.shape}")

        dist = rms_pairwise_distance(x, 1, 2)
        self.sum += dist.sum()
        self.count += dist.numel()

    def compute(self) -> Tensor:
        return self.sum / self.count
