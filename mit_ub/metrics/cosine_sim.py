from typing import Final

import torch
import torch.nn.functional as F
import torchmetrics as tm
from lightning_utilities.core.rank_zero import rank_zero_warn
from torch import Tensor

from ..model import compile_is_disabled


EPS: Final = 1e-8


@torch.compile(fullgraph=True, disable=compile_is_disabled())
def average_pairwise_cosine_similarity(x: Tensor, pairwise_dim: int, embed_dim: int, eps: float = EPS) -> Tensor:
    r"""Compute the average pairwise cosine similarity without manifesting the full pairwise matrix.

    To avoid quadratic memory usage we compute average cosine similarity as the squared norm of the mean vector.

    Args:
        x: The input tensor.
        pairwise_dim: The dimension over which to compute the average pairwise cosine similarity.
        embed_dim: The dimension to normalize the vectors to before computing the cosine similarity.
        eps: A small constant to avoid division by zero.
    """
    x.shape[pairwise_dim]
    x = F.normalize(x, dim=embed_dim, eps=eps)
    y = x.mean(pairwise_dim, keepdim=True).norm(dim=embed_dim, keepdim=True).pow(2).squeeze(embed_dim, pairwise_dim)
    return y


class AveragePairwiseCosineSimilarity(tm.Metric):
    r"""Tracks the average pairwise cosine similarity of embeddings.

    Args:
        dim: The dimension of the embeddings.

    Shapes:
        - x: :math:`(..., D)`
        - output: Scalar
    """

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.add_state("sum", default=torch.zeros(dim, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, x: Tensor) -> None:
        x = F.normalize(x, dim=-1, eps=EPS)
        assert isinstance(self.sum, Tensor)
        assert isinstance(self.count, Tensor)
        x = x.view(-1, self.dim)
        self.sum += x.sum(0)
        self.count += self.count.new_tensor(x.shape[0])

    def compute(self) -> Tensor:
        assert isinstance(self.sum, Tensor)
        assert isinstance(self.count, Tensor)
        count = int(self.count.item())
        if count == 0:
            rank_zero_warn(f"compute() method of {self.__class__.__name__} called with no examples")
            return self.sum.new_tensor(0.0)

        # Average and norm to compute final similarity
        y = self.sum / count
        y = y.norm(dim=-1).pow(2)
        return y


class ExampleSimilarity(AveragePairwiseCosineSimilarity):
    r"""Tracks the average pairwise cosine similarity of token averages across examples.

    This is useful for tracking the similarity of the embeddings of different examples.
    Inputs are averaged over the sequence dimension before computing the cosine similarity.

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


class TokenSimilarity(AveragePairwiseCosineSimilarity):
    r"""Tracks the average pairwise cosine similarity of embeddings within each example.

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

        sim = average_pairwise_cosine_similarity(x, 1, 2)
        self.sum += sim.sum()
        self.count += sim.numel()

    def compute(self) -> Tensor:
        return self.sum / self.count
