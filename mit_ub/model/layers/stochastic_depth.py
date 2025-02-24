import torch
from torch import Tensor


# NOTE: Do not attempt to torch.compile these methods. Since they change the shape of the input
# and have diverging train/inference behaviors, torch.compile has all kinds of issues.


def stochastic_depth_indices(x: Tensor, p: float) -> Tensor:
    r"""Generates a tensor of batch indices to retain when applying stochastic depth"""
    B = x.shape[0]
    num_keep = max(1, int((1 - p) * B))  # p is drop rate, so keep (1-p) elements
    indices = torch.randperm(B, device=x.device)[:num_keep] if p > 0.0 else torch.arange(B, device=x.device)
    return indices


def apply_stochastic_depth(x: Tensor, indices: Tensor, training: bool = True) -> Tensor:
    y = x[indices] if training and x.shape[0] > indices.shape[0] else x
    return y


def unapply_stochastic_depth(x: Tensor, indices: Tensor, size: int, training: bool = True) -> Tensor:
    if training and x.shape[0] < size:
        result = x.new_zeros(size, *x.shape[1:])
        result = torch.index_add(result, 0, indices, x)
        return result
    else:
        return x
