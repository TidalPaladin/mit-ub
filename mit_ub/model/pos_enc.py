from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mlp import relu2


@torch.compile(fullgraph=True)
def create_grid(
    dims: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    normalize: bool = True,
) -> Tensor:
    r"""Create a grid of coordinate values given the size of each dimension.

    Args:
        dims:
            The length of each dimension
        proto:
            If provided, a source tensor with which to match device / requires_grad
        normalize:
            If true, normalize coordinate values on the range :math:`\[-1, 1\]`

    Shapes:
        * Output - :math:`(1, L, C)` where :math:`C` is ``len(dims)`` and :math:`L` is ``product(dims)``
    """
    if normalize:
        lens = [torch.linspace(-1, 1, d, device=device, dtype=dtype) for d in dims]
    else:
        lens = [torch.arange(d, device=device, dtype=dtype) for d in dims]
    grid = torch.stack(torch.meshgrid(lens, indexing="ij"), dim=-1)
    return grid.view(1, -1, len(dims))


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "shape_padding": True,
        "triton.cudagraph_trees": True,
    },
)
def relative_factorized_position_forward(
    dims: Sequence[int],
    w1: Tensor,
    b1: Tensor | None,
    w2: Tensor,
    b2: Tensor | None,
    w_norm: Tensor | None,
    b_norm: Tensor | None,
    activation: Callable[[Tensor], Tensor] = relu2,
    dropout: float = 0.0,
    training: bool = True,
) -> Tensor:
    """
    Perform the forward pass for the relative factorized position encoding.

    Args:
        dims:
            The length of each dimension
        w1:
            The weight tensor for the first linear layer
        b1:
            The bias tensor for the first linear layer
        w2:
            The weight tensor for the second linear layer
        b2:
            The bias tensor for the second linear layer
        w_norm:
            The weight tensor for the layer normalization
        b_norm:
            The bias tensor for the layer normalization
        activation:
            The activation function to be applied after the first linear layer
        dropout:
            The dropout probability
        training:
            Whether the model is in training mode

    Shapes:
        * dims - :math:`(C,)` where :math:`C` is the number of dimensions
        * w1 - :math:`(H, C)` where :math:`H` is the hidden dimension
        * b1 - :math:`(H,)` or None
        * w2 - :math:`(D, H)` where :math:`D` is the output dimension
        * b2 - :math:`(D,)` or None
        * w_norm - :math:`(D,)`
        * b_norm - :math:`(D,)` or None
        * Output - :math:`(1, L, D)` where :math:`L` is the product of dims and :math:`D` is the output dimension
    """
    # TODO: Make a faster kernel that doesn't need a grid of input coords. Just compute normalized
    # coodinates based on the block of the output we're computing, no need to read coords from DRAM.
    lens = [torch.linspace(-1, 1, d, device=w1.device, dtype=w1.dtype) for d in dims]
    grid = torch.stack(torch.meshgrid(lens, indexing="ij"), dim=-1).view(1, -1, len(dims))
    y = F.linear(grid, w1, b1)
    y = activation(y)
    y = F.dropout(y, p=dropout, training=training)
    y = F.linear(y, w2, b2)
    y = F.layer_norm(y, normalized_shape=(y.shape[-1],), weight=w_norm, bias=b_norm)
    return y


class RelativeFactorizedPosition(nn.Module):
    """
    Computes relative factorized position encodings.

    A grid of positions in the interval :math:`[-1, 1]` is first created.
    This grid is then projected into a higher-dimensional space using a multi-layer perceptron (MLP).
    The output is then normalized using a layer normalization. This computation is performed in float32 precision
    to ensure stability at high resolution, and mamtul precision is set to 'high' for this step.

    Args:
        d_in:
            Input dimension size
        d_out:
            Output dimension size
        dropout:
            Dropout probability to be applied in the MLP

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.0, activation: Callable[[Tensor], Tensor] = relu2):
        super().__init__()
        self._d_in = d_in
        self._d_out = d_out
        self.fc1 = nn.Linear(d_in, 2 * d_out, bias=True)
        self.fc2 = nn.Linear(2 * d_out, d_out, bias=True)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, dims: Sequence[int]) -> Tensor:
        return relative_factorized_position_forward(
            dims,
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.norm.weight,
            self.norm.bias,
            self.activation,
            dropout=self.dropout.p,
            training=self.training,
        )
