import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..activations import DEFAULT_POS_ENC_ACTIVATION, Activation
from ..helpers import compile_backend, compile_is_disabled


@torch.compile(fullgraph=True, backend=compile_backend(), disable=compile_is_disabled())
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
    backend=compile_backend(),
    disable=compile_is_disabled(),
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
    activation: Activation = DEFAULT_POS_ENC_ACTIVATION,
    dropout: float = 0.0,
    training: bool = False,
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
    # NOTE: Scale by 1/sqrt(1/3) make uniform distribution have unit variance.
    scale = math.sqrt(3)
    lens = [torch.linspace(-scale, scale, d, device=w1.device, dtype=w1.dtype) for d in dims]
    grid = torch.stack(torch.meshgrid(lens, indexing="ij"), dim=-1).view(1, -1, len(dims))

    y = F.linear(grid, w1, b1)
    y = activation(y)
    y = F.dropout(y, p=dropout, training=training, inplace=True)
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

    def __init__(
        self,
        d_in: int,
        d_out: int,
        dropout: float = 0.0,
        activation: Activation = DEFAULT_POS_ENC_ACTIVATION,
        dim_feedforward: int | None = None,
    ):
        super().__init__()
        self._dim_feedforward = dim_feedforward or 2 * d_out
        self.dropout = dropout
        self.activation = activation
        self.w_in = nn.Parameter(torch.empty(self.dim_feedforward, d_in))
        self.w_out = nn.Parameter(torch.empty(d_out, self.dim_feedforward))
        self.b_in = nn.Parameter(torch.empty(self.dim_feedforward))
        self.b_out = nn.Parameter(torch.empty(d_out))
        self.w_norm = nn.Parameter(torch.empty(d_out))
        self.b_norm = nn.Parameter(torch.empty(d_out))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for weight in (self.w_in, self.w_out):
            # nn.init.xavier_uniform_(weight)
            nn.init.trunc_normal_(weight, std=0.02)

        for bias in (self.b_in, self.b_out, self.b_norm):
            if bias is not None:
                nn.init.zeros_(bias)

        nn.init.ones_(self.w_norm)

    @property
    def d_in(self) -> int:
        return self.w_in.shape[1]

    @property
    def d_out(self) -> int:
        return self.w_out.shape[0]

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward

    def extra_repr(self) -> str:
        return (
            f"in={self.d_in}, "
            f"hidden={self.dim_feedforward}, "
            f"out={self.d_out}, "
            f"dropout={self.dropout}, "
            f"act={self.activation.__name__}"
        )

    def forward(self, dims: Sequence[int]) -> Tensor:
        return relative_factorized_position_forward(
            dims,
            self.w_in,
            self.b_in,
            self.w_out,
            self.b_out,
            self.w_norm,
            self.b_norm,
            self.activation,
            dropout=self.dropout,
            training=self.training,
        )
