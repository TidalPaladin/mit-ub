from typing import Sequence, cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from torch import Tensor

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

    Shapes:
        * Input - :math:`(C,)` where :math:`C` is the number of input dimensions
        * Output - :math:`(1, L, D)` where :math:`L` is the product of input dimensions and :math:`D` is the output dimension
    """

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.linear = te.Linear(d_in, d_out)
        self.mlp = te.LayerNormMLP(d_out, d_out, activation="srelu")

    def forward(self, dims: Sequence[int]) -> Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            mp = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("high")
            with torch.no_grad():
                grid = create_grid(dims, device=cast(Tensor, self.linear.weight).device)
            result = self.linear(grid)
            torch.set_float32_matmul_precision(mp)

        result = self.mlp(result)
        return result
