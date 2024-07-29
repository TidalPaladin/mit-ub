from typing import Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor


class PositionEncoder(nn.Module):
    r"""Base class for positional encodings"""

    def __init__(self):
        super().__init__()

    @torch.jit.export  # type: ignore
    def from_grid(
        self,
        dims: Sequence[int],
        batch_size: int = 1,
        proto: Optional[Tensor] = None,
        requires_grad: bool = True,
        normalize: bool = True,
        add_noise: bool = False,
    ):
        r"""Creates positional encodings for a coordinate space with lengths given in ``dims``.
        Args:
            dims:
                Forwarded to :func:`create_grid`
            batch_size:
                Batch size, for matching the coordinate grid against a batch of vectors that need
                positional encoding.
            proto:
                Forwarded to :func:`create_grid`
            requires_grad:
                Forwarded to :func:`create_grid`
            normalize:
                Forwarded to :func:`create_grid`
            add_noise:
                Forwarded to :func:`create_grid`

        Keyword Args:
            Forwarded to :func:`create_grid`
        Shapes:
            * Output - :math:`(L, N, D)` where :math:`D` is the embedding size, :math:`L` is ``product(dims)``,
              and :math:`N` is ``batch_size``.
        """
        grid = self.create_grid(dims, proto, requires_grad, normalize, add_noise)
        pos_enc = self(grid).expand(batch_size, -1, -1)
        return pos_enc

    @staticmethod
    def create_grid(
        dims: Sequence[int],
        proto: Optional[Tensor] = None,
        requires_grad: bool = True,
        normalize: bool = True,
        add_noise: bool = False,
    ) -> Tensor:
        r"""Create a grid of coordinate values given the size of each dimension.
        Args:
            dims:
                The length of each dimension
            proto:
                If provided, a source tensor with which to match device / requires_grad
            requires_grad:
                Optional override for requires_grad
            normalize:
                If true, normalize coordinate values on the range :math:`\[-1, 1\]`
            add_noise:
                If true, add noise to the grid. The noise is sampled from a uniform distribution on
                the range :math:`\[-0.5, 0.5\]` and is applied prior to normalization.

        Shapes:
            * Output - :math:`(1, L, C)` where :math:`C` is ``len(dims)`` and :math:`L` is ``product(dims)``
        """
        if proto is not None:
            device = proto.device
            dtype = proto.dtype
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        with torch.no_grad():
            lens = [torch.arange(d, device=device, dtype=dtype) for d in dims]
            grid = torch.stack(torch.meshgrid(lens, indexing="ij"), dim=0)

            # Add noise to the grid (handled differently for normalized and unnormalized grids)
            if add_noise and not normalize:
                grid.add_(torch.rand_like(grid).sub_(0.5))

            C = grid.shape[0]
            if normalize:
                scale = grid.view(C, -1).amax(dim=-1, keepdim=True)
                grid = grid.view(C, -1).div_(scale).sub_(0.5).mul_(2).view_as(grid)
                # In the normalized case we don't want the injected noise to affect the scale parameter above.
                # We could handle this with various checks but it is more efficient to have a separate pathway here.
                if add_noise:
                    noise = torch.rand_like(grid).view(C, -1).sub_(0.5).div_(scale / 2).view_as(grid)
                    grid.add_(noise).clip_(min=-1, max=1)

            # This is cleaner but not scriptable
            # grid = rearrange(grid, "c ... -> () (...) c")
            grid = grid.view(C, -1).movedim(0, -1).unsqueeze_(0)

        requires_grad = requires_grad or (proto is not None and proto.requires_grad)
        grid.requires_grad_()
        return grid


class RelativeFactorizedPosition(PositionEncoder):

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self._d_out = d_out
        self.proj = nn.Linear(d_in, d_out)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)
