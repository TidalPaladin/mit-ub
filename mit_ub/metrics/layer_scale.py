from typing import Iterator

import torch
import torch.nn as nn
import torchmetrics as tm
from torch import Tensor

from ..model.layers.layer_scale import LayerScale


def iterate_layer_scales(module: nn.Module) -> Iterator[Tensor]:
    r"""Iterate over all layer scales in a module."""
    assert isinstance(module, nn.Module)
    for m in module.modules():
        if isinstance(m, LayerScale):
            yield m.gamma.detach()


class MaxLayerScale(tm.MaxMetric):
    r"""Track the maximum absolute value of layer scale across all layers."""

    def update(self, module: nn.Module) -> None:
        max_scales = torch.tensor([t.abs().max() for t in iterate_layer_scales(module)])
        super().update(max_scales.max())


class MeanLayerScale(tm.MeanMetric):
    r"""Track the mean absolute value of layer scale across all layers."""

    def update(self, module: nn.Module) -> None:
        for t in iterate_layer_scales(module):
            super().update(t.abs().mean())
