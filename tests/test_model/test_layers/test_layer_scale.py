import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from mit_ub.model.layers.layer_scale import LayerScale, has_layer_scale


@pytest.mark.parametrize(
    "module, expected",
    [
        (nn.Linear(10, 10), False),
        (LayerScale(10), True),
        (nn.Sequential(nn.Linear(10, 10), LayerScale(10)), True),
        (nn.Sequential(nn.Linear(10, 10), nn.Linear(10, 10)), False),
    ],
)
def test_has_layer_scale(module: nn.Module, expected: bool):
    assert has_layer_scale(module) == expected


class TestLayerScale:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("inplace", [False, True])
    def test_forward(self, device, inplace):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D).to(device)
        layer = LayerScale(D, inplace=inplace).to(device)
        y = layer(x.clone())
        assert y.shape == x.shape

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D, requires_grad=True).to(device)
        layer = LayerScale(D).to(device)
        y = layer(x)
        y.sum().backward()

    def test_zero_gamma(self):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        x = torch.randn(B, L, D)
        layer = LayerScale(D, gamma=0.0)
        assert_close(layer(x), torch.zeros_like(x))

    def test_extra_repr(self):
        layer = LayerScale(32, gamma=1e-5, inplace=True)
        result = str(layer)
        assert result == "LayerScale(gamma=1e-05, inplace=True)"
