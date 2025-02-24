import math

import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.layers.mlp import NormType, mlp_forward
from mit_ub.model.layers.pos_enc import RelativeFactorizedPosition, create_grid, relative_factorized_position_forward


@pytest.mark.parametrize("normalize", [True, False])
def test_create_grid(normalize):
    dims = (4, 4)
    grid = create_grid(dims, normalize=normalize)
    assert grid.shape == (1, 16, 2)
    if normalize:
        assert torch.all(grid[0, 0] == torch.tensor([-1.0, -1.0]))
        assert torch.all(grid[0, -1] == torch.tensor([1.0, 1.0]))
    else:
        assert torch.all(grid[0, 0] == torch.tensor([0, 0]))
        assert torch.all(grid[0, -1] == torch.tensor([3, 3]))


def test_relative_factorized_position_forward():
    C, D = 2, 16
    dims = (4, 4)
    w_in = torch.randn(2 * D, C)
    b_in = torch.randn(2 * D)
    w_out = torch.randn(D, 2 * D)
    b_out = torch.randn(D)
    norm_w = torch.randn(D)
    norm_b = torch.randn(D)

    torch.random.manual_seed(0)
    actual = relative_factorized_position_forward(
        dims, w_in, b_in, w_out, b_out, norm_w, norm_b, F.relu, training=False
    )

    grid = create_grid(dims, normalize=True)
    grid = grid * math.sqrt(3)
    torch.random.manual_seed(0)
    # MLP uses output dropout, position encoding does not. So we must set training=False.
    expected = mlp_forward(grid, w_in, w_out, b_in, b_out, activation=F.relu, training=False)
    expected = F.layer_norm(expected, (D,), weight=norm_w, bias=norm_b)

    assert_close(actual, expected, atol=0.001, rtol=0)


class TestRelativeFactorizedPosition:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("norm_type", [NormType.LAYER_NORM, NormType.RMS_NORM])
    def test_forward(self, device, norm, norm_type):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D, norm=norm, norm_type=norm_type).to(device)
        out = layer((8, 8))
        L = 64
        assert out.shape == (1, L, D)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_backward(self, device):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to(device)
        out = layer((8, 8))
        out.sum().backward()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward_deterministic(self, device):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D, dropout=0.1).to(device)
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert not torch.allclose(out1, out2)

        layer.eval()
        out1 = layer((8, 8))
        out2 = layer((8, 8))
        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize("norm_type", [NormType.LAYER_NORM, NormType.RMS_NORM])
    def test_reset_parameters(self, mocker, norm_type):
        C, D = 2, 16
        spy = mocker.spy(RelativeFactorizedPosition, "reset_parameters")
        layer = RelativeFactorizedPosition(C, D, dropout=0.1, norm=True, norm_type=norm_type)
        spy.assert_called_once()
        if norm_type == NormType.LAYER_NORM:
            assert layer.b_norm is not None

        weights_original = {name: param.clone() for name, param in layer.named_parameters()}
        layer.reset_parameters()
        weights_reset = {name: param for name, param in layer.named_parameters()}

        for name, param in weights_original.items():
            # Ignore constant weights or biases
            if (param == 0).all() or (param == 1).all():
                continue
            assert not torch.allclose(param, weights_reset[name], equal_nan=True)

    def test_extra_repr(self):
        layer = RelativeFactorizedPosition(2, 16)
        result = str(layer)
        assert (
            result
            == "RelativeFactorizedPosition(in=2, hidden=64, out=16, dropout=0.0, act=relu2, norm=False, norm_type=layernorm)"
        )
