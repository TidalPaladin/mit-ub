import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.layers.stochastic_depth import (
    apply_stochastic_depth,
    stochastic_depth_indices,
    unapply_stochastic_depth,
)


class TestStochasticDepth:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize(
        "p,B,exp",
        [
            (0.25, 10, 7),
            (0.99, 10, 1),
        ],
    )
    def test_get_indices(self, device, p, B, exp):
        indices = stochastic_depth_indices(torch.randn(B, 10, device=device), p)
        assert len(indices) == exp

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B = 10
        p = 0.25
        x = torch.randn(B, 10, device=device)
        indices = stochastic_depth_indices(x, p)
        y = apply_stochastic_depth(x, indices)
        assert y.shape[0] == int((1 - p) * B)

        z = unapply_stochastic_depth(y, indices, B)
        assert_close(x[indices], z[indices])
        assert_close(z.sum(), z[indices].sum())

    def test_forward_inference(self):
        B = 10
        p = 0.25
        x = torch.randn(B, 10)
        indices = stochastic_depth_indices(x, p)
        y = apply_stochastic_depth(x, indices, training=False)
        assert_close(x, y)

        z = unapply_stochastic_depth(y, indices, B, training=False)
        assert_close(x, z)

    def test_forward_p_zero(self):
        B = 10
        p = 0.0
        x = torch.randn(B, 10)
        indices = stochastic_depth_indices(x, p)
        y = apply_stochastic_depth(x, indices, training=True)
        assert_close(x, y)

        z = unapply_stochastic_depth(y, indices, B, training=True)
        assert_close(x, z)

        y = apply_stochastic_depth(x, indices, training=False)
        assert_close(x, y)

        z = unapply_stochastic_depth(y, indices, B, training=False)
        assert_close(x, z)

    def test_contiguous(self):
        B = 10
        p = 0.25
        x = torch.randn(1, 10).expand(B, -1)
        indices = stochastic_depth_indices(x, p)
        y = apply_stochastic_depth(x, indices)
        assert y.shape[0] == int((1 - p) * B)

        z = unapply_stochastic_depth(y, indices, B)
        assert_close(x[indices], z[indices])
        assert_close(z.sum(), z[indices].sum())
