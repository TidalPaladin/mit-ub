import pytest
import torch
from torch.testing import assert_close

from mit_ub.data.invert import invert, invert_


@pytest.mark.cuda
class TestInvert:

    def test_invert_pointwise_prob_0(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = invert(x, invert_prob=0.0)
        assert_close(y, x)
        assert_close(y, x_orig)

    def test_invert_pointwise_prob_1(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = invert(x, invert_prob=1.0)
        assert_close(x, x_orig)
        expected = x.neg().add_(1.0)
        assert_close(y, expected)

    def test_invert_determinism(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        y1 = invert(x, invert_prob=0.5, seed=0)
        y2 = invert(x, invert_prob=0.5, seed=0)
        assert_close(y1, y2)


@pytest.mark.cuda
class TestInvertInplace:

    def test_invert_pointwise_prob_0(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        invert_(x, invert_prob=0.0)
        assert_close(x, x_orig)

    def test_invert_pointwise_prob_1(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        expected = x.neg().add_(1.0)
        invert_(x, invert_prob=1.0)
        assert_close(x, expected)

    def test_invert_determinism(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        y1 = x.clone()
        y2 = x.clone()
        invert_(y1, invert_prob=0.5, seed=0)
        invert_(y2, invert_prob=0.5, seed=0)
        assert_close(y1, y2)
