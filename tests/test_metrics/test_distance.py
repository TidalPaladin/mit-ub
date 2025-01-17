import pytest
import torch
from torch.testing import assert_close

from mit_ub.metrics.distance import ExampleRMSDistance, RMSPairwiseDistance, TokenRMSDistance, rms_pairwise_distance


@pytest.mark.parametrize(
    "batch, tokens",
    [
        (10, 128),
        (1, 128),
        (10, 1),
        (1, 1),
    ],
)
def test_rms_pairwise_distance(batch, tokens):
    B, L, D = batch, tokens, 32
    torch.manual_seed(0)
    x = torch.randn(B, L, D)

    actual = rms_pairwise_distance(x, 1, 2)

    # Compute expected using explicit pairwise distances
    x_i = x.unsqueeze(2)  # B, L, 1, D
    x_j = x.unsqueeze(1)  # B, 1, L, D
    expected = ((x_i - x_j).pow(2).sum(-1).mean(dim=(1, 2))).sqrt()
    assert_close(expected, actual)


class TestExampleRMSDistance:

    def test_update_single_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)

        metric = ExampleRMSDistance(D)
        metric.update(x)

        actual = rms_pairwise_distance(x.mean(1), 0, 1)
        expected = metric.compute()
        assert_close(expected, actual)

    def test_update_multiple_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)
        x1 = x[:16]
        x2 = x[16:]

        metric = ExampleRMSDistance(D)
        metric.update(x1)
        metric.update(x2)

        actual = rms_pairwise_distance(x.mean(1), 0, 1)
        expected = metric.compute()
        assert_close(expected, actual)


class TestRMSPairwiseDistance:

    def test_update_single_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)

        metric = RMSPairwiseDistance(D)
        metric.update(x)

        actual = rms_pairwise_distance(x.view(-1, D), 0, 1).mean()
        expected = metric.compute()
        assert_close(expected, actual)

    def test_update_multiple_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)
        x1 = x[:16]
        x2 = x[16:]

        metric = RMSPairwiseDistance(D)
        metric.update(x1)
        metric.update(x2)

        actual = rms_pairwise_distance(x.view(-1, D), 0, 1).mean()
        expected = metric.compute()
        assert_close(expected, actual)


class TestTokenRMSDistance:

    def test_update_single_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)

        metric = TokenRMSDistance(D)
        metric.update(x)

        actual = rms_pairwise_distance(x, 1, 2).mean()
        expected = metric.compute()
        assert_close(expected, actual)

    def test_update_multiple_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)
        x1 = x[:16]
        x2 = x[16:]

        metric = TokenRMSDistance(D)
        metric.update(x1)
        metric.update(x2)

        actual = rms_pairwise_distance(x, 1, 2).mean()
        expected = metric.compute()
        assert_close(expected, actual)
