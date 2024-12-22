import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.metrics.cosine_sim import (
    AveragePairwiseCosineSimilarity,
    ExampleSimilarity,
    TokenSimilarity,
    average_pairwise_cosine_similarity,
)


@pytest.mark.parametrize(
    "batch, tokens",
    [
        (10, 128),
        (1, 128),
        (10, 1),
        (1, 1),
    ],
)
def test_average_pairwise_cosine_similarity(batch, tokens):
    B, L, D = batch, tokens, 32
    torch.manual_seed(0)
    x = torch.randn(B, L, D)

    actual = average_pairwise_cosine_similarity(x, 1, 2)
    expected = F.cosine_similarity(x.view(B, L, 1, D), x.view(B, 1, L, D), dim=-1).mean(dim=(1, 2))
    assert_close(expected, actual)


class TestExampleSimilarity:

    def test_update_single_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)

        metric = ExampleSimilarity(D)
        metric.update(x)

        actual = average_pairwise_cosine_similarity(x.mean(1), 0, 1)
        expected = metric.compute()
        assert_close(expected, actual)

    def test_update_multiple_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)
        x1 = x[:16]
        x2 = x[16:]

        metric = ExampleSimilarity(D)
        metric.update(x1)
        metric.update(x2)

        actual = average_pairwise_cosine_similarity(x.mean(1), 0, 1)
        expected = metric.compute()
        assert_close(expected, actual)


class TestAveragePairwiseCosineSimilarity:

    def test_update_single_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)

        metric = AveragePairwiseCosineSimilarity(D)
        metric.update(x)

        actual = average_pairwise_cosine_similarity(x.view(-1, D), 0, 1).mean()
        expected = metric.compute()
        assert_close(expected, actual)

    def test_update_multiple_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)
        x1 = x[:16]
        x2 = x[16:]

        metric = AveragePairwiseCosineSimilarity(D)
        metric.update(x1)
        metric.update(x2)

        actual = average_pairwise_cosine_similarity(x.view(-1, D), 0, 1).mean()
        expected = metric.compute()
        assert_close(expected, actual)


class TestTokenSimilarity:

    def test_update_single_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)

        metric = TokenSimilarity(D)
        metric.update(x)

        actual = average_pairwise_cosine_similarity(x, 1, 2).mean()
        expected = metric.compute()
        assert_close(expected, actual)

    def test_update_multiple_batch(self):
        B, L, D = 32, 128, 64
        torch.manual_seed(0)
        x = torch.randn(B, L, D)
        x1 = x[:16]
        x2 = x[16:]

        metric = TokenSimilarity(D)
        metric.update(x1)
        metric.update(x2)

        actual = average_pairwise_cosine_similarity(x, 1, 2).mean()
        expected = metric.compute()
        assert_close(expected, actual)
