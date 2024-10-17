import torch
import pytest

from mit_ub.model.pool import MultiHeadAttentionPool
from torch.testing import assert_close

class TestMultiHeadAttentionPool:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_queries", [1, 2])
    def test_forward(self, device, num_queries):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        q = torch.randn(B, L, D).to(device)
        layer = MultiHeadAttentionPool(D, 2, num_queries).to(device)
        y = layer(q)
        assert y.shape == (B, num_queries, D)

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        D = 32
        spy = mocker.spy(MultiHeadAttentionPool, "reset_parameters")
        layer = MultiHeadAttentionPool(D, 2, 1)
        spy.assert_called_once()

        weight_init = {k: v.clone() for k, v in layer.named_parameters()}
        layer.reset_parameters()
        weight_reset = {k: v.clone() for k, v in layer.named_parameters()}

        for k, v in weight_init.items():
            if (v == 0).all() or (v == 1).all():
                continue
            assert not torch.allclose(v, weight_reset[k], equal_nan=True)

    def test_forward_deterministic(self):
        B, L, D = 2, 8, 32
        torch.random.manual_seed(0)
        layer = MultiHeadAttentionPool(D, 2, 1)
        x = torch.randn(B, L, D)

        # Training, non-determinstic
        layer.train()
        out1 = layer(x)
        out2 = layer(x)
        assert_close(out1, out2)

        # Evaluation, deterministic
        layer.eval()
        out1 = layer(x)
        out2 = layer(x)
        assert_close(out1, out2)
