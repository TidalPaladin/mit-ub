import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.layers.pool import MultiHeadAttentionPool, SimpleAttentionPool


class TestMultiHeadAttentionPool:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_queries", [1, 2])
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_forward(self, device, num_queries, qk_norm):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        q = torch.randn(B, L, D).to(device)
        layer = MultiHeadAttentionPool(D, 2, num_queries, qk_norm=qk_norm).to(device)
        y = layer(q)
        assert y.shape == (B, num_queries, D)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_queries", [1, 2])
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_backward(self, device, num_queries, qk_norm):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        q = torch.randn(B, L, D, requires_grad=True).to(device)
        layer = MultiHeadAttentionPool(D, 2, num_queries, qk_norm=qk_norm).to(device)
        y = layer(q)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None

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


class TestSimpleAttentionPool:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_queries", [1, 2])
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_forward(self, device, num_queries, qk_norm):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        q = torch.randn(B, L, D).to(device)
        layer = SimpleAttentionPool(D, 2).to(device)
        y = layer(q)
        assert y.shape == (B, D)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("num_queries", [1, 2])
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_backward(self, device, num_queries, qk_norm):
        torch.random.manual_seed(0)
        B, L, D = 2, 8, 32
        q = torch.randn(B, L, D, requires_grad=True).to(device)
        layer = SimpleAttentionPool(D, 2).to(device)
        y = layer(q)
        y.sum().backward()
        for param in layer.parameters():
            assert param.grad is not None

    def test_reset_parameters(self, mocker):
        torch.random.manual_seed(0)
        D = 32
        spy = mocker.spy(SimpleAttentionPool, "reset_parameters")
        layer = SimpleAttentionPool(D, 2)
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
        layer = SimpleAttentionPool(D, 2)
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
