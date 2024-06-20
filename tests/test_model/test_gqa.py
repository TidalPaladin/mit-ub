import pytest
import torch

from mit_ub.model.gqa import GroupedQueryAttention


class TestGroupedQueryAttention:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, L, D = 1, 32, 64
        x = torch.randn(B, L, D).to(device)
        gqa = GroupedQueryAttention(D, 4, 2).to(device)
        out = gqa(x)
        assert out.shape == (B, L, D)
