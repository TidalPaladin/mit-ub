import pytest
import torch
from torch.testing import assert_close

from mit_ub.data.noise import apply_noise_batched


@pytest.mark.cuda
class TestNoise:

    def test_zero_prob(self):
        torch.random.manual_seed(0)
        x = torch.rand(8, 3, 224, 224).cuda()
        x_no_noise = apply_noise_batched(x, prob=0.0, salt_pepper_prob=0.0)
        assert_close(x, x_no_noise)

    def test_one_prob(self):
        torch.random.manual_seed(0)
        x = torch.rand(8, 3, 224, 224).cuda()
        x_no_noise = apply_noise_batched(x, prob=1.0, salt_pepper_prob=1.0)
        assert not torch.allclose(x, x_no_noise)

    @pytest.mark.parametrize("clip", [True, False])
    def test_clip(self, clip):
        torch.random.manual_seed(0)
        x = torch.randn(8, 3, 224, 224).cuda()
        y = apply_noise_batched(x, prob=1.0, clip=clip)
        if clip:
            assert y.min() >= 0.0
            assert y.max() <= 1.0
        else:
            assert y.min() < 0.0
            assert y.max() > 1.0

    def test_determinstic(self):
        torch.random.manual_seed(0)
        x = torch.randn(8, 3, 224, 224).cuda()
        y1 = apply_noise_batched(x, seed=0)
        y2 = apply_noise_batched(x, seed=0)
        y3 = apply_noise_batched(x, seed=1)
        assert_close(y1, y2)
        assert not torch.allclose(y1, y3)
