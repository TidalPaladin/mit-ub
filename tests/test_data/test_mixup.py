import pytest
import torch
import torch.nn.functional as F
import timeit
from torch.testing import assert_close

from mit_ub.data.mixup import is_mixed, is_mixed_with_unknown, mixup, mixup_dense_label, sample_mixup_parameters, fused_mixup, cross_entropy_mixup


@pytest.mark.cuda
class TestMixUp:

    def test_mixup_pointwise_prob_0(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = fused_mixup(x, mixup_prob=0.0)
        assert_close(y, x)
        assert_close(y, x_orig)

    def test_mixup_pointwise_prob_1(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y = fused_mixup(x, mixup_prob=1.0)
        assert_close(x, x_orig)
        assert not (y == x_orig).view(4, -1).all(dim=-1).any()

    def test_mixup_determinism(self):
        torch.random.manual_seed(0)
        x = torch.randn(4, 3, 32, 32, device="cuda")
        x_orig = x.clone()
        y1 = fused_mixup(x, mixup_prob=1.0, seed=0)
        y2 = fused_mixup(x, mixup_prob=1.0, seed=0)
        y3 = fused_mixup(x, mixup_prob=1.0, seed=1)
        assert_close(y1, y2)
        assert not torch.allclose(y1, y3)

    @pytest.mark.parametrize("batch,nclass", [
        (32, 2,),
        (1024, 1024,),
    ])
    def test_cross_entropy_mixup_prob_0(self, batch, nclass):
        torch.random.manual_seed(0)
        logits = torch.randn(batch, nclass, device="cuda")
        labels = torch.randint(0, nclass, (batch,), device="cuda")
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.cross_entropy(logits, labels, reduction="none")
        assert_close(loss, expected)

    def test_cross_entropy_mixup_prob_0_stability(self):
        torch.random.manual_seed(0)
        logits = torch.randn(4, 3, device="cuda") * 1000
        labels = torch.randint(0, 3, (4,), device="cuda")
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.cross_entropy(logits, labels, reduction="none")
        assert_close(loss, expected)

    def test_cross_entropy_mixup(self):
        torch.random.manual_seed(0)
        logits = torch.randn(4, 3, device="cuda")
        labels = torch.randint(0, 3, (4,), device="cuda")
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        assert loss.shape == (4,)
        assert loss.dtype == logits.dtype

    @pytest.mark.parametrize("batch,nclass", [
        (32, 2,),
        (1024, 1024,),
    ])
    def test_cross_entropy_mixup_speed(self, batch, nclass):
        torch.random.manual_seed(0)
        logits = torch.randn(batch, nclass, device="cuda")
        labels = torch.randint(0, nclass, (batch,), device="cuda")

        # Baseline using cross entropy with no mixup
        torch.cuda.synchronize()
        start = timeit.default_timer()
        F.cross_entropy(logits, labels, reduction="none")
        torch.cuda.synchronize()
        end = timeit.default_timer()
        ce_baseline = end - start

        # Baseline using cross entropy with mixup
        torch.cuda.synchronize()
        start = timeit.default_timer()
        weight = sample_mixup_parameters(batch, 0.5, 1.0, device=logits.device)
        labels_mixed = mixup_dense_label(labels, weight, num_classes=nclass)
        F.cross_entropy(logits, labels_mixed, reduction="none")
        torch.cuda.synchronize()
        end = timeit.default_timer()
        ce_mixup = end - start

        # Using our mixup implementation
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        actual = end - start
        print(f"Time taken: {end - start} seconds")
        assert actual <= 2 * ce_baseline, f"Mixup is slower than baseline: {actual} vs {ce_baseline}"
        