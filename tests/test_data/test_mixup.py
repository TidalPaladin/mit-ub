import pytest
import torch
import torch.nn.functional as F
import timeit
from torch.testing import assert_close

from mit_ub.data.mixup import fused_mixup, cross_entropy_mixup, _get_weights, is_mixed_with_unknown


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
        y1 = fused_mixup(x, mixup_prob=1.0, seed=0)
        y2 = fused_mixup(x, mixup_prob=1.0, seed=0)
        y3 = fused_mixup(x, mixup_prob=1.0, seed=1)
        assert_close(y1, y2)
        assert not torch.allclose(y1, y3)

    def test_mixup(self):
        torch.random.manual_seed(0)
        B = 32
        x = torch.randn(B, 1, device="cuda")
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        weights = _get_weights(B, mixup_prob, mixup_alpha, seed)
        weights = weights.view(B, 1).expand_as(x)
        other = x.roll(-1, 0)
        expected = x * weights + other * (1 - weights)

        y = fused_mixup(x, mixup_prob, mixup_alpha, seed)
        assert_close(y, expected)

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
        logits = torch.randn(4, 3, device="cuda")
        logits[0, 0] = -1000
        logits[1, 1] = 1000
        labels = torch.randint(0, 3, (4,), device="cuda")
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.0, mixup_alpha=1.0, seed=0)
        expected = F.cross_entropy(logits, labels, reduction="none")
        assert_close(loss, expected)

    def test_cross_entropy_mixup(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda")
        labels = torch.randint(0, C, (B,), device="cuda")
        mixup_prob = 0.5
        mixup_alpha = 1.0
        seed = 0
        weights = _get_weights(B, mixup_prob, mixup_alpha, seed)
        weights = weights.view(B, 1).expand_as(logits)
        labels_one_hot = F.one_hot(labels, num_classes=C)
        labels_one_hot_other = labels_one_hot.roll(-1, 0)
        mixed_labels = labels_one_hot * weights +  labels_one_hot_other * (1 - weights)
        expected = F.cross_entropy(logits, mixed_labels, reduction="none")

        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        assert_close(loss, expected)

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

        # Using our mixup implementation
        torch.cuda.synchronize()
        start = timeit.default_timer()
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        torch.cuda.synchronize()
        end = timeit.default_timer()
        actual = end - start
        print(f"Time taken: {end - start} seconds")
        assert actual <= 10 * ce_baseline, f"Mixup is slower than baseline: {actual} vs {ce_baseline}"

    def test_ignore_unknown(self):
        torch.random.manual_seed(0)
        B, C = 32, 8
        logits = torch.randn(B, C, device="cuda")
        labels = torch.randint(0, C, (B,), device="cuda")
        seed = 0
        mixup_prob = 0.5
        mixup_alpha = 1.0
        unknown_stride = 4

        # To reject unknowns in the baseline case we set one hot to a large negative value
        # and mask any mixed results that are negative along the class dimension
        labels_one_hot = F.one_hot(labels, num_classes=C)
        labels_one_hot[::unknown_stride] = -1000
        labels_one_hot_other = labels_one_hot.roll(-1, 0)
        weights = _get_weights(B, mixup_prob, mixup_alpha, seed).view(B, 1).expand_as(logits)
        mixed_labels = labels_one_hot * weights +  labels_one_hot_other * (1 - weights)
        valid = mixed_labels.sum(-1) > 0
        _logits = logits[valid]
        _mixed_labels = mixed_labels[valid]
        expected = F.cross_entropy(_logits, _mixed_labels, reduction="none")

        labels[::unknown_stride] = -1
        loss = cross_entropy_mixup(logits, labels, mixup_prob=0.5, mixup_alpha=1.0, seed=0)
        loss = loss[loss != -1.0]
        assert_close(loss, expected)