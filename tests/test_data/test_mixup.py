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

    def test_cross_entropy_mixup_prob_0(self):
        torch.random.manual_seed(0)
        logits = torch.randn(4, 3, device="cuda")
        labels = torch.randint(0, 3, (4,), device="cuda")
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
        assert False




@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "mixup_prob,mixup_alpha",
    [
        (0.0, 1.0),  # No mixup
        (1.0, 0.5),  # Always mixup with alpha=0.5
        (0.5, 2.0),  # 50% mixup with alpha=2.0
    ],
)
def test_mixup(device: str, mixup_prob: float, mixup_alpha: float):
    # Create test input
    x = torch.randn(4, 3, 32, 32, device=device)
    x_orig = x.clone()

    # Sample mixup parameters
    weight = sample_mixup_parameters(x.shape[0], mixup_prob, mixup_alpha, device=x.device)

    # Apply mixup
    result = mixup(x, weight)

    # Check shape and dtype preserved
    assert result.shape == x.shape
    assert result.dtype == x.dtype

    # Check values are bounded by input range
    assert result.min() >= torch.min(x.min(), x.roll(1, 0).min())
    assert result.max() <= torch.max(x.max(), x.roll(1, 0).max())

    # For mixup_prob=0, should be unchanged
    if mixup_prob == 0:
        assert torch.allclose(result, x_orig)

    # For mixup_prob=1, should be mixed
    if mixup_prob == 1:
        assert not torch.allclose(result, x_orig)

    # Test that weight=0 gives original input
    zero_weight = torch.zeros(x.shape[0], device=device)
    assert torch.allclose(mixup(x, zero_weight), x)

    # Test that weight=1 gives rolled input
    one_weight = torch.ones(x.shape[0], device=device)
    assert torch.allclose(mixup(x, one_weight), x.roll(1, 0))


@pytest.mark.parametrize(
    "mixup_prob,mixup_alpha",
    [
        (0.0, 1.0),  # No mixup
        (1.0, 0.5),  # Always mixup with alpha=0.5
        (0.5, 2.0),  # 50% mixup with alpha=2.0
    ],
)
def test_is_mixed(mixup_prob: float, mixup_alpha: float):
    # Sample mixup parameters
    torch.random.manual_seed(0)
    N = 10
    weight = sample_mixup_parameters(N, mixup_prob, mixup_alpha)

    # For mixup_prob=0, should be unchanged
    if mixup_prob == 0:
        assert not is_mixed(weight).any()

    # For mixup_prob=1, should be mixed
    elif mixup_prob == 1:
        assert is_mixed(weight).all()

    else:
        assert is_mixed(weight).any()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "mixup_prob,mixup_alpha",
    [
        (0.0, 1.0),  # No mixup
        (1.0, 0.5),  # Always mixup with alpha=0.5
        (0.5, 2.0),  # 50% mixup with alpha=2.0
    ],
)
def test_mixup_dense_label(device: str, mixup_prob: float, mixup_alpha: float):
    # Sample mixup parameters
    B, N = 4, 10
    weight = sample_mixup_parameters(B, mixup_prob, mixup_alpha, device=torch.device(device))

    # Test mixup_dense_label
    labels = torch.tensor([0, 1, 2, 3], device=device)
    labels_one_hot = F.one_hot(labels, num_classes=N).float()
    mixed_labels = mixup_dense_label(labels, weight, num_classes=N)

    # Check shape and dtype
    assert mixed_labels.shape == (B, N)
    assert mixed_labels.dtype == labels_one_hot.dtype

    # For mixup_prob=0, should be unchanged
    if mixup_prob == 0:
        assert torch.allclose(mixed_labels, labels_one_hot)

    # For mixup_prob=1, should be mixed
    if mixup_prob == 1:
        assert not torch.allclose(mixed_labels, labels_one_hot)

    # Test that weight=0 gives original labels
    zero_weight = torch.zeros(labels.shape[0], device=device)
    assert torch.allclose(mixup_dense_label(labels, zero_weight, num_classes=N), labels_one_hot)

    # Test that weight=1 gives rolled labels
    one_weight = torch.ones(labels.shape[0], device=device)
    assert torch.allclose(mixup_dense_label(labels, one_weight, num_classes=N), labels_one_hot.roll(1, 0))


def test_is_mixed_with_unknown_binary():
    torch.random.manual_seed(0)
    label = torch.tensor([0, 1, -10, 0, 1, -10])
    mask = label != -10
    weight = sample_mixup_parameters(label.shape[0], 0.8, 1.0)

    result = mask & ~is_mixed_with_unknown(weight, mask)
    mixed_label = mixup(label, weight)
    assert (mixed_label[result] >= 0).all()
    assert (mixed_label[~result] < 0).all()


def test_is_mixed_with_unknown_categorical():
    torch.random.manual_seed(0)
    num_classes = 4
    label = torch.tensor([0, 1, -10, 2, 3, -10])
    mask = label != -10
    weight = sample_mixup_parameters(label.shape[0], 0.8, 1.0)

    result = mask & ~is_mixed_with_unknown(weight, mask)
    mixed_label = mixup_dense_label(label, weight, num_classes=num_classes)
    assert (mixed_label[result].sum(-1) == 1).all()
    assert (mixed_label[~result].sum(-1) < 1).all()


def test_mixup_siglip():
    torch.random.manual_seed(0)
    B = 8
    label = torch.eye(B)
    weight = sample_mixup_parameters(B, 0.8, 1.0)

    mixed_label = mixup(label, weight)
    assert mixed_label.max() <= 1.0
    assert mixed_label.min() >= 0.0
    assert (mixed_label.sum(-1) == 1).all()


@pytest.mark.cuda
def test_mixup_speed():
    x = torch.randn(16, 1, 512, 384, device="cuda")
    x_orig = x.clone()
    mixup_prob = 0.2
    mixup_alpha = 1.0

    torch.cuda.synchronize()
    start = timeit.default_timer()
    weight = sample_mixup_parameters(x.shape[0], mixup_prob, mixup_alpha, device=x.device)
    result = mixup(x, weight)
    torch.cuda.synchronize()
    end = timeit.default_timer()
    print(f"Time taken: {end - start} seconds")
    assert False

