import pytest
import torch
import torch.nn.functional as F

from mit_ub.data.mixup import is_mixed, mixup, mixup_dense_label, sample_mixup_parameters


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
