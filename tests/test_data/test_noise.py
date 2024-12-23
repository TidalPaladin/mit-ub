import pytest
import torch

from mit_ub.data.noise import RandomNoise, multiplicative_noise, salt_pepper_noise, uniform_noise


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "min,max,clip,expected_range",
    [
        (-0.2, 0.2, True, (0.8, 1.0)),
        (-0.5, 0.5, False, (-0.5, 1.5)),
    ],
)
def test_uniform_noise(min: float, max: float, clip: bool, expected_range: tuple[float, float], device: str):
    x = torch.ones(10, 10, device=device)
    result = uniform_noise(x, min=min, max=max, clip=clip)

    # Check shape and dtype preserved
    assert result.shape == x.shape
    assert result.dtype == x.dtype

    # Check values are in expected range
    assert result.min() >= expected_range[0]
    assert result.max() <= expected_range[1]

    # Check noise was actually applied
    assert not torch.allclose(result, x)

    # Check different random values each time
    result2 = uniform_noise(x, min=min, max=max, clip=clip)
    assert not torch.allclose(result, result2)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "prob,expected_ratio", [(0.01, (0.005, 0.015)), ((0.01, 0.05), (0.01, 0.05)), (0.1, (0.09, 0.11))]
)
def test_salt_pepper_noise(prob: float | tuple[float, float], expected_ratio: tuple[float, float], device: str):
    x = torch.zeros(1000, 1000, device=device)
    result = salt_pepper_noise(x, prob=prob)

    # Check shape and dtype preserved
    assert result.shape == x.shape
    assert result.dtype == x.dtype

    # Check values are only 0 or 1
    assert torch.all((result == 0) | (result == 1))

    # Check different random values each time
    result2 = salt_pepper_noise(x, prob=prob)
    assert not torch.allclose(result, result2)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "scale,clip,expected_range",
    [
        (0.2, True, (0.0, 1.0)),
        (0.5, False, (-float("inf"), float("inf"))),
    ],
)
def test_multiplicative_noise(scale: float, clip: bool, expected_range: tuple[float, float], device: str):
    x = torch.ones(10, 10, device=device)
    result = multiplicative_noise(x, scale=scale, clip=clip)

    # Check shape and dtype preserved
    assert result.shape == x.shape
    assert result.dtype == x.dtype

    # Check values are in expected range
    assert result.min() >= expected_range[0]
    assert result.max() <= expected_range[1]

    # Check noise was actually applied
    assert not torch.allclose(result, x)

    # Check different random values each time
    result2 = multiplicative_noise(x, scale=scale, clip=clip)
    assert not torch.allclose(result, result2)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
def test_random_noise(device: str):
    # NOTE: The decision to apply some noise is random, so we need to set a seed that
    # triggers at least one noise application
    torch.random.manual_seed(1)
    x = torch.rand(1, 10, 10, device=device)
    result = RandomNoise()(x)
    assert not torch.allclose(result, x)
