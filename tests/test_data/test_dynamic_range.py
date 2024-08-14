import pytest
import torch
from torch import Tensor
from torch.testing import assert_close
from torchvision.tv_tensors import Image, Video

from mit_ub.data.dynamic_range import DEFAULT_MAX_VAL, CompressDynamicRange


@pytest.mark.parametrize("input_type", [Image, Video, Tensor])
@pytest.mark.parametrize("input_range", [(0, 1), (-1, 1), (0, 255)])
@pytest.mark.parametrize("max_val", [DEFAULT_MAX_VAL, 0.1, 1.0])
def test_compress_dynamic_range(input_type, input_range, max_val):
    transform = CompressDynamicRange(max_val=max_val)

    # Create input tensor
    if isinstance(input_type, Image):
        shape = (3, 32, 32)
    else:
        shape = (1, 3, 32, 32)
    x = torch.rand(shape) * (input_range[1] - input_range[0]) + input_range[0]
    x = input_type(x)

    # Apply transform
    y = transform(x)

    # Check output type
    assert isinstance(y, input_type)

    # Check output range
    assert y.min() >= 0
    assert y.max() <= max_val

    # Check if normalization is applied correctly
    if input_range != (0, 1):
        assert_close(y / max_val, (x - x.min()) / (x.max() - x.min()), rtol=1, atol=1e-4)
    else:
        assert_close(y, x * max_val, atol=1e-4, rtol=1)


def test_compress_dynamic_range_edge_cases():
    transform = CompressDynamicRange()

    # Test with constant input
    x = Image(torch.full((1, 10, 10), 5.0))
    y = transform(x)
    assert_close(y, torch.full_like(y, DEFAULT_MAX_VAL), rtol=1, atol=1e-4)

    # Test with zero input
    x = Image(torch.zeros(1, 10, 10))
    y = transform(x)
    assert_close(y, torch.zeros_like(y), rtol=1, atol=1e-4)
