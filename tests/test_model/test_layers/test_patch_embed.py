import math

import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.layers.patch_embed import (
    PatchEmbed2d,
    binary_search_for_scale,
    calculate_sizes_for_budget,
    pack,
    size_at_scale,
    token_count,
    token_count_at_scale,
    token_count_foreach,
    tokenized_size,
    tokenized_size_at_scale,
    tokenized_size_foreach,
    unpack,
)


@pytest.mark.parametrize(
    "input_size,expected",
    [
        ((16, 16), (4, 4)),
        ((32, 32), (8, 8)),
        ((64, 64), (16, 16)),
    ],
)
def test_tokenized_size(input_size, expected):
    assert tokenized_size(input_size, (4, 4)) == list(expected)


def test_tokenized_size_foreach():
    input_sizes = [(16, 16), (32, 32), (64, 64)]
    patch_sizes = [(4, 4)] * len(input_sizes)
    expected = [[4, 4], [8, 8], [16, 16]]
    assert tokenized_size_foreach(input_sizes, patch_sizes) == expected


@pytest.mark.parametrize(
    "input_size,expected",
    [
        ((16, 16), 16),
        ((32, 32), 64),
        ((64, 64), 256),
    ],
)
def test_token_count(input_size, expected):
    assert token_count(input_size, (4, 4)) == expected


def test_token_count_foreach():
    input_sizes = [(16, 16), (32, 32), (64, 64)]
    patch_sizes = [(4, 4)] * len(input_sizes)
    assert token_count_foreach(input_sizes, patch_sizes) == 16 + 64 + 256


@pytest.mark.parametrize(
    "input_size,scale,expected",
    [
        ((16, 16), 0.5, (8, 8)),
        ((32, 32), 0.25, (8, 8)),
        ((64, 64), 2, (128, 128)),
    ],
)
def test_size_at_scale(input_size, scale, expected):
    assert size_at_scale(input_size, scale) == list(expected)


@pytest.mark.parametrize(
    "input_size,patch_size,scale,expected",
    [
        ((16, 16), (4, 4), 0.5, (2, 2)),
        ((32, 32), (4, 4), 0.25, (2, 2)),
        ((64, 64), (4, 4), 2, (32, 32)),
    ],
)
def test_tokenized_size_at_scale(input_size, patch_size, scale, expected):
    assert tokenized_size_at_scale(input_size, patch_size, scale) == list(expected)


@pytest.mark.parametrize(
    "input_size,patch_size,scale,expected",
    [
        ((16, 16), (4, 4), 0.5, 4),
        ((32, 32), (4, 4), 0.25, 4),
        ((64, 64), (4, 4), 2, 32 * 32),
    ],
)
def test_token_count_at_scale(input_size, patch_size, scale, expected):
    assert token_count_at_scale(input_size, patch_size, scale) == expected


def test_pack_basic():
    tensors = [torch.randn(2, 3), torch.randn(3, 3), torch.randn(1, 3)]
    packed, cu_seq_lens, max_seq_len = pack(tensors)

    assert packed.shape == (6, 3)  # 2 + 3 + 1 = 6
    assert cu_seq_lens.tolist() == [0, 2, 5, 6]
    assert max_seq_len == 3


def test_pack_unpack_cycle():
    # Create a list of tensors with different sequence lengths
    tensors = [torch.randn(2, 3), torch.randn(3, 3), torch.randn(1, 3)]

    # Pack the tensors
    packed, cu_seq_lens, max_seq_len = pack(tensors)

    # Unpack the tensors
    unpacked = unpack(packed, cu_seq_lens)

    # Verify the cycle
    assert len(unpacked) == len(tensors)
    for orig, unpacked_tensor in zip(tensors, unpacked):
        assert_close(orig, unpacked_tensor)


def test_pack_unpack_cuda():
    # Test on CUDA if available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    tensors = [torch.randn(2, 3, device="cuda"), torch.randn(3, 3, device="cuda"), torch.randn(1, 3, device="cuda")]

    packed, cu_seq_lens, max_seq_len = pack(tensors)
    unpacked = unpack(packed, cu_seq_lens)

    assert len(unpacked) == len(tensors)
    for orig, unpacked_tensor in zip(tensors, unpacked):
        assert_close(orig, unpacked_tensor)


@pytest.mark.parametrize(
    "current_size,patch_size,drop_rate,budget,expected",
    [
        ((16, 16), (4, 4), 0.0, 4, (8, 8)),
        ((16, 16), (4, 4), 0.5, 4, (16, 16)),
        ((32, 32), (4, 4), 0.0, 64, (32, 32)),
        ((32, 32), (4, 4), 0.0, 32, (5, 5)),
        ((32, 32), (4, 4), 0.5, 32, (10, 10)),
    ],
)
def test_binary_search_for_scale(current_size, patch_size, drop_rate, budget, expected):
    result = binary_search_for_scale(current_size, patch_size, drop_rate, budget)
    token_count_result = token_count(result, patch_size)
    assert token_count_result * (1 - drop_rate) <= budget


@pytest.mark.parametrize(
    "sizes,patch_sizes,drop_rates,budget,expected",
    [
        # No downsampling
        (
            [(16, 16), (32, 32), (64, 64)],
            [(4, 4)] * 3,
            [0.0, 0.0, 0.0],
            10000,
            [[16, 16], [32, 32], [64, 64]],
        ),
        # Needs downsampling
        (
            [(16, 16), (32, 32), (64, 64)],
            [(4, 4)] * 3,
            [0.0, 0.0, 0.0],
            100,
            [[16, 16], [16, 16], [32, 32]],
        ),
        (
            [(16, 16), (32, 32), (64, 64)],
            [(2, 2)] * 3,
            [0.0, 0.0, 0.0],
            100,
            [[8, 8], [8, 8], [16, 16]],
        ),
        # With drop tokens
        (
            [(16, 16), (32, 32), (64, 64)],
            [(4, 4)] * 3,
            [0.5, 0.75, 0.5],
            100,
            [[16, 16], [32, 32], [48, 48]],
        ),
    ],
)
def test_calculate_sizes_for_budget(sizes, patch_sizes, drop_rates, budget, expected):
    result = calculate_sizes_for_budget(sizes, patch_sizes, drop_rates, budget)
    token_count_result = sum(
        [
            token_count(size, patch_size) * (1 - drop_rate)
            for size, patch_size, drop_rate in zip(result, patch_sizes, drop_rates)
        ]
    )
    assert token_count_result <= budget
    assert result == expected


class TestPatchEmbed2d:

    @pytest.mark.cuda
    def test_forward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    @pytest.mark.cuda
    def test_backward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None
