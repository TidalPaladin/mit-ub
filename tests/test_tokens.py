import pytest
import torch
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.tokens import apply_mask, create_mask, mask_is_ragged


@pytest.mark.parametrize(
    "mask, exp",
    [
        (([True, True], [True, True]), False),
        (([True, False], [True, True]), True),
        (([False, True], [True, True]), True),
        (([False, False], [True, True]), True),
        (([False, False], [False, False]), False),
    ],
)
def test_is_ragged(mask, exp):
    assert mask_is_ragged(torch.tensor(mask, dtype=torch.bool)) == exp


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize(
    "mask, fill_value, padding_value, exp",
    [
        # Non-ragged
        (([True, True], [True, True]), None, 0, torch.tensor([[0, 1], [0, 1]])),
        (([True, False], [False, True]), None, 0, torch.tensor([[0], [1]])),
        (([False, True], [True, False]), None, 0, torch.tensor([[1], [0]])),
        (([False, False], [False, False]), None, 0, torch.tensor([[], []])),
        (([True, False], [False, True]), -1.0, 0, torch.tensor([[0, -1.0], [-1.0, 1]])),
        (([True, False], [False, True]), torch.tensor(-1.0), 0, torch.tensor([[0, -1.0], [-1.0, 1]])),
        # Ragged
        (([True, False], [True, True]), None, 0, torch.tensor([[0, 0], [0, 1]])),
        (([False, True], [True, True]), None, 0, torch.tensor([[1, 0], [0, 1]])),
        (([True, False], [True, True]), None, -1.0, torch.tensor([[0, -1.0], [0, 1]])),
        (([False, True], [True, True]), None, -1.0, torch.tensor([[1, -1.0], [0, 1]])),
        (([False, True], [True, True]), -1.0, 0, torch.tensor([[-1.0, 1], [0, 1]])),
    ],
)
def test_apply(device, mask, fill_value, padding_value, exp):
    mask = torch.tensor(mask, dtype=torch.bool, device=device)
    N, L = mask.shape
    x = torch.arange(L, device=device).view(1, L, 1).expand(N, L, 1)
    o = apply_mask(mask, x, fill_value, padding_value)
    assert_close(o, exp.view_as(o).type_as(o))


class TestCreateMask:

    @pytest.mark.parametrize(
        "size, exp",
        [
            ((32,), 32),
            ((8, 8), 64),
        ],
    )
    def test_create_size(self, size, exp):
        assert create_mask(size, 0.5).numel() == exp

    @pytest.mark.parametrize(
        "size, ratio, exp",
        [
            ((1000,), 0.5, 500),
            ((100, 10), 0.25, 750),
        ],
    )
    def test_create_ratio(self, size, ratio, exp):
        assert create_mask(size, ratio).sum() == exp

    @pytest.mark.parametrize(
        "batch_size, size",
        [
            (2, (1000,)),
            (4, (100, 10)),
        ],
    )
    def test_create_batch_size(self, batch_size, size):
        mask = create_mask(size, 0.5, batch_size)
        assert mask.shape[0] == batch_size
        assert (mask.sum(-1) == mask.sum(-1)[0, None]).all()

    def test_create_scale(self):
        ratio = 0.5
        torch.random.manual_seed(0)
        mask = create_mask((8, 8), ratio, scale=2)
        mask_grid = mask.view(1, 1, 8, 8)
        target_size = (4, 4)
        # Average pool the mask as a a float. Mask should be all 1.0 or 0.0 within a block,
        # so pooled entries should be all 1.0 or 0.0
        pooled = F.adaptive_avg_pool2d(mask_grid.float(), target_size).view(*target_size)
        assert ((pooled == 1.0) | (pooled == 0.0)).all()

    @pytest.mark.cuda
    def test_create_device(self):
        size = (16, 16)
        ratio = 0.25
        scale = 2
        batch_size = 2
        mask = create_mask(size, ratio, batch_size, scale=scale, device=torch.device("cuda"))
        assert mask.device.type == "cuda"
