import pytest
import torch

from mit_ub.model.layers.pos_enc import RelativeFactorizedPosition, create_grid


@pytest.mark.parametrize("normalize", [True, False])
def test_create_grid(normalize):
    dims = (4, 4)
    grid = create_grid(dims, normalize=normalize)
    assert grid.shape == (1, 16, 2)
    if normalize:
        assert torch.all(grid[0, 0] == torch.tensor([-1.0, -1.0]))
        assert torch.all(grid[0, -1] == torch.tensor([1.0, 1.0]))
    else:
        assert torch.all(grid[0, 0] == torch.tensor([0, 0]))
        assert torch.all(grid[0, -1] == torch.tensor([3, 3]))


class TestRelativeFactorizedPosition:

    @pytest.mark.cuda
    def test_forward(self):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to("cuda")
        out = layer((8, 8))
        L = 64
        assert out.shape == (1, L, D)

    def test_backward(self):
        C, D = 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to("cuda")
        out = layer((8, 8))
        out.sum().backward()
