import pytest
import torch

from mit_ub.model.pos_enc import PositionEncoder


class TestPositionEncoder:

    @pytest.mark.parametrize("normalize", [True, False])
    def test_create_grid(self, normalize):
        dims = (4, 4)
        grid = PositionEncoder.create_grid(dims, normalize=normalize)
        assert grid.shape == (1, 16, 2)
        if normalize:
            assert torch.all(grid[0, 0] == torch.tensor([-1.0, -1.0]))
            assert torch.all(grid[0, -1] == torch.tensor([1.0, 1.0]))
        else:
            assert torch.all(grid[0, 0] == torch.tensor([0, 0]))
            assert torch.all(grid[0, -1] == torch.tensor([3, 3]))

    def test_create_grid_with_noise(self):
        torch.random.manual_seed(0)
        dims = (4, 4)
        grid1 = PositionEncoder.create_grid(dims, normalize=False, add_noise=True)
        grid2 = PositionEncoder.create_grid(dims, normalize=False)
        assert grid1.shape == grid2.shape == (1, 16, 2)
        assert ((grid1 - grid2).abs() <= 0.5).all()
