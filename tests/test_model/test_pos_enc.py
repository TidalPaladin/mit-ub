import pytest
import torch
from torchtune.modules.peft.lora import LoRALinear

from mit_ub.model.lora import LoRATarget
from mit_ub.model.pos_enc import PositionEncoder, RelativeFactorizedPosition


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

    @pytest.mark.parametrize("size", [(4, 4), (32, 32)])
    def test_create_grid_with_noise(self, size):
        torch.random.manual_seed(0)
        grid1 = PositionEncoder.create_grid(size, normalize=False, add_noise=True)
        grid2 = PositionEncoder.create_grid(size, normalize=False)
        assert grid1.shape == grid2.shape == (1, size[0] * size[1], 2)
        assert ((grid1 - grid2).abs() <= 0.5).all()

    @pytest.mark.parametrize("size", [(4, 4), (32, 32)])
    def test_create_normalized_grid_with_noise(self, size):
        torch.random.manual_seed(0)
        grid1 = PositionEncoder.create_grid(size, normalize=True, add_noise=True)
        grid2 = PositionEncoder.create_grid(size, normalize=True)
        assert grid1.shape == grid2.shape == (1, size[0] * size[1], 2)
        assert ((grid1 - grid2).abs() <= 1 / (max(size) - 1)).all()


class TestRelativeFactorizedPosition:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_forward(self, device):
        B, L, C, D = 2, 32, 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to(device)
        grid = torch.randn(B, L, C, device=device)
        out = layer(grid)
        assert out.shape == (B, L, D)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_lora(self, device):
        B, L, C, D = 2, 32, 2, 16
        torch.random.manual_seed(0)
        layer = RelativeFactorizedPosition(C, D).to(device)
        grid = torch.randn(B, L, C, device=device, requires_grad=True)

        layer = layer.apply_lora(target=[LoRATarget.POSITION], rank=4, alpha=16)
        assert isinstance(layer.proj.fc1, LoRALinear)
        assert isinstance(layer.proj.fc2, LoRALinear)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out = layer(grid)
        assert out.shape == (B, L, D)

        for name, param in layer.named_parameters():
            assert param.requires_grad == ("lora_a" in name or "lora_b" in name)
