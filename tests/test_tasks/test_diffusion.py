import pytest
import torch
from torch.testing import assert_close

from mit_ub.tasks.diffusion import DiffusionSchedule


class TestDiffusionSchedule:

    @pytest.mark.parametrize("max_noise_level", [None, 0.5])
    def test_add_subtract_noise(self, max_noise_level):
        T = 100
        B = 10
        schedule = DiffusionSchedule(num_timesteps=T, beta_start=0.0001, beta_end=0.02, max_noise_level=max_noise_level)
        x = torch.rand(B, 1, 64, 64)
        t = torch.randint(0, T, (B,))
        noise = schedule.create_noise(x)
        noised_x, noise = schedule.add_noise(x, t, noise)
        recovered_x = schedule.subtract_noise(noised_x, t, noise)
        assert_close(x, recovered_x)
