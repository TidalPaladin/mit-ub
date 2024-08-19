import pytest
import torch
import pytorch_lightning as pl
from torch.testing import assert_close

from mit_ub.tasks.diffusion import Diffusion, DiffusionSchedule

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


class TestDiffusion:
    @pytest.fixture
    def task(self, optimizer_init, backbone):
        if backbone == "vit-dummy":
            pytest.skip("Only AdaptiveViT is supported now")
        return Diffusion(backbone, optimizer_init=optimizer_init)

    def test_fit(self, task, datamodule, logger):
        trainer = pl.Trainer(
            accelerator="cpu",
            fast_dev_run=True,
            logger=logger,
        )
        trainer.fit(task, datamodule=datamodule)
