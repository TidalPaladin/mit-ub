from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..model.helpers import compile_is_disabled


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def add_noise(alphas_cumprod: Tensor, x: Tensor, t: Tensor, noise: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Add noise to the input image for the given timestep.

    Args:
        alphas_cumprod: Tensor of alphas_cumprod
        x: Input image tensor
        t: Timestep tensor
        noise: Noise tensor

    Returns:
        Tuple of (noised image, noise)
    """
    alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    noised_x = torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise
    return noised_x, noise


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def subtract_noise(alphas_cumprod: Tensor, x: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """
    Subtract noise from the input image for the given timestep.

    Args:
        x: Noised image tensor
        t: Timestep tensor
        noise: Noise tensor

    Returns:
        Denoised image tensor
    """
    alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    denoised_x = (x - torch.sqrt(1 - alphas_cumprod_t) * noise) / torch.sqrt(alphas_cumprod_t)
    return denoised_x


@torch.compile(fullgraph=True, mode="reduce-overhead", disable=compile_is_disabled())
def subtract_noise_one_step(alphas: Tensor, alphas_cumprod: Tensor, x: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """
    Subtract noise from the input image for the given timestep.

    Args:
        x: Noised image tensor
        t: Timestep tensor
        noise: Noise tensor

    Returns:
        Denoised image tensor
    """
    alpha_t = alphas[t].view(-1, 1, 1, 1)
    alpha_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
    alpha_cumprod_prev_t = torch.where(
        (t > 0).view(-1, 1, 1, 1), alphas_cumprod[t.clip(min=1) - 1].view(-1, 1, 1, 1), torch.ones_like(alpha_cumprod_t)
    )

    # Compute the mean for the reverse process
    mean = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t)) * noise)

    # Compute the variance for the reverse process
    variance = torch.sqrt((1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t) * (1 - alpha_t))

    # If t > 0, add noise to the mean for stochastic sampling; otherwise return the mean as the final output
    result = torch.where((t > 0).view(-1, 1, 1, 1), mean + variance * torch.randn_like(mean), mean)

    return result


class DiffusionSchedule(nn.Module):
    r"""Defines a diffusion schedule for adding noise to an input.

    Args:
        num_timesteps: Number of timesteps in the diffusion schedule
        beta_start: Starting beta value
        beta_end: Ending beta value
        max_noise_level: Maximum noise level to add to the image. Should be a float between 0 and 1.
            Values closer to 0 will result in less noise being added to the image.

    Returns:
        Tuple of (noised image, noise, timestep)
    """

    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float, max_noise_level: float | None = None):
        super().__init__()
        self.num_timesteps = int(num_timesteps)

        betas = torch.linspace(beta_start, beta_end, num_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        if max_noise_level is not None:
            assert 0 <= max_noise_level <= 1, "max_noise_level must be between 0 and 1"
            alphas_cumprod.mul_(max_noise_level).add_(1 - max_noise_level)
            alphas.mul_(max_noise_level).add_(1 - max_noise_level)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas", alphas)

    @torch.no_grad()
    def create_noise(self, x: Tensor) -> Tensor:
        """Generate Gaussian noise with the same shape as input tensor."""
        return torch.randn_like(x)

    def add_noise(self, x: Tensor, timestep: Tensor, noise: Tensor) -> Tuple[Tensor, Tensor]:
        return add_noise(self.alphas_cumprod.to(x.device), x, timestep, noise)

    def subtract_noise(self, x: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        return subtract_noise(self.alphas_cumprod.to(x.device), x, timestep, noise)

    def subtract_noise_one_step(self, x: Tensor, timestep: Tensor, noise: Tensor) -> Tensor:
        return subtract_noise_one_step(self.alphas.to(x.device), self.alphas_cumprod.to(x.device), x, timestep, noise)

    def forward(self, x: Tensor, timestep: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        # If manual timestep isn't provided, generate one randomly
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        noise = self.create_noise(x)
        noised_x, noise = add_noise(self.alphas_cumprod.to(x.device), x, timestep, noise)
        return noised_x, noise, timestep
