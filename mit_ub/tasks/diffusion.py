import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, cast

import numpy as np
import torch
import torch.nn as nn
import torchmetrics as tm
from deep_helpers.structs import State
from deep_helpers.tasks import Task
from einops import rearrange
from PIL import Image
from torch import Tensor
from torch_dicom.datasets import DicomPathDataset
from torchmetrics.image import PeakSignalNoiseRatio

from ..model import BACKBONES, AdaptiveViT, TransformerDecoderLayer, ViT


class DiffusionSchedule(nn.Module):
    r"""Defines a diffusion schedule for adding noise to an image.

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
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    @torch.no_grad()
    def create_noise(self, x: Tensor) -> Tensor:
        """Generate Gaussian noise with the same shape as input tensor."""
        return torch.randn_like(x)

    def add_noise(self, x: Tensor, t: Tensor, noise: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Add noise to the input image for the given timestep.

        Args:
            x: Input image tensor
            t: Timestep tensor
            noise: Noise tensor

        Returns:
            Tuple of (noised image, noise)
        """
        alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        noised_x = torch.sqrt(alphas_cumprod_t) * x + torch.sqrt(1 - alphas_cumprod_t) * noise
        return noised_x, noise

    def forward(self, x: Tensor, timestep: Tensor | None = None) -> Tuple[Tensor, Tensor, Tensor]:
        # If manual timestep isn't provided, generate one randomly
        if timestep is None:
            timestep = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device)
        noise = self.create_noise(x)
        noised_x, noise = self.add_noise(x, timestep, noise)
        return noised_x, noise, timestep


class Diffusion(Task):
    alphas_cumprod: Tensor

    def __init__(
        self,
        backbone: str,
        decoder_depth: int = 4,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        max_noise_level: float | None = None,
        optimizer_init: Dict[str, Any] = {},
        lr_scheduler_init: Dict[str, Any] = {},
        lr_interval: str = "epoch",
        lr_monitor: str = "train/total_loss_epoch",
        named_datasets: bool = False,
        checkpoint: Optional[str] = None,
        strict_checkpoint: bool = True,
        log_train_metrics_interval: int = 1,
        log_train_metrics_on_epoch: bool = False,
        weight_decay_exemptions: Set[str] = set(),
    ):
        super().__init__(
            optimizer_init,
            lr_scheduler_init,
            lr_interval,
            lr_monitor,
            named_datasets,
            checkpoint,
            strict_checkpoint,
            log_train_metrics_interval,
            log_train_metrics_on_epoch,
            weight_decay_exemptions,
        )
        self.backbone = cast(ViT, self.prepare_backbone(backbone))
        if not isinstance(self.backbone, AdaptiveViT):
            raise ValueError("Backbone must be an instance of AdaptiveViT")

        # Create diffusion schedule
        self.diffusion_schedule = DiffusionSchedule(num_timesteps, beta_start, beta_end, max_noise_level)

        # Cross attention decoder
        self.time_embed = nn.Embedding(num_timesteps, self.backbone.kv_dim)
        diffusion_dim_ff = 4 * self.backbone.kv_dim
        self.diffusion_decoder = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    self.backbone.kv_dim,
                    self.backbone.nhead,
                    self.backbone.dim,
                    diffusion_dim_ff,
                    dropout=0.1,
                    activation=nn.SiLU(),
                    self_attn=False,
                )
                for _ in range(decoder_depth)
            ]
        )

        # Predictor head
        pixels_per_token = math.prod(self.backbone.patch_size_2d) * self.backbone.in_channels
        self.diffusion_head = nn.Linear(self.backbone.kv_dim, pixels_per_token)
        nn.init.trunc_normal_(self.diffusion_head.weight, std=0.01, a=-1e-3, b=1e-3)

        self.criterion = nn.MSELoss()
        self.save_hyperparameters()

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    def create_metrics(self, state: State) -> tm.MetricCollection:
        return tm.MetricCollection({"psnr": PeakSignalNoiseRatio()})

    def forward(self, x: Tensor, t: Tensor) -> Dict[str, Tensor]:
        # Run the backbone / encoder
        coarse_features, fine_features = self.backbone.forward_with_kv(x, reshape=False)

        # Get embedding vector for the timestep and add it to the fine features
        N = t.numel()
        time_embed = self.time_embed(t).view(N, 1, -1)
        fine_features = fine_features + time_embed

        # Run the decoder
        for layer in self.diffusion_decoder:
            fine_features = layer(fine_features, coarse_features)

        # Get the prediction and reshape to image
        prediction = self.diffusion_head(fine_features)
        H, W = self.backbone.tokenizer.kv_size(x.shape[-2:])
        Hp, Wp = self.backbone.patch_size_2d
        prediction = rearrange(prediction, "b (h w) (hp wp c) -> b c (h hp) (w wp)", h=H, w=W, hp=Hp, wp=Wp)

        return {"noise": prediction}

    def step(
        self,
        batch: Any,
        batch_idx: int,
        state: State,
        metrics: Optional[tm.MetricCollection] = None,
    ) -> Dict[str, Any]:
        x: Tensor = batch["img"]

        # generate noised image and noise tensor
        with torch.no_grad():
            noised_x, noise, timestep = self.diffusion_schedule(x)

        # run forward pass to predict noise
        pred_dict = self(noised_x, timestep)
        pred: Tensor = pred_dict["noise"]

        # compute loss between predicted noise and actual noise
        assert pred.shape == noise.shape, f"Prediction shape {pred.shape} does not match target shape {noise.shape}"
        loss = self.criterion(pred, noise)

        output = {
            "log": {
                "loss_diffusion": loss,
            },
            "noised_x": noised_x,
            "noise": noise,
            "timestep": timestep,
        }

        return output


@torch.no_grad()
def preview(
    path: Path,
    output: Path,
    timestep: int,
    num_timesteps: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    max_noise_level: float | None = None,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"File {path} does not exist")
    if not output.parent.is_dir():
        raise NotADirectoryError(f"Directory {output.parent} does not exist")
    if not 0 <= timestep < num_timesteps:
        raise ValueError(f"Timestep {timestep} must be between 0 and {num_timesteps - 1}")
    schedule = DiffusionSchedule(num_timesteps, beta_start, beta_end, max_noise_level)

    # Get the image
    # pipeline = LightningInferencePipeline(dicom_paths=[path,])
    ds = DicomPathDataset([path], img_size=(512, 384))
    x = ds[0]["img"]

    # Run schedule
    noised_x, _, _ = schedule(x, x.new_tensor([timestep], dtype=torch.long))

    noised_x = noised_x.squeeze().clip_(min=0, max=1).mul_(2**16).numpy().astype(np.uint16)

    # Convert tensor to PIL Image and save
    noised_x_pil = Image.fromarray(noised_x)
    noised_x_pil.save(output.with_suffix(".png"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to the input image")
    parser.add_argument("output", type=Path, help="Path to the output image")
    parser.add_argument("-t", "--timestep", type=int, default=0, help="Timestep to preview")
    parser.add_argument(
        "-n", "--num_timesteps", type=int, default=100, help="Number of timesteps in the diffusion schedule"
    )
    parser.add_argument("-s", "--beta_start", type=float, default=0.0001, help="Starting beta value")
    parser.add_argument("-e", "--beta_end", type=float, default=0.02, help="Ending beta value")
    parser.add_argument(
        "-m", "--max_noise_level", type=float, default=None, help="Maximum noise level to add to the image"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    preview(
        args.path, args.output, args.timestep, args.num_timesteps, args.beta_start, args.beta_end, args.max_noise_level
    )
