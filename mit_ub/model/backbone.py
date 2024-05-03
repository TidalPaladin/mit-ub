import torch
from torch import Tensor
import torch.nn as nn

from einops.layers.torch import Rearrange

class PatchEmbed(nn.Module):

    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)
        self.reshape = Rearrange('B C H W -> B (H W) C')
        self.pos_emb = nn.Linear(2, out_channels)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        x = self.conv(x)

        H, W = x.shape[-2:]
        grid = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        grid = torch.stack(grid, dim=-1)
        grid = grid.flatten(1)
        pos_emb = self.pos_emb(grid)

        x = self.reshape(x) + pos_emb
        return x

