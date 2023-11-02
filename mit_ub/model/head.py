from copy import copy

import torch.nn as nn
from einops.layers.torch import Rearrange
from gpvit.layers import MLPMixer
from torch import Tensor


class MLPMixerPooling(MLPMixer):
    """
    MLP Mixer pooling head for group propagation tokens.

    Args:
        dim: The dimension of the input tensor.
        token_dim: The hidden dimension of the token-mixing MLP.
        num_patches: The number of patches to be considered for token mixing.
        channel_dim: The hidden dimension of the channel-mixing MLP.
        dropout: Dropout rate for the MLPs. Defaults to 0.0.
        activation: Activation function for the MLPs. Defaults to nn.GELU().
    """

    def __init__(
        self,
        dim: int,
        token_dim: int,
        num_patches: int,
        channel_dim: int,
        dropout: float = 0.0,
        activation: nn.Module = nn.GELU(),
    ):
        super().__init__(dim, token_dim, num_patches, channel_dim, dropout, activation)
        self.token_mixing = nn.Sequential(
            nn.Conv1d(num_patches, token_dim, 1),
            copy(activation),
            nn.Dropout(dropout),
            nn.Conv1d(token_dim, 1, 1),
            Rearrange("b () d -> b d"),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(self.token_mixing(x))
        x = self.norm2(x + self.channel_mixing(x))
        return x
