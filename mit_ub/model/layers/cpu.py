import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SReLU(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x).pow(2)


def _act_func(activation: str):
    funcs = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "srelu": SReLU(),
    }
    if activation not in funcs:
        raise NotImplementedError("Activation type " + activation + " is not supported!")
    return funcs[activation]


class LayerNormLinear(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        normalization: str = "LayerNorm",
        zero_centered_gamma: bool = False,
    ):
        super().__init__()
        self.normalization = normalization
        self.zero_centered_gamma = zero_centered_gamma

        self.layer_norm_weight = nn.Parameter(torch.empty(in_features))
        if normalization == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(torch.empty(in_features))
        else:
            self.register_parameter("layer_norm_bias", None)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.layer_norm_weight, 0.0 if self.zero_centered_gamma else 1.0)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)

        nn.init.trunc_normal_(self.weight, std=0.02)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        w_norm = self.layer_norm_weight + 1 if self.zero_centered_gamma else self.layer_norm_weight
        if self.normalization == "LayerNorm":
            x = F.layer_norm(x, x.shape[-1:], w_norm, self.layer_norm_bias)
        elif self.normalization == "RMSNorm":
            x = F.rms_norm(x, x.shape[-1:], w_norm)
        else:
            raise NotImplementedError("Normalization type " + self.normalization + " is not supported!")

        x = F.linear(x, self.weight, self.bias)
        return x


class LayerNormMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        bias: bool = True,
        normalization: str = "LayerNorm",
        activation: str = "gelu",
        zero_centered_gamma: bool = False,
    ):
        super().__init__()
        self.normalization = normalization
        self.zero_centered_gamma = zero_centered_gamma
        self.bias = bias
        self.activation = _act_func(activation)

        self.layer_norm_weight = nn.Parameter(torch.empty(hidden_size))
        if normalization == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("layer_norm_bias", None)

        self.fc1_weight = nn.Parameter(torch.empty(ffn_hidden_size, hidden_size))
        self.fc2_weight = nn.Parameter(torch.empty(hidden_size, ffn_hidden_size))

        if bias:
            self.fc1_bias = nn.Parameter(torch.empty(ffn_hidden_size))
            self.fc2_bias = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("fc1_bias", None)
            self.register_parameter("fc2_bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.layer_norm_weight, 0.0 if self.zero_centered_gamma else 1.0)
        if self.layer_norm_bias is not None:
            nn.init.zeros_(self.layer_norm_bias)

        nn.init.trunc_normal_(self.fc1_weight, std=0.02)
        nn.init.trunc_normal_(self.fc2_weight, std=0.02)

        if self.bias:
            nn.init.zeros_(self.fc1_bias)
            nn.init.zeros_(self.fc2_bias)

    def forward(self, x: Tensor) -> Tensor:
        w_norm = self.layer_norm_weight + 1 if self.zero_centered_gamma else self.layer_norm_weight
        if self.normalization == "LayerNorm":
            x = F.layer_norm(x, x.shape[-1:], w_norm, self.layer_norm_bias)
        elif self.normalization == "RMSNorm":
            x = F.rms_norm(x, x.shape[-1:], w_norm)
        else:
            raise NotImplementedError("Normalization type " + self.normalization + " is not supported!")

        x = F.linear(x, self.fc1_weight, self.fc1_bias)
        x = self.activation(x)
        x = F.linear(x, self.fc2_weight, self.fc2_bias)
        return x
