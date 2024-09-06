from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from triton_helpers.benchmark import CLI, KernelExecutor

from .kernel import relu2


@dataclass
class Baseline(KernelExecutor):

    def prepare_inputs(self, L: int, **kwargs) -> Dict[str, Tensor | None]:
        x = torch.randn((L,), **kwargs)
        return {"x": x}

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(x)
        return y * y


@dataclass
class Triton(Baseline):

    def forward(self, x: Tensor) -> Tensor:
        return relu2(x)


if __name__ == "__main__":
    CLI.entrypoint(
        "ReLU2",
        [
            Baseline("baseline"),
            Triton("triton"),
            # Not working since Triton upgrade
            # Triton("triton-pointwise", method="pointwise"),
        ],
        dims={
            "L": (tuple(int(2**i) for i in range(7, 21)), "values"),
        },
    )
