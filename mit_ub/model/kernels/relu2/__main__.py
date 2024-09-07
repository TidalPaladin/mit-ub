from dataclasses import dataclass
from typing import Dict, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from triton_helpers.benchmark import CLI, KernelExecutor

from .kernel import ReLU2 as relu2_triton
from .kernel import relu2 as relu2_compiled


@dataclass
class Baseline(KernelExecutor):

    def prepare_inputs(self, L: int, **kwargs) -> Dict[str, Tensor | None]:
        x = torch.randn((L,), **kwargs)
        return {"x": x}

    def forward(self, x: Tensor) -> Tensor:
        y = F.relu(x)
        return y * y


@dataclass
class Compiled(KernelExecutor):

    def prepare_inputs(self, L: int, **kwargs) -> Dict[str, Tensor | None]:
        x = torch.randn((L,), **kwargs)
        return {"x": x}

    def forward(self, x: Tensor) -> Tensor:
        return relu2_compiled(x)


@dataclass
class Triton(Baseline):

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, relu2_triton.apply(x))


if __name__ == "__main__":
    CLI.entrypoint(
        "ReLU2",
        [
            Baseline("baseline"),
            Compiled("torch-compile"),
            Triton("triton"),
        ],
        dims={
            "L": (tuple(int(2**i) for i in range(7, 21)), "values"),
        },
    )
