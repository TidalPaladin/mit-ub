from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor
from triton_helpers.benchmark import CLI, KernelExecutor

from .kernel import euclidean_distance


@dataclass
class Baseline(KernelExecutor):

    def prepare_inputs(self, M: int, K: int, **kwargs) -> Dict[str, Tensor | None]:
        N = M
        a = torch.randn((M, K), **kwargs)
        b = torch.randn((N, K), **kwargs)
        return {"a": a, "b": b}

    def forward(self, a: Tensor, b: Tensor, c: Tensor | None = None) -> Tensor:
        assert a.shape[-1] == b.shape[-1]
        K = a.shape[-1]
        result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).sum(-1).sqrt_()
        if c is not None:
            result.add_(c)
        return result


@dataclass
class Triton(Baseline):
    method: str = "matmul-nodiag"

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return euclidean_distance(a, b, method=self.method)


if __name__ == "__main__":
    CLI.entrypoint(
        "EuclideanDistance",
        [
            Baseline("baseline"),
            Triton("triton-matmul", method="matmul"),
            Triton("triton-matmul-nodiag"),
            # Not working since Triton upgrade
            # Triton("triton-pointwise", method="pointwise"),
        ],
        dims={
            "M": ((128, 256, 512, 1024, 2048, 4096, 8192, 16384), "values"),
            "K": ((2, 3, 4), "values"),
        },
    )
