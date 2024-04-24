from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor
from triton_helpers.benchmark import CLI, KernelExecutor

from .kernel import euclidean_distance


@dataclass
class Baseline(KernelExecutor):
    weighted: bool = False

    def prepare_inputs(self, M: int, K: int, **kwargs) -> Dict[str, Tensor | None]:
        N = M
        a = torch.randn((M, K), **kwargs)
        b = torch.randn((N, K), **kwargs)
        w = torch.randn((K,), **kwargs).abs() if self.weighted else None
        return {"a": a, "b": b, "w": w}

    def forward(self, a: Tensor, b: Tensor, w: Tensor | None, c: Tensor | None = None) -> Tensor:
        assert a.shape[-1] == b.shape[-1]
        M, K = a.shape
        N, K = b.shape
        if w is not None:
            assert w.shape == (K,)
            result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).mul(w.view(1, 1, K)).sum(-1).sqrt_()
        else:
            result = (a.view(-1, 1, K) - b.view(1, -1, K)).pow(2).sum(-1).sqrt_()
        if c is not None:
            result.add_(c)
        return result


@dataclass
class Triton(Baseline):
    weighted: bool = False
    matmul: bool = True

    def forward(self, a: Tensor, b: Tensor, w: Tensor | None) -> Tensor:
        return euclidean_distance(a, b, w, matmul=self.matmul)


if __name__ == "__main__":
    CLI.entrypoint(
        "EuclideanDistance",
        [Baseline("baseline"), Triton("triton-matmul"), Triton("triton-vector", matmul=False)],
        dims={
            "M": ((128, 256, 512, 1024, 2048, 4096, 8192, 16384), "values"),
            "K": ((2, 3), "values"),
        },
    )
