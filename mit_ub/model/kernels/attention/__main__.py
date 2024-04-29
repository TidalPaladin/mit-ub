from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor
from triton_helpers.benchmark import CLI, KernelExecutor

from .kernel import attention


@dataclass
class Baseline(KernelExecutor):

    def prepare_inputs(self, H: int, QK: int, D: int, D_pos: int, **kwargs) -> Dict[str, Tensor | None]:
        B = 1
        Q = K = QK
        q = torch.randn((B, H, Q, D), **kwargs)
        k = torch.randn((B, H, K, D), **kwargs)
        v = torch.randn((B, H, K, D), **kwargs)
        return {"q": q, "k": k, "v": v}

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v)


@dataclass
class FlashAttention(Baseline):
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
            return F.scaled_dot_product_attention(q, k, v)


@dataclass
class MemEffAttention(Baseline):
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
            return F.scaled_dot_product_attention(q, k, v)


@dataclass
class Triton(Baseline):
    pos_enc: bool = False
    full_precision: bool = True

    def prepare_inputs(self, H: int, QK: int, D: int, D_pos: int, **kwargs) -> Dict[str, Tensor | None]:
        Q = K = QK
        result = super().prepare_inputs(H, QK, D, D_pos, **kwargs)
        if self.pos_enc:
            pos_q = torch.randn((1, H, Q, D_pos), **kwargs)
            pos_k = torch.randn((1, H, K, D_pos), **kwargs)
        else:
            pos_q = pos_k = None
        result.update({"pos_q": pos_q, "pos_k": pos_k})
        return result

    def forward(self, q: Tensor, k: Tensor, v: Tensor, pos_q: Tensor | None, pos_k: Tensor | None) -> Tensor:
        return attention(q, k, v, pos_q, pos_k, full_precision=self.full_precision)


if __name__ == "__main__":
    CLI.entrypoint(
        "EuclideanDistance",
        [
            Baseline("baseline"),
            FlashAttention("flash"),
            MemEffAttention("mem-eff"),
            Triton("triton"),
            Triton("triton-fast", full_precision=False),
            Triton("triton-pos", pos_enc=True),
            Triton("triton-pos-fast", pos_enc=True, full_precision=False),
        ],
        dims={
            "QK": ((128, 256, 512, 1024, 2048, 4096, 8192), "values"),
            "D": ((32,), "values"),
            "D_pos": ((2,), "values"),
            "H": ((12,), "values"),
        },
    )
