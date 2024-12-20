import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..helpers import compile_is_disabled


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "shape_padding": True,
    },
    disable=compile_is_disabled(),
)
def multi_head_attention_pool(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_k: Tensor, w_v: Tensor,
    b_k: Tensor | None, b_v: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int,
    dropout: float = 0.0,
    training: bool = False,
    # fmt: on
) -> Tensor:
    k = rearrange(F.linear(kv, w_k, b_k), "b l (h d) -> b h l d", h=num_heads)
    v = rearrange(F.linear(kv, w_v, b_v), "b l (h d) -> b h l d", h=num_heads)

    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if training else 0.0)

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    return F.linear(o, w_out, b_out)


class MultiHeadAttentionPool(nn.Module):

    def __init__(self, dim: int, num_heads: int, num_queries: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.query = nn.Parameter(torch.empty(1, num_heads, num_queries, dim // num_heads))
        self.w_k = nn.Parameter(torch.empty(dim, dim))
        self.w_v = nn.Parameter(torch.empty(dim, dim))
        self.b_k = nn.Parameter(torch.empty(dim))
        self.b_v = nn.Parameter(torch.empty(dim))
        self.w_out = nn.Parameter(torch.empty(dim, dim))
        self.b_out = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name == "query":
                nn.init.trunc_normal_(param, std=0.02)
            elif name.startswith("w_"):
                nn.init.xavier_uniform_(param)
            elif name.startswith("b_"):
                nn.init.zeros_(param)
            else:
                raise ValueError(f"Unsure how to initialize {name}")

    def forward(self, x: Tensor) -> Tensor:
        return multi_head_attention_pool(
            # fmt: off
            self.query, x,
            self.w_k, self.w_v,
            self.b_k, self.b_v,
            self.w_out, self.b_out,
            self.num_heads,
            dropout=self.dropout,
            training=self.training,
            # fmt: on
        )
