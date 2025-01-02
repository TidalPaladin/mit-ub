from enum import StrEnum

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ..helpers import compile_is_disabled, max_autotune


class PoolType(StrEnum):
    ATTENTION = "attention"
    SIMPLE_ATTENTION = "simple-attention"
    ATTENTION_QK_NORM = "attention-qk-norm"
    AVG = "avg"
    MAX = "max"


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": max_autotune(),
        "triton.cudagraph_trees": max_autotune(),
        "shape_padding": True,
    },
    disable=compile_is_disabled(),
)
def multi_head_attention_pool(
    # fmt: off
    q: Tensor, kv: Tensor,
    w_k: Tensor, w_v: Tensor,
    b_k: Tensor | None, b_v: Tensor | None,
    w_norm: Tensor | None, b_norm: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int,
    dropout: float = 0.0,
    training: bool = False,
    # fmt: on
) -> Tensor:
    k = rearrange(F.linear(kv, w_k, b_k), "b l (h d) -> b h l d", h=num_heads)
    v = rearrange(F.linear(kv, w_v, b_v), "b l (h d) -> b h l d", h=num_heads)

    if w_norm is not None and b_norm is not None:
        q = F.layer_norm(q, (q.shape[-1],), w_norm, b_norm)
        k = F.layer_norm(k, (k.shape[-1],), w_norm, b_norm)

    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout if training else 0.0)

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    return F.linear(o, w_out, b_out)


class MultiHeadAttentionPool(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_queries: int = 1,
        dropout: float = 0.0,
        qk_norm: bool = False,
    ):
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

        if qk_norm:
            head_dim = dim // num_heads
            self.w_norm = nn.Parameter(torch.empty(head_dim))
            self.b_norm = nn.Parameter(torch.empty(head_dim))
        else:
            self.register_parameter("w_norm", None)
            self.register_parameter("b_norm", None)

        self.reset_parameters()

    def reset_parameters(self):
        for name, param in self.named_parameters():
            if name == "query":
                nn.init.trunc_normal_(param, std=0.02)
            elif name == "w_norm":
                nn.init.ones_(param)
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
            self.w_norm, self.b_norm,
            self.w_out, self.b_out,
            self.num_heads,
            dropout=self.dropout,
            training=self.training,
            # fmt: on
        )


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": max_autotune(),
        "triton.cudagraph_trees": max_autotune(),
        "shape_padding": True,
    },
    disable=compile_is_disabled(),
)
def simple_attention_pool(x: Tensor, w: Tensor, num_heads: int) -> Tensor:
    weights = F.linear(x, w).softmax(dim=1)
    weights = rearrange(weights, "b l h -> b l h ()")
    x = rearrange(x, "b l (h d) -> b l h d", h=num_heads)
    y = (weights * x).sum(dim=1)
    return rearrange(y, "b h d -> b (h d)")


class SimpleAttentionPool(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.dropout = dropout
        self.w = nn.Parameter(torch.empty(num_heads, dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w)

    def forward(self, x: Tensor) -> Tensor:
        return simple_attention_pool(x, self.w, self.num_heads)


class AveragePool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1)


class MaxPool(nn.Module):

    def forward(self, x: Tensor) -> Tensor:
        return x.amax(dim=1)


def get_global_pooling_layer(pool_type: PoolType, *args, **kwargs) -> nn.Module:
    match pool_type:
        case PoolType.ATTENTION:
            return MultiHeadAttentionPool(*args, **kwargs)
        case PoolType.SIMPLE_ATTENTION:
            return SimpleAttentionPool(*args, **kwargs)
        case PoolType.ATTENTION_QK_NORM:
            return MultiHeadAttentionPool(*args, **kwargs, qk_norm=True)
        case PoolType.AVG:
            return AveragePool()
        case PoolType.MAX:
            return MaxPool()
        case _:
            raise ValueError(f"Unknown pooling type: {pool_type}")
