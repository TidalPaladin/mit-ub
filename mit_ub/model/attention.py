import os
from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


TORCH_COMPILE = os.getenv("TORCH_COMPILE", "1").lower() == "1"


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "triton.cudagraph_trees": True,
        "shape_padding": True,
    },
    disable=not TORCH_COMPILE,
)
def multi_head_self_attention_forward(
    # fmt: off
    x: Tensor,
    w_in: Tensor, b_in: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int,
    norm_w: Tensor | None = None, norm_b: Tensor | None = None,
    dropout: float = 0.0,
    eps: float = 1e-8,
    output_dropout: bool = False,
    # fmt: on
) -> Tensor:
    # QKV projection
    x = F.linear(x, w_in, b_in)
    q, k, v = rearrange(x, "b l (h d c) -> b h l d c", h=num_heads, c=3).unbind(dim=-1)
    head_dim = q.size(-1)

    # QK normalization
    if norm_w is not None:
        q = F.layer_norm(q, (head_dim,), norm_w, norm_b, eps)
        k = F.layer_norm(k, (head_dim,), norm_w, norm_b, eps)

    # SDPA
    scale = 1.0 if norm_w is not None else None
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False, scale=scale)

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=dropout > 0.0 and output_dropout)

    return o


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "triton.cudagraph_trees": True,
        "shape_padding": True,
    },
    disable=not TORCH_COMPILE,
)
def grouped_query_self_attention_forward(
    # fmt: off
    x: Tensor,
    w_in: Tensor, b_in: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int, num_kv_heads: int,
    norm_w: Tensor | None = None, norm_b: Tensor | None = None,
    dropout: float = 0.0,
    eps: float = 1e-8,
    output_dropout: bool = False,
    # fmt: on
) -> Tensor:
    # QKV projection
    head_dim = x.size(-1) // num_heads
    x = F.linear(x, w_in, b_in)
    q, k, v = x.split([num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    q = rearrange(cast(Tensor, q), "b l (h d) -> b h l d", h=num_heads)
    k = rearrange(cast(Tensor, k), "b l (h d) -> b h l d", h=num_kv_heads)
    v = rearrange(cast(Tensor, v), "b l (h d) -> b h l d", h=num_kv_heads)

    # QK normalization
    if norm_w is not None:
        q = F.layer_norm(q, (head_dim,), norm_w, norm_b, eps)
        k = F.layer_norm(k, (head_dim,), norm_w, norm_b, eps)

    # KV expansion
    ratio = num_heads // num_kv_heads
    k = k.repeat_interleave(ratio, 1, output_size=num_heads)
    v = v.repeat_interleave(ratio, 1, output_size=num_heads)

    # SDPA
    scale = 1.0 if norm_w is not None else None
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False, scale=scale)

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=dropout > 0.0 and output_dropout)

    return o


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "triton.cudagraph_trees": True,
        "shape_padding": True,
    },
    disable=not TORCH_COMPILE,
)
def multi_head_attention_forward(
    # fmt: off
    q: Tensor, k: Tensor, v: Tensor,
    w_q: Tensor, w_k: Tensor, w_v: Tensor,
    b_q: Tensor | None, b_k: Tensor | None, b_v: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int,
    norm_w: Tensor | None = None, norm_b: Tensor | None = None,
    dropout: float = 0.0,
    eps: float = 1e-8,
    output_dropout: bool = False,
    # fmt: on
) -> Tensor:
    q = rearrange(F.linear(q, w_q, b_q), "b l (h d) -> b h l d", h=num_heads)
    k = rearrange(F.linear(k, w_k, b_k), "b l (h d) -> b h l d", h=num_heads)
    v = rearrange(F.linear(v, w_v, b_v), "b l (h d) -> b h l d", h=num_heads)
    head_dim = q.size(-1)

    # QK normalization
    if norm_w is not None:
        q = F.layer_norm(q, (head_dim,), norm_w, norm_b, eps)
        k = F.layer_norm(k, (head_dim,), norm_w, norm_b, eps)

    # SDPA
    scale = 1.0 if norm_w is not None else None
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False, scale=scale)

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=dropout > 0.0 and output_dropout)

    return o


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "triton.cudagraph_trees": True,
        "shape_padding": True,
    },
    disable=not TORCH_COMPILE,
)
def grouped_query_attention_forward(
    # fmt: off
    q: Tensor, k: Tensor, v: Tensor,
    w_q: Tensor, w_k: Tensor, w_v: Tensor,
    b_q: Tensor | None, b_k: Tensor | None, b_v: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int, num_kv_heads: int,
    norm_w: Tensor | None = None, norm_b: Tensor | None = None,
    dropout: float = 0.0,
    eps: float = 1e-8,
    output_dropout: bool = False,
    # fmt: on
) -> Tensor:
    q = rearrange(F.linear(q, w_q, b_q), "b l (h d) -> b h l d", h=num_heads)
    k = rearrange(F.linear(k, w_k, b_k), "b l (h d) -> b h l d", h=num_kv_heads)
    v = rearrange(F.linear(v, w_v, b_v), "b l (h d) -> b h l d", h=num_kv_heads)
    head_dim = q.size(-1)

    # QK normalization
    if norm_w is not None:
        q = F.layer_norm(q, (head_dim,), norm_w, norm_b, eps)
        k = F.layer_norm(k, (head_dim,), norm_w, norm_b, eps)

    # KV expansion
    ratio = num_heads // num_kv_heads
    k = k.repeat_interleave(ratio, 1, output_size=num_heads)
    v = v.repeat_interleave(ratio, 1, output_size=num_heads)

    # SDPA
    scale = 1.0 if norm_w is not None else None
    o = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=False, scale=scale)

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=dropout > 0.0 and output_dropout)

    return o


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
        qk_norm: bool = False,
        kdim: int | None = None,
        vdim: int | None = None,
        output_dropout: bool = False,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._head_dim = embed_dim // num_heads
        self._num_kv_heads = num_kv_heads
        self._kv_dim = self._head_dim * num_kv_heads
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.output_dropout = output_dropout

        # TODO: It is required to pass kdim and vdim for non-self attention. This should
        # be handled better.
        if kdim is None and vdim is None:
            self.w_in = nn.Parameter(torch.empty(embed_dim + 2 * self._kv_dim, embed_dim))
            self.register_parameter("w_q", None)
            self.register_parameter("w_k", None)
            self.register_parameter("w_v", None)
        else:
            self.w_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.w_k = nn.Parameter(torch.empty(self._kv_dim, kdim or embed_dim))
            self.w_v = nn.Parameter(torch.empty(self._kv_dim, vdim or embed_dim))
            self.register_parameter("w_in", None)

        if qk_norm:
            self.norm_w = nn.Parameter(torch.empty(self.head_dim))
            self.norm_b = nn.Parameter(torch.empty(self.head_dim))
        else:
            self.register_parameter("norm_w", None)
            self.register_parameter("norm_b", None)

        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.w_in is not None:
            nn.init.trunc_normal_(self.w_in, std=0.02)
        else:
            nn.init.trunc_normal_(self.w_q, std=0.02)
            nn.init.trunc_normal_(self.w_k, std=0.02)
            nn.init.trunc_normal_(self.w_v, std=0.02)

        if self.qk_norm:
            nn.init.ones_(self.norm_w)
            nn.init.zeros_(self.norm_b)

        nn.init.trunc_normal_(self.w_out, std=0.02)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def num_kv_heads(self) -> int:
        return self._num_kv_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def is_gqa(self) -> bool:
        return self._num_kv_heads != self._num_heads

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.is_gqa:
            if q is k and q is v:
                return grouped_query_self_attention_forward(
                    # fmt: off
                    q,
                    self.w_in, None,
                    self.w_out, None,
                    self.num_heads, self.num_kv_heads,
                    self.norm_w, self.norm_b,
                    self.dropout if self.training else 0.0,
                    output_dropout=self.output_dropout,
                    # fmt: on
                )
            else:
                return grouped_query_attention_forward(
                    # fmt: off
                    q, k, v,
                    self.w_q, self.w_k, self.w_v,
                    None, None, None,
                    self.w_out, None,
                    self.num_heads, self.num_kv_heads,
                    self.norm_w, self.norm_b,
                    self.dropout if self.training else 0.0,
                    output_dropout=self.output_dropout,
                    # fmt: on
                )

        else:
            if q is k and q is v:
                return multi_head_self_attention_forward(
                    # fmt: off
                    q,
                    self.w_in, None,
                    self.w_out, None,
                    self.num_heads,
                    self.norm_w, self.norm_b,
                    self.dropout if self.training else 0.0,
                    output_dropout=self.output_dropout,
                    # fmt: on
                )
            else:
                return multi_head_attention_forward(
                    # fmt: off
                    q, k, v,
                    self.w_q, self.w_k, self.w_v,
                    None, None, None,
                    self.w_out, None,
                    self.num_heads,
                    self.norm_w, self.norm_b,
                    self.dropout if self.training else 0.0,
                    output_dropout=self.output_dropout,
                    # fmt: on
                )
