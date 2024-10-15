from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .compile import compile_is_disabled


@torch.compile(
    fullgraph=True,
    options={
        "max_autotune": True,
        "triton.cudagraph_trees": True,
        "shape_padding": True,
        "epilogue_fusion": True,
    },
    disable=compile_is_disabled(),
)
def attention_forward(
    # fmt: off
    q: Tensor, k: Tensor | None, v: Tensor | None,
    w_q: Tensor, w_k: Tensor | None, w_v: Tensor | None,
    b_q: Tensor | None, b_k: Tensor | None, b_v: Tensor | None,
    w_out: Tensor, b_out: Tensor | None,
    num_heads: int, num_kv_heads: int,
    norm_w: Tensor | None = None, norm_b: Tensor | None = None,
    dropout: float = 0.0,
    eps: float = 1e-5,
    output_dropout: bool = True,
    norm: bool = False,
    pre_norm_w: Tensor | None = None,
    pre_norm_b: Tensor | None = None,
    # fmt: on
) -> Tensor:
    """
    Perform scaled dot product attention forward pass.

    Args:
        q: Query tensor, or packed QKV tensor
        k: Key tensor, or ``None`` when ``q`` is packed QKV tensor
        v: Value tensor, or ``None`` when ``q`` is packed QKV tensor
        w_q: Query projection weight, or a packed QKV weight
        w_k: Key projection weight, or ``None`` when ``q`` is packed QKV tensor
        w_v: Value projection weight, or ``None`` when ``q`` is packed QKV tensor
        b_q: Query projection bias, or ``None`` when ``q`` is packed QKV tensor
        b_k: Key projection bias, or ``None`` when ``q`` is packed QKV tensor
        b_v: Value projection bias, or ``None`` when ``q`` is packed QKV tensor
        w_out: Output projection weight
        b_out: Output projection bias
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads. Set to ``num_heads`` for multi-head attention.
        norm_w: Normalization weight of shape :math:`(head_dim,)` or None
        norm_b: Normalization bias of shape :math:`(head_dim,)` or None
        dropout: Dropout probability
        eps: Epsilon value for layer normalization
        output_dropout: Whether to apply dropout to the output

    Shapes:
        q, k, v: :math:`(B, L, D)`, or :math:`(B, L, 3*D)` when packed QKV tensor
        w_q, w_k, w_v, w_out: :math:`(D, D)`, or :math:`(3*D, D)` when packed QKV tensor
        b_q, b_k, b_v, b_out: :math:`(D,)`, or :math:`(3*D,)` when packed QKV tensor
        norm_w, norm_b: :math:`(head_dim,)`

    Returns:
        Output tensor of shape :math:`(B, L, D)`
    """
    if norm:
        q = F.layer_norm(q, (q.size(-1),), weight=pre_norm_w, bias=pre_norm_b, eps=eps)

    head_dim = q.size(-1) // num_heads
    # Packed QKV projection
    if k is None and v is None and w_k is None and w_v is None and b_k is None and b_v is None:
        x = F.linear(q, w_q, b_q)
        q, k, v = x.split([num_heads * head_dim, num_kv_heads * head_dim, num_kv_heads * head_dim], dim=-1)
    # Non-packed QKV projection
    elif k is not None and v is not None and w_k is not None and w_v is not None:
        q = F.linear(q, w_q, b_q)
        k = F.linear(k, w_k, b_k)
        v = F.linear(v, w_v, b_v)
    else:
        raise ValueError(
            "Invalid input. Either provide q and w_q as QKV packed inputs, or provide q, k, v, w_q, w_k, w_v separately."
        )

    q = rearrange(cast(Tensor, q), "b l (h d) -> b h l d", h=num_heads)
    k = rearrange(cast(Tensor, k), "b l (h d) -> b h l d", h=num_kv_heads)
    v = rearrange(cast(Tensor, v), "b l (h d) -> b h l d", h=num_kv_heads)

    # QK normalization
    if norm_w is not None:
        q = F.layer_norm(q, (head_dim,), weight=norm_w, bias=norm_b, eps=eps)
        k = F.layer_norm(k, (head_dim,), weight=norm_w, bias=norm_b, eps=eps)

    # KV expansion
    if num_kv_heads != num_heads:
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
        bias: bool = True,
        output_dropout: bool = True,
        norm: bool = False,
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
        self.norm = norm

        if norm:
            self.pre_norm_w = nn.Parameter(torch.empty(embed_dim))
            self.pre_norm_b = nn.Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter("pre_norm_w", None)
            self.register_parameter("pre_norm_b", None)

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

        # Bias only applied to output projection
        if bias:
            self.b_out = nn.Parameter(torch.empty(embed_dim))
        else:
            self.register_parameter("b_out", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.w_in is not None:
            nn.init.trunc_normal_(self.w_in, std=0.02)
        else:
            nn.init.trunc_normal_(self.w_q, std=0.02)
            nn.init.trunc_normal_(self.w_k, std=0.02)
            nn.init.trunc_normal_(self.w_v, std=0.02)

        if self.pre_norm_w is not None:
            nn.init.ones_(self.pre_norm_w)
        if self.pre_norm_b is not None:
            nn.init.zeros_(self.pre_norm_b)

        if self.qk_norm:
            nn.init.ones_(self.norm_w)
            nn.init.zeros_(self.norm_b)

        nn.init.trunc_normal_(self.w_out, std=0.02)
        if self.b_out is not None:
            nn.init.zeros_(self.b_out)

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
        if q is k and k is v:
            assert self.w_in is not None
            _k = _v = None
        else:
            _k = k
            _v = v

        w_q = self.w_q if self.w_in is None else self.w_in
        w_k = self.w_k if self.w_in is None else None
        w_v = self.w_v if self.w_in is None else None
        b_q = b_k = b_v = None

        return attention_forward(
            # fmt: off
            q, _k, _v,
            w_q, w_k, w_v,
            b_q, b_k, b_v,
            self.w_out, self.b_out,
            self.num_heads, self.num_kv_heads,
            self.norm_w, self.norm_b,
            self.dropout if self.training else 0.0,
            output_dropout=self.output_dropout,
            norm=self.norm,
            pre_norm_w=self.pre_norm_w,
            pre_norm_b=self.pre_norm_b,
            # fmt: on
        )
