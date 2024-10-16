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
    norm: bool = False,
    pre_norm_w: Tensor | None = None,
    pre_norm_b: Tensor | None = None,
    training: bool = False,
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
        training: Training or inference mode

    Shapes:
        q, k, v: :math:`(B, L, D)`, or :math:`(B, L, 3*D)` when packed QKV tensor
        w_q, w_k, w_v, w_out: :math:`(D, D)`, or :math:`(3*D, D)` when packed QKV tensor
        b_q, b_k, b_v, b_out: :math:`(D,)`, or :math:`(3*D,)` when packed QKV tensor
        norm_w, norm_b: :math:`(head_dim,)`

    Returns:
        Output tensor of shape :math:`(B, L, D)`
    """
    head_dim = q.shape[-1] // num_heads
    if norm:
        q = F.layer_norm(q, q.shape[-1:], weight=pre_norm_w, bias=pre_norm_b, eps=eps)

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
        q = F.layer_norm(q, q.shape[-1:], weight=norm_w, bias=norm_b, eps=eps)
        k = F.layer_norm(k, k.shape[-1:], weight=norm_w, bias=norm_b, eps=eps)

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
    o = F.dropout(o, p=dropout, training=training)

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
        norm: bool = False,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.qk_norm = qk_norm
        self.norm = norm

        # Register optional parameters
        for prefix in ("w", "b"):
            for suffix in ("_in", "_q", "_k", "_v", "_out", "_pre_norm", "_norm"):
                param = f"{prefix}{suffix}"
                self.register_parameter(param, None)

        # Fused pre-norm
        if norm:
            self.w_pre_norm = nn.Parameter(torch.empty(embed_dim))
            self.b_pre_norm = nn.Parameter(torch.empty(embed_dim))

        # Packed QKV projection
        if kdim is None and vdim is None:
            self.w_in = nn.Parameter(torch.empty(embed_dim + 2 * self.kv_dim, embed_dim))
            self.b_in = nn.Parameter(torch.empty(embed_dim + 2 * self.kv_dim)) if bias else None
        # Unpacked QKV projection
        else:
            self.w_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
            self.w_k = nn.Parameter(torch.empty(self.kv_dim, kdim or embed_dim))
            self.w_v = nn.Parameter(torch.empty(self.kv_dim, vdim or embed_dim))
            self.b_q = nn.Parameter(torch.empty(embed_dim)) if bias else None
            self.b_k = nn.Parameter(torch.empty(self.kv_dim)) if bias else None
            self.b_v = nn.Parameter(torch.empty(self.kv_dim)) if bias else None

        # QK normalization
        if qk_norm:
            self.w_norm = nn.Parameter(torch.empty(self.head_dim))
            self.b_norm = nn.Parameter(torch.empty(self.head_dim))

        # Output projection
        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_out = nn.Parameter(torch.empty(embed_dim)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("b_"):
                nn.init.zeros_(param)
            elif "norm" in name:
                nn.init.ones_(param)
            elif name.startswith("w_"):
                nn.init.xavier_uniform_(param)
            else:
                raise ValueError(f"Unsure how to initialize {name}")

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
    def kv_dim(self) -> int:
        return self.head_dim * self.num_kv_heads

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    @property
    def is_gqa(self) -> bool:
        return self._num_kv_heads != self._num_heads

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Self attention
        if q is k and k is v:
            assert self.w_in is not None
            _k = _v = None
        # Cross attention
        else:
            _k = k
            _v = v

        # Handle selection of packed or unpacked QKV
        w_q = self.w_q if self.w_in is None else self.w_in
        w_k = self.w_k if self.w_in is None else None
        w_v = self.w_v if self.w_in is None else None
        b_q = self.b_q if self.b_in is None else self.b_in
        b_k = self.b_k if self.b_in is None else None
        b_v = self.b_v if self.b_in is None else None

        return attention_forward(
            # fmt: off
            q, _k, _v,
            w_q, w_k, w_v,
            b_q, b_k, b_v,
            self.w_out, self.b_out,
            self.num_heads, self.num_kv_heads,
            self.w_norm, self.b_norm,
            self.dropout,
            self.norm,
            pre_norm_w=self.w_pre_norm,
            pre_norm_b=self.b_pre_norm,
            training=self.training,
            # fmt: on
        )
