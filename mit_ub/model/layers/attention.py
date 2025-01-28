from typing import cast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..helpers import Checkpointable, compile_backend, compile_is_disabled, init_weight, max_autotune
from .layer_scale import LayerScale
from .mlp import NormType
from .stochastic_depth import apply_stochastic_depth, stochastic_depth_indices, unapply_stochastic_depth


@torch.compile(
    fullgraph=True,
    backend=compile_backend(),
    options={
        "max_autotune": max_autotune(),
        "triton.cudagraph_trees": max_autotune(),
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
    w_qk_norm: Tensor | None = None, b_qk_norm: Tensor | None = None,
    dropout: float = 0.0,
    w_norm: Tensor | None = None,
    b_norm: Tensor | None = None,
    kv_norm: bool = False,
    training: bool = False,
    mask: Tensor | None = None,
    norm_type: NormType = NormType.LAYER_NORM,
    w_layer_scale: Tensor | None = None,
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
        w_qk_norm: Normalization weight of shape :math:`(head_dim,)` or None
        b_qk_norm: Normalization bias of shape :math:`(head_dim,)` or None
        dropout: Dropout probability
        w_norm: Weight for pre-normalization, or ``None`` for no pre-normalization
        b_norm: Bias for pre-normalization
        kv_norm: Whether to normalize the key and value tensors in non-packed QKV mode. This should
            probably be ``True`` when using encoder-decoder attention on an unnormalized KV.
        training: Training or inference mode
        mask: Attention mask
        norm_type: Type of pre-normalization to apply.
        w_layer_scale: Optional layer scale weight

    Shapes:
        q, k, v: :math:`(B, L, D)`, or :math:`(B, L, 3*D)` when packed QKV tensor
        w_q, w_k, w_v, w_out: :math:`(D, D)`, or :math:`(3*D, D)` when packed QKV tensor
        b_q, b_k, b_v, b_out: :math:`(D,)`, or :math:`(3*D,)` when packed QKV tensor
        w_qk_norm, b_qk_norm: :math:`(head_dim,)`
        mask: :math:`(B, Lq, Lk)`
        w_layer_scale: :math:`(D,)`

    Returns:
        Output tensor of shape :math:`(B, L, D)`
    """
    q.shape[0]
    is_packed = k is None and v is None and w_k is None and w_v is None and b_k is None and b_v is None
    head_dim = q.shape[-1] // num_heads

    # Pre-normalization
    eps = torch.finfo(q.dtype).eps
    if w_norm is not None:
        if norm_type == NormType.LAYER_NORM:
            q = F.layer_norm(q, q.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)
            if not is_packed and kv_norm:
                k = F.layer_norm(k, k.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)  # type: ignore[arg-type]
                v = F.layer_norm(v, v.shape[-1:], weight=w_norm, bias=b_norm, eps=eps)  # type: ignore[arg-type]
        elif norm_type == NormType.RMS_NORM:
            q = F.rms_norm(q, q.shape[-1:], weight=w_norm, eps=eps)
            if not is_packed and kv_norm:
                k = F.rms_norm(k, k.shape[-1:], weight=w_norm, eps=eps)  # type: ignore[arg-type]
                v = F.rms_norm(v, v.shape[-1:], weight=w_norm, eps=eps)  # type: ignore[arg-type]
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")

    # Packed QKV projection
    if is_packed:
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
    if w_qk_norm is not None:
        q = F.layer_norm(q, q.shape[-1:], weight=w_qk_norm, bias=b_qk_norm, eps=eps)
        k = F.layer_norm(k, k.shape[-1:], weight=w_qk_norm, bias=b_qk_norm, eps=eps)

    # KV expansion
    if num_kv_heads != num_heads:
        ratio = num_heads // num_kv_heads
        k = k.repeat_interleave(ratio, 1, output_size=num_heads)
        v = v.repeat_interleave(ratio, 1, output_size=num_heads)

    # SDPA
    scale = 1.0 if w_qk_norm is not None else None
    o = F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout if training else 0.0, is_causal=False, scale=scale
    )

    # output projection
    o = rearrange(o, "b h l d -> b l (h d)")
    o = F.linear(o, w_out, b_out)
    o = F.dropout(o, p=dropout, training=training, inplace=True)

    # Layer scale
    if w_layer_scale is not None:
        o = o * w_layer_scale

    return o


class MultiHeadAttention(nn.Module, Checkpointable):

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
        kv_norm: bool = False,
        norm_type: NormType = NormType.LAYER_NORM,
        layer_scale: float | None = None,
        stochastic_depth: float = 0.0,
    ):
        super().__init__()
        self._embed_dim = embed_dim
        self._num_heads = num_heads
        self._num_kv_heads = num_kv_heads
        self.dropout = dropout
        self.kv_norm = kv_norm
        self.norm_type = norm_type
        self.checkpoint = False
        self.stochastic_depth = stochastic_depth

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        if self.head_dim % 16 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 16")
        if self.kv_dim % 16 != 0:
            raise ValueError(f"kv_dim ({self.kv_dim}) must be divisible by 16")

        # Register optional parameters
        for prefix in ("w", "b"):
            for suffix in ("_q", "_k", "_v", "_out", "_qk_norm", "_norm"):
                param = f"{prefix}{suffix}"
                self.register_parameter(param, None)

        # Fused pre-norm
        if norm:
            self.w_norm = nn.Parameter(torch.empty(embed_dim))
            if norm_type == NormType.LAYER_NORM:
                self.b_norm = nn.Parameter(torch.empty(embed_dim))
            else:
                self.register_parameter("b_norm", None)

        self.w_q = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.w_k = nn.Parameter(torch.empty(self.kv_dim, kdim or embed_dim))
        self.w_v = nn.Parameter(torch.empty(self.kv_dim, vdim or embed_dim))
        self.b_q = nn.Parameter(torch.empty(embed_dim)) if bias else None
        self.b_k = nn.Parameter(torch.empty(self.kv_dim)) if bias else None
        self.b_v = nn.Parameter(torch.empty(self.kv_dim)) if bias else None

        # QK normalization
        if qk_norm:
            self.w_qk_norm = nn.Parameter(torch.empty(self.head_dim))
            self.b_qk_norm = nn.Parameter(torch.empty(self.head_dim))

        # Output projection
        self.w_out = nn.Parameter(torch.empty(embed_dim, embed_dim))
        self.b_out = nn.Parameter(torch.empty(embed_dim)) if bias else None

        # Layer scale
        if layer_scale is not None:
            self.layer_scale = LayerScale(embed_dim, gamma=layer_scale)
        else:
            self.register_module("layer_scale", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for name, param in self.named_parameters():
            if name.startswith("b_"):
                nn.init.zeros_(param)
            elif "norm" in name:
                nn.init.ones_(param)
            elif name.startswith("w_"):
                init_weight(param)
            elif name.startswith("layer_scale"):
                pass
            else:
                raise ValueError(f"Unsure how to initialize {name}")
        if self.layer_scale is not None:
            self.layer_scale.reset_parameters()

    def copy_parameters(self, src: "MultiHeadAttention") -> None:
        r"""Copies parameters from a source attention module.

        This method accounts for the difference in parameters between packed and unpacked QKV projections.
        Any layer of `src` that is incompatible with `self` is ignored. The copy includes noramlization
        layers and the output projection layer.
        """
        # Booleans for compatibility
        attn_compatible = (
            self.embed_dim == src.embed_dim
            and self.head_dim == src.head_dim
            and self.kv_dim == src.kv_dim
            and self.num_heads == src.num_heads
            and self.num_kv_heads == src.num_kv_heads
        )
        embed_compatible = self.embed_dim == src.embed_dim

        # Both are unpacked
        if attn_compatible:
            self.w_q.data = src.w_q.data
            self.w_k.data = src.w_k.data
            self.w_v.data = src.w_v.data
            if self.b_q is not None and src.b_q is not None:
                self.b_q.data = src.b_q.data
            if self.b_k is not None and src.b_k is not None:
                self.b_k.data = src.b_k.data
            if self.b_v is not None and src.b_v is not None:
                self.b_v.data = src.b_v.data

        # Norm and projection
        if embed_compatible and self.w_norm is not None and src.w_norm is not None:
            self.w_norm.data = src.w_norm.data
            if self.b_norm is not None and src.b_norm is not None:
                self.b_norm.data = src.b_norm.data
        if attn_compatible and self.w_qk_norm is not None and src.w_qk_norm is not None:
            self.w_qk_norm.data = src.w_qk_norm.data
            self.b_qk_norm.data = src.b_qk_norm.data
        if embed_compatible and self.w_out is not None and src.w_out is not None:
            self.w_out.data = src.w_out.data
            if self.b_out is not None and src.b_out is not None:
                self.b_out.data = src.b_out.data

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

    @property
    def qk_norm(self) -> bool:
        return self.w_qk_norm is not None

    @property
    def norm(self) -> bool:
        return self.w_norm is not None

    @property
    def bias(self) -> bool:
        return self.b_q is not None

    @property
    def w_layer_scale(self) -> Tensor | None:
        return self.layer_scale.gamma if self.layer_scale is not None else None

    def extra_repr(self) -> str:
        return (
            f"dim={self.embed_dim}, "
            f"heads={self.num_heads}, "
            f"kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, "
            f"kv_dim={self.kv_dim}, "
            f"dropout={self.dropout}, "
            f"norm={self.norm}, "
            f"qk_norm={self.qk_norm}, "
            f"bias={self.bias}, "
            f"kv_norm={self.kv_norm}, "
            f"norm_type={self.norm_type}, "
            f"stochastic_depth={self.stochastic_depth}"
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
        B = q.shape[0]

        if self.training and self.stochastic_depth > 0.0:
            indices = stochastic_depth_indices(q, self.stochastic_depth)
        else:
            indices = None

        is_self_attn = q is k and k is v
        is_packable = (self.w_q.shape == self.w_k.shape) and (self.w_q.shape == self.w_v.shape)
        q = apply_stochastic_depth(q, indices, training=self.training) if indices is not None else q

        # Self attention
        if is_self_attn and is_packable:
            _k = _v = None
        elif is_self_attn:
            _k = _v = q
        # Cross attention
        else:
            _k = k
            _v = v
            if indices is not None:
                _k = apply_stochastic_depth(_k, indices, training=self.training)
                _v = apply_stochastic_depth(_v, indices, training=self.training)

        # Pack weights
        if is_self_attn and is_packable:
            w_q = torch.cat([self.w_q, self.w_k, self.w_v], dim=0)
            w_k = w_v = None
            b_k = b_v = None
            if self.b_q is not None and self.b_k is not None and self.b_v is not None:
                b_q = torch.cat([self.b_q, self.b_k, self.b_v], dim=0)
            else:
                b_q = None
        else:
            w_q = self.w_q
            w_k = self.w_k
            w_v = self.w_v
            b_q = self.b_q
            b_k = self.b_k
            b_v = self.b_v

        if self.checkpoint and torch.is_grad_enabled():
            # Workaround for checkpointing with compile + DDP
            fn = torch.compiler.disable(attention_forward) if torch.distributed.is_initialized() else attention_forward
            result = checkpoint(
                # fmt: off
                fn,
                q, _k, _v,
                w_q, w_k, w_v,
                b_q, b_k, b_v,
                self.w_out, self.b_out,
                self.num_heads, self.num_kv_heads,
                self.w_qk_norm, self.b_qk_norm,
                self.dropout,
                self.w_norm,
                self.b_norm,
                self.kv_norm,
                self.training,
                mask,
                self.norm_type,
                self.w_layer_scale,
                use_reentrant=False,
                # fmt: on
            )
        else:
            result = attention_forward(
                # fmt: off
                q, _k, _v,
                w_q, w_k, w_v,
                b_q, b_k, b_v,
                self.w_out, self.b_out,
                self.num_heads, self.num_kv_heads,
                self.w_qk_norm, self.b_qk_norm,
                self.dropout,
                self.w_norm,
                self.b_norm,
                self.kv_norm,
                self.training,
                mask,
                self.norm_type,
                self.w_layer_scale,
                # fmt: on
            )
        assert isinstance(result, Tensor)

        if indices is not None:
            result = unapply_stochastic_depth(result, indices, B, training=self.training)

        return result
