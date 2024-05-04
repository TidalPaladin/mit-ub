import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .kernel import attention


class MultiheadAttention(nn.MultiheadAttention):

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        pos_q: Tensor | None = None,
        pos_k: Tensor | None = None,
        slopes: Tensor | None = None,
        softmax_scale: float | None = None,
        full_precision: bool = True,
        mask_threshold: float | None = None,
    ) -> Tensor:
        B, H, L, D = q.shape[0], self.num_heads, q.shape[1], q.shape[-1] // self.num_heads

        # Do input projections
        if self._qkv_same_embed_dim and q is k and k is v:
            x = F.linear(q, self.in_proj_weight, self.in_proj_bias)
            q, k, v = (t.view(B, L, -1) for t in x.chunk(3, dim=-1))
        else:
            if self._qkv_same_embed_dim:
                wq, wk, wv = self.in_proj_weight.chunk(3, dim=0)
                bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)
                q = F.linear(q, wq, bq)
                k = F.linear(k, wk, bk)
                v = F.linear(v, wv, bv)
            else:
                bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)
                q = F.linear(q, self.q_proj_weight, bq)
                k = F.linear(k, self.k_proj_weight, bk)
                v = F.linear(v, self.v_proj_weight, bv)

        # Apply kv bias
        if self.bias_k is not None:
            k = k + self.bias_k.view(1, 1, 1, -1)
        if self.bias_v is not None:
            v = v + self.bias_v

        # Reshape for attention
        q = rearrange(q, "b l (d h) -> b h l d", b=B, h=H)
        k = rearrange(k, "b l (d h) -> b h l d", b=B, h=H)
        v = rearrange(v, "b l (d h) -> b h l d", b=B, h=H)

        # Attention and restore shape
        result = attention(q, k, v, pos_q, pos_k, slopes, softmax_scale, full_precision, mask_threshold)
        result = rearrange(result, "b h l d -> b l (d h)")

        # Output projection
        return self.out_proj(result)
