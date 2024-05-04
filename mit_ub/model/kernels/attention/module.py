import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from .kernel import attention


class MultiheadSelfAttention(nn.MultiheadAttention):

    def forward(
        self,
        x: Tensor,
        pos: Tensor | None = None,
        slopes: Tensor | None = None,
        softmax_scale: float | None = None,
        full_precision: bool = True,
        mask_threshold: float | None = None,
    ) -> Tensor:
        x = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        x = rearrange(x, "b l (d h c) -> b h l d c", c=3, h=self.num_heads)
        q, k, v = [t.squeeze(-1) for t in x.chunk(3, dim=-1)]
        result = attention(q, k, v, pos, pos, slopes, softmax_scale, full_precision, mask_threshold)
        result = rearrange(result, "b h l d -> b l (d h)")
        return self.out_proj(result)
