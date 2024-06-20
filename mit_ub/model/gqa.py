import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


# TODO: Fix this, GPT messed it up
class GroupedQueryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_groups: int, dropout: float = 0.0):
        super(GroupedQueryAttention, self).__init__()
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.dim = dim
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).view(batch_size, seq_len, 3, self.num_heads, dim // self.num_heads)
        q, k, v = rearrange(qkv, "b l c h d -> c b h l d").unbind(dim=0)

        q = q.view(batch_size, self.num_heads, self.num_groups, seq_len // self.num_groups, dim // self.num_heads)
        k = k.view(batch_size, self.num_heads, self.num_groups, seq_len // self.num_groups, dim // self.num_heads)
        v = v.view(batch_size, self.num_heads, self.num_groups, seq_len // self.num_groups, dim // self.num_heads)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0)
        out = rearrange(out, "b h g l d -> b (g l) (h d)")
        out = self.fc_out(out)
        return out
