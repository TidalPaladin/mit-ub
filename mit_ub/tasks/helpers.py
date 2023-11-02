from typing import Any

import torch.nn as nn
from ssl_tasks.tokens import TokenMask
from torch import Tensor


def mask_fn(module: nn.Module, args: Any, output: Tensor, mask: TokenMask) -> Tensor:
    output = mask.apply_to_tokens(output, fill_value=0)
    return output
