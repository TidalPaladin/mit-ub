from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from ssl_tasks.mae.task import MAE as MAEBase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from ..model import BACKBONES
from .helpers import mask_fn


class MAE(MAEBase):
    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        dim = cast(Any, self.backbone).dim
        out_dim = cast(Any, self.backbone).in_channels
        patch_h, patch_w = cast(Any, self.backbone).patch_size
        outputs_per_token = out_dim * patch_h * patch_w
        return nn.Conv2d(dim, outputs_per_token, kernel_size=1)

    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        x, _ = self.backbone(x)
        x = self.mae_head(x)
        x = cast(Any, self.backbone).unpatch(x)

        if mask_hook is not None:
            mask_hook.remove()
        return {"mae": x}
