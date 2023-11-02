from functools import partial
from typing import Any, Dict, Optional, Tuple, cast

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from ssl_tasks.jepa.task import JEPA as JEPABase
from ssl_tasks.tokens import TokenMask
from torch import Tensor

from ..model import BACKBONES
from .helpers import mask_fn


class JEPA(JEPABase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        torch.set_float32_matmul_precision("medium")

    def prepare_backbone(self, name: str) -> nn.Module:
        return BACKBONES.get(name).instantiate_with_metadata().fn

    @property
    def img_size(self) -> Tuple[int, int]:
        return cast(Any, self.backbone).img_size

    def create_head(self) -> nn.Module:
        dim = cast(Any, self.backbone).dim
        return nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            Rearrange("b c h w -> b (h w) c"),
        )

    def create_token_mask(self, batch_size: int, device: torch.device = torch.device("cpu")) -> TokenMask:
        return TokenMask.create(self.img_size, cast(Any, self.backbone).patch_size, batch_size, device=device)

    def forward(self, x: Tensor, mask: Optional[TokenMask] = None) -> Dict[str, Tensor]:
        mask_hook = (
            self.backbone.register_mask_hook(partial(mask_fn, mask=mask), prepend=True) if mask is not None else None
        )
        x, _ = self.backbone(x)
        x = self.jepa_head(x)

        if mask_hook is not None:
            mask_hook.remove()
        return {"jepa": x}
