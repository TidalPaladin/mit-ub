from itertools import product
from typing import Final

from gpvit import GPViT
from registry import Registry
from gpvit.train import BACKBONES



#BACKBONES = Registry("backbones")

_IMAGE_SIZES: Final = ((512, 384),)
_PATCH_SIZES: Final = ((16, 16),)
_DEPTH_WIDTH: Final = {"base": (12, 768)}
_MIXER: Final = (
    "transformer",
    "mlpmixer",
)

for img_size, patch_size, mixer, (size_name, (depth, width)) in product(
    _IMAGE_SIZES, _PATCH_SIZES, _MIXER, _DEPTH_WIDTH.items()
):
    name = f"gpvit_{mixer}_{size_name}_i{img_size[0]}x{img_size[1]}_p{patch_size[0]}"
    BACKBONES(
        GPViT,
        name=name,
        in_channels=1,
        dim=width,
        num_group_tokens=48,
        img_size=img_size,
        patch_size=patch_size,
        window_size=(4, 4),
        depth=depth,
        dropout=0.1,
        group_interval=1,
        conv=False,
        group_token_mixer=mixer,
        mixer_repeats=1,
        group_tokens_as_kv=False,
    )

__all__ = ["BACKBONES"]
