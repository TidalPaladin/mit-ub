from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Self, Tuple, Type, cast

import torch
import torch.nn.functional as F
from torch import Tensor

from ...tokens import create_mask
from ..layers import PatchEmbed2d
from ..layers.patch_embed import calculate_sizes_for_budget, pack, unpack
from .vit import ViT, ViTConfig


@dataclass(frozen=True)
class NaViTConfig(ViTConfig):
    token_drop_gamma: float = 0.4
    target_seq_len: int = 4096

    def instantiate(self) -> "NaViT":
        return NaViT(self)

    @property
    def transformer_kwargs(self) -> Dict[str, Any]:
        kwargs = super().transformer_kwargs
        kwargs.update(
            self_attn_mask_type="padding",
            enc_dec_attn_mask_type="padding",
            attn_input_format="thd",
        )
        return kwargs


class NaViT(ViT):
    stem: PatchEmbed2d
    CONFIG_TYPE: ClassVar[Type[ViTConfig]] = NaViTConfig

    def __init__(self, config: NaViTConfig):
        super().__init__(config)
        self.target_seq_len = config.target_seq_len

    @property
    def config(self) -> NaViTConfig:
        return self._config

    def create_mask(
        self,
        input: Tensor,
        unmasked_ratio: float,
        scale: int,
    ) -> Tensor:
        r"""Creates a token mask for the input.

        Args:
            input: Input tensor from which to infer mask properties.
                Should be a raw input prior to tokenization.
            unmasked_ratio: Proportion of tokens to leave unmasked.
            scale: Scale of the mask.

        Shapes:
            - input: :math:`(C, H, W)` or :math:`(C, D, H, W)`
            - output: :math:`(L,)`

        Returns:
            Token mask.
        """
        device = input.device
        original_size = input.shape[1:]
        tokenized_size = self.stem.tokenized_size(cast(Any, original_size))
        mask = create_mask(
            tokenized_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=1,
            scale=scale,
            device=device,
        )
        return mask.flatten()

    @torch.no_grad()
    def prepare_inputs(
        self,
        x: List[Tensor],
        target_seq_len: int | None = None,
        training: bool | None = None,
    ) -> Tuple[List[Tensor], Tensor]:
        r"""Resizes inputs while trying to achieve the target sequence length after accounting
        for masking. Uses a binary search-like approach where at each iteration the largest
        eligible input is downsampled by half until the target sequence length is met.
        When a 0.5 downsample would overshoot the target, uses binary search to find an
        optimal size that preserves aspect ratio while getting closer to the target.

        Args:
            x: List of input tensors, each of shape (C, H, W) or (C, D, H, W)
            target_seq_len: Target sequence length after tokenization
            training: Whether in training mode (affects token drop rates)

        Returns:
            Tuple of resized inputs and drop rates. Drop rates are zero if not in training mode.
        """
        target_seq_len = target_seq_len if target_seq_len is not None else self.target_seq_len
        training = training if training is not None else self.training

        # Initialize drop rates if in training mode
        if training:
            drop_rates = torch.rand(len(x)).mul_(self.config.token_drop_gamma).add_(0.5)
        else:
            drop_rates = torch.zeros(len(x))

        target_sizes = calculate_sizes_for_budget(
            [xi.shape[1:] for xi in x],
            [self.stem.patch_size] * len(x),
            drop_rates.tolist(),
            target_seq_len,
        )

        resized: List[Tensor] = []
        for xi, new_size in zip(x, target_sizes):
            if new_size != xi.shape[1:]:
                resized.append(
                    F.interpolate(
                        xi[None],
                        size=new_size,
                        mode="bilinear" if len(new_size) == 2 else "trilinear",
                        align_corners=False,
                    ).squeeze(0)
                )
            else:
                resized.append(xi)

        return resized, drop_rates

    def forward(
        self,
        x: List[Tensor],
        mask: List[Tensor] | None = None,
    ) -> Tuple[List[Tensor], Tensor]:
        # Unpacked tokenization, mask, and extra tokens
        unpacked = [self.stem(xi[None]).squeeze(0) for xi in x]
        unpacked = [t[m] for t, m in zip(unpacked, mask)] if mask is not None else unpacked
        unpacked = [torch.cat([self.cls_token.view(1, -1), t], dim=0) for t in unpacked]

        # Pack and run transformer blocks
        packed_seqs, cu_seq_lens, max_seq_len = pack(unpacked)
        for block in self.blocks:
            packed_seqs = block(
                packed_seqs,
                cu_seqlens_q=cu_seq_lens,
                cu_seqlens_kv=cu_seq_lens,
                max_seqlen_q=max_seq_len,
                max_seqlen_kv=max_seq_len,
                checkpoint_core_attention=self.config.checkpoint,
            )

        # Unpack, separate CLS token and other tokens
        unpacked = unpack(packed_seqs, cu_seq_lens)
        cls_tokens = torch.stack([t[0, :] for t in unpacked], dim=0)
        other_tokens = [t[1:, :].contiguous() for t in unpacked]

        return other_tokens, cls_tokens

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = NaViTConfig(*args, **kwargs)
        return cls(config)
