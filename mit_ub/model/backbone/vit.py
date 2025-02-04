from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Dict, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from torch import Tensor

from ...tokens import apply_mask, create_mask
from ..config import ModelConfig, SupportsSafeTensors, convert_sequences
from ..layers import PatchEmbed2d
from ..layers.pool import get_global_pooling_layer
from .convnext import tokens_to_grid


@dataclass(frozen=True)
class ViTConfig(ModelConfig):
    # Inputs
    in_channels: int
    patch_size: Sequence[int]

    # Transformer
    depth: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int
    num_gqa_groups: int | None = None
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1
    parallel_attention_mlp: bool = False
    zero_centered_gamma: bool = False
    normalization: str = "RMSNorm"
    bias: bool = True
    activation: str = "srelu"
    drop_path_rate: float = 0.0

    # Optimizations
    fuse_qkv_params: bool = False
    fuse_wgrad_accumulation: bool = False

    # Other
    checkpoint: bool = False

    def __post_init__(self) -> None:
        convert_sequences(self, tuple)

    @property
    def isotropic_output_dim(self) -> int:
        return self.hidden_size

    def instantiate(self) -> "ViT":
        return ViT(self)

    @property
    def transformer_kwargs(self) -> Dict[str, Any]:
        return dict(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            parallel_attention_mlp=self.parallel_attention_mlp,
            zero_centered_gamma=self.zero_centered_gamma,
            normalization=self.normalization,
            bias=self.bias,
            activation=self.activation,
            drop_path_rate=self.drop_path_rate,
            fuse_qkv_params=self.fuse_qkv_params,
            fuse_wgrad_accumulation=self.fuse_wgrad_accumulation,
            # Constants
            self_attn_mask_type="no_mask",
            attn_input_format="bshd",
        )


class ViT(nn.Module, SupportsSafeTensors):
    stem: PatchEmbed2d
    CONFIG_TYPE: ClassVar[Type[ViTConfig]] = ViTConfig

    def __init__(self, config: ViTConfig):
        super().__init__()
        self._config = config

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(config.hidden_size))

        # Stem tokenizer
        self.stem = PatchEmbed2d(
            config.in_channels,
            config.hidden_size,
            cast(Tuple[int, int], tuple(config.patch_size)),
            normalization=config.normalization,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(config.depth)])

    @property
    def config(self) -> ViTConfig:
        return self._config

    def create_encoder_layer(self, i: int = 0, **kwargs) -> te.TransformerLayer:
        """
        Creates a Transformer encoder layer.

        This method initializes a Transformer encoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        _kwargs = self.config.transformer_kwargs
        _kwargs.update(kwargs)
        _kwargs["layer_number"] = i + 1
        _kwargs["layer_type"] = "encoder"
        layer = te.TransformerLayer(**_kwargs)
        return layer

    def create_decoder_layer(self, i: int = 0, **kwargs) -> te.TransformerLayer:
        """
        Creates a Transformer decoder layer.

        This method initializes a Transformer decoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        _kwargs = self.config.transformer_kwargs
        _kwargs.update(kwargs)
        _kwargs["layer_number"] = i + 1
        _kwargs["layer_type"] = "decoder"
        layer = te.TransformerLayer(**_kwargs)
        return layer

    def create_norm(self, dim: int | None = None, **kwargs) -> nn.Module:
        r"""Creates a normalization layer.

        Args:
            dim: Dimension of the normalization layer. By default this will be the model dimension.
        """
        dim = dim or self.config.hidden_size
        return te.LayerNorm(dim, **kwargs) if self.config.normalization == "LayerNorm" else te.RMSNorm(dim, **kwargs)

    def create_head(
        self,
        out_dim: int,
        pool_type: str | None = None,
        use_mlp: bool = False,
        init_method: Callable | None = None,
        **kwargs,
    ) -> nn.Module:
        r"""Creates a head for the model.

        Args:
            out_dim: Dimension of the output.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.
            use_mlp: Whether to use an MLP after pooling.

        Keyword Args:
            Overrides for the MLP
        """
        if pool_type is None:
            if use_mlp:
                layer = nn.Sequential()
                mlp = te.LayerNormMLP(
                    self.config.hidden_size,
                    self.config.ffn_hidden_size,
                    activation=self.config.activation,
                    **kwargs,
                )
                layer.add_module("mlp", mlp)
                linear = te.Linear(self.config.hidden_size, out_dim, init_method=init_method, **kwargs)
                layer.add_module("out", linear)
                return layer
            else:
                return te.LayerNormLinear(
                    self.config.hidden_size,
                    out_dim,
                    init_method=init_method,
                )

        else:
            layer = nn.Sequential()
            pool = get_global_pooling_layer(pool_type)
            layer.add_module("pool", pool)

            if use_mlp:
                mlp = te.LayerNormMLP(
                    self.config.hidden_size,
                    self.config.ffn_hidden_size,
                    activation=self.config.activation,
                )
                layer.add_module("mlp", mlp)
                linear = te.Linear(self.config.hidden_size, out_dim, init_method=init_method)
                layer.add_module("out", linear)
                return layer
            else:
                linear = te.LayerNormLinear(
                    self.config.hidden_size,
                    out_dim,
                    init_method=init_method,
                )
                layer.add_module("out", linear)
                return layer

    def on_load_checkpoint(self, state_dict: Dict[str, Any], *args, **kwargs) -> None:
        r"""Called after loading a checkpoint to perform any necessary post-processing."""

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
            - input: :math:`(B, C, H, W)` or :math:`(B, C, D, H, W)`
            - output: :math:`(B, L)`

        Returns:
            Token mask.
        """
        batch_size = input.shape[0]
        device = input.device
        original_size = input.shape[2:]
        tokenized_size = self.stem.tokenized_size(cast(Any, original_size))
        mask = create_mask(
            tokenized_size,
            mask_ratio=1 - unmasked_ratio,
            batch_size=batch_size,
            scale=scale,
            device=device,
        )

        return mask

    def forward(
        self,
        x: Tensor,
        reshape: bool = True,
        mask: Tensor | None = None,
        mask_fill_value: float | Tensor | None = None,
    ) -> Tuple[Tensor, Tensor]:
        B, C, *original_size = x.shape
        tokenized_size = self.stem.tokenized_size(cast(Any, tuple(original_size)))

        # Tokenize and apply mask
        x = self.stem(x)
        if mask is not None:
            x = apply_mask(mask, x, fill_value=mask_fill_value)

        # Add CLS token
        x = torch.cat([self.cls_token.view(1, 1, -1).expand(B, -1, -1), x], dim=1)

        # Transformer blocks and output norm
        for block in self.blocks:
            x = block(x, checkpoint_core_attention=self.config.checkpoint)

        # Extract CLS token
        cls_token = x[:, 0, :].contiguous()
        x = x[:, 1:, :].contiguous()

        # Reshape to original grid if requested
        if reshape and mask is not None and mask_fill_value is None:
            raise ValueError(
                "Cannot reshape with mask and no fill value. Either specify `reshape=False` or provide a `mask_fill_value`"
            )
        elif reshape:
            x = tokens_to_grid(x, tokenized_size)

        return x, cls_token

    @classmethod
    def from_args(cls, *args, **kwargs) -> Self:
        config = ViTConfig(*args, **kwargs)
        return cls(config)
