from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Self, Sequence, Tuple, Type, cast

import torch
import torch.nn as nn
import transformer_engine.pytorch as te
from torch import Tensor

from ...tokens import apply_mask, create_mask
from ..config import ModelConfig, SupportsSafeTensors, convert_sequences
from ..helpers import set_checkpointing
from ..layers.mlp import NormType
from ..layers.pool import PoolType, get_global_pooling_layer
from ..layers.pos_enc import DEFAULT_POS_ENC_ACTIVATION
from ..stem import PatchEmbed2d, PatchEmbed3d
from .convnext import tokens_to_grid


@dataclass(frozen=True)
class ViTTEConfig(ModelConfig):
    in_channels: int
    dim: int
    patch_size: Sequence[int]
    depth: int
    nhead: int
    dim_feedforward: int
    dropout: float = 0.1
    stochastic_depth: float = 0.0
    bias: bool = False
    activation: str = "srelu"
    num_kv_heads: int | None = None
    norm_type: NormType = cast(NormType, "layernorm")
    checkpoint: bool = False

    def __post_init__(self) -> None:
        convert_sequences(self, tuple)

    def instantiate(self) -> "ViTTE":
        return ViTTE(self)


class ViTTE(nn.Module, SupportsSafeTensors):
    stem: PatchEmbed2d | PatchEmbed3d
    CONFIG_TYPE: ClassVar[Type[ViTTEConfig]] = ViTTEConfig

    def __init__(self, config: ViTTEConfig):
        super().__init__()
        self._config = config

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(config.dim))

        # Stem tokenizer
        stem_act = DEFAULT_POS_ENC_ACTIVATION
        stem_type = PatchEmbed2d if isinstance(config.patch_size, int) or len(config.patch_size) == 2 else PatchEmbed3d
        self.stem = stem_type(
            config.in_channels, config.dim, cast(Any, config.patch_size), dropout=config.dropout, activation=stem_act
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([self.create_encoder_layer(i) for i in range(config.depth)])
        self.embedding_norm = self.create_norm()
        if config.checkpoint:
            set_checkpointing(self, config.checkpoint)

    @property
    def config(self) -> ViTTEConfig:
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
        _kwargs: Dict[str, Any] = dict(
            hidden_size=self.config.dim,
            ffn_hidden_size=self.config.dim_feedforward,
            num_attention_heads=self.config.nhead,
            attention_dropout=self.config.dropout,
            hidden_dropout=self.config.dropout,
            activation=self.config.activation,
            normalization="LayerNorm" if self.config.norm_type == NormType.LAYER_NORM else "RMSNorm",
            drop_path_rate=self.config.stochastic_depth,
            bias=self.config.bias,
            self_attn_mask_type="no_mask",
            attn_input_format="bshd",
            layer_number=i + 1,
            layer_type="encoder",
        )
        _kwargs.update(kwargs)
        layer = te.TransformerLayer(**_kwargs)
        if self.config.checkpoint:
            set_checkpointing(layer, self.config.checkpoint)
        return layer

    def create_decoder_layer(self, i: int = 0, d_kv: int | None = None, **kwargs) -> te.TransformerLayer:
        """
        Creates a Transformer decoder layer.

        This method initializes a Transformer decoder layer with the specified
        parameters. It supports various configurations such as the number of
        attention heads, feedforward dimension, dropout rate, activation functions,
        and more.

        Args:
            i: Index of the encoder layer. Default is 0.
            d_kv: Dimension of the key and value vectors. By default this will be the same as the model dimension.

        Keyword Args:
            Additional keyword arguments to override default layer parameters.
        """
        d_kv = d_kv or self.config.dim
        _kwargs: Dict[str, Any] = dict(
            hidden_size=self.config.dim,
            ffn_hidden_size=self.config.dim_feedforward,
            num_attention_heads=self.config.nhead,
            attention_dropout=self.config.dropout,
            hidden_dropout=self.config.dropout,
            activation=self.config.activation,
            normalization="LayerNorm" if self.config.norm_type == NormType.LAYER_NORM else "RMSNorm",
            drop_path_rate=self.config.stochastic_depth,
            bias=self.config.bias,
            self_attn_mask_type="no_mask",
            attn_input_format="bshd",
            layer_number=i + 1,
            layer_type="decoder",
        )
        _kwargs.update(kwargs)
        layer = te.TransformerLayer(**_kwargs)
        if self.config.checkpoint:
            set_checkpointing(layer, self.config.checkpoint)
        return layer

    def create_norm(self, dim: int | None = None, **kwargs) -> nn.Module:
        r"""Creates a normalization layer.

        Args:
            dim: Dimension of the normalization layer. By default this will be the model dimension.
        """
        dim = dim or self.config.dim
        return (
            te.LayerNorm(dim, **kwargs) if self.config.norm_type == NormType.LAYER_NORM else te.RMSNorm(dim, **kwargs)
        )

    def create_head(
        self,
        out_dim: int,
        pool_type: PoolType | None = None,
        use_mlp: bool = True,
        **kwargs,
    ) -> nn.Module:
        r"""Creates a head for the model.

        The head has the following structure:
            - Norm layer if ``input_norm`` is ``True`` and either ``pool_type`` is not ``None`` or ``use_mlp`` is ``True``.
            - Pooling layer if ``pool_type`` is not ``None``
            - MLP if ``use_mlp`` is ``True``. MLP is pre-normalized if ``pool_type`` is not ``None``.
            - Norm layer
            - ``nn.Linear`` to project to ``out_dim``

        The following MLP options differ from the backbone defaults:
            * ``bias`` is set to ``True``
            * ``stochastic_depth`` is set to ``0.0``
            * ``layer_scale`` is set to ``None``

        Args:
            out_dim: Dimension of the output.
            input_norm: Whether to apply input normalization.
            pool_type: Type of pooling to apply, or ``None`` to skip pooling.

        Keyword Args:
            Overrides for the MLP
        """
        # No MLP or pooling, use fully fused layers
        if pool_type is None and not use_mlp:
            return te.LayerNormLinear(
                self.config.dim,
                out_dim,
                normalization="LayerNorm" if self.config.norm_type == NormType.LAYER_NORM else "RMSNorm",
            )

        # No pooling, just MLP
        elif pool_type is None and use_mlp:
            return te.LayerNormMLP(
                self.config.dim,
                self.config.dim_feedforward,
                activation=self.config.activation,
                normalization="LayerNorm" if self.config.norm_type == NormType.LAYER_NORM else "RMSNorm",
            )

        elif pool_type is not None and not use_mlp:
            layer = nn.Sequential()
            norm = self.create_norm()
            layer.add_module("norm", norm)
            pool = get_global_pooling_layer(
                cast(PoolType, kwargs.get("pool_type", pool_type)),
                self.config.dim,
                num_heads=kwargs.get("nhead", self.config.nhead),
                dropout=kwargs.get("dropout", self.config.dropout),
            )
            layer.add_module("pool", pool)
            linear = te.Linear(self.config.dim, out_dim)
            layer.add_module("out", linear)
            return layer

        else:
            layer = nn.Sequential()
            norm = self.create_norm()
            layer.add_module("norm", norm)
            pool = get_global_pooling_layer(
                cast(PoolType, kwargs.get("pool_type", pool_type)),
                self.config.dim,
                num_heads=kwargs.get("nhead", self.config.nhead),
                dropout=kwargs.get("dropout", self.config.dropout),
            )
            layer.add_module("pool", pool)
            mlp = te.LayerNormMLP(
                self.config.dim,
                self.config.dim_feedforward,
                activation=self.config.activation,
                normalization="LayerNorm" if self.config.norm_type == NormType.LAYER_NORM else "RMSNorm",
            )
            layer.add_module("mlp", mlp)
            linear = te.Linear(self.config.dim, out_dim)
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
            x = block(x, checkpoint_core_attention=self.config.checkpoint and self.training)
        x = self.embedding_norm(x)

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
        config = ViTTEConfig(*args, **kwargs)
        return cls(config)
