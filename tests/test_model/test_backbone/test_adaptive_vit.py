import json
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.testing import assert_close

from mit_ub.model.activations import relu2
from mit_ub.model.backbone import AdaptiveViT, AdaptiveViTConfig, ViT
from mit_ub.model.layers.mlp import MLP, NormType
from mit_ub.model.layers.pool import PoolType
from mit_ub.tokens import create_mask


@pytest.fixture
def config():
    config = AdaptiveViTConfig(
        in_channels=3,
        dim=128,
        patch_size=(16, 16),
        depth=3,
        nhead=128 // 16,
        dim_feedforward=256,
        target_shape=(64, 64),
    )
    return config


class TestAdaptiveViTConfig:

    def test_convert_activations(self, config):
        config = replace(config, activation="relu2", gate_activation="silu")
        assert config.activation == relu2
        assert config.gate_activation == F.silu

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_from_file(self, tmp_path, ext):
        config = {
            "in_channels": 3,
            "dim": 128,
            "dim_feedforward": 256,
            "patch_size": [16, 16],
            "depth": 3,
            "nhead": 128 // 16,
            "activation": "relu2",
            "gate_activation": "silu",
            "target_shape": [64, 64],
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = AdaptiveViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.dim == 128
        assert config.target_shape == (64, 64)

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = AdaptiveViTConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, AdaptiveViT)


class TestAdaptiveViT:

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, config, device, dtype):
        x = torch.randn(1, 3, 224, 224, device=device)
        model = AdaptiveViT(config).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert out.shape[:2] == (1, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward(self, config, device, dtype, checkpoint):
        x = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = AdaptiveViT(config).to(device)
        with torch.autocast(device_type=device, dtype=dtype):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    def test_token_mask(self, config, device):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        model = AdaptiveViT(config).to(device)

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1, device=device)
        x = torch.randn(B, C, H, W, device=device)

        with torch.autocast(device_type=device, dtype=torch.float16):
            out1, _ = model(x, reshape=False)
            out2, _ = model(x, mask=mask, reshape=False)
        assert out1.shape != out2.shape

    def test_forward_deterministic(self, config):
        x = torch.randn(1, 3, 224, 224)
        model = AdaptiveViT(config)

        model.train()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1, cls1 = model(x)
            out2, cls2 = model(x)
        assert not torch.allclose(out1, out2)
        assert not torch.allclose(cls1, cls2)

        model.eval()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1, cls1 = model(x)
            out2, cls2 = model(x)
        assert torch.allclose(out1, out2)
        assert torch.allclose(cls1, cls2)

    def test_load_from_vit(self, config):
        model = AdaptiveViT(config)
        baseline = ViT(config)
        for name, param in baseline.blocks.named_parameters():
            assert param.shape == model.blocks.get_parameter(name).shape

    def test_initialize_like_vit(self, config):
        C, H, W = 3, 256, 256

        # Set the adaptive model to process fixed tokens at native resolution.
        # Layer scale at 1e-9 should shut off the contribution of the dynamic tokens.
        # NOTE: Must set target_shape to match the input size for this to work.
        config = replace(config, layer_scale=1e-9, target_shape=(H, W))
        model = AdaptiveViT(config)
        baseline = ViT(config)

        # Put the baseline weights into the model
        set_weights = []
        for name, base_param in baseline.named_parameters():
            try:
                model_param = model.get_parameter(name)
                model_param.data = base_param.data
                set_weights.append(name)
            except Exception:
                pass

        model.eval()
        baseline.eval()
        x = torch.randn(1, C, H, W)
        y, cls = model(x)
        y_baseline, cls_baseline = baseline(x)
        assert_close(y, y_baseline)
        assert_close(cls, cls_baseline)

    def test_trivial_token_mask(self, config):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        model = AdaptiveViT(config)
        model.eval()

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        mask = torch.ones_like(mask).bool()
        x = torch.randn(B, C, H, W)
        out1, _ = model(x, reshape=False)
        out2, _ = model(x, mask=mask, reshape=False)
        assert_close(out1, out2)

    def test_share_layers(self, config):
        C, H, W = 3, 256, 256
        config = replace(config, share_layers=True)
        model = AdaptiveViT(config)

        for block, dynamic_block in zip(model.blocks, model.dynamic_blocks):
            for name in ("mlp", "cross_attn"):
                dynamic_child = dynamic_block.get_submodule(name)
                fixed_child = block.get_submodule(name)
                assert dynamic_child is fixed_child

        x = torch.randn(1, C, H, W)
        model(x)

    def test_init_dynamic_from_fixed(self, config):
        C, D, L = 3, 128, 32

        model = AdaptiveViT(config)
        model = AdaptiveViT(config)
        model.init_dynamic_from_fixed()
        model.eval()
        seq = torch.randn(1, L, D)
        for block, dynamic_block in zip(model.blocks, model.dynamic_blocks):
            # Check MLP weights initialized
            exp = block.mlp(seq)
            act = dynamic_block.mlp(seq)
            assert_close(act, exp)

            # Check cross-attention weights initialized from self-attention
            exp = block.self_attn(seq, seq, seq)
            act = dynamic_block.cross_attn(seq, seq.clone(), seq.clone())
            assert_close(act, exp)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("pool_type", [PoolType.ATTENTION, None])
    def test_forward_head(self, config, device, dtype, pool_type):
        x = torch.randn(1, 3, 224, 224, device=device)
        model = AdaptiveViT(config).to(device)
        head = model.create_head(out_dim=10, pool_type=pool_type).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            features, _ = model(x, reshape=False)
            out = head(features)
        exp = (1, 1, 10) if pool_type is not None else (1, 196, 10)
        assert out.shape == exp

        # Ensure this is present for targeting weight decay
        assert isinstance(head.get_submodule("mlp"), MLP)

    @pytest.mark.parametrize("norm_type, exp", [(NormType.LAYER_NORM, nn.LayerNorm), (NormType.RMS_NORM, nn.RMSNorm)])
    def test_norms(self, config, norm_type, exp):
        config = replace(config, norm_type=norm_type)
        model = AdaptiveViT(config)
        assert isinstance(model.embedding_norm, exp), f"Embedding norm is not {exp}"
        for layer in model.modules():
            if hasattr(layer, "norm_type"):
                assert layer.norm_type == norm_type, f"Layer norm type is not {norm_type}"

        head = model.create_head(out_dim=10, pool_type=PoolType.ATTENTION)
        assert isinstance(head.get_submodule("input_norm"), exp), f"Head input norm is not {exp}"
        assert isinstance(head.get_submodule("output_norm"), exp), f"Head output norm is not {exp}"
