import json
from dataclasses import replace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.testing import assert_close

from mit_ub.model.activations import relu2
from mit_ub.model.backbone import AdaptiveViT, ConvViT, ConvViTConfig
from mit_ub.model.layers.mlp import MLP, NormType
from mit_ub.model.layers.pool import PoolType


@pytest.fixture
def config():
    config = ConvViTConfig(
        in_channels=3,
        dim=128,
        patch_size=(16, 16),
        depth=3,
        nhead=128 // 16,
        dim_feedforward=256,
        target_shape=(64, 64),
    )
    return config


class TestConvViTConfig:

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

        config = ConvViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.dim == 128

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = ConvViTConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, ConvViT)


class TestConvViT:

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
        model = ConvViT(config).to(device)
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
        model = ConvViT(config).to(device)

        with torch.autocast(device_type=device, dtype=dtype):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_forward_deterministic(self, config):
        x = torch.randn(1, 3, 224, 224)
        config = replace(config, dropout=0.1, stochastic_depth=0.1)
        model = ConvViT(config)

        model.train()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1, cls_token1 = model(x)
            out2, cls_token2 = model(x)
        assert not torch.allclose(out1, out2)
        assert not torch.allclose(cls_token1, cls_token2)

        model.eval()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1, cls_token1 = model(x)
            out2, cls_token2 = model(x)
        assert torch.allclose(out1, out2)
        assert torch.allclose(cls_token1, cls_token2)

    def test_load_from_adaptive_vit(self, config):
        baseline = AdaptiveViT(config)
        model = ConvViT(config)
        for name, param in baseline.blocks.named_parameters():
            assert param.shape == model.blocks.get_parameter(name).shape

    def test_initialize_like_adaptive_vit(self, config):
        C, H, W = 3, 256, 256

        # Set the adaptive model to process fixed tokens at native resolution.
        # Layer scale at 1e-9 should shut off the contribution of the dynamic tokens.
        config = replace(config, layer_scale=1e-9)
        baseline = AdaptiveViT(config)
        model = ConvViT(config)

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
        y, cls_token = model(x)
        y_baseline, cls_token_baseline = baseline(x)
        assert_close(y, y_baseline)
        assert_close(cls_token, cls_token_baseline)

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
        model = ConvViT(config).to(device)
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
        model = ConvViT(config)
        assert isinstance(model.embedding_norm, exp), f"Embedding norm is not {exp}"
        for layer in model.modules():
            if hasattr(layer, "norm_type"):
                assert layer.norm_type == norm_type, f"Layer norm type is not {norm_type}"

        head = model.create_head(out_dim=10, pool_type=PoolType.ATTENTION)
        assert isinstance(head.get_submodule("input_norm"), exp), f"Head input norm is not {exp}"
        assert isinstance(head.get_submodule("output_norm"), exp), f"Head output norm is not {exp}"
