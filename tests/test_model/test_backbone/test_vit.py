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
from mit_ub.model.backbone import ViT, ViTConfig
from mit_ub.model.layers.mlp import MLP, NormType
from mit_ub.model.layers.pool import PoolType
from mit_ub.tokens import create_mask


@pytest.fixture
def config():
    config = ViTConfig(
        in_channels=3,
        dim=128,
        dim_feedforward=256,
        patch_size=(16, 16),
        depth=3,
        nhead=128 // 16,
    )
    return config


class TestViTConfig:

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
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = ViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.dim == 128

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = ViTConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, ViT)


class TestViT:

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
        model = ViT(config).to(device)
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
        model = ViT(config).to(device)
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
        B, C, H, W = 1, 3, 224, 224
        D, patch_size, depth = 128, (16, 16), 3

        model = ViT(config).to(device)

        x = torch.randn(B, C, H, W, device=device)
        mask = model.create_mask(x, unmasked_ratio=0.25, scale=1)
        with torch.autocast(device_type=device, dtype=torch.float16):
            out1, _ = model(x, reshape=False)
            out2, _ = model(x, mask=mask, reshape=False)
        assert out1.shape != out2.shape

    def test_forward_deterministic(self, config):
        x = torch.randn(1, 3, 224, 224)
        config = replace(config, stochastic_depth=0.1, dropout=0.1)
        model = ViT(config)

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

    def test_trivial_token_mask(self, config):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 224, 224
        model = ViT(config)
        model.eval()

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        mask = torch.ones_like(mask).bool()
        x = torch.randn(B, C, H, W)
        out1, _ = model(x, reshape=False)
        out2, _ = model(x, mask=mask, reshape=False)
        assert_close(out1, out2)

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
        model = ViT(config).to(device)
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
        model = ViT(config)
        assert isinstance(model.embedding_norm, exp), f"Embedding norm is not {exp}"
        for layer in model.modules():
            if hasattr(layer, "norm_type"):
                assert layer.norm_type == norm_type, f"Layer norm type is not {norm_type}"

        head = model.create_head(out_dim=10, pool_type=PoolType.ATTENTION)
        assert isinstance(head.get_submodule("input_norm"), exp), f"Head input norm is not {exp}"
        assert isinstance(head.get_submodule("output_norm"), exp), f"Head output norm is not {exp}"
