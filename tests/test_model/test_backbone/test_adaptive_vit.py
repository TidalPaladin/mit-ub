import json
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.testing import assert_close

from mit_ub.model.activations import relu2
from mit_ub.model.backbone import AdaptiveViT, AdaptiveViTConfig, ViT
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
            out = model(x)
        assert out.shape[:2] == (1, 128)

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
            out = model(x)
            out = out.sum()
        out.sum().backward()
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
            out1 = model(x, reshape=False)
            out2 = model(x, mask=mask, reshape=False)
        assert out1.shape != out2.shape

    def test_forward_deterministic(self, config):
        x = torch.randn(1, 3, 224, 224)
        model = AdaptiveViT(config)

        model.train()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1 = model(x)
            out2 = model(x)
        assert not torch.allclose(out1, out2)

        model.eval()
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

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
        config = replace(config, layer_scale_adaptive=1e-9, target_shape=(H, W))
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
        y = model(x)
        y_baseline = baseline(x)
        assert_close(y, y_baseline)

    def test_trivial_token_mask(self, config):
        torch.random.manual_seed(0)
        B, C, H, W = 1, 3, 256, 256
        model = AdaptiveViT(config)
        model.eval()

        mask_size = model.stem.tokenized_size(cast(Any, (H, W)))
        mask = create_mask(mask_size, batch_size=B, mask_ratio=0.25, scale=1)
        mask = torch.ones_like(mask).bool()
        x = torch.randn(B, C, H, W)
        out1 = model(x, reshape=False)
        out2 = model(x, mask=mask, reshape=False)
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
