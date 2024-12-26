import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
import yaml
from torch.testing import assert_close

from mit_ub.model.activations import relu2
from mit_ub.model.backbone import AdaptiveViT, ConvViT, ConvViTConfig


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
        model = ConvViT(config).to(device)

        with torch.autocast(device_type=device, dtype=dtype):
            out = model(x)
            out = out.sum()
        out.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    def test_forward_deterministic(self, config):
        x = torch.randn(1, 3, 224, 224)
        config = replace(config, dropout=0.1, stochastic_depth=0.1)
        model = ConvViT(config)

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
        y = model(x)
        y_baseline = baseline(x)
        assert_close(y, y_baseline)
