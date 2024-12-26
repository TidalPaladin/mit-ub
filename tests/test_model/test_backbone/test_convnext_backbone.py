import json
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
import yaml

from mit_ub.model import ConvNext
from mit_ub.model.activations import relu2
from mit_ub.model.backbone.convnext import ConvNextConfig


@pytest.fixture
def config():
    config = ConvNextConfig(
        in_channels=3,
        depths=(2, 2, 2),
        dims=(32, 48, 64),
        dims_feedforward=(128, 192, 256),
        activation="relu2",
        gate_activation=None,
    )
    return config


class TestConvNextConfig:

    def test_convert_activations(self, config):
        config = replace(config, activation="relu2", gate_activation="silu")
        assert config.activation == relu2
        assert config.gate_activation == F.silu

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_from_file(self, tmp_path, ext):
        config = {
            "in_channels": 3,
            "depths": [2, 2, 2],
            "dims": [32, 48, 64],
            "dims_feedforward": [128, 192, 256],
            "activation": "relu2",
            "gate_activation": "silu",
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = ConvNextConfig.from_file(path)
        assert config.in_channels == 3
        assert config.depths == (2, 2, 2)
        assert config.dims == (32, 48, 64)
        assert config.dims_feedforward == (128, 192, 256)

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = ConvNextConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, ConvNext)


class TestConvNext:

    @pytest.mark.parametrize(
        "config, exp",
        [
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    dims=(32, 48, 64),
                    dims_feedforward=(128, 192, 256),
                ),
                (1, 64, 16, 16),
            ),
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    dims=(32, 48, 64),
                    dims_feedforward=(128, 192, 256),
                    up_depths=(2, 2, 2),
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, device, dtype, config, exp):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W)
        model = ConvNext(config)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = model(x)
        assert out.shape == exp

    def test_forward_deterministic(self):
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W)
        config = ConvNextConfig(
            in_channels=C,
            depths=(2, 2, 2),
            dims=(32, 48, 64),
            dims_feedforward=(128, 192, 256),
            dropout=0.1,
            stochastic_depth=0.1,
        )
        model = ConvNext(config)

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

    @pytest.mark.parametrize(
        "config, exp",
        [
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    dims=(32, 48, 64),
                    dims_feedforward=(128, 192, 256),
                ),
                (1, 64, 16, 16),
            ),
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    dims=(32, 48, 64),
                    dims_feedforward=(128, 192, 256),
                    up_depths=(2, 2, 2),
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_backward(self, device, dtype, config, exp, checkpoint):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = ConvNext(config)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            out = model(x)
        out.sum().backward()
        for param in model.parameters():
            assert param.grad is not None

    def test_fewer_up_levels(self):
        config = ConvNextConfig(
            in_channels=1,
            depths=(2, 2, 2),
            up_depths=(2, 2),
            dims=(32, 48, 64),
            dims_feedforward=(128, 192, 256),
        )

        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W)
        model = ConvNext(config)
        out = model(x)
        assert out.shape == (1, 48, 32, 32)
