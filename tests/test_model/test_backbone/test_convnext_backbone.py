import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file

from mit_ub.model import ConvNext
from mit_ub.model.activations import relu2
from mit_ub.model.backbone.convnext import ConvNextConfig
from mit_ub.model.layers.mlp import MLP, NormType
from mit_ub.model.layers.pool import PoolType


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
        model = ConvNext(config).to(device)
        head = model.create_head(out_dim=10, pool_type=pool_type).to(device)
        with torch.autocast(device_type=device, dtype=dtype, enabled=True):
            features = model(x, reshape=False)
            out = head(features)
        exp = (1, 1, 10) if pool_type is not None else (1, 196, 10)
        assert out.shape == exp

        # Ensure this is present for targeting weight decay
        assert isinstance(head.get_submodule("mlp"), MLP)

    @pytest.mark.parametrize("norm_type, exp", [(NormType.LAYER_NORM, nn.LayerNorm), (NormType.RMS_NORM, nn.RMSNorm)])
    def test_norms(self, config, norm_type, exp):
        config = replace(config, norm_type=norm_type)
        model = ConvNext(config)
        assert isinstance(model.embedding_norm, exp), f"Embedding norm is not {exp}"
        for layer in model.modules():
            if hasattr(layer, "norm_type"):
                assert layer.norm_type == norm_type, f"Layer norm type is not {norm_type}"

        head = model.create_head(out_dim=10, pool_type=PoolType.ATTENTION)
        assert isinstance(head.get_submodule("input_norm"), exp), f"Head input norm is not {exp}"
        assert isinstance(head.get_submodule("output_norm"), exp), f"Head output norm is not {exp}"

    @pytest.fixture
    def checkpoint_model(self, config):
        return ConvNext(config)

    @pytest.fixture
    def safetensors_checkpoint(self, tmp_path, checkpoint_model):
        model = checkpoint_model
        checkpoint_path = tmp_path / "checkpoint.safetensors"
        state_dict = model.state_dict()
        save_file(state_dict, checkpoint_path)
        return checkpoint_path

    @pytest.fixture
    def tar_checkpoint(self, tmp_path, safetensors_checkpoint, checkpoint_model):
        model = checkpoint_model
        model.config.save(tmp_path / "config.yaml")

        tar_path = tmp_path / "checkpoint.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            tar.add(safetensors_checkpoint, arcname="checkpoint.safetensors")
            tar.add(tmp_path / "config.yaml", arcname="config.yaml")

        return tar_path

    def test_load_safetensors(self, checkpoint_model: nn.Module, safetensors_checkpoint: Path):
        # Fill with an irregular value
        for param in checkpoint_model.parameters():
            param.data.fill_(3.0)

        # Load should update the irregular value back to normal
        loaded = checkpoint_model.load_safetensors(safetensors_checkpoint)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()

    def test_load_tar(self, checkpoint_model: nn.Module, tar_checkpoint: Path):
        # Fill with an irregular value
        for param in checkpoint_model.parameters():
            param.data.fill_(3.0)

        # Load should update the irregular value back to normal
        loaded = ConvNext.load_tar(tar_checkpoint)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
