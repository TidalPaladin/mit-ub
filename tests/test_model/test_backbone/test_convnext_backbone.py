import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from mit_ub.model import ConvNext2d
from mit_ub.model.backbone.convnext import ConvNextConfig


@pytest.fixture
def config():
    config = ConvNextConfig(
        in_channels=3,
        depths=(2, 2, 2),
        hidden_sizes=(32, 48, 64),
        ffn_hidden_sizes=(128, 192, 256),
        patch_size=(4, 4),
        kernel_size=(3, 3),
        activation="srelu",
    )
    return config


class TestConvNextConfig:

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_from_file(self, tmp_path, ext):
        config = {
            "in_channels": 3,
            "depths": [2, 2, 2],
            "hidden_sizes": [32, 48, 64],
            "ffn_hidden_sizes": [128, 192, 256],
            "activation": "srelu",
            "patch_size": [4, 4],
            "kernel_size": [3, 3],
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
        assert config.hidden_sizes == (32, 48, 64)
        assert config.ffn_hidden_sizes == (128, 192, 256)
        assert config.activation == "srelu"
        assert config.patch_size == (4, 4)
        assert config.kernel_size == (3, 3)

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = ConvNextConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, ConvNext2d)


class TestConvNext:

    @pytest.mark.parametrize(
        "config, exp",
        [
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 64, 16, 16),
            ),
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    up_depths=(2, 2, 2),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    @pytest.mark.cuda
    def test_forward(self, config, exp):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, device="cuda")
        model = ConvNext2d(config).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(x)
        assert out.shape == exp

    @pytest.mark.parametrize(
        "config, exp",
        [
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 64, 16, 16),
            ),
            (
                ConvNextConfig(
                    in_channels=1,
                    depths=(2, 2, 2),
                    hidden_sizes=(32, 48, 64),
                    ffn_hidden_sizes=(128, 192, 256),
                    up_depths=(2, 2, 2),
                    patch_size=(4, 4),
                    kernel_size=(3, 3),
                ),
                (1, 32, 64, 64),
            ),
        ],
    )
    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, exp, checkpoint):
        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        config = replace(config, checkpoint=checkpoint)
        model = ConvNext2d(config).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out = model(x)
        out.sum().backward()
        for param in model.parameters():
            assert param.grad is not None

    @pytest.mark.cuda
    def test_fewer_up_levels(self):
        config = ConvNextConfig(
            in_channels=1,
            depths=(2, 2, 2),
            up_depths=(2, 2),
            hidden_sizes=(32, 48, 64),
            ffn_hidden_sizes=(128, 192, 256),
            patch_size=(4, 4),
            kernel_size=(3, 3),
        )

        torch.random.manual_seed(42)
        B, C, H, W = 1, 1, 256, 256
        x = torch.randn(B, C, H, W, device="cuda")
        model = ConvNext2d(config).to("cuda")
        out = model(x)
        assert out.shape == (1, 48, 32, 32)

    @pytest.mark.cuda
    @pytest.mark.parametrize("pool_type", ["avg", None])
    def test_forward_head(self, config, pool_type):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = ConvNext2d(config).to("cuda")
        head = model.create_head(out_dim=10, pool_type=pool_type).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            features = model(x, reshape=False)
            out = head(features)
        exp = (1, 10) if pool_type is not None else (1, 196, 10)
        assert out.shape == exp

    @pytest.fixture
    def checkpoint_model(self, config):
        return ConvNext2d(config)

    @pytest.fixture
    def safetensors_checkpoint(self, tmp_path, checkpoint_model):
        model = checkpoint_model
        checkpoint_path = tmp_path / "checkpoint.safetensors"
        state_dict = model.state_dict()
        tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
        save_file(tensors, checkpoint_path)
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
        loaded = checkpoint_model.load_safetensors(safetensors_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()

    def test_load_tar(self, checkpoint_model: nn.Module, tar_checkpoint: Path):
        # Fill with an irregular value
        for param in checkpoint_model.parameters():
            param.data.fill_(3.0)

        # Load should update the irregular value back to normal
        loaded = ConvNext2d.load_tar(tar_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
