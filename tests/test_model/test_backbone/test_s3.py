import json
import tarfile
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from mit_ub.model.backbone import S3, ConvNextConfig, S3Config, ViTConfig


@pytest.fixture
def config():
    # 56x56/8 -> 7x7 output grid
    # 224x224/8 -> 28x28 output grid
    vit_config = ViTConfig(
        in_channels=3,
        patch_size=(8, 8),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
    )
    # 112x112/4 -> 28x28 output grid
    convnext_config = ConvNextConfig(
        in_channels=3,
        patch_size=(4, 4),
        kernel_size=(7, 7),
        depths=(3,),
        hidden_sizes=(32,),
        ffn_hidden_sizes=(128,),
    )
    config = S3Config(
        vit_config=vit_config,
        convnext_config=convnext_config,
        convnext_resolution=(112, 112),  # 1/2 resolution
        resolutions=[
            (224, 224),
            (112, 112),
            (56, 56),
        ],
        token_fracs=0.2,
    )
    return config


class TestS3Config:

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_from_file(self, tmp_path, ext):
        config = {
            "vit_config": {
                "in_channels": 3,
                "hidden_size": 128,
                "ffn_hidden_size": 256,
                "patch_size": [16, 16],
                "depth": 3,
                "num_attention_heads": 128 // 16,
            },
            "convnext_config": {
                "in_channels": 3,
                "patch_size": [4, 4],
                "kernel_size": [7, 7],
                "depths": [3],
                "hidden_sizes": [32],
                "ffn_hidden_sizes": [128],
            },
            "convnext_resolution": [112, 112],
            "resolutions": [[224, 224], [112, 112], [56, 56]],
            "token_fracs": 0.2,
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = S3Config.from_file(path)
        assert config.vit_config.in_channels == 3
        assert config.vit_config.hidden_size == 128
        assert config.convnext_config.in_channels == 3
        assert config.convnext_config.patch_size == (4, 4)
        assert config.convnext_config.kernel_size == (7, 7)
        assert config.vit_resolution == (56, 56)
        assert config.convnext_resolution == (112, 112)
        assert config.resolutions == ((224, 224), (112, 112), (56, 56))
        assert config.token_fracs == (0.2, 0.2, 1.0)

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = S3Config.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, S3)


class TestS3:

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_saliency(self, config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = S3(config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out = model.forward_saliency(x)
        assert out.shape == (1, 1, 28, 28)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = S3(config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token, saliency = model(x)
        expected_out_tokens = 7 * 7 + int(14 * 14 * 0.2) + int(28 * 28 * 0.2)
        assert out.shape == (1, expected_out_tokens, 128)
        assert cls_token.shape == (1, 128)
        assert saliency.shape == (1, 1, 28, 28)

    @pytest.mark.cuda
    def test_backward(self, config):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        model = S3(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token, saliency = model(x)
        (out.sum() + cls_token.sum() + saliency.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.fixture
    def checkpoint_model(self, config):
        return S3(config)

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
        loaded = cast(Any, checkpoint_model).load_safetensors(safetensors_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()

    def test_load_tar(self, checkpoint_model: nn.Module, tar_checkpoint: Path):
        # Fill with an irregular value
        for param in checkpoint_model.parameters():
            param.data.fill_(3.0)

        # Load should update the irregular value back to normal
        loaded = S3.load_tar(tar_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
