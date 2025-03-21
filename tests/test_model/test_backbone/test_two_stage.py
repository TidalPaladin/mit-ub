import json
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import yaml
from safetensors.torch import save_file

from mit_ub.model.backbone import TwoStageViT, TwoStageViTConfig


@pytest.fixture(params=[True, False])
def config(request):
    config = TwoStageViTConfig(
        in_channels=3,
        patch_size=(4, 4),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
        first_stage_size=(32, 32),
        second_stage_cross_attention=request.param,
    )
    return config


class TestTwoStageViTConfig:

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_from_file(self, tmp_path, ext):
        config = {
            "in_channels": 3,
            "hidden_size": 128,
            "ffn_hidden_size": 256,
            "patch_size": [16, 16],
            "depth": 3,
            "num_attention_heads": 128 // 16,
            "activation": "srelu",
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = TwoStageViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.hidden_size == 128

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = TwoStageViTConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, TwoStageViT)


class TestTwoStageViT:

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = TwoStageViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert out.shape[:2] == (1, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = TwoStageViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None or not param.requires_grad, f"{name} has no gradient"
            assert param.grad is None or not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.fixture
    def checkpoint_model(self, config):
        return TwoStageViT(config)

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
        loaded = TwoStageViT.load_tar(tar_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
