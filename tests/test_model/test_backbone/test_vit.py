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

from mit_ub.model.backbone import ViT, ViTConfig


@pytest.fixture
def config():
    config = ViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
    )
    return config


class TestViTConfig:

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

        config = ViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.hidden_size == 128

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = ViTConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, ViT)


class TestEncoderLayer:
    @pytest.mark.cuda
    def test_forward(self, config):
        B, L, D = 4, 32, config.hidden_size
        x = torch.randn(B, L, D, device="cuda")
        layer = ViT(config).create_encoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        assert out.shape == (B, L, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        B, L, D = 4, 32, config.hidden_size
        x = torch.randn(B, L, D, device="cuda")
        layer = ViT(config).create_encoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x, checkpoint_core_attention=checkpoint)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


class TestDecoderLayer:
    @pytest.mark.cuda
    def test_forward(self, config):
        B, Lq, Lk, D = 4, 16, 32, config.hidden_size
        q = torch.randn(B, Lq, D, device="cuda")
        kv = torch.randn(B, Lk, D, device="cuda")
        layer = ViT(config).create_decoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(q, encoder_output=kv)
        assert out.shape == (B, Lq, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        B, Lq, Lk, D = 4, 16, 32, config.hidden_size
        q = torch.randn(B, Lq, D, device="cuda")
        kv = torch.randn(B, Lk, D, device="cuda")
        layer = ViT(config).create_decoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(q, encoder_output=kv, checkpoint_core_attention=checkpoint)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


class TestHeadLayer:
    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_forward_no_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.hidden_size
        x = torch.randn(B, L, D, device="cuda")
        layer = ViT(config).create_head(D, use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        assert out.shape == (B, L, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_forward_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.hidden_size
        x = torch.randn(B, L, D, device="cuda")
        layer = ViT(config).create_head(D, pool_type="avg", use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        assert out.shape == (B, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_backward_no_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.hidden_size
        x = torch.randn(B, L, D, device="cuda")
        layer = ViT(config).create_head(D, use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_backward_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.hidden_size
        x = torch.randn(B, L, D, device="cuda")
        layer = ViT(config).create_head(D, pool_type="avg", use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


class TestViT:

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = ViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert out.shape[:2] == (1, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = ViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.cuda
    @pytest.mark.parametrize("scale", [2, 4])
    def test_forward_hr_conv(self, config, scale):
        x = torch.randn(1, 3, 224 * scale, 224 * scale, device="cuda")
        config = replace(config, hr_conv_scale=scale)
        model = ViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out, cls_token = model(x)
        assert out.shape[:2] == (1, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.cuda
    def test_backward_hr_conv(self, config):
        scale = 2
        x = torch.randn(1, 3, 224 * scale, 224 * scale, device="cuda", requires_grad=True)
        config = replace(config, hr_conv_scale=scale)
        model = ViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.fixture
    def checkpoint_model(self, config):
        return ViT(config)

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
        loaded = ViT.load_tar(tar_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
