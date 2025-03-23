import json
import math
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
import yaml
from einops import rearrange
from safetensors.torch import save_file
from torch.testing import assert_close

from mit_ub.model.backbone import TwoStageViT, TwoStageViTConfig, ViT, WindowedViT, WindowedViTConfig


@pytest.fixture
def config():
    config = TwoStageViTConfig(
        in_channels=3,
        patch_size=(4, 4),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
        first_stage_size=(32, 32),
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
    def test_tile_image(self, config):
        B = 2
        x = torch.randn(B, 3, 224, 224, device="cuda")
        Hp, Wp = config.first_stage_size
        x[0, ..., :Hp, :Wp] = 0
        x[1, ..., -Hp:, -Wp:] = 1
        model = TwoStageViT(config).cuda()
        out = model.tile_image(x)
        L = out.shape[0] // B
        assert (out[0] == 0).all()
        assert (out[L] != 0).any()
        assert (out[-1] == 1).all()

    @pytest.mark.cuda
    def test_untile_sequence(self, config):
        B = 2
        x = torch.randn(B, 3, 224, 224, device="cuda")
        model = TwoStageViT(config).cuda()
        Hp, Wp = config.first_stage_size
        Hl, Wl = model.stage_two_tokenized_size(x.shape[-2:])

        x[0, ..., :Hp, :Wp] = 0
        x[1, ..., -Hp:, -Wp:] = 1
        x = model.tile_image(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        out = model.untile_sequence(x, batch_size=B)

        assert (out[0, : Hp * Wp] == 0.0).all()
        assert (out[1, -Hp * Wp :] == 1.0).all()
        expected_len = Hp * Hp * Hl * Wl
        assert out.shape == (B, expected_len, 3)

    @pytest.mark.cuda
    def test_forward_stage_one(self, config):
        x = torch.randn(2, 3, 224, 224, device="cuda")
        model = TwoStageViT(config).cuda()
        model.eval()

        Hp, Wp = config.first_stage_size
        x2 = x[:, :, :Hp, :Wp]
        x3 = x[:, :, -Hp:, -Wp:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _, expected_cls_token2 = ViT.forward(model, x2, reshape=False)
            _, expected_cls_token3 = ViT.forward(model, x3, reshape=False)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out, cls_tokens = model.forward_stage_one(x)

        assert_close(cls_tokens[:, 0, :], expected_cls_token2)
        assert_close(cls_tokens[:, -1, :], expected_cls_token3)

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
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_batch_independent(self, config, dtype):
        x = torch.randn(2, 3, 224, 224, device="cuda")
        x[0] = float("nan")
        model = TwoStageViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert not out[1].isnan().any()
        assert not cls_token[1].isnan().any()

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_masked(self, config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = TwoStageViT(config).cuda()
        mask = model.create_mask(x, 0.5, 1)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x, mask=mask, reshape=False)
        assert out.shape[:2] == (1, 25)
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


@pytest.fixture
def windowed_config():
    config = WindowedViTConfig(
        in_channels=3,
        patch_size=(4, 4),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
        window_size=(32, 32),
        global_depth=4,
    )
    return config


class TestWindowedViTConfig:

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
            "window_size": [32, 32],
            "global_depth": 4,
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = WindowedViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.hidden_size == 128
        assert config.window_size == (32, 32)
        assert config.global_depth == 4

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, windowed_config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        windowed_config.save(path)

        loaded = WindowedViTConfig.from_file(path)
        assert windowed_config == loaded

    def test_instantiate(self, windowed_config):
        model = windowed_config.instantiate()
        assert isinstance(model, WindowedViT)


class TestWindowedViT:

    @pytest.mark.cuda
    def test_tile_image(self, windowed_config):
        B = 2
        x = torch.randn(B, 3, 224, 224, device="cuda")
        Hp, Wp = windowed_config.window_size
        x[0, ..., :Hp, :Wp] = 0
        x[1, ..., -Hp:, -Wp:] = 1
        model = WindowedViT(windowed_config).cuda()
        out = model.tile_image(x)
        L = out.shape[0] // B
        assert (out[0] == 0).all()
        assert (out[L] != 0).any()
        assert (out[-1] == 1).all()

    @pytest.mark.cuda
    def test_untile_sequence(self, windowed_config):
        B = 2
        x = torch.randn(B, 3, 224, 224, device="cuda")
        model = WindowedViT(windowed_config).cuda()
        Hp, Wp = windowed_config.window_size
        Hl, Wl = model.stem.tokenized_size(x.shape[-2:])

        x[0, ..., :Hp, :Wp] = 0
        x[1, ..., -Hp:, -Wp:] = 1
        x = model.tile_image(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        out = model.untile_sequence(x, batch_size=B)

        assert (out[0, : Hp * Wp] == 0.0).all()
        assert (out[1, -Hp * Wp :] == 1.0).all()
        expected_len = Hl * Wl * math.prod(windowed_config.patch_size)
        assert out.shape == (B, expected_len, 3)

    @pytest.mark.cuda
    def test_forward_stage_one(self, windowed_config):
        x = torch.randn(2, 3, 224, 224, device="cuda")
        model = WindowedViT(windowed_config).cuda()
        model.eval()

        Hp, Wp = windowed_config.window_size
        x2 = x[:, :, :Hp, :Wp]
        x3 = x[:, :, -Hp:, -Wp:]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _, cls_token_1 = ViT.forward(model, x2, reshape=False)
            _, cls_token_2 = ViT.forward(model, x3, reshape=False)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            _, cls_token = model.forward_windowed(x)

        assert_close(cls_token[:, 0, :], cls_token_1)
        assert_close(cls_token[:, -1, :], cls_token_2)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, windowed_config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = WindowedViT(windowed_config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert out.shape[:2] == (1, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_batch_independent(self, windowed_config, dtype):
        x = torch.randn(2, 3, 224, 224, device="cuda")
        x[0] = float("nan")
        model = WindowedViT(windowed_config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert not out[1].isnan().any()
        assert not cls_token[1].isnan().any()

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward_masked(self, windowed_config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = WindowedViT(windowed_config).cuda()
        mask = model.create_mask(x, 0.5, 1)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x, mask=mask, reshape=False)
        expected_len = math.prod(model.stem.tokenized_size(x.shape[-2:])) // 2
        assert out.shape[:2] == (1, expected_len)
        assert cls_token.shape == (1, 128)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, windowed_config, checkpoint):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        config = replace(windowed_config, checkpoint=checkpoint)
        model = WindowedViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None or not param.requires_grad, f"{name} has no gradient"
            assert param.grad is None or not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward_end_to_end(self, windowed_config, checkpoint):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        config = replace(windowed_config, checkpoint=checkpoint, freeze_first_stage=False)
        model = WindowedViT(config).cuda()
        assert all(p.requires_grad for p in model.parameters())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.fixture
    def checkpoint_model(self, windowed_config):
        return WindowedViT(windowed_config)

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
        loaded = WindowedViT.load_tar(tar_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
