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
from safetensors.torch import save_file
from torch.testing import assert_close

from mit_ub.model.backbone import NaViT, NaViTConfig


@pytest.fixture
def config():
    config = NaViTConfig(
        in_channels=3,
        patch_size=(16, 16),
        depth=3,
        hidden_size=128,
        ffn_hidden_size=256,
        num_attention_heads=128 // 16,
    )
    return config


class TestNaViTConfig:

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

        config = NaViTConfig.from_file(path)
        assert config.in_channels == 3
        assert config.hidden_size == 128

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = NaViTConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, NaViT)


class TestNaViT:

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, config, dtype):
        x1 = torch.randn(3, 224, 224, device="cuda")
        x2 = torch.randn(3, 384, 384, device="cuda")
        x = [x1, x2]
        model = NaViT(config).cuda()
        model.eval()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
            out1, cls_token1 = model([x1])
            out2, cls_token2 = model([x2])

        assert_close(out1[0], out[0])
        assert_close(out2[0], out[1])
        assert_close(cls_token1, cls_token[0:1])
        assert_close(cls_token2, cls_token[1:2])

    @pytest.mark.cuda
    def test_create_mask(self, config):
        x = torch.randn(3, 224, 224, device="cuda")
        model = NaViT(config).cuda()
        mask = model.create_mask(x, 0.5, 1)
        assert mask.ndim == 1
        assert mask.numel() == math.prod(model.stem.tokenized_size(x.shape[-2:]))

    @pytest.mark.cuda
    def test_forward_masked(self, config):
        x1 = torch.randn(3, 224, 224, device="cuda")
        x2 = torch.randn(3, 384, 384, device="cuda")
        x = [x1, x2]

        model = NaViT(config).cuda()
        model.eval()

        mask1 = model.create_mask(x1, 0.5, 1)
        mask2 = model.create_mask(x2, 0.5, 1)
        mask = [mask1, mask2]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            out, cls_token = model(x, mask=mask)
            out1, cls_token1 = model([x1], mask=[mask1])
            out2, cls_token2 = model([x2], mask=[mask2])

        assert_close(out1[0], out[0])
        assert_close(out2[0], out[1])
        assert_close(cls_token1, cls_token[0:1])
        assert_close(cls_token2, cls_token[1:2])

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        x1 = torch.randn(3, 224, 224, device="cuda", requires_grad=True)
        x2 = torch.randn(3, 384, 384, device="cuda", requires_grad=True)
        x = [x1, x2]
        config = replace(config, checkpoint=checkpoint)
        model = NaViT(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, cls_token = model(x)
        cls_token.sum().backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.cuda
    def test_prepare_inputs(self, config):
        torch.random.manual_seed(0)
        model = NaViT(config).cuda()
        x1 = torch.randn(3, 224, 224, device="cuda")
        x2 = torch.randn(3, 384, 384, device="cuda")
        x = [x1, x2]

        # Test inference mode (no token dropping)
        model.eval()
        resized, drop_rates = model.prepare_inputs(x)
        assert (drop_rates == 0.0).all()
        # When no resizing needed (target_seq_len is large enough)
        resized, _ = model.prepare_inputs(x, target_seq_len=10000)
        assert len(resized) == len(x)
        for r, orig in zip(resized, x):
            assert r.shape == orig.shape

        # Test training mode with token dropping
        model.train()
        resized, drop_rates = model.prepare_inputs(x)
        assert len(drop_rates) == len(x)
        assert all(0.5 <= r <= 0.5 + config.token_drop_gamma for r in drop_rates)

        # Test resizing behavior when tokens exceed target_seq_len
        small_target = 100  # Small enough to force resizing
        resized, drop_rates = model.prepare_inputs(x, target_seq_len=small_target)

        # Verify resized shapes are multiples of patch size
        for r in resized:
            assert all(dim % config.patch_size[0] == 0 for dim in r.shape[1:])
            # Verify minimum size is respected
            assert all(dim >= patch_dim for dim, patch_dim in zip(r.shape[1:], config.patch_size))

        # Verify total tokens after resize is less than or equal to target_seq_len
        token_counts = [
            int(math.prod(model.stem.tokenized_size(r.shape[1:])) * (1 - drop_rates[i]).item())
            for i, r in enumerate(resized)
        ]
        total_tokens = sum(token_counts)
        assert total_tokens <= small_target

        # Test with very small target to ensure minimum patch size is respected
        tiny_target = 1
        resized, _ = model.prepare_inputs(x, target_seq_len=tiny_target)
        for r in resized:
            assert all(dim >= patch_dim for dim, patch_dim in zip(r.shape[1:], config.patch_size))

    @pytest.fixture
    def checkpoint_model(self, config):
        return NaViT(config)

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
        loaded = NaViT.load_tar(tar_checkpoint, strict=False)
        for param in loaded.parameters():
            assert not (param.data == 3.0).all()
