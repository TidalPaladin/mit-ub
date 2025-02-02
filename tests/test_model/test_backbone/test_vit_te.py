import json
import timeit
from dataclasses import replace

import pytest
import torch
import yaml

from mit_ub.model.backbone import ViT, ViTConfig, ViTTE, ViTTEConfig
from mit_ub.model.layers.pool import PoolType


@pytest.fixture
def config():
    config = ViTTEConfig(
        in_channels=3,
        dim=128,
        dim_feedforward=256,
        patch_size=(16, 16),
        depth=3,
        nhead=128 // 16,
    )
    return config


@pytest.fixture
def config_vit():
    config = ViTConfig(
        in_channels=3,
        dim=128,
        dim_feedforward=256,
        patch_size=(16, 16),
        depth=3,
        nhead=128 // 16,
    )
    return config


@pytest.mark.skip
class TestViTTEConfig:

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_from_file(self, tmp_path, ext):
        config = {
            "in_channels": 3,
            "dim": 128,
            "dim_feedforward": 256,
            "patch_size": [16, 16],
            "depth": 3,
            "nhead": 128 // 16,
            "activation": "srelu",
        }

        path = tmp_path / f"config{ext}"
        with open(path, "w") as f:
            if ext == ".json":
                json.dump(config, f)
            else:
                yaml.dump(config, f)

        config = ViTTEConfig.from_file(path)
        assert config.in_channels == 3
        assert config.dim == 128

    @pytest.mark.parametrize("ext", [".json", ".yaml", ".yml"])
    def test_save(self, config, tmp_path, ext):
        path = tmp_path / f"config{ext}"
        config.save(path)

        loaded = ViTTEConfig.from_file(path)
        assert config == loaded

    def test_instantiate(self, config):
        model = config.instantiate()
        assert isinstance(model, ViTTE)


class TestTEEncoderLayer:
    @pytest.mark.cuda
    def test_forward(self, config):
        B, L, D = 4, 32, config.dim
        x = torch.randn(B, L, D, device="cuda")
        layer = ViTTE(config).create_encoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        assert out.shape == (B, L, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        B, L, D = 4, 32, config.dim
        x = torch.randn(B, L, D, device="cuda")
        layer = ViTTE(config).create_encoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x, checkpoint_core_attention=checkpoint)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


class TestTEDecoderLayer:
    @pytest.mark.cuda
    def test_forward(self, config):
        B, Lq, Lk, D = 4, 16, 32, config.dim
        q = torch.randn(B, Lq, D, device="cuda")
        kv = torch.randn(B, Lk, D, device="cuda")
        layer = ViTTE(config).create_decoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(q, encoder_output=kv)
        assert out.shape == (B, Lq, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        B, Lq, Lk, D = 4, 16, 32, config.dim
        q = torch.randn(B, Lq, D, device="cuda")
        kv = torch.randn(B, Lk, D, device="cuda")
        layer = ViTTE(config).create_decoder_layer().cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(q, encoder_output=kv, checkpoint_core_attention=checkpoint)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


class TestTEHeadLayer:
    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_forward_no_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.dim
        x = torch.randn(B, L, D, device="cuda")
        layer = ViTTE(config).create_head(D, use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        assert out.shape == (B, L, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_forward_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.dim
        x = torch.randn(B, L, D, device="cuda")
        layer = ViTTE(config).create_head(D, pool_type=PoolType.AVG, use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        assert out.shape == (B, D)

    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_backward_no_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.dim
        x = torch.randn(B, L, D, device="cuda")
        layer = ViTTE(config).create_head(D, use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.cuda
    @pytest.mark.parametrize("use_mlp", [True, False])
    def test_backward_pool(self, config, use_mlp):
        B, L, D = 4, 32, config.dim
        x = torch.randn(B, L, D, device="cuda")
        layer = ViTTE(config).create_head(D, pool_type=PoolType.AVG, use_mlp=use_mlp).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = layer(x)
        out.sum().backward()
        for name, param in layer.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"


class TestViTTE:

    @pytest.mark.cuda
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_forward(self, config, dtype):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = ViTTE(config).cuda()
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=True):
            out, cls_token = model(x)
        assert out.shape[:2] == (1, 128)
        assert cls_token.shape == (1, 128)

    @pytest.mark.cuda
    @pytest.mark.parametrize("checkpoint", [False, True])
    def test_backward(self, config, checkpoint):
        x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        config = replace(config, checkpoint=checkpoint)
        model = ViTTE(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out, cls_token = model(x)
        (out.sum() + cls_token.sum()).backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{name} has no gradient"
            assert not param.grad.isnan().any(), f"{name} has nan gradient"

    @pytest.mark.skip
    @pytest.mark.cuda
    def test_forward_benchmark(self, config, config_vit):
        x = torch.randn(1, 3, 224, 224, device="cuda")
        model = ViT(config_vit).cuda()
        model_te = ViTTE(config).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            # Run them both once to warm up
            model(x)
            model_te(x)

            torch.cuda.synchronize()
            start = timeit.default_timer()
            y, _ = model(x)
            y.sum()
            torch.cuda.synchronize()
            elapsed = timeit.default_timer() - start
            print(f"ViT forward time: {elapsed:.3f}s")

            torch.cuda.synchronize()
            start = timeit.default_timer()
            y_te, _ = model_te(x)
            y_te.sum()
            torch.cuda.synchronize()
            elapsed_te = timeit.default_timer() - start
            print(f"ViTTE forward time: {elapsed_te:.3f}s")
            assert False

    # @pytest.mark.cuda
    # @pytest.mark.parametrize("checkpoint", [False, True])
    # def test_backward_benchmark(self, config, checkpoint):
    #    x = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
    #    config = replace(config, checkpoint=checkpoint)
    #    model = ViTTE(config).cuda()
    #    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    #        out, cls_token = model(x)
    #    (out.sum() + cls_token.sum()).backward()
    #    for name, param in model.named_parameters():
    #        assert param.grad is not None, f"{name} has no gradient"
    #        assert not param.grad.isnan().any(), f"{name} has nan gradient"
