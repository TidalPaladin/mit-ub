from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.layers import NormType
from mit_ub.model.layers.attention import MultiHeadAttention


class TestMultiHeadAttention:

    def test_torch_equivalence(self):
        B, L, D = 2, 8, 32
        nhead = 2
        torch.random.manual_seed(0)

        model = MultiHeadAttention(
            D,
            nhead,
            nhead,
            dropout=0.0,
            bias=False,
            kdim=D,
            vdim=D,
        )
        model.eval()

        q = torch.randn(B, L, D)
        k = torch.randn(B, L, D)
        v = torch.randn(B, L, D)

        baseline_out = F.multi_head_attention_forward(
            # fmt: off
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
            D, nhead,
            None, None,
            None, None,
            False,
            0.0,
            model.w_out, None,
            training=False,
            need_weights=False,
            use_separate_proj_weight=True,
            q_proj_weight=model.w_q,
            k_proj_weight=model.w_k,
            v_proj_weight=model.w_v,
            average_attn_weights=True,
            # fmt: on
        )[0].transpose(0, 1)

        out = model(q, k, v)
        assert_close(baseline_out, out)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("norm_type", [NormType.LAYER_NORM, NormType.RMS_NORM])
    @pytest.mark.parametrize("stochastic_depth", [0.0, 0.25])
    @pytest.mark.parametrize("layer_scale", [None, 0.1])
    @pytest.mark.parametrize(
        "num_heads,num_kv_heads,Lq,Lk,Dq,Dk",
        [
            (32, 32, 32, 32, 128, 128),
            (32, 16, 32, 32, 128, 128),
            (32, 32, 32, 16, 128, 128),
            (32, 32, 32, 32, 128, 64),
            (32, 8, 32, 32, 128, 64),
        ],
    )
    def test_forward(
        self, device, norm, norm_type, num_heads, num_kv_heads, Lq, Lk, Dq, Dk, layer_scale, stochastic_depth
    ):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq or Lk != Lq else None,
            vdim=Dk if Dk != Dq or Lk != Lq else None,
            dropout=0.1,
            norm_type=norm_type,
            layer_scale=layer_scale,
            stochastic_depth=stochastic_depth,
        ).to(device)

        B = 2
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device) if Lk != Lq or Dk != Dq else q
        v = torch.randn(B, Lk, Dk, device=device) if Lk != Lq or Dk != Dq else q

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(q, k, v)

        assert out.shape == q.shape
        assert not out.isnan().any()

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("checkpoint", [False, True])
    @pytest.mark.parametrize("layer_scale", [None, 0.1])
    @pytest.mark.parametrize(
        "num_heads,num_kv_heads,Lq,Lk,Dq,Dk",
        [
            (32, 32, 32, 32, 128, 128),
            (32, 16, 32, 32, 128, 128),
            (32, 32, 32, 16, 128, 128),
            (32, 32, 32, 32, 128, 64),
            (32, 8, 32, 32, 128, 64),
        ],
    )
    def test_backward(self, device, norm, checkpoint, layer_scale, num_heads, num_kv_heads, Lq, Lk, Dq, Dk):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq or Lk != Lq else None,
            vdim=Dk if Dk != Dq or Lk != Lq else None,
            dropout=0.1,
            layer_scale=layer_scale,
        ).to(device)
        model.checkpoint = checkpoint
        model.train()

        B = 2
        q = torch.randn(B, Lq, Dq, device=device, requires_grad=True)
        k = torch.randn(B, Lk, Dk, device=device, requires_grad=True) if Lk != Lq or Dk != Dq else q
        v = torch.randn(B, Lk, Dk, device=device, requires_grad=True) if Lk != Lq or Dk != Dq else q

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(q, k, v)

        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    @pytest.mark.parametrize(
        "num_heads,num_kv_heads,Dq,Dk",
        [
            (32, 32, 128, 128),
            (32, 16, 128, 128),
            (32, 32, 128, 128),
            (32, 32, 128, 64),
            (32, 8, 128, 64),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("layer_scale", [None, 1.0])
    def test_reset_parameters(self, norm, layer_scale, num_heads, num_kv_heads, Dq, Dk):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq else None,
            vdim=Dk if Dk != Dq else None,
            layer_scale=layer_scale,
        )

        weights_original = {name: param.clone() for name, param in model.named_parameters()}
        model.reset_parameters()
        weights_reset = {name: param for name, param in model.named_parameters()}

        for name, param in weights_original.items():
            # Ignore constant weights or biases
            if (param == 0).all() or (param == 1).all():
                continue
            assert not torch.allclose(param, weights_reset[name], equal_nan=True)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize(
        "num_heads,num_kv_heads,Lq,Lk,Dq,Dk",
        [
            (32, 32, 32, 32, 128, 128),
            (32, 16, 32, 32, 128, 128),
            (32, 32, 32, 16, 128, 128),
            (32, 32, 32, 32, 128, 64),
            (32, 8, 32, 32, 128, 64),
        ],
    )
    def test_forward_deterministic(self, device, norm, num_heads, num_kv_heads, Lq, Lk, Dq, Dk):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq or Lk != Lq else None,
            vdim=Dk if Dk != Dq or Lk != Lq else None,
            dropout=0.1,
            stochastic_depth=0.25,
        ).to(device)

        B = 8
        q = torch.randn(B, Lq, Dq, device=device)
        k = torch.randn(B, Lk, Dk, device=device) if Lk != Lq or Dk != Dq else q
        v = torch.randn(B, Lk, Dk, device=device) if Lk != Lq or Dk != Dq else q

        model.train()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out1 = model(q, k, v)
            out2 = model(q, k, v)
        assert not torch.allclose(out1, out2)

        model.eval()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out1 = model(q, k, v)
            out2 = model(q, k, v)
        assert torch.allclose(out1, out2)

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.cuda),
        ],
    )
    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("qk_norm", [False, True])
    @pytest.mark.parametrize(
        "num_heads,L,D",
        [
            (32, 32, 128),
            (16, 16, 64),
        ],
    )
    def test_packed_matches_unpacked(self, device, norm, qk_norm, num_heads, L, D):
        packed = MultiHeadAttention(
            D,
            num_heads,
            num_heads,
            qk_norm=qk_norm,
            norm=norm,
            dropout=0.1,
            kv_norm=True,
        )
        unpacked = deepcopy(packed)
        w_q, w_k, w_v = packed.w_in.split([D, D, D], dim=0)
        b_q, b_k, b_v = packed.b_in.split([D, D, D], dim=0)  # type: ignore
        unpacked.w_q = nn.Parameter(w_q)
        unpacked.w_k = nn.Parameter(w_k)
        unpacked.w_v = nn.Parameter(w_v)
        unpacked.b_q = nn.Parameter(b_q)
        unpacked.b_k = nn.Parameter(b_k)
        unpacked.b_v = nn.Parameter(b_v)
        unpacked.w_in = None  # type: ignore
        unpacked.b_in = None

        sum_packed = sum(p.sum() for p in packed.parameters())
        sum_unpacked = sum(p.sum() for p in unpacked.parameters())
        assert_close(sum_packed, sum_unpacked)

        B = 2
        x = torch.randn(B, L, D, device=device)

        packed = packed.to(device)
        unpacked = unpacked.to(device)
        packed.eval()
        unpacked.eval()

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            packed_out = packed(x, x, x)
            unpacked_out = unpacked(x, x.clone(), x.clone())

        # Fused LayerNorm differs between self and cross attention
        assert_close(packed_out, unpacked_out)

    def test_fused_norm(self):
        num_heads, num_kv_heads, Lq, Lk, Dq, Dk = 32, 32, 32, 32, 128, 128
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            kdim=Dk if Dk != Dq or Lk != Lq else None,
            vdim=Dk if Dk != Dq or Lk != Lq else None,
            norm=True,
        )

        B = 2
        q = torch.randn(B, Lq, Dq)
        k = torch.randn(B, Lk, Dk) if Lk != Lq or Dk != Dq else q
        v = torch.randn(B, Lk, Dk) if Lk != Lq or Dk != Dq else q

        w_norm = model.w_pre_norm
        b_norm = model.b_pre_norm
        actual = model(q, k, v)

        model.w_pre_norm = model.b_pre_norm = None  # type: ignore
        _q = F.layer_norm(q, q.shape[-1:], weight=w_norm, bias=b_norm)
        k = F.layer_norm(k, k.shape[-1:], weight=w_norm, bias=b_norm) if k is not q else _q
        v = F.layer_norm(v, v.shape[-1:], weight=w_norm, bias=b_norm) if v is not q else _q
        q = _q
        baseline = model(q, k, v)

        assert_close(baseline, actual)

    def test_extra_repr(self):
        layer = MultiHeadAttention(32, 8, 8)
        result = str(layer)
        exp = "dim=32, heads=8, kv_heads=8, head_dim=4, kv_dim=32, dropout=0.0, norm=False, qk_norm=False, bias=True, kv_norm=False, norm_type=layernorm"
        assert exp in result

    def test_copy_parameters_self(self):
        D, H, HKV = 128, 128 // 32, 128 // 32
        src = MultiHeadAttention(D, H, HKV, norm=True, qk_norm=True, kv_norm=True)
        dst = MultiHeadAttention(D, H, HKV, norm=True, qk_norm=True, kv_norm=True)
        dst.copy_parameters(src)
        src.eval()
        dst.eval()
        L = 32
        seq = torch.randn(1, L, D)
        exp = src(seq, seq, seq)
        act = dst(seq, seq, seq)
        assert_close(act, exp)

    @pytest.mark.parametrize("norm", [False, True])
    @pytest.mark.parametrize("qk_norm", [False, True])
    def test_copy_parameters_cross(self, norm, qk_norm):
        D, H, HKV = 128, 128 // 32, 128 // 32
        src = MultiHeadAttention(D, H, HKV, norm=norm, qk_norm=qk_norm, kv_norm=True)
        dst = MultiHeadAttention(D, H, HKV, norm=norm, qk_norm=qk_norm, kdim=D, vdim=D, kv_norm=True)
        dst.copy_parameters(src)

        sum_src = sum(p.sum() for p in src.parameters())
        sum_dst = sum(p.sum() for p in dst.parameters())
        assert_close(sum_src, sum_dst)

        src.eval()
        dst.eval()
        L = 32
        seq = torch.randn(1, L, D)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            exp = src(seq, seq, seq)
            act = dst(seq, seq.clone(), seq.clone())
        assert_close(act, exp)

    @pytest.mark.parametrize(
        "d1,h1,hkv1,norm1,qk1,d2,h2,hkv2,norm2,qk2",
        [
            (32, 8, 8, False, False, 32, 8, 8, False, False),
            (32, 8, 8, False, False, 32, 8, 8, True, False),
            (32, 8, 8, False, False, 32, 8, 8, False, True),
            (32, 8, 8, False, False, 32, 8, 8, True, True),
            (32, 8, 8, False, False, 32, 8, 8, False, False),
            (32, 8, 8, True, False, 32, 8, 8, False, False),
            (32, 8, 8, False, True, 32, 8, 8, False, False),
        ],
    )
    def test_copy_parameters_no_error(self, d1, h1, hkv1, norm1, qk1, d2, h2, hkv2, norm2, qk2):
        src = MultiHeadAttention(d1, h1, hkv1, norm=norm1, qk_norm=qk1)
        dst = MultiHeadAttention(d2, h2, hkv2, norm=norm2, qk_norm=qk2)
        dst.copy_parameters(src)

    def test_mask(self):
        B, Lq, Lk, D = 2, 8, 6, 32
        nhead = 2
        torch.random.manual_seed(0)

        model = MultiHeadAttention(
            D,
            nhead,
            nhead,
            dropout=0.0,
            bias=False,
            kdim=D,
            vdim=D,
        )
        model.eval()

        q = torch.randn(B, Lq, D)
        k = torch.randn(B, Lk, D)
        v = torch.randn(B, Lk, D)

        # Set the first value token to a high value but not nan or inf
        mask = torch.full((B, Lq, Lk), True, dtype=torch.bool)
        v[:, 0, :] = 1000
        mask[:, :, 0] = False

        out1 = model(q, k, v, mask)
        out2 = model(q, k, v)
        assert out1.max() < 100
        assert out2.max() > 100
