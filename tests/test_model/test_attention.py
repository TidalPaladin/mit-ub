import pytest
from sklearn import base
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing import assert_close

from mit_ub.model.attention import MultiHeadAttention, attention_forward




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
        nn.MultiheadAttention

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
    def test_forward(self, device, norm, num_heads, num_kv_heads, Lq, Lk, Dq, Dk):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq or Lk != Lq else None,
            vdim=Dk if Dk != Dq or Lk != Lq else None,
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
    def test_backward(self, device, norm, num_heads, num_kv_heads, Lq, Lk, Dq, Dk):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq or Lk != Lq else None,
            vdim=Dk if Dk != Dq or Lk != Lq else None,
        ).to(device)

        B = 2
        q = torch.randn(B, Lq, Dq, device=device, requires_grad=True)
        k = torch.randn(B, Lk, Dk, device=device, requires_grad=True) if Lk != Lq or Dk != Dq else q
        v = torch.randn(B, Lk, Dk, device=device, requires_grad=True) if Lk != Lq or Dk != Dq else q

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            out = model(q, k, v)

        out.sum().backward()

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
    def test_reset_parameters(self, norm, num_heads, num_kv_heads, Dq, Dk):
        model = MultiHeadAttention(
            Dq,
            num_heads,
            num_kv_heads,
            qk_norm=norm,
            kdim=Dk if Dk != Dq else None,
            vdim=Dk if Dk != Dq else None,
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
        ).to(device)

        B = 2
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
