import pytest
import torch
from torch.testing import assert_close

from mit_ub.model.attention import MultiHeadAttention, attention_forward


def test_attention_equivalence():
    B, L, D = 1, 32, 128
    nhead = D // 16

    x = torch.randn(B, L, D)
    q = x.clone()
    kv = x.clone()

    w_q = torch.randn(D, D)
    w_k = torch.randn(D, D)
    w_v = torch.randn(D, D)
    w_in = torch.cat([w_q, w_k, w_v], dim=0)

    b_q = torch.randn(D)
    b_k = torch.randn(D)
    b_v = torch.randn(D)
    b_in = torch.cat([b_q, b_k, b_v], dim=0)

    w_out = torch.randn(D, D)
    b_out = torch.randn(D)
    norm_w = torch.randn(16)

    out_mhsa = attention_forward(
        # fmt: off
        x, None, None,
        w_in, None, None,
        b_in, None, None,
        w_out, b_out,
        nhead, nhead,
        norm_w, None,
        dropout=0.0,
        # fmt: on
    )
    out_mha = attention_forward(
        # fmt: off
        q, kv, kv,
        w_q, w_k, w_v,
        b_q, b_k, b_v,
        w_out, b_out,
        nhead, nhead,
        norm_w, None,
        dropout=0.0,
        # fmt: on
    )
    assert_close(out_mhsa, out_mha, atol=0.01, rtol=0)


class TestMultiHeadAttention:

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

        out_norm = model(q, k, v)
        model.w_pre_norm = None  # type: ignore
        model.b_pre_norm = None  # type: ignore
        out_no_norm = model(q, k, v)
        assert not torch.allclose(out_norm, out_no_norm)
