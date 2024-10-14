import pytest
import torch
import torch.nn as nn

from mit_ub.model.attention import (
    MultiHeadAttention,
    grouped_query_attention_forward,
    grouped_query_self_attention_forward,
    multi_head_attention_forward,
    multi_head_self_attention_forward,
)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("norm", [False, True])
def test_forward_mhsa(device, norm):
    B, L, D = 1, 32, 128
    nhead = D // 16
    x = torch.randn(B, L, D, device=device)
    model = nn.MultiheadAttention(D, nhead, dropout=0.0).to(device)
    norm_w = torch.randn(16, device=device) if norm else None
    with torch.autocast(device_type=device, dtype=torch.float16):
        out = multi_head_self_attention_forward(
            # fmt: off
            x,
            model.in_proj_weight, None,
            model.out_proj.weight, model.out_proj.bias,
            nhead,
            norm_w, None
            # fmt: on
        )

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("norm", [False, True])
def test_forward_gqsa(device, norm):
    B, L, D = 1, 32, 128
    head_dim = 32
    nhead = D // head_dim
    num_kv_heads = nhead // 2
    kv_dim = head_dim * num_kv_heads

    x = torch.randn(B, L, D, device=device)
    w_in = torch.randn(D + 2 * kv_dim, D, device=device)
    w_out = torch.randn(D, D, device=device)
    b_out = torch.randn(D, device=device)
    norm_w = torch.randn(head_dim, device=device) if norm else None
    with torch.autocast(device_type=device, dtype=torch.float16):
        out = grouped_query_self_attention_forward(
            # fmt: off
            x,
            w_in, None,
            w_out, b_out,
            nhead, num_kv_heads,
            norm_w, None
            # fmt: on
        )

    assert out.shape == x.shape


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("norm", [False, True])
def test_forward_mha(device, norm):
    B, L, D = 1, 32, 128
    nhead = D // 16
    q = torch.randn(B, L, D, device=device)
    kv = torch.randn(B, L, D, device=device)
    w_q = torch.randn(D, D, device=device)
    w_k = torch.randn(D, D, device=device)
    w_v = torch.randn(D, D, device=device)
    b_q = torch.randn(D, device=device) if norm else None
    b_k = torch.randn(D, device=device) if norm else None
    b_v = torch.randn(D, device=device) if norm else None
    w_out = torch.randn(D, D, device=device)
    b_out = torch.randn(D, device=device) if norm else None
    norm_w = torch.randn(16, device=device) if norm else None
    with torch.autocast(device_type=device, dtype=torch.float16):
        out = multi_head_attention_forward(
            # fmt: off
            q, kv, kv,
            w_q, w_k, w_v,
            b_q, b_k, b_v,
            w_out, b_out,
            nhead,
            norm_w, None
            # fmt: on
        )

    assert out.shape == q.shape


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.cuda),
    ],
)
@pytest.mark.parametrize("norm", [False, True])
def test_forward_gqa(device, norm):
    B, L, D = 1, 32, 128
    head_dim = 32
    nhead = D // head_dim
    num_kv_heads = nhead // 2
    kv_dim = head_dim * num_kv_heads

    q = torch.randn(B, L, D, device=device)
    kv = torch.randn(B, L, D, device=device)
    w_q = torch.randn(D, D, device=device)
    w_k = torch.randn(kv_dim, D, device=device)
    w_v = torch.randn(kv_dim, D, device=device)
    b_q = torch.randn(D, device=device) if norm else None
    b_k = torch.randn(kv_dim, device=device) if norm else None
    b_v = torch.randn(kv_dim, device=device) if norm else None
    w_out = torch.randn(D, D, device=device)
    b_out = torch.randn(D, device=device) if norm else None
    norm_w = torch.randn(head_dim, device=device) if norm else None
    with torch.autocast(device_type=device, dtype=torch.float16):
        out = grouped_query_attention_forward(
            # fmt: off
            q, kv, kv,
            w_q, w_k, w_v,
            b_q, b_k, b_v,
            w_out, b_out,
            nhead, num_kv_heads,
            norm_w, None
            # fmt: on
        )

    assert out.shape == q.shape


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
