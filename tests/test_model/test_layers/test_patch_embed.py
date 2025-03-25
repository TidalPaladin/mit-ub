import math

import pytest
import torch
import transformer_engine.pytorch as te
from torch.testing import assert_close

from mit_ub.model.layers.patch_embed import PatchEmbed2d
from mit_ub.tokens import apply_mask


class TestPatchEmbed2d:

    @pytest.mark.cuda
    def test_forward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x)
        assert y.shape == (B, math.prod((H // 4, W // 4)), D_model)

    @pytest.mark.cuda
    def test_backward(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, requires_grad=True, device="cuda")
        y = layer(x)
        y.sum().backward()
        assert x.grad is not None

    @pytest.mark.cuda
    @pytest.mark.parametrize(
        "target",
        [
            (8, 8),
            (16, 16),
        ],
    )
    def test_resize_patch_weights(self, target):
        C, D = 3, 32
        layer = PatchEmbed2d(C, D, (4, 4)).to("cuda")
        resized = layer.resize_patch_weights(target)
        assert resized.shape[:2] == layer.patch.weight.shape[:2]
        assert resized.shape[2:] == target

    @pytest.mark.cuda
    def test_forward_flexible(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y1 = layer(x)
            y2 = layer(x, patch_size=(8, 8))

        assert y1.shape[0] == y2.shape[0] == B
        assert y1.shape[1] == y2.shape[1] * 4
        assert y1.shape[2] == y2.shape[2]

    @pytest.mark.cuda
    def test_backward_flexible(self):
        B, C, H, W = 2, 3, 64, 64
        D_model = 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x = torch.randn(B, C, H, W, device="cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            y = layer(x, patch_size=(8, 8))
        y.sum().backward()
        assert layer.patch.weight.grad is not None

    @pytest.mark.cuda
    def test_pack_forward(self):
        C, D_model = 3, 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        layer.eval()

        # Create input sequences of different sizes
        x1 = torch.randn(C, 64, 64, device="cuda")  # Will produce 16x16=256 tokens
        x2 = torch.randn(C, 32, 32, device="cuda")  # Will produce 8x8=64 tokens
        x3 = torch.randn(C, 48, 48, device="cuda")  # Will produce 12x12=144 tokens
        x = [x1, x2, x3]

        # Use different patch sizes for each sequence
        patch_sizes = [(4, 4), (2, 2), (8, 8)]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed_seqs, cu_seq_lens = layer.pack(x, patch_sizes)
            assert cu_seq_lens.dtype == torch.int32
            seq1 = layer.forward(x1[None], patch_sizes[0])
            seq2 = layer.forward(x2[None], patch_sizes[1])
            seq3 = layer.forward(x3[None], patch_sizes[2])

        packed_layer = te.TransformerLayer(
            D_model,
            D_model,
            self_attn_mask_type="padding",
            attn_input_format="thd",
            num_attention_heads=D_model // 16,
        )
        packed_layer.eval()
        layer = te.TransformerLayer(
            D_model,
            D_model,
            self_attn_mask_type="no_mask",
            attn_input_format="bshd",
            num_attention_heads=D_model // 16,
        )
        layer.eval()
        for param1, param2 in zip(packed_layer.parameters(), layer.parameters()):
            param1.data.copy_(param2.data)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out1 = layer(seq1)
            out2 = layer(seq2)
            out3 = layer(seq3)
            out = packed_layer(packed_seqs, cu_seqlens_q=cu_seq_lens, cu_seqlens_kv=cu_seq_lens)

        assert_close(out, torch.cat([out1, out2, out3], dim=1).squeeze(0), atol=0.001, rtol=0)

    @pytest.mark.cuda
    def test_pack_forward_with_mask(self):
        C, D_model = 3, 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        layer.eval()

        # Create input sequences of different sizes
        x1 = torch.randn(C, 64, 64, device="cuda")
        mask1 = torch.randint(0, 2, (1, 256), dtype=torch.bool, device="cuda")
        x2 = torch.randn(C, 32, 32, device="cuda")  # Will produce 8x8=64 tokens
        mask2 = torch.randint(0, 2, (1, 64), dtype=torch.bool, device="cuda")
        x3 = torch.randn(C, 48, 48, device="cuda")  # Will produce 12x12=144 tokens
        mask3 = torch.randint(0, 2, (1, 144), dtype=torch.bool, device="cuda")
        x = [x1, x2, x3]
        mask = [mask1, mask2, mask3]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed_seqs, cu_seq_lens = layer.pack(x, mask=mask)
            assert cu_seq_lens.dtype == torch.int32
            seq1 = apply_mask(mask1, layer.forward(x1[None]))
            seq2 = apply_mask(mask2, layer.forward(x2[None]))
            seq3 = apply_mask(mask3, layer.forward(x3[None]))

        expected = torch.cat([seq1, seq2, seq3], dim=1).squeeze(0)
        assert_close(packed_seqs, expected)

    @pytest.mark.cuda
    def test_pack_backward(self):
        C, D_model = 3, 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")
        x1 = torch.randn(C, 64, 64, device="cuda")  # Will produce 16x16=256 tokens
        x2 = torch.randn(C, 32, 32, device="cuda")  # Will produce 8x8=64 tokens
        x3 = torch.randn(C, 48, 48, device="cuda")  # Will produce 12x12=144 tokens
        x = [x1, x2, x3]
        patch_sizes = [(4, 4), (4, 4), (4, 4)]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            packed_seqs, seq_lens = layer.pack(x, patch_sizes)

        packed_seqs.sum().backward()
        assert layer.patch.weight.grad is not None

    @pytest.mark.cuda
    def test_pack_unpack(self):
        C, D_model = 3, 64
        layer = PatchEmbed2d(C, D_model, (4, 4)).to("cuda")

        # Create input sequences of different sizes
        x1 = torch.randn(C, 64, 64, device="cuda")  # Will produce 16x16=256 tokens
        x2 = torch.randn(C, 32, 32, device="cuda")  # Will produce 8x8=64 tokens
        x3 = torch.randn(C, 48, 48, device="cuda")  # Will produce 12x12=144 tokens
        x = [x1, x2, x3]
        patch_sizes = [(4, 4), (4, 4), (4, 4)]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            original_seqs = [layer.forward(xi[None], pi) for xi, pi in zip(x, patch_sizes)]
            packed_seqs, seq_lens = layer.pack(x, patch_sizes)
            unpacked_seqs = layer.unpack(packed_seqs, seq_lens)

        # Verify each unpacked sequence matches the original
        assert len(unpacked_seqs) == len(original_seqs)
        for orig, unpacked in zip(original_seqs, unpacked_seqs):
            assert_close(orig, unpacked, rtol=1e-3, atol=1e-3)  # Use larger tolerance for bfloat16
