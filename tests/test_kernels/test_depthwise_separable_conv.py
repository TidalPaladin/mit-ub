import os
import torch
import torch.nn.functional as F
import pytest
from torch.testing import assert_close
from torch.utils.cpp_extension import load

@pytest.fixture
def extension():
    # Load the CUDA extension.
    # Assumes the source files are in the csrc directory relative to the test file.
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    extension = load(
        name="depthwise_conv",
        sources=[os.path.join(cur_dir, "../../csrc/conv.cu")],
        extra_cflags=["-O2"],
        extra_cuda_cflags=["-O2"],
        verbose=True
    )

@pytest.mark.parametrize("B,H,W,C,C_out,kernel_size,pad", [
    (2, 8, 8, 3, 5, 3, 1),
    (1, 16, 16, 4, 4, 3, 1)
])
def test_depthwise_separable_conv(extension, tmp_path, B, H, W, C, C_out, kernel_size, pad):
    """
    Tests custom depthwise separable convolution against a reference PyTorch implementation.

    Shapes:
      - input: (B, H*W, C)
      - weight_dw: (C, kernel_size, kernel_size)
      - weight_pw: (C, C_out)
    """
    # Create input tensor and weights on CUDA.
    input_tensor = torch.randn(B, H * W, C, device="cuda")
    weight_dw = torch.randn(C, kernel_size, kernel_size, device="cuda")
    bias_dw = torch.randn(C, device="cuda")
    weight_pw = torch.randn(C, C_out, device="cuda")
    bias_pw = torch.randn(C_out, device="cuda")
    
    # Run the custom extension.
    output = extension.depthwise_separable_conv_forward(
        input_tensor, weight_dw, bias_dw, weight_pw, bias_pw, H, W, pad, pad
    )
    
    # Reference implementation using PyTorch's conv2d.
    # Reshape input to (B, C, H, W) for conv2d.
    x = input_tensor.reshape(B, H, W, C).permute(0, 3, 1, 2)
    # Depthwise convolution: use groups==C.
    depthwise_out = F.conv2d(
        x, weight=weight_dw.view(C, 1, kernel_size, kernel_size),
        bias=bias_dw, padding=pad, groups=C
    )
    # Pointwise convolution: 1x1 convolution.
    # Permute to get weight of shape (C_out, C, 1, 1).
    pointwise_out = F.conv2d(
        depthwise_out,
        weight=weight_pw.permute(1, 0).view(C_out, C, 1, 1),
        bias=bias_pw
    )
    # Reshape back to (B, H*W, C_out).
    output_ref = pointwise_out.permute(0, 2, 3, 1).reshape(B, H * W, C_out)
    
    # Compare the outputs.
    assert_close(output, output_ref, atol=1e-5, rtol=1e-5) 