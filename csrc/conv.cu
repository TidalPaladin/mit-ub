#include "conv.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

/**
 * @brief CUDA kernel for depthwise convolution.
 *
 * Each thread computes one output element for the intermediate tensor.
 *
 * Shapes:
 *   - input: (B, H*W, C) flattened; index conversion done via H and W.
 *   - weight_dw: (C, kH, kW)
 *   - bias_dw: (C)
 *   - output: (B, H, W, C)
 */
template <typename scalar_t>
__global__ void depthwise_conv_kernel(
    const scalar_t* __restrict__ input,   // (B, H*W, C)
    const scalar_t* __restrict__ weight_dw, // (C, kH, kW)
    const scalar_t* __restrict__ bias_dw,   // (C)
    scalar_t* __restrict__ output,          // (B, H, W, C)
    int B, int H, int W, int C,
    int kH, int kW,
    int padH, int padW) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * C;
    if (idx >= total)
        return;

    // Convert flat index idx into (b, h, w, c)
    int tmp = idx;
    int c = tmp % C;
    tmp /= C;
    int w = tmp % W;
    tmp /= W;
    int h = tmp % H;
    tmp /= H;
    int b = tmp;

    // Depthwise convolution sum for a single output element.
    scalar_t sum = 0;
    for (int i = 0; i < kH; ++i) {
        for (int j = 0; j < kW; ++j) {
            int in_h = h + i - padH;
            int in_w = w + j - padW;
            if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                // Input is stored as (B, H*W, C) => compute index accordingly.
                int input_idx = b * (H * W * C) + (in_h * W + in_w) * C + c;
                int weight_idx = c * (kH * kW) + i * kW + j;
                sum += input[input_idx] * weight_dw[weight_idx];
            }
        }
    }
    sum += bias_dw[c];
    int out_idx = b * (H * W * C) + (h * W + w) * C + c;
    output[out_idx] = sum;
}

/**
 * @brief CUDA kernel for pointwise convolution.
 *
 * Each thread computes one final output element.
 *
 * Shapes:
 *   - input: Intermediate tensor from depthwise conv (B, H, W, C).
 *   - weight_pw: (C, C_out)
 *   - bias_pw: (C_out)
 *   - output: (B, H, W, C_out)
 */
template <typename scalar_t>
__global__ void pointwise_conv_kernel(
    const scalar_t* __restrict__ input,   // (B, H, W, C)
    const scalar_t* __restrict__ weight_pw, // (C, C_out)
    const scalar_t* __restrict__ bias_pw,   // (C_out)
    scalar_t* __restrict__ output,          // (B, H, W, C_out)
    int B, int H, int W, int C, int C_out) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * H * W * C_out;
    if (idx >= total)
        return;
    
    // Convert flat index into (b, h, w, co)
    int tmp = idx;
    int co = tmp % C_out;
    tmp /= C_out;
    int w = tmp % W;
    tmp /= W;
    int h = tmp % H;
    tmp /= H;
    int b = tmp;

    scalar_t sum = bias_pw[co];
    // Pointwise convolution: sum over C channels.
    for (int c = 0; c < C; ++c) {
        int input_idx = b * (H * W * C) + (h * W + w) * C + c;
        int weight_idx = c * C_out + co;
        sum += input[input_idx] * weight_pw[weight_idx];
    }
    int out_idx = b * (H * W * C_out) + (h * W + w) * C_out + co;
    output[out_idx] = sum;
}

torch::Tensor depthwise_separable_conv_forward(
    torch::Tensor input,
    torch::Tensor weight_dw,
    torch::Tensor bias_dw,
    torch::Tensor weight_pw,
    torch::Tensor bias_pw,
    int H, int W, int padH, int padW)
{
    // Get input dimensions
    int B = input.size(0);
    int C = input.size(2);  // input shape is (B, H*W, C)
    int spatial = H * W;
    
    // Determine kernel spatial dimensions (kH, kW) from depthwise weights (C, kH, kW)
    int kH = weight_dw.size(1);
    int kW = weight_dw.size(2);
    
    // Allocate intermediate tensor for depthwise output: shape (B, H, W, C)
    auto intermediate = torch::empty({B, H, W, C}, input.options());
    
    // Launch depthwise convolution kernel.
    int total_depthwise = B * H * W * C;
    int threads = 256;
    int blocks = (total_depthwise + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "depthwise_conv_forward_cuda", ([&] {
        depthwise_conv_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            weight_dw.data_ptr<scalar_t>(),
            bias_dw.data_ptr<scalar_t>(),
            intermediate.data_ptr<scalar_t>(),
            B, H, W, C,
            kH, kW,
            padH, padW);
    }));
    
    // Determine pointwise output channel count from weight_pw: shape (C, C_out)
    int C_out = weight_pw.size(1);
    // Allocate output tensor: shape (B, H, W, C_out)
    auto output = torch::empty({B, H, W, C_out}, input.options());
    
    int total_pointwise = B * H * W * C_out;
    blocks = (total_pointwise + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pointwise_conv_forward_cuda", ([&] {
        pointwise_conv_kernel<scalar_t><<<blocks, threads>>>(
            intermediate.data_ptr<scalar_t>(),
            weight_pw.data_ptr<scalar_t>(),
            bias_pw.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, H, W, C, C_out);
    }));
    
    // Reshape output to (B, H*W, C_out) to match the expected PyTorch layout.
    return output.reshape({B, spatial, C_out});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("depthwise_separable_conv_forward", &depthwise_separable_conv_forward,
          "Depthwise separable convolution forward");
} 