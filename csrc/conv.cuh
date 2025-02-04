#ifndef CONV_CUH
#define CONV_CUH

#include <torch/extension.h>

/**
 * @brief Depthwise separable convolution forward function.
 *
 * @param input Input tensor of shape (B, H*W, C).
 * @param weight_dw Depthwise weights of shape (C, kH, kW).
 * @param bias_dw Depthwise bias of shape (C).
 * @param weight_pw Pointwise weights of shape (C, C_out).
 * @param bias_pw Pointwise bias of shape (C_out).
 * @param H Height of the feature map.
 * @param W Width of the feature map.
 * @param padH Padding along the height.
 * @param padW Padding along the width.
 * @return torch::Tensor Output tensor of shape (B, H*W, C_out).
 */
torch::Tensor depthwise_separable_conv_forward(
    torch::Tensor input,
    torch::Tensor weight_dw,
    torch::Tensor bias_dw,
    torch::Tensor weight_pw,
    torch::Tensor bias_pw,
    int H, int W, int padH, int padW
);

#endif 