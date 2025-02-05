#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

template <typename scalar_t>
__global__ void fused_noise_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const float* __restrict__ uniform_noise_params,     // [min, max]
    const float* __restrict__ multiplicative_params,    // [min, max] 
    const float* __restrict__ salt_pepper_params,       // [prob, min, max]
    const float uniform_prob,
    const float multiplicative_prob,
    const float salt_pepper_prob,
    const bool clip,
    const int64_t numel,
    const int64_t seed
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) return;

    // Initialize RNG state
    curandState state;
    curand_init(seed, idx, 0, &state);
    
    // Copy input to output
    scalar_t val = input[idx];
    
    // Apply uniform noise
    if (curand_uniform(&state) < uniform_prob) {
        const float center = (uniform_noise_params[0] + uniform_noise_params[1]) / 2.0f;
        const float min_range = uniform_noise_params[0] + 
            (center - uniform_noise_params[0]) * curand_uniform(&state);
        const float max_range = center + 
            (uniform_noise_params[1] - center) * curand_uniform(&state);
        const float noise = min_range + (max_range - min_range) * curand_uniform(&state);
        val += noise;
    }
    
    // Apply multiplicative noise
    if (curand_uniform(&state) < multiplicative_prob) {
        const float center = (multiplicative_params[0] + multiplicative_params[1]) / 2.0f;
        const float min_range = multiplicative_params[0] + 
            (center - multiplicative_params[0]) * curand_uniform(&state);
        const float max_range = center + 
            (multiplicative_params[1] - center) * curand_uniform(&state);
        const float noise = min_range + (max_range - min_range) * curand_uniform(&state);
        val *= noise;
    }
    
    // Apply salt and pepper noise
    if (curand_uniform(&state) < salt_pepper_prob) {
        const float pixel_prob = salt_pepper_params[1] + 
            (salt_pepper_params[2] - salt_pepper_params[1]) * curand_uniform(&state);
        if (curand_uniform(&state) < pixel_prob) {
            val = static_cast<scalar_t>(curand_uniform(&state) < 0.5f ? 0.0f : 1.0f);
        }
    }
    
    // Clip if requested
    if (clip) {
        val = min(max(val, scalar_t(0)), scalar_t(1));
    }
    
    output[idx] = val;
}

torch::Tensor fused_noise_cuda(
    const torch::Tensor& input,
    const torch::Tensor& uniform_noise_params,
    const torch::Tensor& multiplicative_params,
    const torch::Tensor& salt_pepper_params,
    const float uniform_prob,
    const float multiplicative_prob, 
    const float salt_pepper_prob,
    const bool clip,
    const int64_t seed
) {
    auto output = torch::empty_like(input);
    const int64_t numel = input.numel();
    const int threads = 256;
    const int blocks = (numel + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_noise_cuda", ([&] {
        fused_noise_kernel<scalar_t><<<blocks, threads>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            uniform_noise_params.data_ptr<float>(),
            multiplicative_params.data_ptr<float>(),
            salt_pepper_params.data_ptr<float>(),
            uniform_prob,
            multiplicative_prob,
            salt_pepper_prob,
            clip,
            numel,
            seed
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_noise", &fused_noise_cuda, "Fused noise operations (CUDA)");
}
