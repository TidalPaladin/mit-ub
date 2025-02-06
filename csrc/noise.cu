/*
Implements a fused kernel for applying various noise types to a batch of images.
Each noise type is either applied or not applied to each entry of the batch independently.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

// Initialize curand states for each thread
__global__ void setup_noise_seq_curand(curandState *states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

template <typename scalar_t>
__global__ void fused_noise_kernel(const scalar_t *__restrict__ input, scalar_t *__restrict__ output,
                                   const float uniform_noise_min, const float uniform_noise_max,
                                   const float multiplicative_min, const float multiplicative_max,
                                   const float salt_pepper_min, const float salt_pepper_max, const float uniform_prob,
                                   const float multiplicative_prob, const float salt_pepper_prob, const bool clip,
                                   const int64_t batch_size, const int64_t seq_len, const int64_t seed) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= seq_len || batch_idx >= batch_size) return;

    // Load input value
    const int64_t idx = batch_idx * seq_len + seq_idx;
    scalar_t val = input[idx];

    // Compute masks for each noise type for this batch entry
    curandState batch_state;
    curand_init(seed + batch_idx, 0, 0, &batch_state);
    const scalar_t uniform_mask = curand_uniform(&batch_state) < uniform_prob;
    const scalar_t mult_mask = curand_uniform(&batch_state) < multiplicative_prob;
    const scalar_t sp_mask = curand_uniform(&batch_state) < salt_pepper_prob;

    // Initialize pointwise-unique random state
    curandState seq_state;
    const unsigned long seq_seed = seed + batch_idx * seq_len + seq_idx;
    curand_init(seq_seed, 0, 0, &seq_state);

    // Compute multiplicative factor (will be 1.0 if not applied)
    const float mult_center = (multiplicative_min + multiplicative_max) / 2.0f;
    const float mult_min_range = multiplicative_min + (mult_center - multiplicative_min) * curand_uniform(&seq_state);
    const float mult_max_range = mult_center + (multiplicative_max - mult_center) * curand_uniform(&seq_state);
    const float mult_noise = mult_min_range + (mult_max_range - mult_min_range) * curand_uniform(&seq_state);
    const float mult_factor = __fmaf_rn(mult_noise - 1.0f, mult_mask, 1.0f);

    // Compute additive factor (will be 0.0 if not applied)
    const float unif_center = (uniform_noise_min + uniform_noise_max) / 2.0f;
    const float unif_min_range = uniform_noise_min + (unif_center - uniform_noise_min) * curand_uniform(&seq_state);
    const float unif_max_range = unif_center + (uniform_noise_max - unif_center) * curand_uniform(&seq_state);
    const float unif_noise = unif_min_range + (unif_max_range - unif_min_range) * curand_uniform(&seq_state);
    const float add_factor = unif_noise * uniform_mask;

    // Compute salt & pepper value (will not be used if not applied)
    const float sp_prob = salt_pepper_min + (salt_pepper_max - salt_pepper_min) * curand_uniform(&seq_state);
    const float sp_trigger = curand_uniform(&seq_state) < sp_prob;
    const float sp_value = curand_uniform(&seq_state) < 0.5f ? 0.0f : 1.0f;

    // Fused multiply-add for noise application
    val = __fmaf_rn(val, mult_factor, add_factor);

    // Blend with salt & pepper value if applied
    const float sp_blend = sp_mask * sp_trigger;
    val = val * (1.0f - sp_blend) + sp_value * sp_blend;

    // Clip using min/max intrinsics if requested
    if (clip) {
        val = __saturatef(val);
    }

    output[idx] = val;
}

torch::Tensor fused_noise_cuda(const torch::Tensor &input, const float uniform_noise_min, const float uniform_noise_max,
                               const float multiplicative_min, const float multiplicative_max,
                               const float salt_pepper_min, const float salt_pepper_max, const float uniform_prob,
                               const float multiplicative_prob, const float salt_pepper_prob, const bool clip,
                               const int64_t seed, const bool inplace) {
    // Prepare output and infer dimensions
    auto output = inplace ? input : torch::empty_like(input);
    const int64_t batch_size = input.size(0);
    const int64_t seq_len = input.numel() / batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "fused_noise_cuda", ([&] {
            // Calculate grid dimensions
            int min_grid_size;
            int block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)fused_noise_kernel<scalar_t>, 0, 0);
            const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
            const dim3 blocks(blocks_x, batch_size);

            fused_noise_kernel<scalar_t><<<blocks, block_size>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), uniform_noise_min, uniform_noise_max,
                multiplicative_min, multiplicative_max, salt_pepper_min, salt_pepper_max, uniform_prob,
                multiplicative_prob, salt_pepper_prob, clip, batch_size, seq_len, seed);
        }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_noise", &fused_noise_cuda, "Fused noise operations (CUDA)"); }
