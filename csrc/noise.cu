/*
Implements a fused kernel for applying various noise types to a batch of images.
Each noise type is either applied or not applied to each entry of the batch independently.
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

// Initialize weights for each noise operation for each batch entry
__global__ void setup_noise_batch_weights(float *__restrict__ weights, const int64_t batch_size, const int64_t seed,
                                          const float uniform_prob, const float multiplicative_prob,
                                          const float salt_pepper_prob) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed, id, 0, &state);

    const float uniform = curand_uniform(&state) < uniform_prob;
    const float multiplicative = curand_uniform(&state) < multiplicative_prob;
    const float salt_pepper = curand_uniform(&state) < salt_pepper_prob;

    weights[id] = uniform;
    weights[id + batch_size] = multiplicative;
    weights[id + 2 * batch_size] = salt_pepper;
}

// Initialize curand states for each thread
__global__ void setup_noise_seq_curand(curandState *states, unsigned long seed) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, id, 0, &states[id]);
}

template <typename scalar_t>
__global__ void fused_noise_kernel(const scalar_t *__restrict__ input, scalar_t *__restrict__ output,
                                   const float *__restrict__ weights, const float uniform_noise_min,
                                   const float uniform_noise_max, const float multiplicative_min,
                                   const float multiplicative_max, const float salt_pepper_min,
                                   const float salt_pepper_max, const float uniform_prob,
                                   const float multiplicative_prob, const float salt_pepper_prob, const bool clip,
                                   const int64_t batch_size, const int64_t seq_len, curandState *states) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= seq_len || batch_idx >= batch_size) return;

    // Load input value
    const int64_t idx = batch_idx * seq_len + seq_idx;
    scalar_t val = input[idx];

    // Load the weights for this batch entry
    const float uniform_mask = weights[batch_idx];
    const float mult_mask = weights[batch_idx + batch_size];
    const float sp_mask = weights[batch_idx + 2 * batch_size];

    // Get random state for this thread
    curandState seq_state = states[seq_idx];
    skipahead(batch_idx, &seq_state);

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

    float *weights;
    curandState *states;

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "fused_noise_cuda", ([&] {
            // Calculate grid dimensions
            int min_grid_size;
            int block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)fused_noise_kernel<scalar_t>, 0, 0);
            const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
            const dim3 blocks(blocks_x, batch_size);

            // Initialize weights for each batch entry
            int block_size_batch_setup;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_batch_setup,
                                               (void *)setup_noise_batch_weights, 0, 0);
            const unsigned int blocks_b = (batch_size + block_size_batch_setup - 1) / block_size_batch_setup;
            cudaMalloc((void **)&weights, 3 * batch_size * sizeof(float));
            setup_noise_batch_weights<<<blocks_b, block_size_batch_setup>>>(weights, batch_size, seed, uniform_prob,
                                                                            multiplicative_prob, salt_pepper_prob);

            // Initialize curand states
            int block_size_seq_setup;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size_seq_setup, (void *)setup_noise_seq_curand, 0,
                                               0);
            const unsigned int blocks_s = (seq_len + block_size_seq_setup - 1) / block_size_seq_setup;
            cudaMalloc((void **)&states, blocks_s * block_size_seq_setup * sizeof(curandState));
            setup_noise_seq_curand<<<blocks_s, block_size_seq_setup>>>(states, seed);

            fused_noise_kernel<scalar_t><<<blocks, block_size>>>(
                input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), weights, uniform_noise_min, uniform_noise_max,
                multiplicative_min, multiplicative_max, salt_pepper_min, salt_pepper_max, uniform_prob,
                multiplicative_prob, salt_pepper_prob, clip, batch_size, seq_len, states);
        }));

    // Free memory
    cudaFree(weights);
    cudaFree(states);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("fused_noise", &fused_noise_cuda, "Fused noise operations (CUDA)"); }
