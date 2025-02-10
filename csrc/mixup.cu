/*
Implements a fused kernel for applying MixUp to a batch of images or categorical labels.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#define WARP_SIZE 32
#define UNKNOWN_LABEL -1

__device__ __forceinline__ float sample_beta(curandState *state, const float alpha) {
    const float divisor = 1.0f / alpha;
    float u = powf(curand_uniform(state), divisor);
    float v = powf(curand_uniform(state), divisor);
    return u / (u + v);
}

template <typename scalar_t>
__global__ void get_weights_kernel(scalar_t *__restrict__ output, const float mixup_prob, const float mixup_alpha,
                                   const int64_t batch_size, const int64_t seed) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;
    curandState batch_state;
    curand_init(seed + batch_idx, 0, 0, &batch_state);
    const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
    const scalar_t weight = sample_beta(&batch_state, mixup_alpha);
    if (!apply_mixup) {
        output[batch_idx] = 1.0f;
    } else {
        output[batch_idx] = weight;
    }
}

torch::Tensor get_weights(const int64_t batch_size, const float mixup_prob, const float mixup_alpha,
                          const int64_t seed) {
    auto output = torch::empty({batch_size}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "get_weights", ([&] {
                                   get_weights_kernel<scalar_t><<<batch_size, 1>>>(
                                       output.data_ptr<scalar_t>(), mixup_prob, mixup_alpha, batch_size, seed);
                               }));
    return output;
}

template <typename scalar_t>
__global__ void mixup_kernel(const scalar_t *__restrict__ input, scalar_t *__restrict__ output, const float mixup_prob,
                             const float mixup_alpha, const int64_t batch_size, const int64_t seq_len,
                             const int64_t seed) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= seq_len || batch_idx >= batch_size) return;

    // Load input value
    const int64_t idx = batch_idx * seq_len + seq_idx;
    scalar_t val = input[idx];

    // Decide if MixUp should be applied to this batch entry
    curandState batch_state;
    curand_init(seed + batch_idx, 0, 0, &batch_state);
    const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;
    if (!apply_mixup) {
        output[idx] = val;
        return;
    }

    // Generate a mixup weight for this batch entry
    const scalar_t weight = sample_beta(&batch_state, mixup_alpha);

    // Load the value to be mixed
    const int64_t mixup_batch_idx = (batch_idx + 1) % batch_size;
    const int64_t mixup_idx = mixup_batch_idx * seq_len + seq_idx;
    scalar_t mixup_val = input[mixup_idx];

    // Apply the mixup weight
    // w * x + (1 - w) * y = w * (x - y) + y
    val = __fmaf_rn(weight, val - mixup_val, mixup_val);

    output[idx] = val;
}

torch::Tensor mixup(const torch::Tensor &input, const float mixup_prob, const float mixup_alpha, const int64_t seed) {
    // Prepare output and infer dimensions
    auto output = torch::empty_like(input);
    const int64_t batch_size = input.size(0);
    const int64_t seq_len = input.numel() / batch_size;

    AT_DISPATCH_FLOATING_TYPES(
        input.scalar_type(), "mixup", ([&] {
            int min_grid_size;
            int block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)mixup_kernel<scalar_t>, 0, 0);
            const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
            const dim3 blocks(blocks_x, batch_size);

            mixup_kernel<scalar_t><<<blocks, block_size>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                                           mixup_prob, mixup_alpha, batch_size, seq_len, seed);
        }));

    return output;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_max(scalar_t val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_sum(scalar_t val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void cross_entropy_mixup_fwd_kernel(const scalar_t *__restrict__ logits, const int64_t *__restrict__ labels,
                                               scalar_t *__restrict__ output, const float mixup_prob,
                                               const float mixup_alpha, const int64_t batch_size,
                                               const int64_t num_classes, const int64_t seed) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    // Initialize RNG state for this batch element
    curandState batch_state;
    curand_init(seed + batch_idx, 0, 0, &batch_state);

    // Determine if we apply MixUp
    const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;

    // Get the label for current batch
    const int label_idx = static_cast<int>(labels[batch_idx]);
    scalar_t loss = 0.0f;

    // Check for unknown label, set loss to -1
    if (label_idx == UNKNOWN_LABEL) {
        output[batch_idx] = -1.0f;
        return;
    }

    // Softmax using online softmax trick
    scalar_t max_val = -INFINITY;
    scalar_t sum_exp = 0.0f;
    for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
        scalar_t logit = logits[batch_idx * num_classes + c];
        scalar_t old_max = max_val;
        max_val = fmaxf(old_max, warp_max(logit));
        max_val = __shfl_sync(0xffffffff, max_val, 0);
        scalar_t update = warp_sum(expf(logit - max_val));
        update = __shfl_sync(0xffffffff, update, 0);
        sum_exp = __fma_rn(sum_exp, expf(old_max - max_val), update);
    }
    const scalar_t log_sum_exp = logf(sum_exp);

    if (apply_mixup) {
        // Generate MixUp weight using same method as image MixUp
        const scalar_t weight = sample_beta(&batch_state, mixup_alpha);

        // Get the next batch's label for mixing
        const int mixup_batch_idx = (batch_idx + 1) % batch_size;
        const int mixup_label_idx = static_cast<int>(labels[mixup_batch_idx]);
        if (mixup_label_idx == UNKNOWN_LABEL) {
            output[batch_idx] = -1.0f;
            return;
        }

        // Original label contribution
        const scalar_t log_softmax_val_orig = logits[batch_idx * num_classes + label_idx] - max_val - log_sum_exp;
        loss -= weight * log_softmax_val_orig;

        // Mixed label contribution
        const scalar_t log_softmax_val_mix = logits[batch_idx * num_classes + mixup_label_idx] - max_val - log_sum_exp;
        loss -= (1.0f - weight) * log_softmax_val_mix;
    } else {
        // Standard cross entropy without MixUp
        const scalar_t log_softmax_val = logits[batch_idx * num_classes + label_idx] - max_val - log_sum_exp;
        loss = -log_softmax_val;
    }

    if (threadIdx.x == 0) {
        output[batch_idx] = loss;
    }
}

torch::Tensor cross_entropy_mixup_fwd(const torch::Tensor &logits, const torch::Tensor &labels, const float mixup_prob,
                                      const float mixup_alpha, const int64_t seed) {
    TORCH_CHECK(logits.dim() == 2, "Logits must be 2D tensor");
    TORCH_CHECK(labels.dim() == 1, "Labels must be 1D tensor");
    TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
    TORCH_CHECK(labels.scalar_type() == torch::kInt64, "Labels must be torch.long");

    const auto batch_size = logits.size(0);
    const auto num_classes = logits.size(1);
    auto output = torch::zeros({batch_size}, logits.options());

    const size_t block_size = WARP_SIZE;
    const size_t num_blocks = batch_size;
    const dim3 blocks(num_blocks);

    AT_DISPATCH_FLOATING_TYPES(logits.scalar_type(), "cross_entropy_mixup_fwd", ([&] {
                                   cross_entropy_mixup_fwd_kernel<scalar_t>
                                       <<<blocks, block_size>>>(logits.data_ptr<scalar_t>(), labels.data_ptr<int64_t>(),
                                                                output.data_ptr<scalar_t>(), mixup_prob, mixup_alpha,
                                                                batch_size, num_classes, seed);
                               }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_weights", &get_weights, "Get MixUp weights (CUDA)");
    m.def("mixup", &mixup, "MixUp operation (CUDA)");
    m.def("cross_entropy_mixup_fwd", &cross_entropy_mixup_fwd, "Cross-entropy with MixUp (CUDA)");
}
