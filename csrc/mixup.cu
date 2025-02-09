/*
Implements a fused kernel for applying MixUp to a batch of images or categorical labels.
*/

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

__device__ float sample_beta(curandState *state, const float alpha) {
    const float divisor = 1.0f / alpha;
    float u = powf(curand_uniform(state), divisor);
    float v = powf(curand_uniform(state), divisor);
    return u / (u + v);
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
__global__ void cross_entropy_mixup_fwd_kernel(const scalar_t *__restrict__ logits, const int64_t *__restrict__ labels,
                                               scalar_t *__restrict__ output, const float mixup_prob,
                                               const float mixup_alpha, const int64_t batch_size,
                                               const int64_t num_classes, const int64_t seed) {
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    // Initialize RNG state for this batch element
    curandState batch_state;
    curand_init(seed + batch_idx, 0, 0, &batch_state);

    // Determine if we apply MixUp
    const bool apply_mixup = curand_uniform(&batch_state) < mixup_prob;

    // Get the label for current batch
    const int label_idx = static_cast<int>(labels[batch_idx]);
    scalar_t loss = 0.0f;

    // Calculate softmax denominator first
    scalar_t max_val = logits[batch_idx * num_classes];
    for (int c = 1; c < num_classes; c++) {
        max_val = max(max_val, logits[batch_idx * num_classes + c]);
    }

    scalar_t sum_exp = 0.0f;
    for (int c = 0; c < num_classes; c++) {
        sum_exp += exp(logits[batch_idx * num_classes + c] - max_val);
    }
    const scalar_t log_sum_exp = log(sum_exp);

    if (apply_mixup) {
        // Generate MixUp weight using same method as image MixUp
        const scalar_t weight = sample_beta(&batch_state, mixup_alpha);

        // Get the next batch's label for mixing
        const int mixup_batch_idx = (batch_idx + 1) % batch_size;
        const int mixup_label_idx = static_cast<int>(labels[mixup_batch_idx]);

        // Calculate cross entropy with mixed labels
        for (int c = 0; c < num_classes; c++) {
            const scalar_t target_prob = (c == label_idx) ? weight : ((c == mixup_label_idx) ? (1.0f - weight) : 0.0f);
            if (target_prob > 0) {
                const scalar_t log_softmax_val = logits[batch_idx * num_classes + c] - max_val - log_sum_exp;
                loss -= target_prob * log_softmax_val;
            }
        }
    } else {
        // Standard cross entropy without MixUp
        const scalar_t log_softmax_val = logits[batch_idx * num_classes + label_idx] - max_val - log_sum_exp;
        loss = -log_softmax_val;
    }

    output[batch_idx] = loss;
}

torch::Tensor cross_entropy_mixup_fwd(const torch::Tensor &logits, const torch::Tensor &labels, const float mixup_prob,
                                      const float mixup_alpha, const int64_t seed) {
    TORCH_CHECK(logits.dim() == 2, "Logits must be 2D tensor");
    TORCH_CHECK(labels.dim() == 1, "Labels must be 1D tensor");
    TORCH_CHECK(logits.size(0) == labels.size(0), "Batch sizes must match");
    TORCH_CHECK(labels.scalar_type() == torch::kInt64, "Labels must be torch.long");

    const auto batch_size = logits.size(0);
    const auto num_classes = logits.size(1);
    auto output = torch::empty({batch_size}, logits.options());

    AT_DISPATCH_FLOATING_TYPES(
        logits.scalar_type(), "cross_entropy_mixup_fwd", ([&] {
            int min_grid_size;
            int block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                               (void *)cross_entropy_mixup_fwd_kernel<scalar_t>, 0, 0);
            const int grid_size = (batch_size + block_size - 1) / block_size;

            cross_entropy_mixup_fwd_kernel<scalar_t><<<grid_size, block_size>>>(
                logits.data_ptr<scalar_t>(), labels.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), mixup_prob,
                mixup_alpha, batch_size, num_classes, seed);
        }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mixup", &mixup, "MixUp operation (CUDA)");
    m.def("cross_entropy_mixup_fwd", &cross_entropy_mixup_fwd, "Cross-entropy with MixUp (CUDA)");
}
