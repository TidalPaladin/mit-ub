#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <torch/extension.h>

#define WARP_SIZE 32

template <typename scalar_t>
__global__ void invert_kernel(const scalar_t *__restrict__ input, scalar_t *__restrict__ output,
                              const float invert_prob, const int64_t batch_size, const int64_t seq_len,
                              const int64_t seed) {
    const int batch_idx = blockIdx.y;
    const int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx >= seq_len || batch_idx >= batch_size) return;

    // Load input value
    const int64_t idx = batch_idx * seq_len + seq_idx;
    scalar_t val = input[idx];

    // Decide if this batch entry should be inverted
    curandState batch_state;
    curand_init(seed + batch_idx, 0, 0, &batch_state);
    const bool apply_invert = curand_uniform(&batch_state) < invert_prob;
    if (!apply_invert) {
        output[idx] = val;
        return;
    } else {
        output[idx] = 1.0f - val;
    }
}

torch::Tensor invert(const torch::Tensor &input, const float invert_prob, const int64_t seed) {
    // Prepare output and infer dimensions
    auto output = torch::empty_like(input);
    const int64_t batch_size = input.size(0);
    const int64_t seq_len = input.numel() / batch_size;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "invert", ([&] {
            int min_grid_size;
            int block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)invert_kernel<scalar_t>, 0, 0);
            const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
            const dim3 blocks(blocks_x, batch_size);

            invert_kernel<scalar_t><<<blocks, block_size>>>(input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
                                                            invert_prob, batch_size, seq_len, seed);
        }));

    return output;
}

void invert_(torch::Tensor &input, const float invert_prob, const int64_t seed) {
    // Prepare output and infer dimensions
    const int64_t batch_size = input.size(0);
    const int64_t seq_len = input.numel() / batch_size;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "invert", ([&] {
            int min_grid_size;
            int block_size;
            cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, (void *)invert_kernel<scalar_t>, 0, 0);
            const unsigned int blocks_x = (seq_len + block_size - 1) / block_size;
            const dim3 blocks(blocks_x, batch_size);

            invert_kernel<scalar_t><<<blocks, block_size>>>(input.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
                                                            invert_prob, batch_size, seq_len, seed);
        }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("invert", &invert, "Invert operation");
    m.def("invert_", &invert_, "Invert operation in-place");
}
