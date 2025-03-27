#include <torch/extension.h>

#define REDUCTION_SCALE 0.5

template <typename T>
using Size = std::vector<T>;
template <typename T>
using SizeList = std::vector<Size<T>>;

template <typename T>
std::vector<T> tokenized_size(const std::vector<T>& input_size, const std::vector<T>& patch_size) {
    std::vector<T> tokenized_size(input_size.size());
    for (size_t i = 0; i < input_size.size(); i++) {
        tokenized_size[i] = input_size[i] / patch_size[i];
    }
    return tokenized_size;
}

template <typename T>
std::vector<std::vector<T>> tokenized_size_foreach(const std::vector<std::vector<T>>& input_sizes, const std::vector<std::vector<T>>& patch_sizes) {
    std::vector<std::vector<T>> tokenized_sizes(input_sizes.size());
    for (size_t i = 0; i < input_sizes.size(); i++) {
        tokenized_sizes[i] = tokenized_size(input_sizes[i], patch_sizes[i]);
    }
    return tokenized_sizes;
}

template <typename T>
T token_count(const std::vector<T>& input_size, const std::vector<T>& patch_size) {
    auto sizes = tokenized_size(input_size, patch_size);
    if (sizes.size() == 0) {
        return 0;
    }
    T count = 1;
    for (size_t i = 0; i < sizes.size(); i++) {
        count *= sizes[i];
    }
    return count;
}

template <typename T>
T token_count_foreach(const std::vector<std::vector<T>>& input_sizes, const std::vector<std::vector<T>>& patch_sizes) {
    T total = 0;
    for (size_t i = 0; i < input_sizes.size(); i++) {
        total += token_count(input_sizes[i], patch_sizes[i]);
    }
    return total;
}

template <typename T>
std::vector<T> size_at_scale(const std::vector<T>& input_size, const double scale) {
    std::vector<T> new_sizes(input_size.size());
    for (size_t i = 0; i < input_size.size(); i++) {
        new_sizes[i] = input_size[i] * scale;
    }
    return new_sizes;
}

template <typename T>
std::vector<T> tokenized_size_at_scale(const std::vector<T>& input_size, const std::vector<T>& patch_size, const double scale) {
    std::vector<T> new_sizes(input_size.size());
    for (size_t i = 0; i < input_size.size(); i++) {
        new_sizes[i] = input_size[i] * scale;
    }
    return tokenized_size(new_sizes, patch_size);
}

template <typename T>
T token_count_at_scale(const std::vector<T>& input_size, const std::vector<T>& patch_size, const double scale) {
    std::vector<T> new_sizes(input_size.size());
    for (size_t i = 0; i < input_size.size(); i++) {
        new_sizes[i] = input_size[i] * scale;
    }
    return token_count(new_sizes, patch_size);
}

std::tuple<torch::Tensor, torch::Tensor, int32_t> pack(const std::vector<torch::Tensor>& tensors) {
    TORCH_CHECK(tensors.size() > 0, "tensors must not be empty");
    TORCH_CHECK(tensors[0].dim() == 2, "tensors must be 2D");
    std::vector<int32_t> seq_lens(tensors.size() + 1);
    int32_t max_seq_len = 0;
    for (size_t i = 0; i < tensors.size(); i++) {
        seq_lens[i + 1] = tensors[i].size(0);
        max_seq_len = std::max(max_seq_len, seq_lens[i + 1]);
    }
    auto packed = torch::cat(tensors);
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(packed.device());
    auto cu_seq_lens = torch::tensor(seq_lens, options).cumsum_(0);
    return std::make_tuple(packed, cu_seq_lens, max_seq_len);
}

std::vector<torch::Tensor> unpack(const torch::Tensor& packed, const torch::Tensor& cu_seq_lens) {
    TORCH_CHECK(packed.dim() == 2, "packed must be 2D");
    TORCH_CHECK(cu_seq_lens.dim() == 1, "cu_seq_lens must be 1D");
    std::vector<torch::Tensor> tensors(cu_seq_lens.size(0) - 1);
    for (size_t i = 0; i < cu_seq_lens.size(0) - 1; i++) {
        tensors[i] = packed.slice(0, cu_seq_lens[i].item<int32_t>(), cu_seq_lens[i + 1].item<int32_t>());
    }
    return tensors;
}

template <typename T>
inline Size<T> binary_search_for_scale(
    const Size<T>& current_size, 
    const Size<T>& patch_size, 
    const double drop_rate, 
    const size_t budget
) {
    TORCH_CHECK(current_size.size() == patch_size.size(), "current_size and patch_size must have the same number of dimensions");
    TORCH_CHECK(drop_rate >= 0.0 && drop_rate <= 1.0, "drop_rate must be between 0.0 and 1.0");
    double min_scale = REDUCTION_SCALE;
    double max_scale = 1.0;
    double mid_scale;
    auto current_token_count = token_count(current_size, patch_size) * (1.0 - drop_rate);
    auto new_size = current_size;

    while (current_token_count > budget) {
        // Compute the new size (as a multiple of the patch size) at the midpoint
        mid_scale = (min_scale + max_scale) / 2.0;
        new_size = size_at_scale(current_size, mid_scale);
        for (size_t i = 0; i < new_size.size(); i++) {
            new_size[i] = (new_size[i] / patch_size[i]) * patch_size[i];
            if (new_size[i] < patch_size[i]) {
                new_size[i] = patch_size[i];
            }
        }
        
        // Add an escape in case we're retrying the same scale.
        auto new_token_count = token_count(new_size, patch_size) * (1.0 - drop_rate);
        if (new_token_count == current_token_count) {
            break;
        }
        
        // Update the current token count and scale.
        current_token_count = new_token_count;
        if (new_token_count > budget) {
            max_scale = mid_scale;
        } else {
            min_scale = mid_scale;
        }
    }

    // If we're within the budget, return the new size.
    if (current_token_count <= budget) {
        return new_size;
    }
    // Otherwise, return the size at scale REDUCTION_SCALE.
    else {
        return size_at_scale(new_size, REDUCTION_SCALE);
    }
}


template <typename T>
SizeList<T> calculate_sizes_for_budget(
    const SizeList<T>& input_sizes, const SizeList<T>& patch_sizes, 
    const std::vector<double>& drop_rates, const size_t budget
) {
    // Initialize starting sizes and token counts (accounting for drop tokens)
    int64_t current_token_count = 0;
    for (size_t i = 0; i < input_sizes.size(); i++) {
        current_token_count += token_count(input_sizes[i], patch_sizes[i]) * (1.0 - drop_rates[i]);
    }
    if (current_token_count <= budget) {
        return input_sizes;
    }
    SizeList<T> current_sizes(input_sizes.size());
    for (size_t i = 0; i < input_sizes.size(); i++) {
        current_sizes[i] = input_sizes[i];
    }

    // Each iteration we select the largest token count input and scale it by 0.5.
    // Once we drop below the budget, we then see if we can choose a better scale for the last input between [0.5, 1.0]
    // that is closer to the budget.
    const size_t MAX_ITERATIONS = input_sizes.size() * 10;
    std::vector<bool> downsampled_in_round(input_sizes.size(), false);
    for (size_t iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
        // If every input has already been downsampled in this round, we need to start a new round.
        if (std::all_of(downsampled_in_round.begin(), downsampled_in_round.end(), [](bool b) { return b; })) {
            downsampled_in_round = std::vector<bool>(input_sizes.size(), false);
        }

        // Determine which inputs are eligible to be downsampled.
        std::vector<size_t> eligible_indices;
        for (size_t i = 0; i < input_sizes.size(); i++) {
            if (!downsampled_in_round[i]) {
                eligible_indices.push_back(i);
            }
        }

        // Find the largest eligible input and mark it as downsampled.
        size_t largest_idx = *std::max_element(eligible_indices.begin(), eligible_indices.end(), [&](size_t a, size_t b) {
            return token_count(current_sizes[a], patch_sizes[a]) * (1.0 - drop_rates[a]) < token_count(current_sizes[b], patch_sizes[b]) * (1.0 - drop_rates[b]);
        });
        downsampled_in_round[largest_idx] = true;


        // Calculate the new token count after downsampling the largest eligible input.
        current_sizes[largest_idx] = size_at_scale(current_sizes[largest_idx], REDUCTION_SCALE);
        current_token_count = 0;
        for (size_t i = 0; i < input_sizes.size(); i++) {
            current_token_count += token_count(current_sizes[i], patch_sizes[i]) * (1.0 - drop_rates[i]);
        }

        // If the new token count is within the budget, we need to check if we can do better.
        if (current_token_count <= budget) {
            int64_t budget_for_scale = budget;
            for (size_t i = 0; i < input_sizes.size(); i++) {   
                if (i != largest_idx) {
                    budget_for_scale -= token_count(current_sizes[i], patch_sizes[i]) * (1.0 - drop_rates[i]);
                }
            }

            const auto original_size = size_at_scale(current_sizes[largest_idx], 1 / REDUCTION_SCALE);
            current_sizes[largest_idx] = binary_search_for_scale(original_size, patch_sizes[largest_idx], drop_rates[largest_idx], budget_for_scale);
            return current_sizes;
        }
    }

    return current_sizes;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tokenized_size", &tokenized_size<size_t>, "Get the tokenized size of the input");
    m.def("tokenized_size_foreach", &tokenized_size_foreach<size_t>, "Get the tokenized size of the input for each input");
    m.def("token_count", &token_count<size_t>, "Get the token count of the input");
    m.def("token_count_foreach", &token_count_foreach<size_t>, "Get the token count of the input for each input");
    m.def("size_at_scale", &size_at_scale<size_t>, "Get the size of the input at a given scale");
    m.def("tokenized_size_at_scale", &tokenized_size_at_scale<size_t>, "Get the tokenized size of the input at a given scale");
    m.def("token_count_at_scale", &token_count_at_scale<size_t>, "Get the token count of the input at a given scale");
    m.def("pack", &pack, "Pack a list of tensors into a single tensor");
    m.def("unpack", &unpack, "Unpack a single tensor into a list of tensors");
    m.def("binary_search_for_scale", &binary_search_for_scale<size_t>, "Binary search for the optimal scale");
    m.def("calculate_sizes_for_budget", &calculate_sizes_for_budget<size_t>, "Calculate the sizes for a given budget");
}
