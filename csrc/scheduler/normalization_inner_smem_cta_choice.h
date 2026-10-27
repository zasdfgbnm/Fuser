// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace nvfuser::normalization_inner::non_tma::smem_cta_choice {

struct Buffer {
  std::string tv;
  int64_t logical_bits;
  int64_t dtype_bits;
};

struct Input {
  int64_t reduction_numel;
  int64_t vector_width;
  int64_t max_threads_per_block;
  int64_t warp_size;
  int64_t persistent_buffer_bits;
  int64_t smem_cap_bits;
  int64_t workspace_bits;
  int64_t static_reserve_bits;
  int64_t alignment_bits;
  int64_t max_vector_bits;
  std::vector<Buffer> buffers;
};

struct Plan {
  int64_t seed;
  int64_t batch;
  int64_t actual_threads;
  int64_t padded_threads;
  int64_t roundup_overhead_bits;
  int64_t available_persistent_bits;
  int64_t required_smem_bits;
  bool legal;
};

struct Menu {
  std::vector<Plan> geometry;
  std::vector<Plan> candidates;

  bool hasNativeSeed(int64_t max_threads_per_block) const {
    return !candidates.empty() &&
        candidates.front().seed == max_threads_per_block;
  }
};

inline int64_t ceilDivPositive(int64_t numerator, int64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

inline std::optional<int64_t> narrow(__int128 value) {
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(value);
}

inline std::optional<int64_t> alignBits(int64_t bits, int64_t alignment_bits) {
  if (alignment_bits < 1 ||
      (alignment_bits & (alignment_bits - 1)) != 0) {
    return std::nullopt;
  }
  auto adjusted = narrow(static_cast<__int128>(bits) + alignment_bits - 1);
  if (!adjusted) {
    return std::nullopt;
  }
  return *adjusted & ~(alignment_bits - 1);
}

// Mirrors roundUpSharedMemory and sharedMemoryRoundUpOverheadBit with injected
// resource facts. The ordered buffer list preserves each tensor's logical
// size and storage dtype, which can differ for an upcast buffer.
inline std::optional<Plan> preview(const Input& input, int64_t seed) {
  if (input.reduction_numel < 1 || input.vector_width < 1 ||
      input.reduction_numel % input.vector_width != 0 ||
      input.max_threads_per_block < 1 || input.warp_size < 1 ||
      seed < input.warp_size || seed > input.max_threads_per_block ||
      input.persistent_buffer_bits < 0 || input.smem_cap_bits < 0 ||
      input.workspace_bits < 0 || input.static_reserve_bits < 0 ||
      input.max_vector_bits < 1) {
    return std::nullopt;
  }
  const int64_t after_vectorize =
      input.reduction_numel / input.vector_width;
  const int64_t batch = ceilDivPositive(after_vectorize, seed);
  const int64_t actual_threads = ceilDivPositive(after_vectorize, batch);
  auto padded = narrow(
      static_cast<__int128>(ceilDivPositive(actual_threads, input.warp_size)) *
      input.warp_size);
  if (!padded || actual_threads < 1 || *padded > seed ||
      *padded > input.max_threads_per_block) {
    return std::nullopt;
  }

  __int128 unaligned_overhead = 0;
  for (const auto& buffer : input.buffers) {
    if (buffer.logical_bits < 0 || buffer.dtype_bits < 1) {
      return std::nullopt;
    }
    const int64_t dim_size = buffer.logical_bits / buffer.dtype_bits;
    const int64_t max_vector_factor =
        input.max_vector_bits / buffer.dtype_bits;
    int64_t max_rounded_bits = 0;
    for (int64_t factor = 1; factor <= max_vector_factor;) {
      if (dim_size % factor == 0) {
        const int64_t after_vect = dim_size / factor;
        const int64_t threads =
            std::max(ceilDivPositive(after_vect, batch), *padded);
        const auto batch_factor =
            narrow(static_cast<__int128>(batch) * factor);
        if (!batch_factor) {
          return std::nullopt;
        }
        const auto elements =
            narrow(static_cast<__int128>(*batch_factor) * threads);
        if (!elements) {
          return std::nullopt;
        }
        const auto rounded = narrow(
            static_cast<__int128>(*elements) * buffer.dtype_bits);
        if (!rounded) {
          return std::nullopt;
        }
        max_rounded_bits = std::max(max_rounded_bits, *rounded);
      }
      if (factor > max_vector_factor / 2) {
        break;
      }
      factor *= 2;
    }
    auto aligned = alignBits(max_rounded_bits, input.alignment_bits);
    if (!aligned) {
      return std::nullopt;
    }
    unaligned_overhead += static_cast<__int128>(*aligned) -
        buffer.logical_bits;
  }
  auto raw_overhead = narrow(unaligned_overhead);
  if (!raw_overhead) {
    return std::nullopt;
  }
  auto overhead = alignBits(*raw_overhead, input.alignment_bits);
  if (!overhead) {
    return std::nullopt;
  }
  auto available = narrow(
      static_cast<__int128>(input.smem_cap_bits) - input.workspace_bits -
      *overhead - input.static_reserve_bits);
  auto required = narrow(
      static_cast<__int128>(input.persistent_buffer_bits) +
      input.workspace_bits + *overhead + input.static_reserve_bits);
  if (!available || !required) {
    return std::nullopt;
  }
  return Plan{
      seed,
      batch,
      actual_threads,
      *padded,
      *overhead,
      *available,
      *required,
      input.persistent_buffer_bits <= *available};
}

inline Menu enumerate(const Input& input) {
  Menu menu;
  if (input.max_threads_per_block < 1 || input.warp_size < 1) {
    return menu;
  }
  for (int64_t seed = input.max_threads_per_block;
       seed >= input.warp_size;
       seed /= 2) {
    const auto plan = preview(input, seed);
    if (!plan) {
      continue;
    }
    menu.geometry.push_back(*plan);
    if (plan->legal &&
        std::none_of(
            menu.candidates.begin(),
            menu.candidates.end(),
            [&](const auto& existing) { return existing.batch == plan->batch; })) {
      menu.candidates.push_back(*plan);
    }
  }
  return menu;
}

} // namespace nvfuser::normalization_inner::non_tma::smem_cta_choice
