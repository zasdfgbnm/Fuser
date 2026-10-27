// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/reduction_non_tma.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "debug.h"
#include "instrumentation.h"
#include "multidevice/utils.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/mark_aliases.h"
#include "scheduler/reduction.h"
#include "scheduler/reduction_inner3d_iter_unroll_continuation.h"
#include "scheduler/reduction_inner_remainder_choice.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/domain_map.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"
#include "scheduler/vectorize_helper.h"

namespace nvfuser {
namespace reduction {
namespace non_tma {
namespace {

// Rounds x up to a power of 2 or a multiple of multiple
int64_t roundUpPow2OrMultipleOf(const int64_t x, const int64_t multiple) {
  auto round_up_pow2 = scheduler_utils::lastPow2(x);
  if (round_up_pow2 < x) {
    round_up_pow2 *= 2;
  }
  auto round_up_multiple =
      x % multiple == 0 ? x : x + (multiple - x % multiple);
  return std::max(std::min(round_up_multiple, round_up_pow2), (int64_t)1);
}

// Rounds x down to a power of 2 or a multiple of multiple, whichever is bigger
int64_t roundDownPow2OrMultipleOf(const int64_t x, const int64_t multiple) {
  auto round_down_pow2 = scheduler_utils::lastPow2(x);

  auto round_down_multiple = x % multiple == 0 ? x : x - x % multiple;
  return std::max(std::max(round_down_multiple, round_down_pow2), (int64_t)1);
}

int64_t clamp(const int64_t val, const int64_t min_val, const int64_t max_val) {
  return std::min(std::max(val, min_val), max_val);
}

// A short 2D reduction is wholly covered by its CTA x dimension. Rebuild the
// complete block and output-grid plan for an enlarged CTA target. The native
// remainder scan then has no work and RD3.grid has only g1; later output
// unrolling can only shrink the grid. Keep the original 128/256/floor menu for
// shapes outside this provable domain.
struct ShortInnerCtaPlan {
  int64_t block_x;
  int64_t block_y;
  int64_t padded_block_x;
  int64_t output_grid;
  int64_t physical_grid_x;
  bool split_grid_x;
};

std::optional<ShortInnerCtaPlan> shortInnerCtaPlan(
    int64_t reduction_numel,
    int64_t iteration_numel,
    int64_t vector_width,
    int64_t target_inner_unroll,
    int64_t target_threads_per_block,
    int64_t warp_size,
    int64_t max_threads_per_block,
    int64_t max_threads_dim_y) {
  if (reduction_numel < 1 || iteration_numel < 1 || vector_width < 1 ||
      reduction_numel % vector_width != 0 || target_inner_unroll < 1 ||
      warp_size != 32 || max_threads_per_block < 512 ||
      max_threads_per_block > 1024 || target_threads_per_block < 64 ||
      target_threads_per_block > max_threads_per_block ||
      max_threads_dim_y < 1) {
    return std::nullopt;
  }
  const int64_t after_vect = reduction_numel / vector_width;
  if (after_vect > 64 || after_vect > target_threads_per_block) {
    return std::nullopt;
  }

  // Replaying the native seed: x=after_vect, inner_unroll=1, and the
  // reduction remainder is one. Neither the logical minimum warp target nor
  // the remainder Choice can change these values for this domain.
  const int64_t block_x = after_vect;
  const int64_t block_y =
      std::min(iteration_numel, target_threads_per_block / block_x);
  if (block_y < 1 || block_y > max_threads_dim_y) {
    return std::nullopt;
  }
  const bool pad_x =
      block_x > 16 && block_x * block_y < max_threads_per_block;
  const int64_t padded_x =
      pad_x ? ceilDiv(block_x, warp_size) * warp_size : block_x;
  if (padded_x * block_y > max_threads_per_block) {
    return std::nullopt;
  }
  const int64_t godim = positiveCeilDiv(iteration_numel, block_y);
  if (godim < 1 || scheduler_utils::x_grid_limit < 1) {
    return std::nullopt;
  }
  return ShortInnerCtaPlan{
      block_x,
      block_y,
      padded_x,
      godim,
      std::min(godim, scheduler_utils::x_grid_limit),
      godim > scheduler_utils::x_grid_limit};
}

// Reduce x, y, z until it's product is less than max value, reduce round robin
// starting with x
void reduceProductTo(int64_t& z, int64_t& y, int64_t& x, const int64_t max) {
  NVF_ERROR(max > 1);
  if (z * y * x > max) {
    z = scheduler_utils::safeDiv(z, 2);
  }
  if (z * y * x > max) {
    y = scheduler_utils::safeDiv(y, 2);
  }
  if (z * y * x > max) {
    x = scheduler_utils::safeDiv(x, 2);
  }
  if (z * y * x > max) {
    reduceProductTo(x, y, z, max);
  }
}

// Parallelization strategy:
// [] indicates optional
// Reduction dim: Serial, [BIDx or unroll], TIDx, Vect
// Iteration dim: Serial, BIDy, [TIDy], [unroll]
std::unique_ptr<ReductionParams> inner2dReductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const int64_t max_vectorize_factor,
    const bool has_mufu_computation) {
  // Get device properties
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = dev_prop->warpSize;
  const int64_t max_threads_per_block = dev_prop->maxThreadsPerBlock;
  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  const int64_t sm_count = dev_prop->multiProcessorCount;
  const int64_t target_threads_per_sm = max_threads_per_sm / 2;

  // RD3.width asks the local model for the performance width within the
  // layout-derived native maximum. RD3.unroll separately chooses the inner
  // unroll target, and RD3.cta chooses the CTA target before native
  // remainder and grid planning.
  const auto jev_mode = scheduler::jev::mode();
  int64_t target_vect_unroll = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    target_vect_unroll = reduction_scheduler_utils::getVectUnroll(
        max_dtype_size_bit_for_vectorization,
        max_vectorize_factor,
        n_tensor_inputs,
        target_threads_per_sm,
        has_mufu_computation);
  }

  int64_t selected_vect_factor = 0;
  if (jev_mode != scheduler::jev::Mode::Off) {
    NVF_ERROR(max_vectorize_factor > 0, "RD3.width requires a positive layout limit");
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t width = 1;; width *= 2) {
      options.push_back(
          {"w" + std::to_string(width),
           "Vectorize " + std::to_string(width) + " element(s) per thread"});
      if (width > max_vectorize_factor / 2) {
        break;
      }
    }
    // Preserve source min(lastPow2(target), layout_max) if a synthetic
    // non-power-of-two layout cap reaches this path.
    if (scheduler_utils::lastPow2(max_vectorize_factor) !=
        max_vectorize_factor) {
      options.push_back(
          {"w" + std::to_string(max_vectorize_factor),
           "Use the native layout width cap"});
    }

    int memory_clock_khz = 0;
    const int64_t memory_bus_bits = has_mufu_computation
        ? 0
        : static_cast<int64_t>(dev_prop->memoryBusWidth);
    if (!has_mufu_computation) {
      NVF_ERROR(
          cudaDeviceGetAttribute(
              &memory_clock_khz,
              cudaDevAttrMemoryClockRate,
              at::cuda::current_device()) == cudaSuccess,
          "RD3.width could not read CUDA memory clock");
    }
    std::string state = "reduction.inner_width.v1\n";
    state += "layout_max=" + std::to_string(max_vectorize_factor) + '\n';
    state += "max_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) + '\n';
    state += "n_tensor_inputs=" + std::to_string(n_tensor_inputs) + '\n';
    state += "has_mufu=" + std::to_string(has_mufu_computation ? 1 : 0) + '\n';
    state += "sm_count=" + std::to_string(sm_count) + '\n';
    state += "max_threads_per_sm=" +
        std::to_string(max_threads_per_sm) + '\n';
    state += "memory_bus_bits=" + std::to_string(memory_bus_bits) + '\n';
    state += "memory_clock_khz=" + std::to_string(memory_clock_khz) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "w" + std::to_string(std::min(
                     scheduler_utils::lastPow2(target_vect_unroll),
                     max_vectorize_factor))
        : std::string();
    const auto selected = scheduler::jev::chooseDecision(
        "RD3.width", state, options, native_choice);
    bool matched = false;
    for (const auto& option : options) {
      if (selected == option.id) {
        selected_vect_factor = std::stoll(selected.substr(1));
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "RD3.width response is outside native layout bounds");
  }
  // redu = [i-remainder, i-Unroll, TIDx, Vect]
  int64_t inner_unroll = 1, bdimx = 1, vect_factor = 1;
  auto getInnerRemainder = [&]() {
    return positiveCeilDiv(
        positiveCeilDiv(total_reduction_numel / vect_factor, bdimx),
        inner_unroll);
  };

  // Max vectorization factor
  vect_factor = jev_mode == scheduler::jev::Mode::Off
      ? std::min(
            scheduler_utils::lastPow2(target_vect_unroll),
            (int64_t)max_vectorize_factor)
      : selected_vect_factor;
  const int64_t four_warps = 4 * threads_per_warp;
  // Off preserves the native quotient, including values larger than the
  // reduction extent. Record uses it to identify the native menu choice.
  int64_t target_inner_unroll = jev_mode == scheduler::jev::Mode::Remote
      ? 0
      : target_vect_unroll / vect_factor;
  int64_t after_vect = total_reduction_numel / vect_factor;
  if (jev_mode != scheduler::jev::Mode::Off) {
    NVF_ERROR(after_vect >= 1, "RD3.unroll requires a nonempty reduction");
    NVF_ERROR(
        max_dtype_size_bit_for_vectorization <= 128,
        "RD3.unroll requires a positive native vector/unroll target");
    NVF_ERROR(
        jev_mode != scheduler::jev::Mode::Record || target_inner_unroll >= 1,
        "RD3.unroll native target must be positive");

    // The model's structural range is independent of getVectUnroll's native
    // preference. Record must preserve an original native quotient above the
    // reduction extent instead of replacing it with a menu endpoint.
    const int64_t hard_max = after_vect;
    if (jev_mode == scheduler::jev::Mode::Record &&
        target_inner_unroll > hard_max) {
      scheduler::jev::recordCoverageSkip(
          "RD3.unroll", "native_target_exceeds_reduction_extent");
    } else {
      int memory_clock_khz = 0;
      const int64_t memory_bus_bits = has_mufu_computation
          ? 0
          : static_cast<int64_t>(dev_prop->memoryBusWidth);
      if (!has_mufu_computation) {
        NVF_ERROR(
            cudaDeviceGetAttribute(
                &memory_clock_khz,
                cudaDevAttrMemoryClockRate,
                at::cuda::current_device()) == cudaSuccess,
            "RD3.unroll could not read CUDA memory clock");
      }
      std::string state_base = "reduction.inner_unroll.v1\n";
      state_base += "reduction_numel=" +
          std::to_string(total_reduction_numel) + '\n';
      state_base += "vector_width=" + std::to_string(vect_factor) + '\n';
      state_base += "layout_max=" + std::to_string(max_vectorize_factor) + '\n';
      state_base += "max_dtype_bits=" +
          std::to_string(max_dtype_size_bit_for_vectorization) + '\n';
      state_base += "n_tensor_inputs=" + std::to_string(n_tensor_inputs) + '\n';
      state_base += "has_mufu=" +
          std::to_string(has_mufu_computation ? 1 : 0) + '\n';
      state_base += "sm_count=" + std::to_string(sm_count) + '\n';
      state_base += "max_threads_per_sm=" +
          std::to_string(max_threads_per_sm) + '\n';
      state_base += "memory_bus_bits=" + std::to_string(memory_bus_bits) + '\n';
      state_base += "memory_clock_khz=" +
          std::to_string(memory_clock_khz) + '\n';

      target_inner_unroll = chooseInnerUnrollTarget(
          state_base,
          hard_max,
          jev_mode == scheduler::jev::Mode::Record ? target_inner_unroll : 0,
          jev_mode);
    }
  }

  // Empirical CTA size, both values allow thread uses 255 registers, so we
  // don't need to set maxrregcount which may affect occupancy since it is
  // derived by assuming one block per sm. This is due to the fact that when
  // maxrregcount is set, the compiler are more aggressive in register
  // allocation. When has expensive ops, use a smaller CTA size for higher
  // opportunity to reach higher occupancy.
  int64_t target_threads_per_block = 0;

  // When iteration dim is small, the number of blocks is small since each block
  // reduces one row (if iteration dim is extremely small, may do grid reduction
  // where multiple blocks are used for one rwo). Then, we need a large CTA size
  // to saturate SMs. But don't go over max_threads_per_block / 2 to reduce
  // communication cost and allow more than 1 block per SM.
  int64_t max_blocks = total_iteration_numel;
  int64_t min_threads_per_block = scheduler_utils::roundUpPow2(
      positiveCeilDiv(target_threads_per_sm * sm_count, max_blocks));
  min_threads_per_block =
      std::min(min_threads_per_block, max_threads_per_block / 2);
  // Only off/record need the native empirical starting target. Remote mode
  // sends the MUFU fact and occupancy inputs so the service can compute it.
  if (jev_mode != scheduler::jev::Mode::Remote) {
    target_threads_per_block = std::max(
        has_mufu_computation ? int64_t{128} : int64_t{256},
        min_threads_per_block);
  }

  std::optional<ShortInnerCtaPlan> selected_short_cta_plan;
  if (jev_mode != scheduler::jev::Mode::Off) {
    // Keep both existing performance policies in the menu. Short reductions
    // can additionally offer targets on either side of the occupancy floor:
    // their entire remainder, block and grid plan is checked above.
    NVF_ERROR(
        max_threads_per_block >= 512 && total_iteration_numel >= 1 &&
            target_threads_per_sm >= 1 && sm_count >= 1,
        "RD3.cta requires a positive iteration extent and a block limit of at "
        "least 512 threads");
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t starting_target : {int64_t{128}, int64_t{256}}) {
      const int64_t candidate =
          std::max(starting_target, min_threads_per_block);
      const std::string id = "t" + std::to_string(candidate);
      if (options.empty() || options.back().id != id) {
        options.push_back(
            {id, "Target " + std::to_string(candidate) +
                     " threads per CTA before native remainder and grid planning"});
      }
    }
    for (int64_t candidate : {int64_t{64}, int64_t{512}, int64_t{1024}}) {
      if (!shortInnerCtaPlan(
              total_reduction_numel,
              total_iteration_numel,
              vect_factor,
              target_inner_unroll,
              candidate,
              threads_per_warp,
              max_threads_per_block,
              dev_prop->maxThreadsDim[1])) {
        continue;
      }
      const std::string id = "t" + std::to_string(candidate);
      if (std::none_of(options.begin(), options.end(), [&](const auto& option) {
            return option.id == id;
          })) {
        options.push_back(
            {id,
             "Use a complete short-reduction plan with a " +
                 std::to_string(candidate) + "-thread CTA target"});
      }
    }
    std::sort(options.begin(), options.end(), [](const auto& lhs, const auto& rhs) {
      return std::stoll(lhs.id.substr(1)) < std::stoll(rhs.id.substr(1));
    });

    std::string state = "reduction.inner_cta.v2\n";
    state += "reduction_numel=" +
        std::to_string(total_reduction_numel) + '\n';
    state += "iteration_numel=" +
        std::to_string(total_iteration_numel) + '\n';
    state += "vector_width=" + std::to_string(vect_factor) + '\n';
    state += "target_inner_unroll=" +
        std::to_string(target_inner_unroll) + '\n';
    state += "warp_size=" + std::to_string(threads_per_warp) + '\n';
    state += "has_mufu=" + std::to_string(has_mufu_computation ? 1 : 0) + '\n';
    state += "sm_count=" + std::to_string(sm_count) + '\n';
    state += "max_threads_per_sm=" +
        std::to_string(max_threads_per_sm) + '\n';
    state += "max_threads_per_block=" +
        std::to_string(max_threads_per_block) + '\n';
    state += "max_threads_dim_y=" +
        std::to_string(dev_prop->maxThreadsDim[1]) + '\n';
    state += "max_grid_x=" +
        std::to_string(scheduler_utils::x_grid_limit) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "t" + std::to_string(target_threads_per_block)
        : std::string();
    const auto selected = scheduler::jev::chooseDecision(
        "RD3.cta", state, options, native_choice);
    bool matched = false;
    for (const auto& option : options) {
      if (selected == option.id) {
        target_threads_per_block = std::stoll(selected.substr(1));
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "RD3.cta response exceeds the preflighted CTA set");
    selected_short_cta_plan = shortInnerCtaPlan(
        total_reduction_numel,
        total_iteration_numel,
        vect_factor,
        target_inner_unroll,
        target_threads_per_block,
        threads_per_warp,
        max_threads_per_block,
        dev_prop->maxThreadsDim[1]);
    NVF_ERROR(
        target_threads_per_block <= max_threads_per_block &&
            (target_threads_per_block <= max_threads_per_block / 2 ||
             selected_short_cta_plan.has_value()),
        "RD3.cta target exceeds the preflighted physical block limit");
  }

  // RD3.warp chooses the empirical minimum number of reduction threads used
  // by the seed and remainder search. getL1L2WarpSize returns any integer in
  // [1, 16], not just a power of two. Keep every native value in the menu.
  // This is the 2D path only; the 3D planner has a separate use of the same
  // utility. The bounds below cover the complete downstream plan for *each*
  // candidate, without running speculative scheduler/API calls: x <= CTA,
  // unroll >= 1, and the remainder scan has at most one seed plus
  // (CTA - four_warps + 1) widths. Warp padding is less than 2x x, while
  // CTA <= half the physical block limit. Native grid bounds remain in force.
  // The remaining integer guards protect shape arithmetic representability;
  // neither factor count nor width-times-factor work is a legality limit.
  int64_t min_warp_size = 0;
  const int64_t l2_cache_bytes = dev_prop->l2CacheSize;
  const bool bounded_warp_plan =
      threads_per_warp == 32 && total_reduction_numel >= 1 &&
      total_iteration_numel >= 1 &&
      total_iteration_numel <= scheduler_utils::y_grid_limit &&
      n_tensor_inputs >= 1 && max_dtype_size_bit_for_vectorization >= 1 &&
      max_dtype_size_bit_for_vectorization <= 128 &&
      (max_dtype_size_bit_for_vectorization &
       (max_dtype_size_bit_for_vectorization - 1)) == 0 &&
      n_tensor_inputs <= std::numeric_limits<int64_t>::max() /
          max_dtype_size_bit_for_vectorization &&
      total_reduction_numel <=
          std::numeric_limits<int64_t>::max() / total_iteration_numel &&
      target_inner_unroll >= 1 && target_inner_unroll <= after_vect &&
      target_threads_per_block >= four_warps &&
      target_threads_per_block <= max_threads_per_block / 2 &&
      target_threads_per_block <= 32768 && l2_cache_bytes >= 1 &&
      l2_cache_bytes <= std::numeric_limits<int64_t>::max() / 8;
  bool choose_warp = bounded_warp_plan;
  if (choose_warp) {
    const int64_t input_bits =
        n_tensor_inputs * max_dtype_size_bit_for_vectorization;
    const int64_t n_elems = total_reduction_numel * total_iteration_numel;
    choose_warp =
        input_bits <= std::numeric_limits<int64_t>::max() / 1024 &&
        n_elems <= std::numeric_limits<int64_t>::max() / input_bits;
  }
  if (jev_mode != scheduler::jev::Mode::Remote || !choose_warp) {
    min_warp_size = reduction_scheduler_utils::getL1L2WarpSize(
        total_reduction_numel,
        total_iteration_numel,
        n_tensor_inputs,
        max_dtype_size_bit_for_vectorization);
  }
  if (jev_mode != scheduler::jev::Mode::Off && choose_warp) {
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t warp = 1; warp <= 16; ++warp) {
      options.push_back(
          {"m" + std::to_string(warp),
           "Use " + std::to_string(warp) +
               " minimum reduction threads in the 2D block plan"});
    }
    std::string state = "reduction.inner_warp_target.v1\n";
    state += "reduction_numel=" + std::to_string(total_reduction_numel) + '\n';
    state += "iteration_numel=" + std::to_string(total_iteration_numel) + '\n';
    state += "n_tensor_inputs=" + std::to_string(n_tensor_inputs) + '\n';
    state += "max_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) + '\n';
    state += "l2_cache_bytes=" + std::to_string(l2_cache_bytes) + '\n';
    state += "vector_width=" + std::to_string(vect_factor) + '\n';
    state += "after_vect=" + std::to_string(after_vect) + '\n';
    state += "target_inner_unroll=" +
        std::to_string(target_inner_unroll) + '\n';
    state += "target_threads_per_block=" +
        std::to_string(target_threads_per_block) + '\n';
    state += "warp_size=" + std::to_string(threads_per_warp) + '\n';
    state += "max_threads_per_block=" +
        std::to_string(max_threads_per_block) + '\n';
    state += "max_grid_y=" +
        std::to_string(scheduler_utils::y_grid_limit) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "m" + std::to_string(min_warp_size)
        : std::string();
    const auto selected = scheduler::jev::chooseDecision(
        "RD3.warp", state, options, native_choice);
    const auto found = std::find_if(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == selected;
        });
    NVF_ERROR(found != options.end(), "RD3.warp response exceeds 2D plan bounds");
    min_warp_size = std::stoll(selected.substr(1));
  }

  // Put inner remainder to bdimx, but don't go over 4 warps
  // to leave some room for unrolling.
  bdimx = std::max(getInnerRemainder(), min_warp_size);
  bdimx = std::min(bdimx, four_warps);

  // Put inner remainder to unroll, but don't go over target_inner_unroll
  inner_unroll = std::min(getInnerRemainder(), target_inner_unroll);

  // Put inner remainder to bdimx, but don't go over target_threads_per_block
  bdimx = std::min(
      positiveCeilDiv(after_vect, inner_unroll), target_threads_per_block);

  // if unroll can't cover the whole reduction, adjust bdimx and unroll to
  // remove quantization effect. e.g. at 20736, vect = 8, bdimx = 512, unroll =
  // 5, then iteration = 1.01, the 2nd iteration only has 1% effective threads.
  // Check all possible bdimx and unroll, choose the one with smallest remainder
  const bool remainder_scan_needed = getInnerRemainder() > 1;
  std::string_view remainder_no_choice_reason;
  if (remainder_scan_needed) {
    const int64_t seed_bdimx = bdimx;
    if (jev_mode != scheduler::jev::Mode::Remote) {
      int64_t after_vect_bdimx = positiveCeilDiv(after_vect, bdimx);
      int64_t current_remainder = after_vect_bdimx % inner_unroll;
      if (jev_mode == scheduler::jev::Mode::Off) {
        // Preserve the original native scan in Off mode.
        for (auto tmp_bdimx = four_warps; tmp_bdimx <= target_threads_per_block;
             tmp_bdimx += min_warp_size) {
          after_vect_bdimx = positiveCeilDiv(after_vect, tmp_bdimx);
          for (auto factor = target_inner_unroll; factor > 1; factor--) {
            int64_t remainder = after_vect_bdimx % factor;
            if (remainder == 0) {
              inner_unroll = factor;
              bdimx = tmp_bdimx;
              break;
            }
            if (remainder < current_remainder) {
              current_remainder = remainder;
              inner_unroll = factor;
              bdimx = tmp_bdimx;
            }
          }
        }
      } else {
        InnerRemainderScanState scan{bdimx, inner_unroll, current_remainder};
        for (auto tmp_bdimx = four_warps; tmp_bdimx <= target_threads_per_block;
             tmp_bdimx += min_warp_size) {
          scanInnerRemainderWidthByPlateaus(
              positiveCeilDiv(after_vect, tmp_bdimx),
              target_inner_unroll,
              tmp_bdimx,
              scan);
        }
        bdimx = scan.block_x;
        inner_unroll = scan.unroll;
      }
    }

    if (jev_mode != scheduler::jev::Mode::Off &&
        target_inner_unroll >= 2 && target_threads_per_block >= four_warps) {
      // Enumerate only native CTA widths. The seed may be off the warp-step
      // search grid, so it is always retained as one legal candidate.
      std::vector<int64_t> widths{seed_bdimx};
      for (int64_t x = four_warps; x <= target_threads_per_block;
           x += min_warp_size) {
        if (x != seed_bdimx) {
          widths.push_back(x);
        }
      }
      if (widths.size() > 1) {
        std::string fields;
        fields += "after_vect=" + std::to_string(after_vect) + '\n';
        fields += "target_inner_unroll=" +
            std::to_string(target_inner_unroll) + '\n';
        fields += "target_threads_per_block=" +
            std::to_string(target_threads_per_block) + '\n';
        fields += "min_warp_size=" + std::to_string(min_warp_size) + '\n';
        fields += "warp_size=" + std::to_string(threads_per_warp) + '\n';
        int64_t native_index = 0;
        if (jev_mode == scheduler::jev::Mode::Record) {
          const auto found = std::find(widths.begin(), widths.end(), bdimx);
          NVF_ERROR(found != widths.end());
          native_index = std::distance(widths.begin(), found) + 1;
        }
        const int64_t chosen_index = chooseInnerRemainderRange(
            "RD3.remainder_x",
            "reduction.inner_remainder_x.v1\n" + fields,
            1,
            static_cast<int64_t>(widths.size()),
            native_index,
            jev_mode);
        const int64_t selected_bdimx = widths.at(chosen_index - 1);

        std::string factor_state =
            "reduction.inner_remainder_u.v1\n" + fields;
        factor_state += "chosen_bdimx=" +
            std::to_string(selected_bdimx) + '\n';
        const int64_t selected_unroll = chooseInnerRemainderRange(
            "RD3.remainder_u",
            factor_state,
            1,
            target_inner_unroll,
            jev_mode == scheduler::jev::Mode::Record ? inner_unroll : 0,
            jev_mode);
        if (jev_mode == scheduler::jev::Mode::Remote) {
          bdimx = selected_bdimx;
          inner_unroll = selected_unroll;
        }
        NVF_ERROR(
            bdimx >= 1 && bdimx <= target_threads_per_block &&
                inner_unroll >= 1 && inner_unroll <= target_inner_unroll,
            "RD3 remainder plan exceeds native CTA or reduction bounds");
      } else {
        remainder_no_choice_reason = "single_cta_width";
      }
    } else if (jev_mode != scheduler::jev::Mode::Off) {
      remainder_no_choice_reason = target_inner_unroll < 2
          ? "inner_unroll_target_below_two"
          : "cta_target_below_four_warps";
    }
  } else if (jev_mode != scheduler::jev::Mode::Off) {
    remainder_no_choice_reason = "reduction_covered_before_remainder_scan";
  }
  if (jev_mode != scheduler::jev::Mode::Off &&
      !remainder_no_choice_reason.empty()) {
    // Record has the native scan result; Remote has only its actual seed when
    // no menu was offered. Keep those origins distinct in the outcome log.
    const std::string_view selection_source = !remainder_scan_needed
        ? "initial_remainder_plan"
        : (jev_mode == scheduler::jev::Mode::Record
               ? "native_remainder_scan"
               : "remote_remainder_seed");
    try {
      scheduler::jev::recordNoChoiceDecision(
          "RD3.remainder_x",
          "x" + std::to_string(bdimx),
          remainder_no_choice_reason,
          selection_source);
      scheduler::jev::recordNoChoiceDecision(
          "RD3.remainder_u",
          "u" + std::to_string(inner_unroll),
          remainder_no_choice_reason,
          selection_source);
    } catch (...) {
      // Passive outcome observation cannot change scheduling.
    }
  }

  // Set iteration dims, iter = [BIDy, o-Unroll, TIDy]. The native preference
  // for outer unroll is one: without an iteration-domain grouped warp
  // reduction, each block repeats the warp reduction for every output unroll.
  int64_t bdimy = 1, outer_unroll = 1, godim = 1;

  // fill bdimy, but don't go over target_threads_per_block
  bdimy = std::min(total_iteration_numel, target_threads_per_block / bdimx);

  // Put what is left to godim
  godim = positiveCeilDiv(
      positiveCeilDiv(total_iteration_numel, bdimy), outer_unroll);
  if (selected_short_cta_plan) {
    NVF_ERROR(
        bdimx == selected_short_cta_plan->block_x &&
            bdimy == selected_short_cta_plan->block_y &&
            inner_unroll == 1 && getInnerRemainder() == 1 &&
            godim == selected_short_cta_plan->output_grid &&
            selected_short_cta_plan->physical_grid_x >= 1 &&
            selected_short_cta_plan->split_grid_x ==
                (godim > scheduler_utils::x_grid_limit),
        "RD3.cta selected plan differs from the native remainder/grid plan");
  }

  // When iteration dim is small, may have unused SMs, to increase SM usage
  // needs to shift from block reduction to grid reduction.
  int64_t grdim = 1;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    while (grdim <= sm_count / godim / 2 &&
           getInnerRemainder() / grdim >= 2) {
      grdim *= 2;
    }
  }
  if (jev_mode != scheduler::jev::Mode::Off) {
    const int64_t inner_remainder = getInnerRemainder();
    NVF_ERROR(
        godim >= 1 && inner_remainder >= 1 && sm_count >= 1,
        "RD3.grid requires positive grid and reduction dimensions");

    // Each doubling is permitted only while its preceding factor has at
    // least two reduction pieces and the resulting grid fits one SM wave.
    // This also keeps every cross-grid plan's BIDy dimension in range.
    std::vector<scheduler::jev::ChoiceOption> options;
    options.push_back({"g1", "Use one block per output tile"});
    if (godim <= scheduler_utils::y_grid_limit) {
      for (int64_t candidate = 1;
           candidate <= sm_count / godim / 2 &&
           inner_remainder / candidate >= 2;) {
        candidate *= 2;
        options.push_back(
            {"g" + std::to_string(candidate),
             "Use " + std::to_string(candidate) +
                 " blocks per output tile for grid reduction"});
      }
    }

    std::string state = "reduction.inner_grid.v1\n";
    state += "godim=" + std::to_string(godim) + '\n';
    state += "inner_remainder=" + std::to_string(inner_remainder) + '\n';
    state += "sm_count=" + std::to_string(sm_count) + '\n';
    state += "y_grid_limit=" +
        std::to_string(scheduler_utils::y_grid_limit) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "g" + std::to_string(grdim)
        : std::string();
    const auto selected = scheduler::jev::chooseDecision(
        "RD3.grid", state, options, native_choice);
    const auto found = std::find_if(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == selected;
        });
    NVF_ERROR(found != options.end(), "RD3.grid response exceeds grid bounds");
    grdim = std::stoll(selected.substr(1));
  }

  // A grdim > 1 from either the native or the bounded Choice path implies
  // godim * grdim <= sm_count. Keep the legacy ragged-wave cleanup below,
  // although its > sm_count guard is unreachable for this inner 2D path.
  if (grdim > 1 && // If we have less than 8 waves of blocks
      grdim * godim < sm_count * 8 &&
      // And we don't have an even divisible number of blocks
      (grdim * godim) % sm_count != 0 &&
      // And we have more than one wave
      grdim * godim > sm_count) {
    // round waves down
    auto waves = std::max((godim * grdim) / sm_count, (int64_t)1);
    auto new_grdim = std::max((waves * sm_count) / godim, (int64_t)1);
    if (
        // If difference is less than 25% of the original grdim
        (new_grdim - grdim) * 4 < grdim &&
        // and difference is less than 25% of the original number of blocks
        ((new_grdim * godim) - (grdim * godim)) * 4 < grdim * godim) {
      grdim = new_grdim;
    }
  }

  if (grdim > 1) {
    // Grid reductions do not support unrolling iteration dimension, revert if
    // set. Recalculate godim.
    if (outer_unroll) {
      outer_unroll = 1;
      godim = positiveCeilDiv(total_iteration_numel, bdimy * outer_unroll);
    }
    // This could mess up parallelization which could be redone, but that would
    // require iterating over this entire function.
  }

  // RD3.iter_unroll offers complete output tiles only after the grid
  // reduction decision. A cross-grid reduction cannot unroll this axis, and
  // changing godim before RD3.grid would alter that decision's legal set.
  // Cap the speculative per-thread output work at four; the other bounds are
  // structural and the grid split below remains native.
  if (jev_mode != scheduler::jev::Mode::Off && grdim == 1 &&
      total_iteration_numel / bdimy >= 2) {
    std::vector<scheduler::jev::ChoiceOption> options;
    const int64_t max_factor = std::min<int64_t>(
        4, total_iteration_numel / bdimy);
    for (int64_t factor = 1; factor <= max_factor; factor *= 2) {
      options.push_back(
          {"u" + std::to_string(factor),
           "Unroll " + std::to_string(factor) +
               " output(s) per thread in the completed block plan"});
    }
    std::string state = "reduction.inner_iter_unroll.v1\n";
    state += "iteration_numel=" +
        std::to_string(total_iteration_numel) + '\n';
    state += "bdimy=" + std::to_string(bdimy) + '\n';
    state += "pre_unroll_godim=" + std::to_string(godim) + '\n';
    state += "grid_reduction_factor=" + std::to_string(grdim) + '\n';
    state += "max_grid_x=" +
        std::to_string(scheduler_utils::x_grid_limit) + '\n';
    const auto selected = scheduler::jev::chooseDecision(
        "RD3.iter_unroll",
        state,
        options,
        jev_mode == scheduler::jev::Mode::Record ? "u1" : "");
    const auto found = std::find_if(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == selected;
        });
    NVF_ERROR(
        found != options.end(),
        "RD3.iter_unroll response exceeds the native block-plan bounds");
    outer_unroll = std::stoll(selected.substr(1));
    godim = positiveCeilDiv(
        positiveCeilDiv(total_iteration_numel, bdimy), outer_unroll);
    NVF_ERROR(godim >= 1, "RD3.iter_unroll produced an invalid output grid");
  }

  auto rparams = std::make_unique<ReductionParams>();
  rparams->schedule_3d = false;
  rparams->fastest_dim = true;
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->cross_grid_inner_reduction = grdim > 1;
  rparams->multiple_reds_per_blk = bdimy > 1;
  bool pad_bdimx = bdimx > 16 && bdimx * bdimy < max_threads_per_block;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;

  if (rparams->pad_inner_reduction_to_warp) {
    // Adjust bdimx based on padding
    auto min_warp_size =
        (int64_t)at::cuda::getCurrentDeviceProperties()->warpSize;
    bdimx = bdimx % min_warp_size == 0
        ? bdimx
        : bdimx + min_warp_size - bdimx % min_warp_size;
  }
  if (selected_short_cta_plan) {
    NVF_ERROR(
        grdim == 1 && bdimx == selected_short_cta_plan->padded_block_x &&
            bdimx * bdimy <= max_threads_per_block,
        "RD3.cta selected plan violates the physical CTA or grid bound");
  }
  rparams->unroll_factor_inner_reduction = vect_factor;
  rparams->unroll_factor_top_of_vectorization = inner_unroll;
  rparams->vectorize_inner_reduction = vect_factor > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  rparams->unroll_factor_iter_dom = outer_unroll;

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.

  if (rparams->cross_grid_inner_reduction) {
    rparams->grid_dim_inner_reduction = ParallelType::BIDx;
    rparams->split_grid_dim_inner_reduction = true;
    gdimx = std::min(grdim, scheduler_utils::x_grid_limit);

    rparams->grid_dim_iter_dom = ParallelType::BIDy;
    if (godim > scheduler_utils::y_grid_limit) {
      NVF_ERROR(
          false,
          "Shouldn't do grid reduction when the iteration dimension is larger "
          "than y_grid_limit. iteration dim: ",
          total_iteration_numel);
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimy = scheduler_utils::y_grid_limit;
    }

  } else {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      bdimy > 1 ? bdimy : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Inner 2D Reduction Stats ========\n"
            << "total_reduction_numel: " << total_reduction_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "target_vect_unroll: " << target_vect_unroll << "\n"
            << "vectorize_factor: " << vect_factor << "\n"
            << "inner_unroll: " << inner_unroll << "\n"
            << "outer_unroll: " << outer_unroll << "\n"
            << "inner iteration: " << getInnerRemainder() << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_dtype_size_bit_for_vectorization: "
            << max_dtype_size_bit_for_vectorization << "\n"
            << "block(" << bdimx << ", " << bdimy << ", " << 1 << ")"
            << std::endl;
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

// Prove a logical-warp candidate before exposing it to the API. This bounded
// subset covers the full native 3D block and grid arithmetic, including every
// later RD5.iter_unroll option. No Choice is made while probing a candidate.
bool inner3dWarpPlanIsSafe(
    int64_t reduction_numel,
    int64_t iteration_numel,
    int64_t inner_numel,
    int64_t logical_warp,
    int64_t sm_count,
    int64_t max_threads_per_sm,
    int64_t max_threads_per_block,
    int64_t requested_target_threads = 0,
    bool iteration_first = false,
    int64_t* completed_block_x = nullptr,
    int64_t target_waves = 4) {
  // Keep every intermediate product bounded, and fix the vector work budget
  // to four. Other shapes retain the native heuristic without a warp request.
  if (reduction_numel <= inner_numel || reduction_numel > 4096 ||
      inner_numel < 4 || inner_numel > 128 ||
      reduction_numel % inner_numel != 0 || inner_numel % 4 != 0 ||
      iteration_numel < 1 || iteration_numel > 128 ||
      logical_warp < 1 || logical_warp > 16 || sm_count < 1 ||
      sm_count > 256 || max_threads_per_sm < 256 ||
      max_threads_per_sm > 4096 || max_threads_per_block < 256 ||
      max_threads_per_block > 2048 ||
      (target_waves != 2 && target_waves != 4 && target_waves != 8)) {
    return false;
  }
  const auto div = [](int64_t numerator, int64_t denominator) {
    return ceilDiv(numerator, denominator);
  };
  const int64_t n_elems = reduction_numel * iteration_numel;
  int64_t target_threads = std::max(logical_warp, div(reduction_numel, 8));
  int64_t target_blocks = div(n_elems, logical_warp);
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;
  if (target_blocks > sm_count) {
    int64_t available = std::max(n_elems / (logical_warp * sm_count), int64_t{1});
    bool flip = !iteration_first;
    while (available > 1 &&
           (target_unroll < 4 || target_iterations < 4)) {
      if (target_unroll * 2 <= 4 && flip) {
        target_unroll *= 2;
      }
      if (target_iterations * 2 <= 4 && !flip) {
        target_iterations *= 2;
      }
      available = std::max(
          n_elems /
              (logical_warp * sm_count * target_unroll * target_iterations),
          int64_t{1});
      flip = !flip;
    }
    target_blocks =
        div(n_elems, logical_warp * target_unroll * target_iterations);
  }
  target_blocks = std::min(target_blocks, sm_count * target_waves);
  if (target_blocks * target_unroll * target_iterations < n_elems) {
    target_threads = std::min(
        div(n_elems, target_blocks * target_unroll),
        div(max_threads_per_sm, target_waves));
  }
  target_threads = div(target_threads, logical_warp) * logical_warp;
  if (requested_target_threads != 0) {
    if (requested_target_threads < logical_warp ||
        requested_target_threads % logical_warp != 0 ||
        requested_target_threads > max_threads_per_block) {
      return false;
    }
    target_threads = requested_target_threads;
  }
  if (target_threads < 1 || target_threads > max_threads_per_block) {
    return false;
  }

  int64_t inner_unroll = 4;
  int64_t x = std::min(
      std::max(div(inner_numel, inner_unroll), logical_warp),
      target_threads);
  if (x * inner_unroll != inner_numel) {
    x = x < logical_warp ? scheduler_utils::lastPow2(x)
                         : x - x % logical_warp;
  }
  if (x < 1) {
    return false;
  }
  int64_t y = std::max(logical_warp / x, int64_t{1});
  int64_t z = std::min(
      std::min(
          std::max(target_threads / (x * y), int64_t{1}),
          div(reduction_numel, inner_numel)),
      scheduler_utils::z_block_limit);
  y = std::max(target_threads / (x * z), int64_t{1});
  if (x * y * z < logical_warp && inner_unroll > 1) {
    x = std::min(std::max(inner_numel, logical_warp), target_threads);
    inner_unroll = std::min(div(inner_numel, x), int64_t{4});
    y = std::max(logical_warp / x, int64_t{1});
    z = std::min(
        std::max(target_threads / (x * y), int64_t{1}),
        div(reduction_numel, inner_numel));
    y = std::max(target_threads / (x * z), int64_t{1});
  }
  // Multiple outputs per CTA plus either grid reduction triggers the native
  // 3D-to-2D fallback. Constrain this offer to one output per CTA, where that
  // fallback is impossible for every subsequent iteration-unroll choice.
  if (y != 1 || z < 1 || z > scheduler_utils::z_block_limit) {
    return false;
  }
  inner_unroll = std::min(scheduler_utils::lastPow2(inner_unroll), int64_t{4});
  const int64_t outer_unroll = inner_unroll < 4
      ? std::min(div(4, inner_unroll),
                 div(div(reduction_numel, inner_numel), z))
      : 1;
  const int64_t remainder = div(
      reduction_numel,
      x * inner_unroll * z * outer_unroll * target_iterations);
  const int64_t inner_remainder =
      div(inner_numel, x * inner_unroll * target_iterations);
  const int64_t budget = inner_unroll * outer_unroll < 4
      ? div(4, inner_unroll * outer_unroll)
      : 1;
  const int64_t hard_limit = std::min(budget, div(iteration_numel, y));

  const int64_t padded_x = x > 16 && x * y < max_threads_per_block
      ? div(x, 32) * 32
      : x;
  if (padded_x * y * z > max_threads_per_block ||
      padded_x > max_threads_per_block) {
    return false;
  }
  for (int64_t iter_unroll = 1; iter_unroll <= hard_limit; ++iter_unroll) {
    int64_t godim = div(iteration_numel, y * iter_unroll);
    int64_t gridim = 1;
    int64_t grodim = 1;
    if ((godim < target_blocks && remainder >= 0) || remainder >= 8) {
      const int64_t grdim = std::min(remainder, x * y * 8);
      gridim = inner_remainder;
      grodim = std::max(grdim / gridim, int64_t{1});
      grodim = std::max(
          std::min(remainder / inner_remainder, grodim), int64_t{1});
    }
    if (grodim > 1 || gridim > 1) {
      godim = div(iteration_numel, y); // native grid-unroll rollback
    }
    // These extents need neither grid splitting nor ragged-wave cleanup.
    if (gridim < 1 || grodim < 1 ||
        gridim > scheduler_utils::x_grid_limit ||
        grodim > scheduler_utils::z_grid_limit ||
        godim > scheduler_utils::y_grid_limit ||
        gridim * godim > sm_count) {
      return false;
    }
  }
  if (completed_block_x != nullptr) {
    *completed_block_x = padded_x;
  }
  return true;
}

// Check one outer-reduction unroll against the fixed upstream CTA plan. In
// this bounded domain the output CTA dimension is one, so no later grid plan
// can enter the unsupported 3D-to-2D fallback. Probe every later output
// unroll, including the native grid-reduction rollback, before offering it.
bool inner3dOuterUnrollPlanIsSafe(
    int64_t reduction_numel,
    int64_t iteration_numel,
    int64_t inner_numel,
    int64_t inner_unroll,
    int64_t max_unroll,
    int64_t block_x,
    int64_t block_y,
    int64_t block_z,
    int64_t target_iterations,
    int64_t target_blocks,
    int64_t sm_count,
    int64_t max_threads_per_block,
    int64_t outer_unroll) {
  if (reduction_numel <= inner_numel || reduction_numel > 4096 ||
      inner_numel < 1 || inner_numel > 128 ||
      reduction_numel % inner_numel != 0 || iteration_numel < 1 ||
      iteration_numel > 128 || inner_unroll < 1 || inner_unroll > 16 ||
      max_unroll < 1 || max_unroll > 16 || block_x < 1 ||
      block_x > 2048 || block_y != 1 || block_z < 1 ||
      block_z > scheduler_utils::z_block_limit || target_iterations < 1 ||
      target_iterations > 16 || target_blocks < 1 ||
      target_blocks > 2048 || sm_count < 1 || sm_count > 256 ||
      max_threads_per_block < 256 || max_threads_per_block > 2048) {
    return false;
  }
  const int64_t native_limit = inner_unroll < max_unroll
      ? std::min(
            ceilDiv(max_unroll, inner_unroll),
            ceilDiv(ceilDiv(reduction_numel, inner_numel), block_z))
      : 1;
  if (outer_unroll < 1 || outer_unroll > native_limit) {
    return false;
  }
  const int64_t physical_x =
      block_x > 16 && block_x * block_y < max_threads_per_block
      ? ceilDiv(block_x, int64_t{32}) * 32
      : block_x;
  if (physical_x * block_y * block_z > max_threads_per_block) {
    return false;
  }
  const int64_t reduction_remainder = ceilDiv(
      reduction_numel,
      block_x * inner_unroll * block_z * outer_unroll * target_iterations);
  const int64_t inner_remainder = ceilDiv(
      inner_numel, block_x * inner_unroll * target_iterations);
  const int64_t reduction_unroll = inner_unroll * outer_unroll;
  const int64_t output_budget = reduction_unroll < max_unroll
      ? ceilDiv(max_unroll, reduction_unroll)
      : 1;
  const int64_t output_limit =
      std::min(output_budget, ceilDiv(iteration_numel, block_y));
  for (int64_t output_unroll = 1; output_unroll <= output_limit;
       ++output_unroll) {
    int64_t output_grid =
        ceilDiv(iteration_numel, block_y * output_unroll);
    int64_t inner_grid = 1;
    int64_t outer_grid = 1;
    if (output_grid < target_blocks || reduction_remainder >= 8) {
      const int64_t reduction_grid =
          std::min(reduction_remainder, block_x * block_y * 8);
      inner_grid = inner_remainder;
      outer_grid = std::max(reduction_grid / inner_grid, int64_t{1});
      outer_grid = std::max(
          std::min(reduction_remainder / inner_remainder, outer_grid),
          int64_t{1});
    }
    if (inner_grid > 1 || outer_grid > 1) {
      output_grid = ceilDiv(iteration_numel, block_y);
    }
    // This also excludes the ragged-wave rewrite; the complete grid is
    // therefore identical to the native arithmetic for every offered factor.
    if (inner_grid < 1 || inner_grid > scheduler_utils::x_grid_limit ||
        outer_grid < 1 || outer_grid > scheduler_utils::z_grid_limit ||
        output_grid < 1 || output_grid > scheduler_utils::y_grid_limit ||
        inner_grid * output_grid > sm_count) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<ReductionParams> inner3dReductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t inner_most_dimension_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const size_t vectorize_factor,
    const bool has_mufu_computation) {
  // Set some targets for parallelization

  const int64_t n_elems = total_reduction_numel * total_iteration_numel;

  // WARNING: At some point we may want to generate heuristics for another
  // device that is not the current device.
  const int64_t max_threads_per_sm =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      (int64_t)128 / (int64_t)max_dtype_size_bit_for_vectorization,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      scheduler_utils::lastPow2(
          std::max((int64_t)n_tensor_inputs >> 2, (int64_t)1)));

  // The cache-derived logical warp is a performance target, and may be any
  // integer from one through sixteen. Offer all such targets only after each
  // complete 3D plan passes the native CTA, grid, and fallback preflight.
  const auto jev_mode_warp = scheduler::jev::mode();
  const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
  const bool bounded_warp_input =
      dev_prop->warpSize == 32 && dev_prop->l2CacheSize > 0 &&
      n_tensor_inputs == 1 && max_dtype_size_bit_for_vectorization == 32 &&
      vectorize_factor == 4;
  bool offer_warp = bounded_warp_input;
  if (jev_mode_warp != scheduler::jev::Mode::Off && offer_warp) {
    for (int64_t candidate = 1; candidate <= 16; ++candidate) {
      if (!inner3dWarpPlanIsSafe(
              total_reduction_numel,
              total_iteration_numel,
              inner_most_dimension_numel,
              candidate,
              device_multiprocessor_count,
              max_threads_per_sm,
              dev_prop->maxThreadsPerBlock)) {
        offer_warp = false;
        break;
      }
    }
  }
  int64_t min_warp_size = 0;
  if (jev_mode_warp != scheduler::jev::Mode::Remote || !offer_warp) {
    min_warp_size = reduction_scheduler_utils::getL1L2WarpSize(
        total_reduction_numel,
        total_iteration_numel,
        n_tensor_inputs,
        max_dtype_size_bit_for_vectorization);
  }
  if (jev_mode_warp != scheduler::jev::Mode::Off && offer_warp) {
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t warp = 1; warp <= 16; ++warp) {
      options.push_back(
          {"m" + std::to_string(warp),
           "Use " + std::to_string(warp) +
               " logical reduction threads in the completed 3D plan"});
    }
    const std::string state =
        "reduction.inner3d_warp_target.v1\nreduction_numel=" +
        std::to_string(total_reduction_numel) +
        "\niteration_numel=" + std::to_string(total_iteration_numel) +
        "\ninner_numel=" + std::to_string(inner_most_dimension_numel) +
        "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
        "\nmax_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) +
        "\nvector_width=" + std::to_string(vectorize_factor) +
        "\nsm_count=" + std::to_string(device_multiprocessor_count) +
        "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
        "\nmax_threads_per_block=" +
        std::to_string(dev_prop->maxThreadsPerBlock) +
        "\nl2_cache_bytes=" + std::to_string(dev_prop->l2CacheSize) +
        "\nwarp_size=" + std::to_string(dev_prop->warpSize);
    const std::string native_choice =
        jev_mode_warp == scheduler::jev::Mode::Record
        ? "m" + std::to_string(min_warp_size)
        : "";
    const auto selected = scheduler::jev::chooseDecision(
        "RD5.warp", state, options, native_choice);
    const auto found = std::find_if(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == selected;
        });
    NVF_ERROR(found != options.end(), "RD5.warp exceeds preflighted 3D plans");
    min_warp_size = std::stoll(selected.substr(1));
  }

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;

  // Try to set a minmum amount of work for each thread, as cross thread
  // communication is slow so it shouldn't be done for every element in the
  // reduction.
  int64_t min_target_iterations = std::max(
      (int64_t)256 / (int64_t)max_dtype_size_bit_for_vectorization, (int64_t)1);

  // Start trying to break parallelization up across threads,
  // unrolling/iterations, and blocks.

  // target_threads_in_block is the cap on a thread block, the minimum is based
  // on min_warp_size
  int64_t target_threads_in_block = std::max(
      min_warp_size, ceilDiv(total_reduction_numel, min_target_iterations));

  // If we have one warp per block, check if that's enough to saturate the SMs
  target_blocks = ceilDiv(n_elems, min_warp_size);

  // If we have more than a wave of blocks, put parallelism into unrolling and
  // target iterations
  if (target_blocks > device_multiprocessor_count) {
    auto available_unroll = std::max(
        n_elems / (min_warp_size * device_multiprocessor_count), (int64_t)1);

    // Spread across unrolling and iterations, want a balance of the two so flip
    // back and forth to alternate adding to them.
    bool flip = true;

    while (available_unroll > 1 &&
           (target_unroll < max_unroll ||
            // Prefer unrolling
            target_iterations < max_unroll)) {
      if (target_unroll * 2 <= max_unroll && flip) {
        target_unroll *= 2;
      }

      if (target_iterations * 2 <= max_unroll && !flip) {
        target_iterations *= 2;
      }

      available_unroll = std::max(
          n_elems /
              (min_warp_size * device_multiprocessor_count * target_unroll *
               target_iterations),
          (int64_t)1);

      flip = !flip;
    }

    // Recompute target blocks
    target_blocks =
        ceilDiv(n_elems, min_warp_size * target_unroll * target_iterations);
  }

  // The alternating work split favors reduction unroll on its first step.
  // Starting with serial iterations produces the symmetric alternative. Both
  // have the same product and therefore the same target-block count, but the
  // reduction-unroll factor changes the later CTA and grid construction.
  // Offer it only when both complete bounded 3D plans pass native limits and
  // produce distinct physical x dimensions. The ordinary native loop above
  // stays exact in Off and Record modes.
  bool iteration_first = false;
  std::string_view work_split_no_choice_reason;
  if (jev_mode_warp != scheduler::jev::Mode::Off && bounded_warp_input &&
      target_unroll != target_iterations && min_warp_size >= 1 &&
      min_warp_size <= 16) {
    int64_t native_x = 0;
    int64_t alternate_x = 0;
    const bool native_safe = inner3dWarpPlanIsSafe(
        total_reduction_numel,
        total_iteration_numel,
        inner_most_dimension_numel,
        min_warp_size,
        device_multiprocessor_count,
        max_threads_per_sm,
        dev_prop->maxThreadsPerBlock,
        0,
        false,
        &native_x);
    const bool alternate_safe = inner3dWarpPlanIsSafe(
        total_reduction_numel,
        total_iteration_numel,
        inner_most_dimension_numel,
        min_warp_size,
        device_multiprocessor_count,
        max_threads_per_sm,
        dev_prop->maxThreadsPerBlock,
        0,
        true,
        &alternate_x);
    if (native_safe && alternate_safe && native_x != alternate_x) {
      const std::string state =
          "reduction.inner3d_work_split.v1\nreduction_numel=" +
          std::to_string(total_reduction_numel) +
          "\niteration_numel=" + std::to_string(total_iteration_numel) +
          "\ninner_numel=" + std::to_string(inner_most_dimension_numel) +
          "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
          "\nmax_dtype_bits=" +
          std::to_string(max_dtype_size_bit_for_vectorization) +
          "\nvector_width=" + std::to_string(vectorize_factor) +
          "\nsm_count=" + std::to_string(device_multiprocessor_count) +
          "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
          "\nmax_threads_per_block=" +
          std::to_string(dev_prop->maxThreadsPerBlock) +
          "\nlogical_warp=" + std::to_string(min_warp_size);
      const std::vector<scheduler::jev::ChoiceOption> options = {
          {"unroll_first",
           "Assign the first available work factor to reduction unroll"},
          {"iter_first",
           "Assign the first available work factor to serial iterations"}};
      const auto selected = scheduler::jev::chooseDecision(
          "RD5.work_split", state, options,
          jev_mode_warp == scheduler::jev::Mode::Record ? "unroll_first"
                                                         : "");
      NVF_ERROR(
          selected == "unroll_first" || selected == "iter_first",
          "RD5.work_split response exceeds preflighted 3D plans");
      if (selected == "iter_first") {
        std::swap(target_unroll, target_iterations);
        iteration_first = true;
      }
    } else {
      work_split_no_choice_reason = !native_safe
          ? "native_plan_not_preflighted"
          : (!alternate_safe ? "alternate_plan_not_preflighted"
                             : "same_physical_grid_x");
    }
  } else if (jev_mode_warp != scheduler::jev::Mode::Off) {
    work_split_no_choice_reason = !bounded_warp_input
        ? "outside_bounded_warp_domain"
        : (target_unroll == target_iterations
               ? "equal_work_factors"
               : "logical_warp_outside_preflight_range");
  }
  if (!work_split_no_choice_reason.empty()) {
    scheduler::jev::recordNoChoiceDecision(
        "RD5.work_split",
        "unroll_first",
        work_split_no_choice_reason,
        "native_unroll_first_branch");
  }

  // The wave cap couples the block count and the per-CTA SM thread budget.
  // Offer one cap for each distinct pair consumed by the descendant planner.
  // Equal CTAs alone are insufficient: their target-block counts can differ.
  int64_t target_waves = 4;
  const bool bounded_wave_domain =
      bounded_warp_input && total_reduction_numel > inner_most_dimension_numel &&
      total_reduction_numel <= 4096 && inner_most_dimension_numel >= 4 &&
      inner_most_dimension_numel <= 128 &&
      total_reduction_numel % inner_most_dimension_numel == 0 &&
      inner_most_dimension_numel % 4 == 0 && total_iteration_numel >= 1 &&
      total_iteration_numel <= 128 && min_warp_size >= 1 &&
      min_warp_size <= 16 && device_multiprocessor_count >= 1 &&
      device_multiprocessor_count <= 256 && max_threads_per_sm >= 256 &&
      max_threads_per_sm <= 4096 && dev_prop->maxThreadsPerBlock >= 256 &&
      dev_prop->maxThreadsPerBlock <= 2048;
  if (jev_mode_warp != scheduler::jev::Mode::Off) {
    if (!bounded_wave_domain) {
      scheduler::jev::recordNoChoiceDecision(
          "RD5.waves",
          "waves4",
          "outside bounded 3D wave domain",
          "native_four_wave_cap");
    } else {
      std::vector<Inner3DWaveCandidate> hard_valid_candidates;
      for (int64_t waves : {2, 4, 8}) {
        if (inner3dWarpPlanIsSafe(
                total_reduction_numel,
                total_iteration_numel,
                inner_most_dimension_numel,
                min_warp_size,
                device_multiprocessor_count,
                max_threads_per_sm,
                dev_prop->maxThreadsPerBlock,
                0,
                iteration_first,
                nullptr,
                waves)) {
          hard_valid_candidates.push_back(
              {waves,
               previewInner3DWaveTarget(
                   n_elems,
                   target_blocks,
                   target_threads_in_block,
                   target_unroll,
                   target_iterations,
                   min_warp_size,
                   device_multiprocessor_count,
                   max_threads_per_sm,
                   waves)});
        }
      }
      const bool native_preflighted = std::any_of(
          hard_valid_candidates.begin(),
          hard_valid_candidates.end(),
          [](const auto& candidate) { return candidate.waves == 4; });
      // A wave cap matters only through its completed target-block count and
      // rounded target threads. Prefer waves4 within its equivalence class so
      // Record can still emit the exact native winner.
      const auto distinct_candidates =
          distinctInner3DWaveCandidates(hard_valid_candidates);
      std::vector<scheduler::jev::ChoiceOption> options;
      for (const auto& candidate : distinct_candidates) {
        const auto waves = candidate.waves;
        const auto target = candidate.target;
        options.push_back(
            {"waves" + std::to_string(waves),
             "Cap target blocks at " + std::to_string(waves) +
                 " SM waves and target SM threads / " +
                 std::to_string(waves) + " per CTA; target blocks " +
                 std::to_string(target.capped_target_blocks) +
                 "; rounded target threads " +
                 std::to_string(target.rounded_target_threads_in_block),
             "{\"target_blocks\":" +
                 std::to_string(target.capped_target_blocks) +
                 ",\"rounded_target_threads_per_block\":" +
                 std::to_string(target.rounded_target_threads_in_block) +
                 ",\"subchoices_pending\":true}"});
      }
      if (!native_preflighted) {
        if (jev_mode_warp == scheduler::jev::Mode::Record) {
          scheduler::jev::recordCoverageSkip(
              "RD5.waves", "complete-plan preflight excludes native cap");
        } else {
          NVF_ERROR(false, "RD5.waves lacks a complete native plan");
        }
      } else if (options.size() == 1) {
        scheduler::jev::recordNoChoiceDecision(
            "RD5.waves",
            "waves4",
            hard_valid_candidates.size() == 1
                ? "only one hard-valid wave cap"
                : "wave caps have equal block and thread targets",
            "native_four_wave_cap");
      } else {
        const std::string state =
            std::string(iteration_first ? "reduction.inner3d_waves.v4\n"
                                        : "reduction.inner3d_waves.v3\n") +
            "reduction_numel=" + std::to_string(total_reduction_numel) +
            "\niteration_numel=" + std::to_string(total_iteration_numel) +
            "\ninner_numel=" +
            std::to_string(inner_most_dimension_numel) +
            "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
            "\nmax_dtype_bits=" +
            std::to_string(max_dtype_size_bit_for_vectorization) +
            "\nvector_width=" + std::to_string(vectorize_factor) +
            "\nsm_count=" + std::to_string(device_multiprocessor_count) +
            "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
            "\nmax_threads_per_block=" +
            std::to_string(dev_prop->maxThreadsPerBlock) +
            "\nlogical_warp=" + std::to_string(min_warp_size) +
            (iteration_first ? "\nwork_split=iter_first" : "");
        const auto selected = scheduler::jev::chooseDecision(
            "RD5.waves",
            state,
            options,
            jev_mode_warp == scheduler::jev::Mode::Record ? "waves4" : "");
        const auto found = std::find_if(
            options.begin(), options.end(), [&](const auto& option) {
              return option.id == selected;
            });
        NVF_ERROR(found != options.end(), "RD5.waves exceeds preflighted plans");
        target_waves = std::stoll(selected.substr(5));
      }
    }
  }

  const auto wave_target = previewInner3DWaveTarget(
      n_elems,
      target_blocks,
      target_threads_in_block,
      target_unroll,
      target_iterations,
      min_warp_size,
      device_multiprocessor_count,
      max_threads_per_sm,
      target_waves);
  target_blocks = wave_target.capped_target_blocks;
  target_threads_in_block = wave_target.rounded_target_threads_in_block;

  // The four-wave/quarter-SM CTA target is a performance preference. Keep
  // native arithmetic above for Record mode. Offer every multiple of the
  // logical warp in the first 255 such targets, up to the physical block
  // limit, whose *complete* downstream 3D plan passes native checks. The
  // finite envelope respects the Choice API's 255-option limit. In
  // particular, preflight covers physical CTA size, all later output-unroll
  // choices, grid extents, and the unsupported 3D-to-2D fallback.
  bool cta_choice_emitted = false;
  std::string_view cta_no_choice_reason = "cta_menu_outside_bounded_domain";
  if (jev_mode_warp != scheduler::jev::Mode::Off && bounded_warp_input &&
      min_warp_size >= 1 && min_warp_size <= 16 &&
      target_threads_in_block >= min_warp_size &&
      target_threads_in_block <= dev_prop->maxThreadsPerBlock) {
    const int64_t native_target = target_threads_in_block;
    const int64_t candidate_limit =
        std::min(int64_t{255} * min_warp_size,
                 int64_t{dev_prop->maxThreadsPerBlock});
    std::vector<scheduler::jev::ChoiceOption> options;
    std::string hard_valid_targets;
    for (int64_t candidate = min_warp_size; candidate <= candidate_limit;
         candidate += min_warp_size) {
      if (!inner3dWarpPlanIsSafe(
              total_reduction_numel,
              total_iteration_numel,
              inner_most_dimension_numel,
              min_warp_size,
              device_multiprocessor_count,
              max_threads_per_sm,
              dev_prop->maxThreadsPerBlock,
              candidate,
              iteration_first,
              nullptr,
              target_waves)) {
        continue;
      }
      if (!hard_valid_targets.empty()) {
        hard_valid_targets += ",";
      }
      hard_valid_targets += std::to_string(candidate);
      options.push_back(
          {"t" + std::to_string(candidate),
           "Target " + std::to_string(candidate) +
               " threads in the completed 3D CTA plan"});
    }
    const bool native_target_preflighted = std::any_of(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == "t" + std::to_string(native_target);
        });
    if (options.size() > 1 && native_target_preflighted) {
      const std::string state =
          std::string(target_waves == 4
                          ? (iteration_first
                                 ? "reduction.inner3d_cta_target.v2\n"
                                 : "reduction.inner3d_cta_target.v1\n")
                          : (iteration_first
                                 ? "reduction.inner3d_cta_target.v4\n"
                                 : "reduction.inner3d_cta_target.v3\n")) +
          "reduction_numel=" +
          std::to_string(total_reduction_numel) +
          "\niteration_numel=" + std::to_string(total_iteration_numel) +
          "\ninner_numel=" + std::to_string(inner_most_dimension_numel) +
          "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
          "\nmax_dtype_bits=" +
          std::to_string(max_dtype_size_bit_for_vectorization) +
          "\nvector_width=" + std::to_string(vectorize_factor) +
          "\nsm_count=" + std::to_string(device_multiprocessor_count) +
          "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
          "\nmax_threads_per_block=" +
          std::to_string(dev_prop->maxThreadsPerBlock) +
          "\nlogical_warp=" + std::to_string(min_warp_size) +
          (iteration_first ? "\nwork_split=iter_first" : "") +
          (target_waves == 4
               ? ""
               : "\ntarget_waves=" + std::to_string(target_waves)) +
          "\nhard_valid_targets=" + hard_valid_targets;
      cta_choice_emitted = true;
      const auto selected = scheduler::jev::chooseDecision(
          "RD5.cta", state, options,
          jev_mode_warp == scheduler::jev::Mode::Record
              ? "t" + std::to_string(native_target)
              : "");
      const auto found = std::find_if(
          options.begin(), options.end(), [&](const auto& option) {
            return option.id == selected;
          });
      NVF_ERROR(found != options.end(), "RD5.cta exceeds preflighted 3D plans");
      target_threads_in_block = std::stoll(selected.substr(1));
    } else {
      cta_no_choice_reason = options.size() <= 1
          ? "fewer_than_two_hard_valid_cta_targets"
          : "native_cta_target_not_preflighted";
    }
  }

  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

  // Cross grid inner reduction, number of blocks to cross-grid on
  int64_t gridim = 1;
  // Cross grid outer reduction, number of blocks to cross-grid on
  int64_t grodim = 1;
  // Blocks for outputs
  int64_t godim = 1;

  // Threads for reduction
  int64_t bdimx = 1;
  // Threads for outputs
  int64_t bdimy = 1;
  // Threads for outer reduction dimension
  int64_t bdimz = 1;

  // Unroll amount
  int64_t inner_reduction_unroll_factor = 1;
  int64_t outer_reduction_unroll_factor = 1;
  int64_t iter_unroll_factor = 1;

  inner_reduction_unroll_factor =
      vectorize_factor > 1 ? (int64_t)vectorize_factor : 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(
      std::max(
          ceilDiv(inner_most_dimension_numel, inner_reduction_unroll_factor),
          (int64_t)min_warp_size),
      target_threads_in_block);

  // If we're not just barely covering the dimension, round to a more friendly
  // number
  if (bdimx * inner_reduction_unroll_factor != inner_most_dimension_numel) {
    // Round bdimx down to multiple of warp size or power 2
    if (bdimx < min_warp_size) {
      bdimx = scheduler_utils::lastPow2(bdimx);
    } else {
      bdimx = bdimx - bdimx % min_warp_size;
    }
  }

  // Put everything else in bdimy for now
  bdimy = std::max(min_warp_size / bdimx, (int64_t)1);

  // If 3D fill the rest of the threads into bdimz
  bdimz = std::min(
      std::min(
          std::max(target_threads_in_block / (bdimx * bdimy), (int64_t)1),
          ceilDiv(total_reduction_numel, inner_most_dimension_numel)),
      scheduler_utils::z_block_limit);

  // If 3D doesn't fill out the threads, adjust to add to bdimy
  bdimy = std::max(target_threads_in_block / (bdimx * bdimz), (int64_t)1);

  // If we don't have a full warp and have an unroll factor, move unroll into
  // bdimx
  if (bdimx * bdimy * bdimz < min_warp_size &&
      inner_reduction_unroll_factor > 1) {
    bdimx = std::min(
        std::max(inner_most_dimension_numel, min_warp_size),
        target_threads_in_block);

    inner_reduction_unroll_factor =
        std::min(ceilDiv(inner_most_dimension_numel, bdimx), max_unroll);

    // Readjust bdimy and bdimz
    bdimy = std::max(min_warp_size / bdimx, (int64_t)1);

    bdimz = std::min(
        std::max(target_threads_in_block / (bdimx * bdimy), (int64_t)1),
        ceilDiv(total_reduction_numel, inner_most_dimension_numel));

    bdimy = std::max(target_threads_in_block / (bdimx * bdimz), (int64_t)1);
  }

  godim = ceilDiv(total_iteration_numel, bdimy);

  bool vectorize = false;
  // Preserve the unrounded work factor so the remote RD5.width service can
  // reproduce the native vector width from the layout cap and this fixed plan.
  const int64_t inner_work_per_thread = inner_reduction_unroll_factor;

  // Move unrolling factor into vectorization upto vectorization limit.
  if (vectorize_factor > 1 && inner_reduction_unroll_factor > 1) {
    vectorize = true;
    inner_reduction_unroll_factor = std::min(
        scheduler_utils::lastPow2(inner_reduction_unroll_factor),
        (int64_t)vectorize_factor);
  }

  // Attempt to put some unrolling into the outer reduction if inner hasn't
  // taken the max unrolling
  if (inner_reduction_unroll_factor < max_unroll) {
    outer_reduction_unroll_factor = std::min(
        ceilDiv(max_unroll, inner_reduction_unroll_factor),
        ceilDiv(
            ceilDiv(total_reduction_numel, inner_most_dimension_numel), bdimz));
  }

  // The native rule spends the remaining per-thread work budget on outer
  // reduction unroll. A smaller factor may expose more reduction blocks or
  // leave work for output unroll. Offer only complete preflighted 3D plans.
  if (jev_mode_warp != scheduler::jev::Mode::Off &&
      outer_reduction_unroll_factor > 1) {
    const int64_t native_factor = outer_reduction_unroll_factor;
    const bool bounded_input =
        total_reduction_numel > inner_most_dimension_numel &&
        total_reduction_numel <= 4096 &&
        inner_most_dimension_numel >= 1 &&
        inner_most_dimension_numel <= 128 &&
        total_reduction_numel % inner_most_dimension_numel == 0 &&
        total_iteration_numel >= 1 && total_iteration_numel <= 128 &&
        inner_reduction_unroll_factor >= 1 &&
        inner_reduction_unroll_factor <= 16 && max_unroll >= 1 &&
        max_unroll <= 16 && bdimx >= 1 && bdimx <= 2048 && bdimy == 1 &&
        bdimz >= 1 && bdimz <= scheduler_utils::z_block_limit &&
        target_iterations >= 1 && target_iterations <= 16 &&
        target_blocks >= 1 && target_blocks <= 2048 &&
        device_multiprocessor_count >= 1 &&
        device_multiprocessor_count <= 256 &&
        dev_prop->maxThreadsPerBlock >= 256 &&
        dev_prop->maxThreadsPerBlock <= 2048;
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t candidate = 1;
         bounded_input && candidate <= native_factor;
         ++candidate) {
      if (inner3dOuterUnrollPlanIsSafe(
              total_reduction_numel,
              total_iteration_numel,
              inner_most_dimension_numel,
              inner_reduction_unroll_factor,
              max_unroll,
              bdimx,
              bdimy,
              bdimz,
              target_iterations,
              target_blocks,
              device_multiprocessor_count,
              dev_prop->maxThreadsPerBlock,
              candidate)) {
        options.push_back(
            {"u" + std::to_string(candidate),
             "Unroll " + std::to_string(candidate) +
                 " outer reduction element(s) per thread"});
      }
    }
    const bool native_preflighted =
        std::any_of(options.begin(), options.end(), [&](const auto& option) {
          return option.id == "u" + std::to_string(native_factor);
        });
    if (!native_preflighted) {
      NVF_ERROR(
          jev_mode_warp != scheduler::jev::Mode::Remote || !bounded_input,
          "RD5.outer_unroll native plan lacks bounded 3D proof");
      if (jev_mode_warp == scheduler::jev::Mode::Record) {
        scheduler::jev::recordCoverageSkip(
            "RD5.outer_unroll", "native outer unroll lacks bounded 3D proof");
      }
    } else if (options.size() == 1) {
      scheduler::jev::recordNoChoiceDecision(
          "RD5.outer_unroll",
          "u" + std::to_string(native_factor),
          "only native outer unroll has a 3D proof",
          "native_outer_unroll");
    } else if (options.size() > 1) {
      const std::string state =
          "reduction.inner3d_outer_unroll.v1\nreduction_numel=" +
          std::to_string(total_reduction_numel) +
          "\niteration_numel=" + std::to_string(total_iteration_numel) +
          "\ninner_numel=" + std::to_string(inner_most_dimension_numel) +
          "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
          "\nmax_dtype_bits=" +
          std::to_string(max_dtype_size_bit_for_vectorization) +
          "\ninner_unroll=" +
          std::to_string(inner_reduction_unroll_factor) +
          "\nblock_x=" + std::to_string(bdimx) +
          "\nblock_y=" + std::to_string(bdimy) +
          "\nblock_z=" + std::to_string(bdimz) +
          "\ntarget_iterations=" + std::to_string(target_iterations) +
          "\ntarget_blocks=" + std::to_string(target_blocks) +
          "\nsm_count=" + std::to_string(device_multiprocessor_count) +
          "\nmax_threads_per_block=" +
          std::to_string(dev_prop->maxThreadsPerBlock);
      const auto selected = scheduler::jev::chooseDecision(
          "RD5.outer_unroll",
          state,
          options,
          jev_mode_warp == scheduler::jev::Mode::Record
              ? "u" + std::to_string(native_factor)
              : "");
      const auto found = std::find_if(
          options.begin(), options.end(), [&](const auto& option) {
            return option.id == selected;
          });
      NVF_ERROR(
          found != options.end(),
          "RD5.outer_unroll exceeds preflighted 3D plans");
      outer_reduction_unroll_factor = std::stoll(selected.substr(1));
    }
  }

  int64_t remainder_in_reduction = ceilDiv(
      total_reduction_numel,
      bdimx * inner_reduction_unroll_factor * bdimz *
          outer_reduction_unroll_factor * target_iterations);

  int64_t remainder_in_inner_dim = ceilDiv(
      inner_most_dimension_numel,
      bdimx * inner_reduction_unroll_factor * target_iterations);

  // RD5.iter_unroll chooses a performance factor after the inner and outer
  // reduction factors are fixed. The source still owns the register-budget
  // bound, grid construction, and the grid-reduction compatibility rollback.
  const auto jev_mode_3d = scheduler::jev::mode();
  const int64_t pre_unroll_godim = godim;
  if (jev_mode_3d != scheduler::jev::Mode::Remote) {
    // If we haven't gotten to the max_unroll case, try to take it out of the
    // iteration domain.
    if (inner_reduction_unroll_factor * outer_reduction_unroll_factor <
        max_unroll) {
      // Don't go over a combined inner/outer unroll of max_unroll.
      auto unroll_available = ceilDiv(
          max_unroll,
          inner_reduction_unroll_factor * outer_reduction_unroll_factor);

      if (unroll_available > 1 && godim > 2 * device_multiprocessor_count) {
        unroll_available = std::min(
            unroll_available,
            ceilDiv(godim, 2 * device_multiprocessor_count));
        iter_unroll_factor = unroll_available;
      }
    }
  }
  if (jev_mode_3d != scheduler::jev::Mode::Off) {
    NVF_ERROR(
        pre_unroll_godim > 0 && device_multiprocessor_count > 0 &&
            max_unroll > 0,
        "RD5.iter_unroll requires positive iteration and device extents");
    const int64_t reduction_unroll =
        inner_reduction_unroll_factor * outer_reduction_unroll_factor;
    const int64_t budget_limit = reduction_unroll < max_unroll
        ? ceilDiv(max_unroll, reduction_unroll)
        : 1;
    // Never unroll beyond the number of pre-unroll output blocks. Every
    // offered factor remains within the same native register budget.
    const int64_t hard_limit =
        std::min({budget_limit, pre_unroll_godim, int64_t{255}});
    NVF_ERROR(
        jev_mode_3d != scheduler::jev::Mode::Record ||
            iter_unroll_factor <= hard_limit,
        "RD5.iter_unroll native factor exceeds the bounded option set");
    if (hard_limit == 1) {
      // A singleton has no performance decision for the API to make.
      scheduler::jev::recordNoChoiceDecision(
          "RD5.iter_unroll",
          "u1",
          "only one hard-legal iteration unroll factor",
          "hard_legal_singleton");
    } else {
      const int64_t native_factor = iter_unroll_factor;
      const auto request = inner3d_iter_continuation::buildRequest(
          {total_iteration_numel,
           bdimy,
           pre_unroll_godim,
           max_dtype_size_bit_for_vectorization,
           n_tensor_inputs,
           inner_reduction_unroll_factor,
           outer_reduction_unroll_factor,
           device_multiprocessor_count,
           total_reduction_numel,
           inner_most_dimension_numel,
           bdimx,
           bdimz,
           target_iterations,
           target_blocks,
           remainder_in_reduction,
           remainder_in_inner_dim,
           scheduler_utils::x_grid_limit,
           scheduler_utils::y_grid_limit,
           scheduler_utils::z_grid_limit},
          hard_limit,
          jev_mode_3d == scheduler::jev::Mode::Record
              ? std::optional<int64_t>{native_factor}
              : std::nullopt);
      if (request.candidates.size() == 1) {
        scheduler::jev::recordNoChoiceDecision(
            "RD5.iter_unroll",
            "u" + std::to_string(native_factor),
            "all iteration unroll factors realize the same grid plan",
            "native_iteration_unroll");
      } else {
        std::vector<scheduler::jev::ChoiceOption> options;
        options.reserve(request.candidates.size());
        for (const auto& candidate : request.candidates) {
          options.push_back(
              {candidate.id, candidate.description, candidate.consequence_json});
        }
        NVF_ERROR(
            jev_mode_3d != scheduler::jev::Mode::Record ||
                !request.native_choice.empty(),
            "RD5.iter_unroll native grid plan is missing from the menu");
        const auto selected = scheduler::jev::chooseDecision(
            "RD5.iter_unroll",
            request.state,
            options,
            request.native_choice);
        const auto found = std::find_if(
            request.candidates.begin(),
            request.candidates.end(),
            [&](const auto& candidate) { return selected == candidate.id; });
        NVF_ERROR(
            found != request.candidates.end(),
            "RD5.iter_unroll response exceeds realized grid plans");
        iter_unroll_factor = found->factor;
      }
    }
  }

  inner3d_iter_continuation::Plan selected_iter_plan{};
  const bool check_iter_plan = jev_mode_3d != scheduler::jev::Mode::Off;
  if (check_iter_plan) {
    selected_iter_plan = inner3d_iter_continuation::preview(
        {total_iteration_numel,
         bdimy,
         iter_unroll_factor,
         target_blocks,
         remainder_in_reduction,
         remainder_in_inner_dim,
         bdimx,
         device_multiprocessor_count,
         scheduler_utils::y_grid_limit});
  }

  godim = ceilDiv(total_iteration_numel, bdimy * iter_unroll_factor);

  // Clang tidy
  constexpr int64_t kEight = 8;
  // Cross grid reduction if we haven't hit our target blocks, and we have manyr
  // reduction elements.
  if ((godim < target_blocks && remainder_in_reduction >= 0) ||
      (remainder_in_reduction >= kEight)) {
    auto grdim = std::min(remainder_in_reduction, bdimx * bdimy * kEight);

    gridim = remainder_in_inner_dim;
    grodim = std::max(grdim / gridim, (int64_t)1);
    grodim = std::max(
        std::min(remainder_in_reduction / remainder_in_inner_dim, grodim),
        (int64_t)1);
  }

  const int64_t raw_inner_grid = gridim;
  // Try to do some cleanup of ragged waves on device, don't do this if we're
  // trying to do a 3D schedule. godim is a remainder of a split, so can only
  // control gridim
  if (grodim == 1 &&
      // If we have less than 8 waves of blocks
      gridim * godim < device_multiprocessor_count * kEight &&
      // And we don't have an even divisible number of blocks
      (gridim * godim) % device_multiprocessor_count != 0 &&
      // And we have more than one wave
      gridim * godim > device_multiprocessor_count) {
    // round waves down
    auto waves =
        std::max((godim * gridim) / device_multiprocessor_count, (int64_t)1);
    auto new_gridim =
        std::max((waves * device_multiprocessor_count) / godim, (int64_t)1);
    if (
        // If difference is less than 25% of the original gridim
        (new_gridim - gridim) * 4 < gridim &&
        // and difference is less than 25% of the original number of blocks
        ((new_gridim * godim) - (gridim * godim)) * 4 < gridim * godim) {
      gridim = new_gridim;
    }
  }

  // RD5.grid offers the raw grid alongside the native ragged-wave cleanup.
  // Both plans use the same completed CTA and reduction unroll. Restrict the
  // offer to one output per CTA and no output unroll, so neither plan can hit
  // the unsupported 3D fallback or change the grid-unroll rollback below.
  // The native grid construction above remains exact in Off and Record modes.
  const bool rd5_grid_eligible =
      jev_mode_3d != scheduler::jev::Mode::Off &&
      raw_inner_grid > gridim && gridim >= 1 &&
      raw_inner_grid <= 1024 && bdimy == 1 && grodim == 1 &&
      iter_unroll_factor == 1 && total_iteration_numel == godim &&
      godim >= 1 && godim <= scheduler_utils::y_grid_limit &&
      device_multiprocessor_count >= 1 &&
      device_multiprocessor_count <= 256;
  if (check_iter_plan) {
    NVF_ERROR(
        godim == selected_iter_plan.pre_rollback_godim &&
            raw_inner_grid == selected_iter_plan.raw_gridim &&
            gridim == selected_iter_plan.clean_gridim &&
            grodim == selected_iter_plan.grodim &&
            rd5_grid_eligible == selected_iter_plan.rd5_grid_eligible,
        "RD5.iter_unroll preview disagrees with grid continuation");
  }
  bool rd5_grid_selected_raw = false;
  if (jev_mode_3d != scheduler::jev::Mode::Off && rd5_grid_eligible) {
    const std::string state =
        "reduction.inner3d_grid_wave.v1\nraw_inner_grid=" +
        std::to_string(raw_inner_grid) +
        "\noutput_grid=" + std::to_string(godim) +
        "\nouter_grid=" + std::to_string(grodim) +
        "\noutput_threads=" + std::to_string(bdimy) +
        "\noutput_unroll=" + std::to_string(iter_unroll_factor) +
        "\nsm_count=" + std::to_string(device_multiprocessor_count) +
        "\nmax_grid_x=" + std::to_string(scheduler_utils::x_grid_limit) +
        "\nmax_grid_y=" + std::to_string(scheduler_utils::y_grid_limit);
    const std::vector<scheduler::jev::ChoiceOption> options = {
        {"raw", "Keep the complete raw inner-reduction grid"},
        {"clean", "Use the native ragged-wave inner-reduction grid"}};
    const auto selected = scheduler::jev::chooseDecision(
        "RD5.grid", state, options,
        jev_mode_3d == scheduler::jev::Mode::Record ? "clean" : "");
    NVF_ERROR(
        selected == "raw" || selected == "clean",
        "RD5.grid response exceeds native bounds");
    if (selected == "raw") {
      gridim = raw_inner_grid;
      rd5_grid_selected_raw = true;
    }
  }

  if (grodim > 1 || gridim > 1) {
    // Grid reductions do not support unrolling iteration dimension, revert if
    // set. Recalculate godim.
    if (iter_unroll_factor) {
      iter_unroll_factor = 1;
      godim = ceilDiv(total_iteration_numel, bdimy * iter_unroll_factor);
    }
    // This could mess up parallelization which could be redone, but that would
    // require iterating over this entire function.
  }
  if (check_iter_plan) {
    NVF_ERROR(
        iter_unroll_factor == selected_iter_plan.final_unroll &&
            godim == selected_iter_plan.final_godim &&
            gridim == (rd5_grid_selected_raw ? selected_iter_plan.raw_gridim
                                             : selected_iter_plan.clean_gridim) &&
            grodim == selected_iter_plan.grodim,
        "RD5.iter_unroll preview disagrees with final grid plan");
  }

  auto rparams = std::make_unique<ReductionParams>();
  rparams->fastest_dim = true;
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->cross_grid_inner_reduction = gridim > 1;
  rparams->multiple_reds_per_blk = bdimy > 1;
  bool pad_bdimx = bdimx > 16 &&
      bdimx * bdimy <
          (int64_t)at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;

  if (rparams->pad_inner_reduction_to_warp) {
    // Adjust bdimx based on padding
    auto min_warp_size =
        (int64_t)at::cuda::getCurrentDeviceProperties()->warpSize;
    bdimx = bdimx % min_warp_size == 0
        ? bdimx
        : bdimx + min_warp_size - bdimx % min_warp_size;
  }

  rparams->unroll_factor_inner_reduction = inner_reduction_unroll_factor;
  rparams->vectorize_inner_reduction = vectorize;

  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  rparams->unroll_factor_iter_dom = iter_unroll_factor;

  rparams->schedule_3d = total_reduction_numel != inner_most_dimension_numel;
  // Outer reduction domain
  if (rparams->schedule_3d) {
    rparams->cross_grid_outer_reduction = grodim > 1;
    if (bdimz > 1) {
      rparams->block_dim_outer_reduction = ParallelType::TIDz;
      rparams->cross_block_outer_reduction = true;
    }
    rparams->unroll_factor_outer_reduction = outer_reduction_unroll_factor;
  }

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimz = LaunchParams::UNINITIALIZED_VAL;

  // If we have a cross grid case we want to have gdimy assigned to godim and
  // gdimx assigned to grdim. Otherwise it's helpful to pull godim into gdimx in
  // case it's larger than gdimy can hold, as not doing so can thrash the cache.

  if (rparams->cross_grid_inner_reduction) {
    rparams->grid_dim_inner_reduction = ParallelType::BIDx;
    rparams->split_grid_dim_inner_reduction = true;
    gdimx = std::min(gridim, scheduler_utils::x_grid_limit);

    rparams->grid_dim_iter_dom = ParallelType::BIDy;
    if (godim > scheduler_utils::y_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimy = scheduler_utils::y_grid_limit;
    }

  } else {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  if (rparams->cross_grid_outer_reduction) {
    if (rparams->cross_block_inner_reduction) {
      rparams->grid_dim_outer_reduction = ParallelType::BIDz;
      gdimz = std::min(grodim, scheduler_utils::z_grid_limit);
      rparams->split_grid_dim_outer_reduction = true;
    } else {
      rparams->grid_dim_outer_reduction = ParallelType::BIDy;
      gdimy = std::min(grodim, scheduler_utils::y_grid_limit);
      rparams->split_grid_dim_outer_reduction = true;
    }
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      gdimz,
      bdimx,
      bdimy > 1 ? bdimy : LaunchParams::UNINITIALIZED_VAL,
      bdimz > 1 ? bdimz : LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Inner 3D Reduction Stats ========\n"
            << "total_reduction_numel: "
            << total_reduction_numel / inner_most_dimension_numel << " * "
            << inner_most_dimension_numel << "\n"
            << "total_iteration_numel: " << total_iteration_numel << "\n"
            << "vectorize_factor: " << vectorize_factor << "\n"
            << "n_tensor_inputs: " << n_tensor_inputs << "\n"
            << "max_dtype_size_bit_for_vectorization: "
            << max_dtype_size_bit_for_vectorization << "\n"
            << "block(" << bdimx << ", " << bdimy << ", " << bdimz << ")"
            << std::endl;
    debug() << rparams->toString() << std::endl;
  }

  // A 3D plan cannot combine multiple reductions per CTA with a grid
  // reduction. Keep this scheduler support check as a native hard gate.
  if (rparams->schedule_3d) {
    if (rparams->multiple_reds_per_blk &&
        (rparams->cross_grid_inner_reduction ||
         rparams->cross_grid_outer_reduction)) {
      if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
        debug() << "\n===== UNSUPPORTED REDUCTION HEURISTIC ========\n";
        debug() << rparams->multiple_reds_per_blk << ", "
                << (rparams->unroll_factor_inner_reduction > 1) << ", "
                << rparams->cross_grid_inner_reduction << std::endl;
      }
      return inner2dReductionHeuristic(
          total_reduction_numel,
          total_iteration_numel,
          (int64_t)n_tensor_inputs,
          (int64_t)max_dtype_size_bit_for_vectorization,
          (int64_t)vectorize_factor,
          has_mufu_computation);
    }
  }

  // The same flattened 2D constructor above is also a legal alternative to a
  // supported 3D plan: scheduleReduction canonicalizes all reduction axes to
  // one axis when schedule_3d is false. Limit the alternative to shapes where
  // even a 2D cross-grid plan cannot exceed BIDy, and to the native RD3
  // Choice implementation's physical CTA and dtype bounds. The 3D default
  // remains available outside that conservative subset.
  const auto jev_mode = scheduler::jev::mode();
  if (rparams->schedule_3d && jev_mode != scheduler::jev::Mode::Off) {
    const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
    const int64_t max_threads_per_block = dev_prop->maxThreadsPerBlock;
    const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
    const int64_t sm_count = dev_prop->multiProcessorCount;
    const bool flat2d_legal =
        total_iteration_numel > 0 &&
        total_iteration_numel <= scheduler_utils::y_grid_limit &&
        512 <= max_threads_per_block &&
        max_threads_per_block <= max_threads_per_sm &&
        max_threads_per_sm <= 65536 && sm_count >= 1 && sm_count <= 65536 &&
        max_dtype_size_bit_for_vectorization > 0 &&
        max_dtype_size_bit_for_vectorization <= 128 &&
        (max_dtype_size_bit_for_vectorization &
         (max_dtype_size_bit_for_vectorization - 1)) == 0 &&
        n_tensor_inputs > 0 &&
        n_tensor_inputs <=
            std::numeric_limits<int64_t>::max() /
                max_dtype_size_bit_for_vectorization &&
        vectorize_factor > 0 &&
        vectorize_factor <= static_cast<size_t>(total_reduction_numel) &&
        vectorize_factor <= static_cast<size_t>(std::max(
            int64_t{1},
            int64_t{128} / max_dtype_size_bit_for_vectorization));
    std::vector<scheduler::jev::ChoiceOption> options = {
        {"3d", "Keep the native inner and outer reduction axes"}};
    if (flat2d_legal) {
      options.push_back(
          {"flat2d", "Merge reduction axes into one native 2D plan"});
    }
    const std::string state =
        "reduction.inner_mapping.v1\nreduction_numel=" +
        std::to_string(total_reduction_numel) +
        "\niteration_numel=" + std::to_string(total_iteration_numel) +
        "\ninner_numel=" + std::to_string(inner_most_dimension_numel) +
        "\nlayout_width=" + std::to_string(vectorize_factor) +
        "\nmax_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) +
        "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
        "\nhas_mufu=" + std::to_string(has_mufu_computation ? 1 : 0) +
        "\nsm_count=" + std::to_string(sm_count) +
        "\nmax_threads_per_sm=" +
        std::to_string(max_threads_per_sm) +
        "\nmax_threads_per_block=" +
        std::to_string(max_threads_per_block) +
        "\ny_grid_limit=" +
        std::to_string(scheduler_utils::y_grid_limit);
    const auto selected = scheduler::jev::chooseDecision(
        "RD4", state, options,
        jev_mode == scheduler::jev::Mode::Record ? "3d" : "");
    NVF_ERROR(
        selected == "3d" || (flat2d_legal && selected == "flat2d"),
        "RD4 response is outside the native mapping set");
    if (selected == "flat2d") {
      return inner2dReductionHeuristic(
          total_reduction_numel,
          total_iteration_numel,
          n_tensor_inputs,
          max_dtype_size_bit_for_vectorization,
          static_cast<int64_t>(vectorize_factor),
          has_mufu_computation);
    }
  }

  // RD5.width changes only the split of the fixed inner work between a
  // contiguous vector load and ordinary unrolling. CTA dimensions, grid
  // dimensions, total work per thread, and the unsupported-3D fallback above
  // were already established by native analysis. Any smaller power-of-two
  // vector width divides the original vector factor, so the source layout
  // and alignment proof still applies.
  const int64_t fixed_inner_factor = rparams->unroll_factor_inner_reduction;
  if (jev_mode != scheduler::jev::Mode::Off &&
      rparams->vectorize_inner_reduction && fixed_inner_factor > 1 &&
      (fixed_inner_factor & (fixed_inner_factor - 1)) == 0 &&
      fixed_inner_factor <= static_cast<int64_t>(vectorize_factor) &&
      vectorize_factor % static_cast<size_t>(fixed_inner_factor) == 0 &&
      inner_work_per_thread >= fixed_inner_factor) {
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t width = 1; width <= fixed_inner_factor; width *= 2) {
      options.push_back(
          {"w" + std::to_string(width),
           "Vectorize " + std::to_string(width) +
               " innermost reduction element(s) per thread"});
    }
    const std::string state =
        "reduction.inner3d_width.v1\nlayout_max=" +
        std::to_string(vectorize_factor) +
        "\ninner_work_per_thread=" +
        std::to_string(inner_work_per_thread) +
        "\ninner_numel=" + std::to_string(inner_most_dimension_numel) +
        "\nreduction_numel=" + std::to_string(total_reduction_numel) +
        "\nmax_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization);
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "w" + std::to_string(fixed_inner_factor)
        : "";
    const auto selected = scheduler::jev::chooseDecision(
        "RD5.width", state, options, native_choice);
    int64_t selected_width = 0;
    for (int64_t width = 1; width <= fixed_inner_factor; width *= 2) {
      if (selected == "w" + std::to_string(width)) {
        selected_width = width;
        break;
      }
    }
    NVF_ERROR(selected_width > 0, "RD5.width response exceeds native bounds");
    if (selected_width == 1) {
      // Keep the original factor as an ordinary unroll so isUnrolled() and
      // cache placement remain unchanged even when vectorization is disabled.
      rparams->vectorize_inner_reduction = false;
    } else if (selected_width != fixed_inner_factor) {
      rparams->unroll_factor_inner_reduction = selected_width;
      rparams->unroll_factor_top_of_vectorization =
          fixed_inner_factor / selected_width;
    }
  }

  // Observe unoffered native targets at the successful 3D return, after the
  // fallback and alternative-plan branches have been resolved. This does not
  // offer a Choice or alter any scheduling parameter.
  if (rparams->schedule_3d && jev_mode != scheduler::jev::Mode::Off) {
    if (!offer_warp) {
      scheduler::jev::recordNoChoiceDecision(
          "RD5.warp",
          "m" + std::to_string(min_warp_size),
          bounded_warp_input ? "complete_3d_plan_preflight_excluded"
                             : "outside_bounded_warp_domain",
          "final_3d_native_target");
    }
    if (!cta_choice_emitted) {
      scheduler::jev::recordNoChoiceDecision(
          "RD5.cta",
          "t" + std::to_string(target_threads_in_block),
          cta_no_choice_reason,
          "final_3d_native_target");
    }
  }

  return rparams;
}

struct OuterReductionParams {
  OuterReductionParams(
      int64_t total_iteration_numel,
      int64_t total_reduction_numel)
      : total_iteration_numel(total_iteration_numel),
        total_reduction_numel(total_reduction_numel) {}
  // iteration dim paras
  // iteration elements = iter_unroll * bdimx * gidim
  int64_t iter_unroll_factor = 1;
  int64_t bdimx = 1;
  int64_t gidim = 1;
  // reduction dim paras
  // reduction elments = redu_unroll * bdimy * grdim * redu_serial
  int64_t redu_unroll_factor = 1;
  int64_t bdimy = 1;
  int64_t grdim = 1;
  int64_t redu_serial = 1;

  // iteration and reduction dim elements
  int64_t total_iteration_numel = -1;
  int64_t total_reduction_numel = -1;

  // Helper to figure out how much is left in the iter or reduction dim
  int64_t iDimAvail() const {
    return ceilDiv(total_iteration_numel, gidim * bdimx * iter_unroll_factor);
  }
  int64_t rDimAvail() const {
    return ceilDiv(total_reduction_numel, grdim * bdimy * redu_unroll_factor);
  };

  std::string toString() const {
    std::stringstream ss;
    ss << "\n===== Outer Reduction Stats ========\n"
       << "total_reduction_numel: " << total_reduction_numel << "\n"
       << "total_iteration_numel: " << total_iteration_numel << "\n"
       << "vectorize_factor: " << iter_unroll_factor << "\n"
       << "redu_unroll_factor: " << redu_unroll_factor << "\n"
       << "grid(" << gidim << ", " << grdim << ", 1)"
       << "\n"
       << "block(" << bdimx << ", " << bdimy << ", 1)" << std::endl;
    return ss.str();
  }
};

// toString omits redu_serial, so use all of the source planner's fields when
// checking that a native preview is identical to the live grid continuation.
std::array<int64_t, 9> outerReductionFields(
    const OuterReductionParams& params) {
  return {
      params.total_iteration_numel,
      params.total_reduction_numel,
      params.iter_unroll_factor,
      params.bdimx,
      params.gidim,
      params.redu_unroll_factor,
      params.bdimy,
      params.grdim,
      params.redu_serial};
}

// compare block reduction with grid reduction
bool isBetterThan(
    const OuterReductionParams& block_params,
    const OuterReductionParams& grid_params,
    int64_t sm_count) {
  NVF_ERROR(
      block_params.grdim == 1,
      "Only support compare block reduction heuristic with grid reduction not "
      "vice versa");

  // use block reduction if its SM usage >= 90% and its iter_unroll_factor is
  // equal or larger than grid reduction. These two conditions ensure high SM
  // usage and efficient global memory access. The corresponding block reduction
  // avoids the overhead of inter-block data exchange through global memory.
  // It is faster than grid reduction even not all SMs are used.
  // TODO: if we know the fusion is memory bound (e.g. pure reduction), we can
  // use a lower threshold. For computation bound (e.g. gelu bwd), relaxing
  // the threshold leads to regression.
  float f_wave = (float)block_params.gidim / (float)sm_count;
  float sm_efficiency = f_wave / std::ceil(f_wave);
  if (sm_efficiency >= 0.9f &&
      block_params.iter_unroll_factor >= grid_params.iter_unroll_factor) {
    return true;
  }

  // prefer block reduction if it uses same or more blocks than grid
  // reduction. This may happen when input size is very small, e.g. 512 x 128.
  // Current grid reduction heuristic start bdimx from 16 and prioritize
  // vectorization. It may not be able to fully utilize all the SMs.
  // This is really a tiny problem. The performance impact is most likely in a
  // range of a few us in <10us kernels
  // This condition is a WAR. Ideally, the grid reduction heuristics should be
  // improved, but given that the impact is likely negligible, we decided to do
  // this quick adjustment.
  if (block_params.gidim * block_params.grdim >=
      grid_params.gidim * grid_params.grdim) {
    return true;
  }

  // use grid reduction
  return false;
}

std::unique_ptr<ReductionParams> heuristicParaToSchedulerPara(
    const OuterReductionParams& params) {
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;

  // In these instances latency of the cleanup may be significant so flip gdimx
  // and gdimy to try and prevent all cleanup from happening at the
  // same time
  // Always disabled for now.
  // bool flip_grid = gidim > 1 && gidim < 8;
  const bool flip_grid = false;
  auto rparams = std::make_unique<ReductionParams>();

  // Configure register count to achieve 50% occupancy with 1024 threads per SM.
  // Typical thread configurations are 512 or 1024 threads per block, both
  // evenly divisible. This setup ensures each thread is limited to 64
  // registers, preventing compilers from allocating more registers per thread,
  // which could reduce occupancy and degrade performance.
  int64_t target_threads_per_sm =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor /
      2;
  int64_t threads_per_block = params.bdimx * params.bdimy;
  int64_t threads_per_sm =
      scheduler_utils::safeDiv(target_threads_per_sm, threads_per_block) *
      threads_per_block;
  int64_t reg_per_thread = getRegPerThreadGivenThreadsPerSM(threads_per_sm);
  rparams->cparams.maxrregcount = reg_per_thread;

  // cross grid implies cross block
  rparams->cross_block_inner_reduction = params.bdimy > 1 || params.grdim > 1;
  rparams->cross_grid_inner_reduction = params.grdim > 1;
  if (rparams->cross_grid_inner_reduction) {
    rparams->split_grid_dim_inner_reduction = true;
    rparams->grid_dim_inner_reduction =
        flip_grid ? ParallelType::BIDx : ParallelType::BIDy;
    if (flip_grid) {
      gdimx = std::min(params.grdim, scheduler_utils::x_grid_limit);
    } else {
      gdimy = std::min(params.grdim, scheduler_utils::y_grid_limit);
    }
  }
  rparams->multiple_reds_per_blk =
      params.bdimx > 1 || params.iter_unroll_factor > 1;

  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDx;
  }

  rparams->grid_dim_iter_dom =
      flip_grid ? ParallelType::BIDy : ParallelType::BIDx;
  if (params.gidim > (flip_grid ? scheduler_utils::y_grid_limit
                                : scheduler_utils::x_grid_limit)) {
    rparams->split_grid_dim_iter_dom_outer = true;
    if (flip_grid) {
      gdimy = scheduler_utils::y_grid_limit;
    } else {
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  rparams->flip_grid = flip_grid;

  if (rparams->cross_block_inner_reduction) {
    if (rparams->block_dim_iter_dom == ParallelType::TIDx) {
      rparams->block_dim_inner_reduction = ParallelType::TIDy;
    } else {
      rparams->block_dim_inner_reduction = ParallelType::TIDx;
    }
  }

  rparams->unroll_factor_inner_reduction = params.redu_unroll_factor;

  rparams->unroll_factor_iter_dom = params.iter_unroll_factor;
  rparams->vectorize_iter_dom = params.iter_unroll_factor > 1;

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      LaunchParams::UNINITIALIZED_VAL,
      rparams->multiple_reds_per_blk ? params.bdimx : params.bdimy,
      rparams->multiple_reds_per_blk ? params.bdimy
                                     : LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << params.toString() << std::endl;
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

OuterReductionParams getBlockOuterReduction(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t max_unroll,
    int64_t sm_count,
    int64_t max_threads_per_block) {
  OuterReductionParams params(total_iteration_numel, total_reduction_numel);

  int64_t sm_count_pow2 = scheduler_utils::lastPow2(sm_count);
  // Step-1, set iteration dim
  // (1) start with bdimx = 8, gidim = 32, iter_unroll = 1.
  // starts bdimx from 8, ensures each warp spans at most into 4 different rows.
  // when each thread reads 16 bytes, each warp does 4 transactions each
  // with 128 bytes. This is the maximum global memory transaction size
  // for each warp.
  // starts gidim from 32 and iter_unroll from 1, defers fully vectorization
  // to SM usage is high enough.
  params.bdimx = std::min(8L, params.iDimAvail());
  params.gidim = std::min(std::min(32L, sm_count_pow2), params.iDimAvail());
  params.iter_unroll_factor = 1;

  // (2) increase iter_unroll to its maximum following two rules:
  // (2.1) ensure divisible split
  // (2.2) leave enough blocks to saturate the device.
  // For example on any GPU with more than 32 SMs
  // bdimx-vect-gidim = 8-1-32  = 256
  // bdimx-vect-gidim = 8-2-32  = 512
  // bdimx-vect-gidim = 8-2-64  = 1024
  // bdimx-vect-gidim = 8-4-64  = 2048
  // bdimx-vect-gidim = 8-4-128 = 4096
  // bdimx-vect-gidim = 8-8-128 = 8192
  int64_t max_iter_unroll = vectorize_factor;
  while (params.iDimAvail() > 1) {
    if (params.iDimAvail() % 2 == 0 &&
        params.iter_unroll_factor * 2 <= max_iter_unroll) {
      params.iter_unroll_factor *= 2;
    }
    if (params.iDimAvail() > 1) {
      params.gidim *= 2;
    }
    if (params.iter_unroll_factor == max_iter_unroll) {
      break;
    }
  }

  // (3) reset gidim, ensures enough blocks to saturate the
  // device but doesn't use more SMs than available.
  params.gidim = std::min(
      ceilDiv(total_iteration_numel, params.bdimx * params.iter_unroll_factor),
      sm_count);

  // (4) increase bdimx to its maximum
  params.bdimx =
      ceilDiv(total_iteration_numel, params.gidim * params.iter_unroll_factor);
  params.bdimx = std::min(
      scheduler_utils::roundUpPow2(params.bdimx),
      scheduler_utils::roundUpToN(params.bdimx, 32));
  params.bdimx = std::min(params.bdimx, max_threads_per_block);

  // (5) re-calculate gidim after bdimx to fix round up differences. Also
  // handles extreme cases where iter dim is larger than iter_unroll_factor x
  // max_threads_per_block x sm count
  params.gidim =
      ceilDiv(total_iteration_numel, params.bdimx * params.iter_unroll_factor);

  // Step-2, set Reduction dim
  // (1) reduction unroll takes what is left by iter unroll
  params.redu_unroll_factor = std::min(
      params.rDimAvail(),
      scheduler_utils::safeDiv(max_unroll, params.iter_unroll_factor));

  // (2) bdimy takes what is left by bdimx.
  params.bdimy =
      std::min(params.rDimAvail(), max_threads_per_block / params.bdimx);

  // Step-3, final check
  // (1) revisit bdimx just in case bdimy doesn't take all the left threads
  while (params.bdimy * params.bdimx * 2 <= max_threads_per_block &&
         params.gidim / 2 >= sm_count_pow2) {
    params.bdimx *= 2;
    params.gidim /= 2;
  }
  return params;
}

struct OuterGridWarpPlans {
  OuterReductionParams baseline;
  OuterReductionParams stride_capped;
  int64_t base_grdim;
  int64_t stride_grdim;
  bool native_use_stride;
};

// Construct both complete grid plans for one logical warp target. This is pure
// arithmetic: candidate preflight must not send nested Choices or emit a
// SchedulerDebug schedule for a plan that will never be selected.
OuterGridWarpPlans buildOuterGridWarpPlans(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t max_dtype_size_bit_for_vectorization,
    int64_t vectorize_factor,
    int64_t max_unroll,
    int64_t sm_count,
    int64_t max_threads_per_sm,
    int64_t n_waves,
    int64_t min_warp_size,
    int64_t sm_cta_thread_divisor = 4,
    bool extend_reduction_grid = true,
    int64_t iteration_unroll_cap = std::numeric_limits<int64_t>::max()) {
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  // Set some targets for parallelization
  int64_t target_threads_in_block = min_warp_size;
  // Start target blocks at roughly a quarter wave if available
  int64_t target_blocks =
      std::min(ceilDiv(sm_count, (int64_t)4), ceilDiv(n_elems, min_warp_size));
  int64_t target_unroll = 1;

  auto available_parallelism =
      [&n_elems, &target_threads_in_block, &target_blocks, &target_unroll]() {
        return ceilDiv(
            n_elems, target_threads_in_block * target_blocks * target_unroll);
      };

  // If there's available parallelism, divide it between threads, blocks, and
  // vectorization
  // Threads are currently at a warp (16 or 32)
  // Blocks are currently at a quarter wave
  // Unroll is currently at 1
  while (
      // and there's parallelism left
      available_parallelism() > 1 &&
      (
          //  There's a place to put it in the block
          target_threads_in_block <
              ceilDiv(max_threads_per_sm, sm_cta_thread_divisor)
          // There's a place to put it in the device
          || target_blocks < sm_count * n_waves
          // There's a place to put it in unrolling
          || target_unroll < max_unroll)) {
    if (target_threads_in_block <
        ceilDiv(max_threads_per_sm, sm_cta_thread_divisor)) {
      target_threads_in_block *= 2;
    }

    if (target_blocks < sm_count * n_waves && available_parallelism() > 1) {
      target_blocks *= 2;
    }

    // Delay increasing unroll until we have more than one block per SM.
    // Assuming each SM can take more than one block.
    if (target_blocks > sm_count && target_unroll < max_unroll &&
        available_parallelism() > 1) {
      target_unroll *= 2;
    }
  }

  // Fill out unrolling if possible
  if (target_unroll < max_unroll && available_parallelism() > 1) {
    target_unroll = std::min(available_parallelism(), max_unroll);
  }

  target_unroll = scheduler_utils::lastPow2(target_unroll);

  // To get to target threads:
  // Prioritize
  // (1) x dim in iter domain
  // (2) unrolling in iter domain
  // (3) y in reduction domain
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions - need to flip unrolling to reduction
  // domain for this

  // Blocks for reductions
  int64_t grdim = 1;
  // Blocks for outputs
  int64_t gidim = 1;

  // Threads for reduction
  int64_t bdimy = 1;
  // Threads for output
  int64_t bdimx = 1;

  // Unroll amount
  int64_t inner_reduction_unroll_factor = 1;
  int64_t iter_unroll_factor = 1;

  // Helper lambda's to figure out how much is left in the iter or reduction dim
  auto iDimAvail = [&]() {
    return ceilDiv(total_iteration_numel, gidim * bdimx * iter_unroll_factor);
  };

  auto rDimAvail = [&]() {
    return ceilDiv(
        total_reduction_numel, grdim * bdimy * inner_reduction_unroll_factor);
  };

  // Start bdimx as a warp
  bdimx = std::min(min_warp_size, total_iteration_numel);

  if (total_iteration_numel > bdimx && total_iteration_numel < bdimx * 2) {
    // If rounding up would require less than 3/4 of the warp
    if ((total_iteration_numel % bdimx) * 4 < bdimx * 3) {
      // Round up to avoid nasty edge effects
      bdimx = total_iteration_numel;
    }
  }

  // gradually increased iter_unroll_factor from 1 to 2, 4, 8, ensure the
  // split is divisible. This improves performance when iteration dim is not
  // power of 2, e.g. 1600 and 4800.
  int64_t max_iter_unroll_factor =
      std::min(
          iteration_unroll_cap,
          std::min(vectorize_factor, std::min(iDimAvail(), target_unroll)));
  while (total_iteration_numel % (bdimx * iter_unroll_factor * 2) == 0 &&
         iter_unroll_factor * 2 <= max_iter_unroll_factor) {
    iter_unroll_factor *= 2;
  }

  // If iteration numel is not something huge like 64k we probably shouldn't do
  // this, maybe it could be 2 * device_multi_count to make sure iter dim is
  if (iDimAvail() > sm_count) {
    // Put more into bdimx
    bdimx = std::min(
        // Leave 2x a full wave of blocks
        ceilDiv(total_iteration_numel, iter_unroll_factor * sm_count),
        // Don't exceed max thread count
        target_threads_in_block);
  }

  // Round bdimx to pow2 since target threads per block is pow2
  bdimx = scheduler_utils::roundUpPow2(bdimx);

  // Fill bdimy with left over threads
  bdimy = std::min(
      scheduler_utils::safeDiv(target_threads_in_block, bdimx),
      total_reduction_numel);

  bdimy = roundDownPow2OrMultipleOf(bdimy, 8);

  // Move parallelization into unrolling the reduction dimension if
  // parallelizing iteration dimension didn't take the available unroll factor.
  if (iter_unroll_factor < max_unroll && rDimAvail() > 2) {
    inner_reduction_unroll_factor = std::min(
        rDimAvail(), scheduler_utils::safeDiv(max_unroll, iter_unroll_factor));

    inner_reduction_unroll_factor =
        scheduler_utils::lastPow2(inner_reduction_unroll_factor);
  }

  gidim = iDimAvail();

  // Try to hit a wave by going cross reduction
  grdim = std::min(rDimAvail(), ceilDiv(sm_count, gidim));
  // Extend to go to target blocks
  if (extend_reduction_grid && gidim * grdim < target_blocks) {
    // What should we use out of the reduction factor to hit target blocks? Make
    // sure we have 2 reductions per thread beyond what's already set as we
    // consider expanding to target block
    grdim = std::min(
        // At least 2 iterations of the reduction per thread on top of unroll
        ceilDiv(rDimAvail() * grdim, 2),
        // Expand to target blocks
        ceilDiv(target_blocks, gidim));
  }

  // Two complete reduction-grid plans share the same hard-valid block,
  // unroll, and output-grid dimensions. The empirical stride cap is 128 KiB
  // in the original executable formula, despite its old 256 KiB comment.
  const int64_t base_grdim = grdim;
  constexpr int64_t kMaxStrideBit = 128L * 1024L * 8L;
  int64_t max_remainder_size = kMaxStrideBit;
  for (const int64_t factor : {
           max_dtype_size_bit_for_vectorization,
           bdimx,
           bdimy,
           iter_unroll_factor,
           inner_reduction_unroll_factor}) {
    max_remainder_size =
        scheduler_utils::safeDiv(max_remainder_size, factor);
  }
  const int64_t stride_grdim = ceilDiv(
      ceilDiv(
          ceilDiv(total_reduction_numel, max_remainder_size), bdimy),
      inner_reduction_unroll_factor);

  // Complete each plan with the original ragged-wave preference. Both plans
  // retain the same native launch/grid splitting bounds downstream.
  auto cleanRaggedWaves = [&](int64_t candidate) {
    if (candidate * gidim < sm_count * 16 &&
        (candidate * gidim) % sm_count != 0 &&
        candidate * gidim > sm_count) {
      auto waves = std::max((gidim * candidate) / sm_count, (int64_t)1);
      auto new_grdim = std::max((waves * sm_count) / gidim, (int64_t)1);
      if ((candidate - new_grdim) * 4 <= candidate &&
          new_grdim * gidim % sm_count > candidate * gidim % sm_count) {
        candidate = new_grdim;
      }
    }
    return candidate;
  };
  const int64_t base_plan = cleanRaggedWaves(base_grdim);
  const int64_t stride_plan = cleanRaggedWaves(stride_grdim);
  auto makePlan = [&](int64_t reduction_grid) {
    OuterReductionParams params(total_iteration_numel, total_reduction_numel);
    params.bdimx = bdimx;
    params.bdimy = bdimy;
    params.grdim = reduction_grid;
    params.gidim = gidim;
    params.iter_unroll_factor = iter_unroll_factor;
    params.redu_unroll_factor = inner_reduction_unroll_factor;
    return params;
  };
  return {
      makePlan(base_plan),
      makePlan(stride_plan),
      base_grdim,
      stride_grdim,
      rDimAvail() > 16 &&
          ceilDiv(total_iteration_numel, min_warp_size) < sm_count * 2};
}

// These bounds are independent of the L2 preference. The grid dimensions may
// exceed CUDA launch limits because heuristicParaToSchedulerPara splits them.
bool outerGridWarpPlansHardValid(
    const OuterGridWarpPlans& plans,
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t max_unroll,
    int64_t max_threads_per_block) {
  const auto valid = [&](const OuterReductionParams& params) {
    if (params.bdimx < 1 || params.bdimy < 1 || params.gidim < 1 ||
        params.grdim < 1 || params.iter_unroll_factor < 1 ||
        params.redu_unroll_factor < 1 ||
        params.bdimx > max_threads_per_block / params.bdimy ||
        params.iter_unroll_factor > vectorize_factor ||
        params.iter_unroll_factor > max_unroll ||
        params.redu_unroll_factor > max_unroll / params.iter_unroll_factor ||
        total_iteration_numel % params.iter_unroll_factor != 0) {
      return false;
    }
    const int64_t iter_per_block =
        params.bdimx * params.iter_unroll_factor;
    const int64_t reduction_per_block =
        params.bdimy * params.redu_unroll_factor;
    return params.gidim ==
        ceilDiv(total_iteration_numel, iter_per_block) &&
        params.grdim <=
        ceilDiv(total_reduction_numel, reduction_per_block);
  };
  return valid(plans.baseline) && valid(plans.stride_capped);
}

// The block planner is pure arithmetic too. Check its completed plan before
// offering a narrower iteration width; no scheduler state is mutated here.
bool outerBlockPlanHardValid(
    const OuterReductionParams& params,
    int64_t total_iteration_numel,
    int64_t vectorize_factor,
    int64_t max_threads_per_block) {
  return params.bdimx > 0 && params.bdimy > 0 && params.gidim > 0 &&
      params.grdim == 1 && params.iter_unroll_factor > 0 &&
      params.redu_unroll_factor > 0 &&
      params.bdimx <= max_threads_per_block / params.bdimy &&
      params.iter_unroll_factor <= vectorize_factor &&
      total_iteration_numel % params.iter_unroll_factor == 0 &&
      params.gidim >= ceilDiv(
          total_iteration_numel,
          params.bdimx * params.iter_unroll_factor);
}

// A width is offered only when a complete block plan and a complete grid
// plan exist for each subsequent wave target. The later RD9.warp Choice
// still checks each logical-warp plan against its native hard bounds.
bool outerWidthHardValid(
    int64_t total_reduction_numel,
    int64_t total_iteration_numel,
    int64_t max_dtype_size_bit_for_vectorization,
    int64_t vectorize_factor,
    int64_t max_unroll,
    int64_t sm_count,
    int64_t max_threads_per_sm,
    int64_t max_threads_per_block) {
  const auto block = getBlockOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_block);
  if (!outerBlockPlanHardValid(
          block,
          total_iteration_numel,
          vectorize_factor,
          max_threads_per_block)) {
    return false;
  }
  for (int64_t waves : {4, 8}) {
    bool has_grid_plan = false;
    for (int64_t warp : {16, 32}) {
      const auto plans = buildOuterGridWarpPlans(
          total_reduction_numel,
          total_iteration_numel,
          max_dtype_size_bit_for_vectorization,
          vectorize_factor,
          max_unroll,
          sm_count,
          max_threads_per_sm,
          waves,
          warp);
      has_grid_plan |= outerGridWarpPlansHardValid(
          plans,
          total_reduction_numel,
          total_iteration_numel,
          vectorize_factor,
          max_unroll,
          max_threads_per_block);
    }
    if (!has_grid_plan) {
      return false;
    }
  }
  return true;
}

OuterReductionParams getGridOuterReduction(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const int64_t vectorize_factor,
    const int64_t max_unroll,
    const int64_t sm_count,
    const int64_t max_threads_per_sm,
    const int64_t max_threads_per_block) {
  // grid or block reduction
  const int64_t n_elems = total_reduction_numel * total_iteration_numel;
  // Try to use 4 * SM blocks to reduce communication cost. But still
  // use 8 * SM blocks if the problem size is large, it increased requested
  // waves, helps hiding memory latency, if still use 4 * SM blocks, about 5%
  // regression for compute bound kernels. Not much change for memory bound.
  int64_t n_waves = 0;
  const auto jev_mode = scheduler::jev::mode();
  if (jev_mode == scheduler::jev::Mode::Off) {
    n_waves = n_elems >= (int64_t)64 * 1024 * 1024 ? 8 : 4;
  } else {
    // RD9 delegates only the empirical wave target. The native algorithm
    // still turns that target into block, unroll, and grid dimensions.
    const std::string state =
        "reduction.outer_waves.v1\nreduction_numel=" +
        std::to_string(total_reduction_numel) +
        "\niteration_numel=" + std::to_string(total_iteration_numel);
    const std::vector<scheduler::jev::ChoiceOption> options = {
        {"waves4", "Target four waves of SM blocks"},
        {"waves8", "Target eight waves of SM blocks"}};
    const std::string native_choice =
        jev_mode == scheduler::jev::Mode::Record
        ? (n_elems >= (int64_t)64 * 1024 * 1024 ? "waves8" : "waves4")
        : "";
    const auto selected = scheduler::jev::chooseDecision(
        "RD9.waves", state, options, native_choice);
    NVF_ERROR(
        selected == "waves4" || selected == "waves8",
        "RD9 wave target is outside native choices");
    n_waves = selected == "waves8" ? 8 : 4;
  }

  // L2 residency is an empirical preference for a 16-thread logical warp.
  // The planner and hard bounds above make either target a legal schedule.
  const int64_t l2_cache_bytes =
      at::cuda::getCurrentDeviceProperties()->l2CacheSize;
  int64_t native_min_warp_size = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    const bool fits_in_l2 =
        n_elems * max_dtype_size_bit_for_vectorization * n_tensor_inputs <
        l2_cache_bytes * 8;
    native_min_warp_size = fits_in_l2 ? 16 : 32;
  }
  const auto makeWarpPlans = [&](int64_t warp_size) {
    return buildOuterGridWarpPlans(
        total_reduction_numel,
        total_iteration_numel,
        max_dtype_size_bit_for_vectorization,
        vectorize_factor,
        max_unroll,
        sm_count,
        max_threads_per_sm,
        n_waves,
        warp_size);
  };
  int64_t min_warp_size = native_min_warp_size;
  if (jev_mode != scheduler::jev::Mode::Off) {
    const auto plans16 = makeWarpPlans(16);
    const auto plans32 = makeWarpPlans(32);
    const bool valid16 = outerGridWarpPlansHardValid(
        plans16,
        total_reduction_numel,
        total_iteration_numel,
        vectorize_factor,
        max_unroll,
        max_threads_per_block);
    const bool valid32 = outerGridWarpPlansHardValid(
        plans32,
        total_reduction_numel,
        total_iteration_numel,
        vectorize_factor,
        max_unroll,
        max_threads_per_block);
    std::vector<scheduler::jev::ChoiceOption> options;
    if (valid16) {
      options.push_back({"warp16", "Use a 16-thread logical warp target"});
    }
    if (valid32) {
      options.push_back({"warp32", "Use a 32-thread logical warp target"});
    }
    NVF_ERROR(!options.empty(), "RD9 has no hard-valid warp target");
    const std::string state =
        "reduction.outer_warp.v1\nreduction_numel=" +
        std::to_string(total_reduction_numel) +
        "\niteration_numel=" + std::to_string(total_iteration_numel) +
        "\ndtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) +
        "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
        "\nl2_cache_bytes=" + std::to_string(l2_cache_bytes) +
        "\nwarp16_valid=" + std::to_string(valid16 ? 1 : 0) +
        "\nwarp32_valid=" + std::to_string(valid32 ? 1 : 0);
    const std::string native_choice =
        jev_mode == scheduler::jev::Mode::Record
        ? (native_min_warp_size == 16 ? "warp16" : "warp32")
        : "";
    if (jev_mode == scheduler::jev::Mode::Record) {
      NVF_ERROR(
          (native_min_warp_size == 16 && valid16) ||
              (native_min_warp_size == 32 && valid32),
          "Native RD9 warp target is outside hard bounds");
    }
    const auto selected = scheduler::jev::chooseDecision(
        "RD9.warp", state, options, native_choice);
    NVF_ERROR(
        (selected == "warp16" && valid16) ||
            (selected == "warp32" && valid32),
        "RD9 warp target is outside hard-valid candidates");
    min_warp_size = selected == "warp16" ? 16 : 32;
  }
  int64_t cta_share = 4;
  if (jev_mode != scheduler::jev::Mode::Off) {
    // The quarter-SM CTA target is an empirical performance preference. Build
    // both complete reduction-grid plans for each alternative before offering
    // it; their block, vector, grid, and unroll bounds remain native checks.
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t share : {2, 4, 8}) {
      const auto candidate = buildOuterGridWarpPlans(
          total_reduction_numel,
          total_iteration_numel,
          max_dtype_size_bit_for_vectorization,
          vectorize_factor,
          max_unroll,
          sm_count,
          max_threads_per_sm,
          n_waves,
          min_warp_size,
          share);
      if (outerGridWarpPlansHardValid(
              candidate,
              total_reduction_numel,
              total_iteration_numel,
              vectorize_factor,
              max_unroll,
              max_threads_per_block)) {
        options.push_back(
            {"q" + std::to_string(share),
             "Target approximately SM thread capacity / " +
                 std::to_string(share) + " threads per CTA"});
      }
    }
    const auto native = std::find_if(
        options.begin(), options.end(), [](const auto& option) {
          return option.id == "q4";
        });
    NVF_ERROR(
        native != options.end(), "Native RD9 CTA plan violates hard bounds");
    const std::string state =
        "reduction.outer_cta.v1\nreduction_numel=" +
        std::to_string(total_reduction_numel) +
        "\niteration_numel=" + std::to_string(total_iteration_numel) +
        "\ndtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) +
        "\nsm_count=" + std::to_string(sm_count) +
        "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
        "\nmax_threads_per_block=" +
        std::to_string(max_threads_per_block) +
        "\nmax_unroll=" + std::to_string(max_unroll) +
        "\nvector_width=" + std::to_string(vectorize_factor) +
        "\nwaves=" + std::to_string(n_waves) +
        "\nwarp=" + std::to_string(min_warp_size);
    const auto selected = scheduler::jev::chooseDecision(
        "RD9.cta",
        state,
        options,
        jev_mode == scheduler::jev::Mode::Record ? "q4" : "");
    const auto found = std::find_if(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == selected;
        });
    NVF_ERROR(found != options.end(), "RD9 CTA target is outside hard bounds");
    cta_share = std::stoll(selected.substr(1));
  }
  auto plans = buildOuterGridWarpPlans(
      total_reduction_numel,
      total_iteration_numel,
      max_dtype_size_bit_for_vectorization,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_sm,
      n_waves,
      min_warp_size,
      cta_share);
  if (jev_mode == scheduler::jev::Mode::Off) {
    return plans.native_use_stride ? plans.stride_capped : plans.baseline;
  }
  NVF_ERROR(
      outerGridWarpPlansHardValid(
          plans,
          total_reduction_numel,
          total_iteration_numel,
          vectorize_factor,
          max_unroll,
          max_threads_per_block),
      "Selected RD9 warp plan violates hard bounds");
  // One SM wave and the native target-block formula are two complete grid
  // plans. The latter is an occupancy preference, not a reduction-coverage
  // requirement. Keep both downstream ragged and stride calculations coupled
  // to whichever hard-valid complete plan the Choice selects.
  auto wave_plans = buildOuterGridWarpPlans(
      total_reduction_numel,
      total_iteration_numel,
      max_dtype_size_bit_for_vectorization,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_sm,
      n_waves,
      min_warp_size,
      cta_share,
      false);
  std::vector<scheduler::jev::ChoiceOption> grid_base_options = {
      {"target", "Apply target-block reduction-grid formula"}};
  if (wave_plans.base_grdim != plans.base_grdim &&
      outerGridWarpPlansHardValid(
          wave_plans,
          total_reduction_numel,
          total_iteration_numel,
          vectorize_factor,
          max_unroll,
          max_threads_per_block)) {
    grid_base_options.push_back(
        {"wave", "Keep one-SM-wave reduction grid"});
  }
  const std::string grid_base_state =
      "reduction.outer_grid_base.v1\nreduction_numel=" +
      std::to_string(total_reduction_numel) +
      "\niteration_numel=" + std::to_string(total_iteration_numel) +
      "\ndtype_bits=" +
      std::to_string(max_dtype_size_bit_for_vectorization) +
      "\nsm_count=" + std::to_string(sm_count) +
      "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
      "\nmax_threads_per_block=" +
      std::to_string(max_threads_per_block) +
      "\nmax_unroll=" + std::to_string(max_unroll) +
      "\nvector_width=" + std::to_string(vectorize_factor) +
      "\nwaves=" + std::to_string(n_waves) +
      "\nwarp=" + std::to_string(min_warp_size) +
      "\ncta_share=" + std::to_string(cta_share);
  const auto grid_base_choice = scheduler::jev::chooseDecision(
      "RD9.grid_base",
      grid_base_state,
      grid_base_options,
      jev_mode == scheduler::jev::Mode::Record ? "target" : "");
  NVF_ERROR(
      grid_base_choice == "target" ||
          (grid_base_choice == "wave" && grid_base_options.size() == 2),
      "RD9 baseline grid choice is outside hard-valid complete plans");
  if (grid_base_choice == "wave") {
    plans = wave_plans;
  }
  // The native grid planner favors the largest aligned iteration unroll and
  // spends the remaining register budget on reduction unroll. Offer smaller
  // iteration factors only as complete, independently checked grid plans.
  const int64_t native_iter_unroll = plans.baseline.iter_unroll_factor;
  NVF_ERROR(
      native_iter_unroll > 0 &&
          scheduler_utils::lastPow2(native_iter_unroll) == native_iter_unroll,
      "RD9 native iteration unroll is not a positive power of two");
  std::vector<scheduler::jev::ChoiceOption> unroll_options;
  std::vector<OuterGridWarpPlans> unroll_plans;
  for (int64_t factor = 1; factor <= native_iter_unroll; factor *= 2) {
    auto candidate = factor == native_iter_unroll
        ? plans
        : buildOuterGridWarpPlans(
              total_reduction_numel,
              total_iteration_numel,
              max_dtype_size_bit_for_vectorization,
              vectorize_factor,
              max_unroll,
              sm_count,
              max_threads_per_sm,
              n_waves,
              min_warp_size,
              cta_share,
              grid_base_choice != "wave",
              factor);
    if (candidate.baseline.iter_unroll_factor == factor &&
        outerGridWarpPlansHardValid(
            candidate,
            total_reduction_numel,
            total_iteration_numel,
            vectorize_factor,
            max_unroll,
            max_threads_per_block)) {
      unroll_options.push_back(
          {"u" + std::to_string(factor),
           "Unroll " + std::to_string(factor) +
               " iteration element(s); recompute reduction unroll"});
      unroll_plans.push_back(std::move(candidate));
    }
    if (factor == native_iter_unroll) {
      break;
    }
  }
  NVF_ERROR(
      !unroll_options.empty() &&
          unroll_options.back().id == "u" + std::to_string(native_iter_unroll),
      "RD9 native iteration unroll violates hard bounds");
  const std::string unroll_state =
      "reduction.outer_unroll_distribution.v1\nreduction_numel=" +
      std::to_string(total_reduction_numel) +
      "\niteration_numel=" + std::to_string(total_iteration_numel) +
      "\ndtype_bits=" +
      std::to_string(max_dtype_size_bit_for_vectorization) +
      "\nsm_count=" + std::to_string(sm_count) +
      "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
      "\nmax_threads_per_block=" + std::to_string(max_threads_per_block) +
      "\nmax_unroll=" + std::to_string(max_unroll) +
      "\nvector_width=" + std::to_string(vectorize_factor) +
      "\nwaves=" + std::to_string(n_waves) +
      "\nwarp=" + std::to_string(min_warp_size) +
      "\ncta_share=" + std::to_string(cta_share) +
      "\ngrid_base=" + grid_base_choice;
  const auto unroll_choice = scheduler::jev::chooseDecision(
      "RD9.unroll",
      unroll_state,
      unroll_options,
      jev_mode == scheduler::jev::Mode::Record
          ? "u" + std::to_string(native_iter_unroll)
          : "");
  const auto unroll_option = std::find_if(
      unroll_options.begin(),
      unroll_options.end(),
      [&](const scheduler::jev::ChoiceOption& option) {
        return option.id == unroll_choice;
      });
  NVF_ERROR(
      unroll_option != unroll_options.end(),
      "RD9 unroll distribution is outside hard-valid complete plans");
  plans = unroll_plans.at(unroll_option - unroll_options.begin());
  // The ragged-wave cleanup is an empirical occupancy preference. Offer the
  // raw grid only after both complete grid variants pass the same native
  // launch, CTA, vectorization, and reduction-coverage bounds.
  auto raw_plans = plans;
  raw_plans.baseline.grdim = plans.base_grdim;
  raw_plans.stride_capped.grdim = plans.stride_grdim;
  const bool raw_valid = outerGridWarpPlansHardValid(
      raw_plans,
      total_reduction_numel,
      total_iteration_numel,
      vectorize_factor,
      max_unroll,
      max_threads_per_block);
  std::vector<scheduler::jev::ChoiceOption> ragged_options = {
      {"clean", "Apply the native ragged-wave grid adjustment"}};
  if (raw_valid &&
      (plans.base_grdim != plans.baseline.grdim ||
       plans.stride_grdim != plans.stride_capped.grdim)) {
    ragged_options.push_back({"raw", "Keep the complete raw reduction grids"});
  }
  const std::string ragged_state =
      "reduction.outer_ragged.v1\nreduction_numel=" +
      std::to_string(total_reduction_numel) +
      "\niteration_numel=" + std::to_string(total_iteration_numel) +
      "\ndtype_bits=" +
      std::to_string(max_dtype_size_bit_for_vectorization) +
      "\nsm_count=" + std::to_string(sm_count) +
      "\nmax_threads_per_block=" + std::to_string(max_threads_per_block) +
      "\nvector_width=" + std::to_string(vectorize_factor) +
      "\nmax_unroll=" + std::to_string(max_unroll) +
      "\nbdimx=" + std::to_string(plans.baseline.bdimx) +
      "\nbdimy=" + std::to_string(plans.baseline.bdimy) +
      "\niter_unroll=" +
      std::to_string(plans.baseline.iter_unroll_factor) +
      "\nredu_unroll=" +
      std::to_string(plans.baseline.redu_unroll_factor) +
      "\ngidim=" + std::to_string(plans.baseline.gidim) +
      "\nbase_grdim=" + std::to_string(plans.base_grdim);
  const auto ragged_choice = scheduler::jev::chooseDecision(
      "RD9.ragged",
      ragged_state,
      ragged_options,
      jev_mode == scheduler::jev::Mode::Record ? "clean" : "");
  NVF_ERROR(
      ragged_choice == "clean" ||
          (ragged_choice == "raw" && ragged_options.size() == 2),
      "RD9 ragged choice is outside hard-valid complete plans");
  if (ragged_choice == "raw") {
    plans = raw_plans;
  }
  const auto& baseline = plans.baseline;
  const auto& stride = plans.stride_capped;
  const std::string state =
      "reduction.outer_grid_stride.v2\nreduction_numel=" +
      std::to_string(total_reduction_numel) +
      "\niteration_numel=" + std::to_string(total_iteration_numel) +
      "\ndtype_bits=" +
      std::to_string(max_dtype_size_bit_for_vectorization) +
      "\nsm_count=" + std::to_string(sm_count) +
      "\nmin_warp_size=" + std::to_string(min_warp_size) +
      "\nbdimx=" + std::to_string(baseline.bdimx) +
      "\nbdimy=" + std::to_string(baseline.bdimy) +
      "\niter_unroll=" + std::to_string(baseline.iter_unroll_factor) +
      "\nredu_unroll=" + std::to_string(baseline.redu_unroll_factor) +
      "\ngidim=" + std::to_string(baseline.gidim) +
      "\nbase_grdim=" + std::to_string(plans.base_grdim) +
      "\nragged_policy=" + ragged_choice;
  std::vector<scheduler::jev::ChoiceOption> options = {
      {"g" + std::to_string(baseline.grdim),
       "Use the baseline reduction grid"}};
  if (stride.grdim != baseline.grdim) {
    options.push_back(
        {"g" + std::to_string(stride.grdim),
         "Use the stride-capped reduction grid"});
  }
  const std::string native_choice =
      jev_mode == scheduler::jev::Mode::Record
      ? "g" + std::to_string(
            plans.native_use_stride ? stride.grdim : baseline.grdim)
      : "";
  const auto selected = scheduler::jev::chooseDecision(
      "RD9.stride", state, options, native_choice);
  NVF_ERROR(
      selected == options.front().id ||
          (options.size() == 2 && selected == options.back().id),
      "RD9 stride plan is outside native candidates");
  return selected == options.front().id ? baseline : stride;
}

// A native grid baseline for comparing the outer block and grid topologies.
// The scope prevents the seven grid-only RD9 sites from producing speculative
// Choice or trace events. This preview does not certify the later Remote
// continuation, whose selected alternatives retain their own hard checks.
OuterReductionParams previewNativeGridOuterReduction(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const int64_t vectorize_factor,
    const int64_t max_unroll,
    const int64_t sm_count,
    const int64_t max_threads_per_sm,
    const int64_t max_threads_per_block) {
  scheduler::jev::ScopedNativeDecisions native_decisions;
  return getGridOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_sm,
      max_threads_per_block);
}

std::unique_ptr<ReductionParams> outerReductionHeuristic(
    const int64_t total_reduction_numel,
    const int64_t total_iteration_numel,
    const int64_t n_tensor_inputs,
    const int64_t max_dtype_size_bit_for_vectorization,
    const size_t max_vectorize_factor,
    const bool has_mufu_computation) {
  // WARNING: Current device for codegen may not be the target device
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t sm_count = (int64_t)dev_prop->multiProcessorCount;
  const int64_t max_threads_per_block = (int64_t)dev_prop->maxThreadsPerBlock;
  const int64_t max_threads_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  // Set register used to store vectorized and unrolled data loaded from gmem.
  // A large value allows more unroll and vectorization, which is beneficial
  // for memory-bound kernels. However, it increases register pressure and may
  // lead to lower occupancy which is bad for compute-bound kernels. In most
  // cases, the scheduler uses 512 threads and to reach an occupancy of 50%,
  // each thread can use up to 64 registers, here only 8 or 32 registers are
  // reserved for unroll and vectorization. The fused ops can have 56 or 32
  // registers for other purposes. Test shows it leads to 50% occupancy for
  // outer reduction without fused ops and 50% occupancy for gelu backward which
  // fused 21 ops including the expensive tanh op.
  // The two original, coupled register-budget policies are performance
  // targets, not allocated registers. The selected maximum unroll feeds the
  // unchanged native block/grid planners. Their final CTA and register caps
  // continue to be calculated from device resources below.
  const auto budgetMaxUnroll = [&](int64_t buffer_reg_count,
                                   int64_t input_factor) {
    return ceilDiv(
        buffer_reg_count * scheduler_utils::bits_per_register /
            max_dtype_size_bit_for_vectorization,
        scheduler_utils::lastPow2(std::max(input_factor, int64_t{1})));
  };
  const int64_t unroll8 = budgetMaxUnroll(8, n_tensor_inputs >> 2);
  const int64_t unroll32 = budgetMaxUnroll(32, n_tensor_inputs);
  const auto jev_mode = scheduler::jev::mode();
  int64_t max_unroll = 0;
  std::string selected_budget;
  if (jev_mode == scheduler::jev::Mode::Off) {
    max_unroll = has_mufu_computation ? unroll8 : unroll32;
  } else {
    NVF_ERROR(
        unroll8 > 0 && unroll32 > 0,
        "RD9 register policies require positive unroll budgets");
    const std::string state =
        "reduction.outer_reg_budget.v1\nn_tensor_inputs=" +
        std::to_string(n_tensor_inputs) +
        "\ndtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) +
        "\nhas_mufu=" + std::to_string(has_mufu_computation ? 1 : 0) +
        "\nbits_per_register=" +
        std::to_string(scheduler_utils::bits_per_register);
    const std::vector<scheduler::jev::ChoiceOption> options = {
        {"b8", "Eight data registers; input factor is input count / 4"},
        {"b32", "32 data registers; input factor is input count"}};
    const std::string native_choice =
        jev_mode == scheduler::jev::Mode::Record
        ? (has_mufu_computation ? "b8" : "b32")
        : "";
    const auto selected = scheduler::jev::chooseDecision(
        "RD9.reg_budget", state, options, native_choice);
    NVF_ERROR(
        selected == "b8" || selected == "b32",
        "RD9 register policy is outside native candidate bounds");
    selected_budget = selected;
    max_unroll = selected == "b8" ? unroll8 : unroll32;
  }

  // The layout-derived factor is the largest aligned width. RD9.width asks
  // for a performance width among its power-of-two divisors. Preflight uses
  // only pure block/grid arithmetic, after the budget Choice has fixed its
  // unroll cap and before any downstream Choice or schedule construction.
  int64_t vectorize_factor = static_cast<int64_t>(max_vectorize_factor);
  if (jev_mode != scheduler::jev::Mode::Off) {
    const bool power_of_two_layout_limit =
        max_vectorize_factor > 0 &&
        max_vectorize_factor <=
            static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
        scheduler_utils::lastPow2(vectorize_factor) == vectorize_factor;
    if (!power_of_two_layout_limit) {
      if (jev_mode == scheduler::jev::Mode::Record) {
        scheduler::jev::recordCoverageSkip(
            "RD9.width", "layout maximum is not a positive power of two");
      } else {
        NVF_ERROR(false, "RD9.width requires a power-of-two layout limit");
      }
    } else {
      std::vector<scheduler::jev::ChoiceOption> options;
      for (int64_t width = 1;; width *= 2) {
        if (outerWidthHardValid(
                total_reduction_numel,
                total_iteration_numel,
                max_dtype_size_bit_for_vectorization,
                width,
                max_unroll,
                sm_count,
                max_threads_per_sm,
                max_threads_per_block)) {
          options.push_back(
              {"w" + std::to_string(width),
               "Vectorize " + std::to_string(width) +
                   " iteration element(s) per thread"});
        }
        if (width == vectorize_factor) {
          break;
        }
      }
      const std::string native_id =
          "w" + std::to_string(max_vectorize_factor);
      const bool native_preflighted = std::any_of(
          options.begin(),
          options.end(),
          [&](const scheduler::jev::ChoiceOption& option) {
            return option.id == native_id;
          });
      if (!native_preflighted && jev_mode == scheduler::jev::Mode::Record) {
        // Conservative preflight may exclude a plan that the original
        // scheduler can still execute. Preserve that native plan in Record.
        scheduler::jev::recordCoverageSkip(
            "RD9.width", "complete-plan preflight excludes native width");
      } else {
        NVF_ERROR(
            native_preflighted,
            "RD9.width has no complete native plan in remote mode");
        const std::string state =
            "reduction.outer_width.v1\nlayout_max=" +
            std::to_string(max_vectorize_factor) +
            "\nreduction_numel=" + std::to_string(total_reduction_numel) +
            "\niteration_numel=" + std::to_string(total_iteration_numel) +
            "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
            "\ndtype_bits=" +
            std::to_string(max_dtype_size_bit_for_vectorization) +
            "\nsm_count=" + std::to_string(sm_count) +
            "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
            "\nmax_threads_per_block=" +
            std::to_string(max_threads_per_block) +
            "\nbits_per_register=" +
            std::to_string(scheduler_utils::bits_per_register) +
            "\nbudget=" + selected_budget;
        const std::string native_choice =
            jev_mode == scheduler::jev::Mode::Record ? native_id : "";
        const auto selected = scheduler::jev::chooseDecision(
            "RD9.width", state, options, native_choice);
        const auto selected_option = std::find_if(
            options.begin(),
            options.end(),
            [&](const scheduler::jev::ChoiceOption& option) {
              return option.id == selected;
            });
        NVF_ERROR(
            selected_option != options.end(),
            "RD9.width response is outside hard-valid candidates");
        vectorize_factor = std::stoll(selected.substr(1));
      }
    }
  }

  // Off retains the original construction order and evaluates the native
  // preference on the realized native grid plan exactly once.
  if (jev_mode == scheduler::jev::Mode::Off) {
    auto grid_params = getGridOuterReduction(
        total_reduction_numel,
        total_iteration_numel,
        n_tensor_inputs,
        max_dtype_size_bit_for_vectorization,
        vectorize_factor,
        max_unroll,
        sm_count,
        max_threads_per_sm,
        max_threads_per_block);
    auto block_params = getBlockOuterReduction(
        total_reduction_numel,
        total_iteration_numel,
        vectorize_factor,
        max_unroll,
        sm_count,
        max_threads_per_block);
    return heuristicParaToSchedulerPara(
        isBetterThan(block_params, grid_params, sm_count) ? block_params
                                                          : grid_params);
  }

  // RD8 sees a complete block plan and an event-free native grid baseline.
  // Only the selected grid branch executes its seven descendant Choices.
  const auto grid_baseline = previewNativeGridOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_sm,
      max_threads_per_block);
  const auto block_params = getBlockOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_block);
  NVF_ERROR(
      block_params.grdim == 1,
      "RD8 requires a completed block-reduction plan");
  const std::string state =
      "reduction.outer_plan.v2\nsm_count=" + std::to_string(sm_count) +
      "\nreduction_numel=" + std::to_string(total_reduction_numel) +
      "\niteration_numel=" + std::to_string(total_iteration_numel) +
      "\nn_tensor_inputs=" + std::to_string(n_tensor_inputs) +
      "\ndtype_bits=" +
      std::to_string(max_dtype_size_bit_for_vectorization) +
      "\nvector_width=" + std::to_string(vectorize_factor) +
      "\nmax_unroll=" + std::to_string(max_unroll) +
      "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
      "\nmax_threads_per_block=" + std::to_string(max_threads_per_block) +
      "\nl2_cache_bytes=" + std::to_string(dev_prop->l2CacheSize) +
      "\nblock_gidim=" + std::to_string(block_params.gidim) +
      "\nblock_grdim=" + std::to_string(block_params.grdim) +
      "\nblock_iter_unroll=" +
      std::to_string(block_params.iter_unroll_factor) +
      "\nblock_bdimx=" + std::to_string(block_params.bdimx) +
      "\nblock_bdimy=" + std::to_string(block_params.bdimy) +
      "\nblock_redu_unroll=" +
      std::to_string(block_params.redu_unroll_factor) +
      "\nblock_redu_serial=" + std::to_string(block_params.redu_serial) +
      "\ngrid_gidim=" + std::to_string(grid_baseline.gidim) +
      "\ngrid_grdim=" + std::to_string(grid_baseline.grdim) +
      "\ngrid_iter_unroll=" +
      std::to_string(grid_baseline.iter_unroll_factor) +
      "\ngrid_bdimx=" + std::to_string(grid_baseline.bdimx) +
      "\ngrid_bdimy=" + std::to_string(grid_baseline.bdimy) +
      "\ngrid_redu_unroll=" +
      std::to_string(grid_baseline.redu_unroll_factor) +
      "\ngrid_redu_serial=" + std::to_string(grid_baseline.redu_serial) +
      "\ngrid_pending_choices=RD9.waves,RD9.warp,RD9.cta,RD9.grid_base,"
      "RD9.unroll,RD9.ragged,RD9.stride";
  const std::vector<scheduler::jev::ChoiceOption> options = {
      {"block",
       "Use completed block reduction plan after shared register budget and "
       "width"},
      {"grid",
       "Use native grid baseline; seven grid-only RD9 choices remain "
       "pending"}};
  const std::string native_choice =
      jev_mode == scheduler::jev::Mode::Record
      ? (isBetterThan(block_params, grid_baseline, sm_count) ? "block" : "grid")
      : "";
  const auto selected =
      scheduler::jev::chooseDecision("RD8", state, options, native_choice);
  NVF_ERROR(
      selected == "block" || selected == "grid",
      "RD8 response is outside the block/grid topology candidates");
  if (selected == "block") {
    return heuristicParaToSchedulerPara(block_params);
  }
  auto grid_params = getGridOuterReduction(
      total_reduction_numel,
      total_iteration_numel,
      n_tensor_inputs,
      max_dtype_size_bit_for_vectorization,
      vectorize_factor,
      max_unroll,
      sm_count,
      max_threads_per_sm,
      max_threads_per_block);
  if (jev_mode == scheduler::jev::Mode::Record) {
    NVF_ERROR(
        outerReductionFields(grid_baseline) == outerReductionFields(grid_params),
        "RD8 native grid preview differs from the live Record continuation");
  }
  return heuristicParaToSchedulerPara(grid_params);
}

std::unique_ptr<ReductionParams> reductionHeuristic(
    const reduction_scheduler_utils::FusionRuntimeProperties& prop) {
  if (prop.fastest_dim_reduction) {
    if (prop.total_reduction_numel == prop.inner_most_dimension_numel) {
      return inner2dReductionHeuristic(
          prop.total_reduction_numel,
          prop.total_iteration_numel,
          (int64_t)prop.n_tensor_inputs,
          (int64_t)prop.max_dtype_size_bit_for_vectorization,
          (int64_t)prop.vectorize_factor,
          prop.has_mufu_computation);
    } else {
      return inner3dReductionHeuristic(
          prop.total_reduction_numel,
          prop.total_iteration_numel,
          prop.inner_most_dimension_numel,
          (int64_t)prop.n_tensor_inputs,
          (int64_t)prop.max_dtype_size_bit_for_vectorization,
          prop.vectorize_factor,
          prop.has_mufu_computation);
    }

  } else {
    // 3D schedules not enabled for outer reductions
    return outerReductionHeuristic(
        prop.total_reduction_numel,
        prop.total_iteration_numel,
        (int64_t)prop.n_tensor_inputs,
        (int64_t)prop.max_dtype_size_bit_for_vectorization,
        prop.vectorize_factor,
        prop.has_mufu_computation);
  }
}

} // namespace

Inner3DWaveTarget previewInner3DWaveTarget(
    int64_t n_elems,
    int64_t uncapped_target_blocks,
    int64_t initial_target_threads,
    int64_t target_unroll,
    int64_t target_iterations,
    int64_t logical_warp,
    int64_t sm_count,
    int64_t max_threads_per_sm,
    int64_t target_waves) {
  const int64_t blocks =
      std::min(uncapped_target_blocks, sm_count * target_waves);
  int64_t threads = initial_target_threads;
  if (blocks * target_unroll * target_iterations < n_elems) {
    threads = std::min(
        ceilDiv(n_elems, blocks * target_unroll),
        ceilDiv(max_threads_per_sm, target_waves));
  }
  if (threads % logical_warp != 0) {
    threads += logical_warp - threads % logical_warp;
  }
  return {blocks, threads};
}

std::vector<Inner3DWaveCandidate> distinctInner3DWaveCandidates(
    const std::vector<Inner3DWaveCandidate>& hard_valid_candidates) {
  std::vector<Inner3DWaveCandidate> distinct;
  for (const auto& candidate : hard_valid_candidates) {
    auto duplicate = std::find_if(
        distinct.begin(), distinct.end(), [&](const auto& existing) {
          return existing.target == candidate.target;
        });
    if (duplicate == distinct.end()) {
      distinct.push_back(candidate);
    } else if (candidate.waves == 4) {
      *duplicate = candidate;
    }
  }
  return distinct;
}

std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  auto heuristic = reductionHeuristic(props);
  heuristic->cparams.index_type = runtime_info.getIndexType();
  return heuristic;
}

// fusion is the input IR that will be modified by this function
void scheduleReduction(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);

  bool unroll = rparams->isUnrolled();

  // Cache inputs if unrolled
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, unroll);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, unroll);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  NVF_ERROR(!reduction_tvs.empty());

  // Registry assumes the reference tv is the first reduction_tv, if this
  // changes registry needs to change.
  auto reduction_tv = reduction_tvs[0];

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    reduction_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(reduction_tv));
  }

  if (isSharded(reduction_tv)) {
    NVF_ERROR(
        !rparams->schedule_3d,
        "Multidevice nvFuser does not support 3D reduction schedules");
  }

  auto dim_analysis = scheduler_utils::canonicalizeReduction(
      fusion, reduction_tv, rparams->fastest_dim && rparams->schedule_3d);

  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;

  NVF_ERROR(
      has_red_axis,
      "Could not find reduction axis in tensor used for reduction scheduler.");

  if (!has_iter_axis) {
    NVF_ERROR(
        rparams->fastest_dim,
        "If all dims are reduction, should be sending it to fastest dim "
        "scheduler.");
  }

  TensorView* reference_tv = reduction_scheduler_utils::scheduleReductionTV(
      rparams, reduction_tv, has_iter_axis);

  // Reduction tensor views and rfactor tensor views are setup. Let's finish off
  // the scheduling, particularly inlining and unrolling.
  NVF_ERROR(
      reference_tv != nullptr && reduction_tv != nullptr,
      "Need these two tensor views to finish the scheduling.");
  const bool is_vectorize =
      rparams->vectorize_inner_reduction || rparams->vectorize_iter_dom;

  // allow iter domain grouped reduction for block and grid outer reductions.
  // TODO: the var name is confusing, should rename
  // [cross_grid/block_inner_reduction] to [cross_grid/block_reduction], see
  // https://github.com/NVIDIA/Fuser/issues/1863
  // grouped welford is only enabled for grid persistent.
  // see validateAndConvertIterDomainGrouping
  const bool has_welford = ir_utils::hasOpsOfType<WelfordOp>(fusion);
  const bool use_iter_grouped_reduction = !rparams->fastest_dim &&
      (has_welford
           ? rparams->cross_grid_inner_reduction && rparams->persistent_kernel
           : rparams->cross_block_inner_reduction);

  scheduler_utils::moveNonConcretizedBroadcastInnermost(fusion, {reference_tv});

  // Propagate transformations before we rfactor the other reductions
  reduction_scheduler_utils::propagateTransformation(reference_tv);
  // If reduction_tv is rfactored, rfactor all reductions.
  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
  }

  const auto& unroll_vectorizable_cached_tvs =
      reduction_scheduler_utils::getCachedTvsToUnrollOrVectorize(
          reference_tv, is_vectorize, cached_inputs, cached_outputs);

  reduction_scheduler_utils::propagateParallelization(
      reduction_tv,
      reference_tv,
      unroll,
      use_iter_grouped_reduction,
      reduction_tvs,
      unroll_vectorizable_cached_tvs);

  // Inline the schedule
  inlineMost();

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  markAliases(fusion);
}
} // namespace non_tma
} // namespace reduction
} // namespace nvfuser
