// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/pointwise_non_tma.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "debug.h"
#include "exceptions.h"
#include "ir/printer.h"
#include "multidevice/utils.h"
#include "scheduler/cache_policy_refiner.h"
#include "scheduler/jev_client.h"
#include "scheduler/mark_aliases.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"
#include "scheduler/vectorize_helper.h"

namespace nvfuser {
namespace pointwise {
namespace non_tma {
namespace {
// constexpr int64_t x_grid_limit = ((int64_t)1 << (int64_t)31) - (int64_t)1;
// Unused at the moment, commenting for clang tidy
constexpr int64_t kThreadX = 128;

std::vector<TensorView*> getUniqueVectorizableInputs(
    const std::vector<TensorView*>& vectorizable_io_tvs) {
  std::vector<TensorView*> inputs;
  for (auto* tv : vectorizable_io_tvs) {
    if (tv->isFusionInput() &&
        std::find(inputs.begin(), inputs.end(), tv) == inputs.end()) {
      inputs.push_back(tv);
    }
  }
  return inputs;
}

// Classify each unique vectorizable input for the non-outer broadcast cost.
// vectorizable_inputs: all vectorizable inputs
// break_point: the break point of the broadcast flags.
// Used to determine the influence of inputs on unroll factor.
// outer broadcast inputs are not counted since outer unroll is used
// and they are only loaded once regardless of the unroll factor due to
// the re-use across different unrolled iterations.
std::vector<bool> getOuterBcastInputFlags(
    const std::vector<TensorView*>& vectorizable_inputs,
    int64_t break_point) {
  if (break_point == 0) {
    return std::vector<bool>(vectorizable_inputs.size(), false);
  }

  // Returns true if tv is outer broadcast tv or is used by outer broadcast
  // op.
  auto isUsedByOuterBcast = [&break_point](TensorView* tv) {
    // If all the dims to the left of the break point are broadcast, then
    // this tv is considered as an outer broadcast.
    const auto& domains = tv->getLogicalDomain();
    if (std::all_of(
            domains.begin(), domains.begin() + break_point, [](IterDomain* id) {
              return id->isBroadcast();
            })) {
      return true;
    }
    // check consumers
    const auto& all_consumers = DependencyCheck::getAllDependentVals({tv});
    for (auto tv : all_consumers) {
      if (tv->definition()->isA<BroadcastOp>()) {
        const auto& bcast_flags =
            tv->definition()->as<BroadcastOp>()->getBroadcastDimFlags();

        if (std::all_of(
                bcast_flags.begin(),
                bcast_flags.begin() + break_point,
                [](bool flag) { return flag; })) {
          return true;
        }
      }
    }
    return false;
  };
  std::vector<bool> flags;
  flags.reserve(vectorizable_inputs.size());
  for (auto tv : vectorizable_inputs) {
    flags.push_back(isUsedByOuterBcast(tv));
  }
  return flags;
}

int64_t getNumOfNonOuterBcastInputs(
    const std::vector<TensorView*>& vectorizable_inputs,
    int64_t break_point) {
  const auto flags = getOuterBcastInputFlags(vectorizable_inputs, break_point);
  const int64_t n_non_bcast_inputs =
      static_cast<int64_t>(std::count(flags.begin(), flags.end(), false));
  // return 1 if no non-outer broadcast inputs to avoid division by 0
  return std::max(n_non_bcast_inputs, 1L);
}

// calculate unroll factor based on inputs and computations.
int64_t getEmpiricalUnrollFactor(
    Fusion* fusion,
    int64_t break_point,
    int64_t vectorization_bits,
    std::vector<TensorView*> vectorizable_inputs) {
  // no need to unroll if no vectorizable inputs
  if (vectorizable_inputs.empty()) {
    return 1;
  }
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // calculate the required bytes in flight to cover the latency.
  // assuming 100% occupancy.
  int64_t required_bits_per_thread =
      scheduler_utils::getRequiredBitsInFlight() /
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  int64_t unroll_factor =
      std::max(1L, required_bits_per_thread / vectorization_bits);
  // If unroll is required, further scale up with computation cost and scale
  // down with input counts. Won't be triggered on A100 and H100.
  if (unroll_factor > 1) {
    int64_t computation_factor =
        scheduler_utils::getComputationCostFactor(fusion);
    unroll_factor *= computation_factor;
    int64_t n_inputs_factor =
        getNumOfNonOuterBcastInputs(vectorizable_inputs, break_point);
    unroll_factor = scheduler_utils::safeDiv(unroll_factor, n_inputs_factor);
  }
  return unroll_factor;
}

// calculate unroll factor based on total blocks to ensure we still
// have 8 waves after unroll.
int64_t getElementBasedUnrollFactor(
    int64_t total_blocks,
    int64_t threads_per_block) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t target_waves = 8L;
  int64_t max_block_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor / threads_per_block;
  int64_t n_waves_wo_unroll = ceilDiv(
      total_blocks, max_block_per_sm * (int64_t)dev_prop->multiProcessorCount);
  int64_t n_elems_limited_unroll =
      scheduler_utils::roundUpPow2(ceilDiv(n_waves_wo_unroll, target_waves));
  return n_elems_limited_unroll;
}

// Block quantization requires a hard unroll factor of one.
bool hasBlockQuantizationOps(
    Fusion* fusion,
    HeuristicDataCache* data_cache) {
  auto block_quantization_ops =
      HeuristicDataCacheEntry<HeuristicCompileTime::BlockQuantizationOps>(
          data_cache,
          [fusion]() {
            auto ops = std::make_unique<std::vector<Expr*>>();
            // Get BlockQuantizationOp operations
            auto block_quant_ops =
                ir_utils::getOpsOfType<BlockQuantizationOp>(fusion);
            ops->insert(
                ops->end(), block_quant_ops.begin(), block_quant_ops.end());
            // Get GroupedBlockQuantizationOp operations
            auto grouped_ops =
                ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion);
            ops->insert(ops->end(), grouped_ops.begin(), grouped_ops.end());
            return ops;
          })
          .get();

  return !block_quantization_ops.empty();
}

// Balance memory latency, TLP and non-divisible tails.
int64_t getUnrollFactor(
    Fusion* fusion,
    int64_t break_point,
    int64_t total_blocks,
    int64_t threads_per_block,
    int64_t vectorization_bits,
    bool divisible_split,
    std::vector<TensorView*> vectorizable_io_tvs,
    HeuristicDataCache* data_cache) {
  // The runtime block-quantization function assumes no unrolling.
  if (hasBlockQuantizationOps(fusion, data_cache)) {
    // Runtime function implementing Block Quantization Op requires unroll
    // factor to be 1
    return 1;
  }

  auto vectorizable_inputs =
      getUniqueVectorizableInputs(vectorizable_io_tvs);

  int64_t empirical_unroll = getEmpiricalUnrollFactor(
      fusion, break_point, vectorization_bits, vectorizable_inputs);

  // limit unroll factor when n_elems is small to ensure enough
  // blocks for thread level parallelism.
  int64_t n_elems_limited_unroll =
      getElementBasedUnrollFactor(total_blocks, threads_per_block);

  // Avoid unrolling when the unroll factor is constrained by `n_elems` and the
  // split is not divisible. Why? While unrolling increases instruction-level
  // parallelism (ILP), it decreases thread-level parallelism (TLP). A
  // non-divisible split further reduces the number of effective threads, which
  // negatively impacts TLP. Therefore, if the kernel lacks sufficient TLP,
  // unrolling should be avoided.
  if (n_elems_limited_unroll < empirical_unroll && !divisible_split) {
    return 1;
  } else {
    return std::min(n_elems_limited_unroll, empirical_unroll);
  }
}

struct ExpensiveOpCounts {
  int64_t tanh = 0;
  int64_t exp = 0;
  int64_t reciprocal = 0;
};

ExpensiveOpCounts countExpensiveOps(Fusion* fusion) {
  ExpensiveOpCounts counts;
  for (auto* expr : fusion->exprs()) {
    if (auto* unary = dynamic_cast<UnaryOp*>(expr)) {
      switch (unary->getUnaryOpType()) {
        case UnaryOpType::Tanh:
          ++counts.tanh;
          break;
        case UnaryOpType::Exp:
          ++counts.exp;
          break;
        case UnaryOpType::Reciprocal:
          ++counts.reciprocal;
          break;
        default:
          break;
      }
    }
  }
  return counts;
}

struct FactorRange {
  int64_t low;
  int64_t high;
  std::string id;
};

std::vector<FactorRange> getUnrollRanges(int64_t low, int64_t high) {
  NVF_ERROR(low >= 1 && high >= low);
  constexpr int64_t kMaxChoices = 255;
  const int64_t span = high - low + 1;
  const int64_t stride = span / kMaxChoices + (span % kMaxChoices != 0);
  std::vector<FactorRange> ranges;
  for (int64_t first = low;;) {
    const int64_t last = first + std::min(stride - 1, high - first);
    const std::string id = first == last
        ? "u" + std::to_string(first)
        : "r" + std::to_string(first) + "_" + std::to_string(last);
    ranges.push_back({first, last, id});
    if (last == high) {
      break;
    }
    first = last + 1;
  }
  return ranges;
}

} // namespace

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  FusionGuard fg(fusion);
  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise heuristics";
  // Incase any buffer is of type DataType::Index
  const auto index_type = prop.index_type;
  params->cparams.index_type = index_type;

  int64_t max_dtype_size_bit_for_vectorization =
      prop.max_dtype_size_bit_for_vectorization;
  const auto& vectorizable_inputs_outputs = prop.vectorizable_inputs_outputs;
  int64_t n_elems = prop.n_elems;
  const auto device_multiprocessor_count = prop.device_multiprocessor_count;
  constexpr int64_t max_vectorization_size_in_bit = 128;

  // See pointwise.h to understand what we're doing for this 2D analysis.
  // Ideal break point location
  int break_point = 0;

  // If break_point, mark if BIDy and BIDx should be positionally reversed
  // relative to root domains
  bool flip_grid_binding = false;

  // Elements on the right of break point (without break point all are on the
  // right)
  int64_t right_elem_count = 0;

  int64_t bdimx = kThreadX;

  // bdimy may be used if the right side of the break point is not large and we
  // need to expand block level parallelism into the left side of the break
  // point.
  int64_t bdimy = 1;

  // In 2D scheduler gdim_left is used to parallelize the left side of the break
  // point.
  int64_t gdim_left = 1;

  // gdim_right is used if there's too much parallelization in the right side of
  // the break point. We will expand grid parallelization into the right side of
  // the break point with gdim_left and use gdim_right for the left side of the
  // break point.
  int64_t gdim_right = 1;

  // PW2.entry is the performance choice to search for a 2D break point at all.
  // Both choices are legal: the default 1D plan requires no view split, while
  // getBreakPoint keeps the view-disjointness checks for every 2D candidate.
  const auto jev_mode = scheduler::jev::mode();
  bool analyze_2d_break_point = false;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    analyze_2d_break_point =
        n_elems * 2 > device_multiprocessor_count * kThreadX;
  }
  if (jev_mode != scheduler::jev::Mode::Off) {
    const std::string state =
        "pointwise.entry.v1\nn_elems=" + std::to_string(n_elems) +
        "\nsm_count=" + std::to_string(device_multiprocessor_count);
    const std::vector<scheduler::jev::ChoiceOption> options = {
        {"analyze", "Analyze hard-legal two-dimensional break points"},
        {"default_1d",
         "Use the one-dimensional default without split analysis"}};
    std::string native_choice;
    if (jev_mode == scheduler::jev::Mode::Record) {
      native_choice = analyze_2d_break_point ? "analyze" : "default_1d";
    }
    const auto selected = scheduler::jev::chooseDecision(
        "PW2.entry", state, options, native_choice);
    analyze_2d_break_point = selected == "analyze";
  }

  // Only calculate break point if its performance choice requests analysis.
  pointwise_utils::BreakPointInfo bp_info;
  if (analyze_2d_break_point) {
    bp_info = pointwise_utils::getBreakPoint(
        fusion,
        prop,
        data_cache,
        /*is_tma =*/false,
        max_vectorization_size_in_bit /
            prop.max_dtype_size_bit_for_vectorization,
        kThreadX);
  } else {
    // Use default 1D scheduling (break_point = 0)
    bp_info.break_point = 0;
    bp_info.flip_grid_binding = false;
    bp_info.right_elem_count = 0;
    bp_info.is_outer_broadcast_dominated = false;
  }

  const auto logical_reorder_map = pointwise_utils::getLogicalReorderMap(
      prop.largest_out, prop.has_reshapes, data_cache);
  int64_t vectorization_factor = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    // Retain the original calculation in off and record modes. Remote mode
    // must never compute the soft width before asking the local model.
    vectorization_factor = vectorize_helper::getVectorizationFactor(
        runtime_info,
        prop.largest_out,
        data_cache,
        bp_info.break_point,
        max_vectorization_size_in_bit,
        prop.min_dtype_size_bit_for_vectorization,
        prop.max_dtype_size_bit_for_vectorization,
        /*n_vectorizable_tensors=*/(int64_t)vectorizable_inputs_outputs.size(),
        // Keep the source's existing one-wave or disabled-wave approximation.
        /*n_waves=*/n_elems > device_multiprocessor_count * kThreadX ? -1 : 1,
        logical_reorder_map);
  }
  if (jev_mode != scheduler::jev::Mode::Off) {
    // Calling the same layout analysis with optional soft constraints disabled
    // gives a correctness-derived upper bound, not the native preferred width.
    const int64_t layout_max = vectorize_helper::getVectorizationFactor(
        runtime_info,
        prop.largest_out,
        data_cache,
        bp_info.break_point,
        max_vectorization_size_in_bit,
        /*min_dtype_size_bit=*/-1,
        /*max_dtype_size_bit=*/-1,
        /*n_vectorizable_tensors=*/-1,
        /*n_waves=*/-1,
        logical_reorder_map);
    const int64_t min_dtype_bits = prop.min_dtype_size_bit_for_vectorization;
    const int64_t byte_min = min_dtype_bits < 8
        ? scheduler_utils::safeDiv(8L, min_dtype_bits)
        : 1;
    NVF_ERROR(
        layout_max >= 1 && layout_max <= max_vectorization_size_in_bit &&
            scheduler_utils::lastPow2(layout_max) == layout_max,
        "PW3.width layout limit must be a power of two in 1..128");
    NVF_ERROR(
        scheduler_utils::lastPow2(byte_min) == byte_min,
        "PW3.width byte minimum must be a power of two");

    std::vector<scheduler::jev::ChoiceOption> options;
    const int64_t first_width = std::min(byte_min, layout_max);
    for (int64_t width = first_width; width <= layout_max; width *= 2) {
      options.push_back(
          {"w" + std::to_string(width),
           "Vectorize " + std::to_string(width) + " element(s) per thread"});
    }
    // Current nvFuser returns layout_max if the byte floor exceeds it; the
    // singleton preserves that existing behavior until the source TODO is
    // resolved. In the normal case all options satisfy both hard constraints.
    std::string state = "pointwise.width.v1\n";
    state += "layout_max=" + std::to_string(layout_max) + '\n';
    state += "min_dtype_bits=" + std::to_string(min_dtype_bits) + '\n';
    state += "max_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) + '\n';
    state += "n_vectorizable_tensors=" +
        std::to_string(vectorizable_inputs_outputs.size()) + '\n';
    state += "n_elems=" + std::to_string(n_elems) + '\n';
    state += "sm_count=" + std::to_string(device_multiprocessor_count) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "w" + std::to_string(vectorization_factor)
        : std::string();
    const std::string selected = scheduler::jev::chooseDecision(
        "PW3.width", state, options, native_choice);
    bool matched = false;
    for (int64_t width = first_width; width <= layout_max; width *= 2) {
      if (selected == "w" + std::to_string(width)) {
        vectorization_factor = width;
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "PW3.width response is outside native hard bounds");
  }
  params->vectorization_factor = vectorization_factor;

  // Calculate block and grid configuration based on break point
  auto config = pointwise_utils::getBlockGridConfig(
      prop, bp_info, vectorization_factor, kThreadX);

  // Extract results
  break_point = config.break_point;
  flip_grid_binding = config.flip_grid_binding;
  right_elem_count = config.right_elem_count;
  bdimx = config.bdimx;
  bdimy = config.bdimy;
  gdim_left = config.gdim_left;
  gdim_right = config.gdim_right;
  bool is_outer_broadcast_dominated = config.is_outer_broadcast_dominated;
  const auto& elem_counts = prop.elem_counts;

  // Each eligible 1D CTA choice defines a complete thread/grid mapping. A
  // non-native choice uses a symbolic split so different launch dimensions
  // can reuse the same scheduled kernel. The native 128 path remains literal.
  if (break_point == 0 && jev_mode != scheduler::jev::Mode::Off) {
    const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
    const int64_t warp_size = dev_prop->warpSize;
    const bool native_cta_valid =
        warp_size > 0 && kThreadX % warp_size == 0 &&
        kThreadX <= dev_prop->maxThreadsDim[0] &&
        kThreadX <= dev_prop->maxThreadsPerBlock &&
        kThreadX <= dev_prop->maxThreadsPerMultiProcessor &&
        dev_prop->maxGridSize[0] > 0 &&
        ceilDiv(ceilDiv(n_elems, vectorization_factor), kThreadX) <=
            dev_prop->maxGridSize[0];
    if (hasBlockQuantizationOps(fusion, data_cache) || !native_cta_valid) {
      scheduler::jev::recordCoverageSkip(
          "PW3.cta_1d",
          !native_cta_valid ? "native_1d_cta_preflight" : "block_quantization");
    } else {
      std::vector<scheduler::jev::ChoiceOption> options;
      for (int64_t candidate = warp_size; candidate <= kThreadX;
           candidate *= 2) {
        if (ceilDiv(ceilDiv(n_elems, vectorization_factor), candidate) >
            dev_prop->maxGridSize[0]) {
          continue;
        }
        options.push_back(
            {"x" + std::to_string(candidate),
             "Use " + std::to_string(candidate) + " threads per 1D CTA"});
      }
      std::string state = "pointwise.launch_1d.v1\n";
      state += "n_elems=" + std::to_string(n_elems) + '\n';
      state += "vector_width=" + std::to_string(vectorization_factor) + '\n';
      state += "warp_size=" + std::to_string(warp_size) + '\n';
      state += "max_threads_x=" +
          std::to_string(dev_prop->maxThreadsDim[0]) + '\n';
      state += "max_threads_per_block=" +
          std::to_string(dev_prop->maxThreadsPerBlock) + '\n';
      state += "max_threads_per_sm=" +
          std::to_string(dev_prop->maxThreadsPerMultiProcessor) + '\n';
      state += "max_grid_x=" +
          std::to_string(dev_prop->maxGridSize[0]) + '\n';
      const auto selected = scheduler::jev::chooseDecision(
          "PW3.cta_1d", state, options,
          jev_mode == scheduler::jev::Mode::Record ? "x128" : "");
      bool matched = false;
      for (const auto& option : options) {
        if (selected == option.id) {
          bdimx = std::stoll(option.id.substr(1));
          matched = true;
          break;
        }
      }
      NVF_ERROR(matched, "PW3.cta_1d response is outside hard CTA bounds");
      params->symbolic_1d_cta = bdimx != kThreadX;
    }
  }

  // get unroll factor:
  int64_t total_blocks = break_point > 0
      ? gdim_left * gdim_right
      : ceilDiv(n_elems / vectorization_factor, bdimx);
  bool divisible_split = break_point > 0
      ? (right_elem_count % (params->vectorization_factor * bdimx) == 0)
      : (n_elems % (params->vectorization_factor * bdimx) == 0);
  int64_t unroll_factor = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    unroll_factor = getUnrollFactor(
        fusion,
        break_point,
        total_blocks,
        break_point > 0 ? kThreadX : bdimx,
        params->vectorization_factor * max_dtype_size_bit_for_vectorization,
        divisible_split,
        vectorizable_inputs_outputs,
        data_cache);
  }
  if (jev_mode != scheduler::jev::Mode::Off) {
    // Any positive integer up to max(1, total_blocks) can be represented by
    // the pointwise split, including non-power-of-two factors and tails. The
    // source's element-based cap also guarantees its native choice is in this
    // range. Partition large ranges to respect Jev's 255-choice limit.
    const bool block_quantization =
        hasBlockQuantizationOps(fusion, data_cache);
    const int64_t hard_max =
        block_quantization ? 1 : std::max<int64_t>(1, total_blocks);
    NVF_ERROR(
        jev_mode != scheduler::jev::Mode::Record ||
            (unroll_factor >= 1 && unroll_factor <= hard_max),
        "PW3.unroll native factor exceeds the schedule's hard range");

    const auto inputs = getUniqueVectorizableInputs(vectorizable_inputs_outputs);
    const auto outer_flags = getOuterBcastInputFlags(inputs, break_point);
    const auto op_counts = countExpensiveOps(fusion);
    const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
    int memory_clock_khz = 0;
    NVF_ERROR(
        cudaDeviceGetAttribute(
            &memory_clock_khz,
            cudaDevAttrMemoryClockRate,
            at::cuda::current_device()) == cudaSuccess,
        "PW3.unroll could not read CUDA memory clock");
    std::string flags = outer_flags.empty() ? "-" : std::string();
    for (size_t i = 0; i < outer_flags.size(); ++i) {
      if (i != 0) {
        flags += ',';
      }
      flags += (outer_flags[i] ? '1' : '0');
    }
    std::string state_base = "pointwise.unroll.v1\n";
    state_base += "axis=" +
        std::string(is_outer_broadcast_dominated ? "outer" : "inner") + '\n';
    state_base += "break_point=" + std::to_string(break_point) + '\n';
    state_base += "n_elems=" + std::to_string(n_elems) + '\n';
    state_base += "right_elems=" + std::to_string(right_elem_count) + '\n';
    state_base += "bdimx=" + std::to_string(bdimx) + '\n';
    state_base += "gdim_left=" + std::to_string(gdim_left) + '\n';
    state_base += "gdim_right=" + std::to_string(gdim_right) + '\n';
    state_base +=
        "vector_width=" + std::to_string(vectorization_factor) + '\n';
    state_base += "max_dtype_bits=" +
        std::to_string(max_dtype_size_bit_for_vectorization) + '\n';
    state_base += "input_outer_bcast_flags=" + flags + '\n';
    state_base += "tanh_ops=" + std::to_string(op_counts.tanh) + '\n';
    state_base += "exp_ops=" + std::to_string(op_counts.exp) + '\n';
    state_base += "reciprocal_ops=" + std::to_string(op_counts.reciprocal) + '\n';
    state_base += "sm_count=" +
        std::to_string(dev_prop->multiProcessorCount) + '\n';
    state_base += "max_threads_per_sm=" +
        std::to_string(dev_prop->maxThreadsPerMultiProcessor) + '\n';
    state_base += "memory_bus_bits=" +
        std::to_string(dev_prop->memoryBusWidth) + '\n';
    state_base += "memory_clock_khz=" +
        std::to_string(memory_clock_khz) + '\n';
    state_base += "block_quantization=" +
        std::to_string(static_cast<int>(block_quantization)) + '\n';

    int64_t range_low = 1;
    int64_t range_high = hard_max;
    for (;;) {
      const auto ranges = getUnrollRanges(range_low, range_high);
      std::vector<scheduler::jev::ChoiceOption> options;
      options.reserve(ranges.size());
      std::string native_choice;
      for (const auto& range : ranges) {
        options.push_back(
            {range.id,
             "Unroll factor in [" + std::to_string(range.low) + ", " +
                 std::to_string(range.high) + "]"});
        if (jev_mode == scheduler::jev::Mode::Record &&
            range.low <= unroll_factor && unroll_factor <= range.high) {
          native_choice = range.id;
        }
      }
      const std::string state = state_base +
          "range_low=" + std::to_string(range_low) + '\n' +
          "range_high=" + std::to_string(range_high) + '\n';
      const auto selected = scheduler::jev::chooseDecision(
          "PW3.unroll", state, options, native_choice);
      const auto selected_range = std::find_if(
          ranges.begin(), ranges.end(), [&](const FactorRange& range) {
            return range.id == selected;
          });
      NVF_ERROR(
          selected_range != ranges.end(),
          "PW3.unroll response is outside native hard bounds");
      if (selected_range->low == selected_range->high) {
        unroll_factor = selected_range->low;
        break;
      }
      range_low = selected_range->low;
      range_high = selected_range->high;
    }
    NVF_ERROR(
        !block_quantization || unroll_factor == 1,
        "PW3.unroll violated the block-quantization guard");
  }

  if (is_outer_broadcast_dominated) {
    params->unroll_factor_outer = unroll_factor;
  } else {
    params->unroll_factor_inner = unroll_factor;
  }
  gdim_left = ceilDiv(gdim_left, params->unroll_factor_outer);
  gdim_right = ceilDiv(gdim_right, params->unroll_factor_inner);

  NVF_ERROR(right_elem_count > 0 || break_point == 0);

  params->break_point = break_point;
  params->flip_grid_binding = flip_grid_binding;
  params->split_block = bdimy > 1;

  params->lparams.bind(bdimx, ParallelType::TIDx);
  if (params->split_block) {
    params->lparams.bind(bdimy, ParallelType::TIDy);
  }
  if ((flip_grid_binding && gdim_right > 65535) ||
      (!flip_grid_binding && gdim_left > 65535)) {
    params->split_grid_y_dim = true;
  }

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    // Get broadcast info for debug output
    TensorView* largest_out = prop.largest_out;
    auto broadcast_info_for_debug =
        scheduler_utils::getBroadcastMultiples(largest_out, index_type);

    debug() << "\n===== Pointwise Stats ========\n"
            << "num_elems: " << n_elems << "\n"
            << "elem_counts: " << elem_counts << "\n"
            << "max_dtype_size_bit_for_vectorization: "
            << max_dtype_size_bit_for_vectorization << "\n"
            << "unroll_factor_inner: " << params->unroll_factor_inner
            << std::endl
            << "unroll_factor_outer: " << params->unroll_factor_outer
            << std::endl
            << "vectorize_factor: " << params->vectorization_factor << std::endl
            << "\n"
            << "logical_reorder_map: ";
    for (auto [i, j] : pointwise_utils::getLogicalReorderMap(
             prop.largest_out, prop.has_reshapes, data_cache)) {
      debug() << "(" << i << ", " << j << "), ";
    }
    debug() << "\nbroadcast_byte_multiples: ";
    for (auto multiple : broadcast_info_for_debug.broadcast_multiples) {
      debug() << "(" << multiple.lhs_multiple << ", " << multiple.rhs_multiple
              << "), ";
    }
    debug() << "\nLHS elems: "
            << (right_elem_count > 0 ? n_elems / right_elem_count : 0)
            << " RHS elems: " << right_elem_count << std::endl;
    debug() << std::endl;
    debug() << params->toString() << std::endl;
  }

  return params;
}

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  auto schedule_info_opt =
      pointwise_utils::commonPointwiseSchedule(fusion, pparams->break_point);
  if (!schedule_info_opt.has_value()) {
    // Zero-dimensional tensors, nothing to schedule
    return;
  }
  auto& schedule_info = schedule_info_opt.value();

  auto& cached_inputs = schedule_info.cached_inputs;
  TensorView* reference_tv = schedule_info.reference_tv;

  int64_t unswitch_pos = 0;
  IterDomain* vectorize_id = nullptr;
  if (pparams->break_point) {
    // 2D parallelization scheme
    int64_t lhs_i = schedule_info.lhs_i;
    int64_t rhs_i = schedule_info.rhs_i;
    NVF_ERROR(rhs_i >= 0 && lhs_i >= 0);

    // Right (inner merged) dimension is at inner most position, left (outer
    // merged) dimension is at lhs_i. Order as [lhs_i, rhs_i, unmerged...]
    reference_tv->reorder({{lhs_i, 0}, {-1, 1}});

    // vectorization without unroll
    if (pparams->unroll_factor_outer == 1 &&
        pparams->unroll_factor_inner == 1 &&
        pparams->vectorization_factor > 1) {
      reference_tv->split(1, pparams->vectorization_factor);
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      reference_tv->split(0, 1);
      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // Here and in the following comments:
      // prefix [i] represent inner dimension
      // prefix [o] represent inner dimension
      // [|] separates the outer and inner dimensions
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      reference_tv->axis(3)->parallelize(ParallelType::TIDx);
      // Vectorization are propagated separately
      vectorize_id = reference_tv->axis(4);

      // [outer, Unswitch | i-remainder, TIDx, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 2}, {2, 1}, {3, 4}, {4, 3}});
      //[outer | i-remainder, Unswitch, Vectorization, TIDx]
    } else {
      // [outer | inner]
      if (pparams->vectorization_factor > 1) {
        reference_tv->split(1, pparams->vectorization_factor);
      }
      // [outer | i-remainder, Vect]
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDx));
      // [outer | i-remainder, TIDx, Vect]

      if (pparams->unroll_factor_inner > 1) {
        reference_tv->split(1, pparams->unroll_factor_inner);
      }
      // [outer| i-remainder, i-Unroll, TIDx, Vect]

      if (pparams->unroll_factor_outer > 1) {
        reference_tv->split(0, pparams->unroll_factor_outer);
      }
      // [o-remainder, o-Unroll, | i-remainder, i-Unroll, TIDx, Vect]

      reference_tv->split(0, 1);
      // [o-remainder, Unswitch, o-Unroll | i-remainder, i-Unroll, TIDx, Vect]

      int i_remainder_pos = pparams->unroll_factor_outer > 1 ? 3 : 2;
      reference_tv->reorder({{i_remainder_pos, 1}});
      // [o-remainder, i-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]

      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Here we do not set axis(3)->parallelize(Unroll) because we do not want
      // it to be propagated. We manually unroll by splitting the inline
      // propagation process into two steps:
      // step 1: inline at the unswitch position for cached inputs and outputs
      // step 2: inline at the inner most dim for the rest of the graph
      int tidx_pos = 3;
      if (pparams->unroll_factor_inner > 1) {
        tidx_pos++;
      }
      if (pparams->unroll_factor_outer > 1) {
        tidx_pos++;
      }
      reference_tv->axis(tidx_pos)->parallelize(ParallelType::TIDx);
      if (pparams->vectorization_factor > 1) {
        // can't use {-1}, there may be deviceId
        vectorize_id = reference_tv->axis(tidx_pos + 1);
      }
      // [o-remainder, i-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
    }

    // Move out of the way to furthest left point
    reference_tv->reorder({{1, 0}});
    // [i-remainder, o-remainder, Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
    if (pparams->split_block) {
      reference_tv->split(1, NamedScalar::getParallelDim(ParallelType::TIDy));
      // [i-remainder, o-remainder, TIDy, Unswitch, o-Unroll, i-Unroll, TIDx,
      // Vect]
      if (pparams->flip_grid_binding) {
        // [BIDy | BIDx, TIDy | Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (pparams->split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx, TIDy | Unswitch, o-Unroll,
          // i-Unroll, TIDx, Vect]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 5;
        } else {
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        }
      } else {
        // [BIDx | BIDy TIDy | Unswitch, o-Unroll, i-Unroll, TIDx, Vect]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        reference_tv->axis(2)->parallelize(ParallelType::TIDy);
        if (pparams->split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535}, TIDy | Unswitch, o-Unroll,
          // i-Unroll, TIDx, Vect]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
          unswitch_pos = 5;
        } else {
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        }
      }
    } else {
      // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
      if (pparams->flip_grid_binding) {
        // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
        reference_tv->axis(1)->parallelize(ParallelType::BIDx);
        if (pparams->split_grid_y_dim) {
          // [i-remainder, BIDy{65535} | BIDx | Unswitch, Unroll, TIDx]
          reference_tv->split(0, 65535);
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        } else {
          // [BIDy | BIDx | Unswitch, Unroll, TIDx, Vect]
          reference_tv->axis(0)->parallelize(ParallelType::BIDy);
          unswitch_pos = 3;
        }
      } else {
        // [BIDx | BIDy | Unswitch, Unroll, TIDx, Vect]
        reference_tv->axis(0)->parallelize(ParallelType::BIDx);
        if (pparams->split_grid_y_dim) {
          // [BIDx | i-remainder, BIDy{65535} | Unswitch, Unroll, TIDx, Vect]
          reference_tv->split(1, 65535);
          reference_tv->axis(2)->parallelize(ParallelType::BIDy);
          unswitch_pos = 4;
        } else {
          // [BIDx | BIDy | Unswitch, Unroll, TIDx, Vect]
          reference_tv->axis(1)->parallelize(ParallelType::BIDy);
          unswitch_pos = 3;
        }
      }
    }
  } else {
    // 1D Scheduler (break_point == 0)
    int64_t lhs_i = schedule_info.lhs_i;
    int64_t rhs_i = schedule_info.rhs_i;
    NVF_ERROR(rhs_i >= 0 && lhs_i == -1);

    // right hand side exists and is the only axis we care to schedule, move
    // it from the inner most position to left most. Order as [rhs_i,
    // unmerged...]
    reference_tv->reorder({{-1, 0}});

    if (pparams->unroll_factor_inner == 1 &&
        pparams->vectorization_factor > 1) {
      // Vectorize
      reference_tv->split(0, pparams->vectorization_factor);
      // Unswitch
      reference_tv->split(0, 1);
      // Threads
      if (pparams->symbolic_1d_cta) {
        reference_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDx));
      } else {
        reference_tv->split(0, kThreadX);
      }

      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::TIDx);
      reference_tv->axis(2)->parallelize(ParallelType::Unswitch);
      // Vectorization are propagated separately
      vectorize_id = reference_tv->axis(3);

      //[BIDx, TIDx, Unswitch, Vectorization]
      // To make consistent with unrolling:
      reference_tv->reorder({{1, 3}, {2, 1}, {3, 2}});
      //[BIDx, Unswitch, Vectorization, TIDx]
    } else {
      // Vectorize
      if (pparams->vectorization_factor > 1) {
        reference_tv->split(0, pparams->vectorization_factor);
      }
      // Threads
      if (pparams->symbolic_1d_cta) {
        reference_tv->split(
            0, NamedScalar::getParallelDim(ParallelType::TIDx));
      } else {
        reference_tv->split(0, kThreadX);
      }
      // Unroll
      if (pparams->unroll_factor_inner > 1) {
        reference_tv->split(0, pparams->unroll_factor_inner);
      }
      // Unswitch
      reference_tv->split(0, 1);

      // [BIDx, Unswitch, Unroll, TIDx, Vect]
      reference_tv->axis(0)->parallelize(ParallelType::BIDx);
      reference_tv->axis(1)->parallelize(ParallelType::Unswitch);
      // Here we do not set axis(2)->parallelize(Unroll) because we do not want
      // it to be propagated. We manually unroll by splitting the inline
      // propagation process into two steps:
      // step 1: inline at the unswitch position for cached inputs and outputs
      // step 2: inline at the inner most dim for the rest of the graph
      int tidx_pos = pparams->unroll_factor_inner > 1 ? 3 : 2;
      reference_tv->axis(tidx_pos)->parallelize(ParallelType::TIDx);
      if (pparams->vectorization_factor > 1) {
        vectorize_id = reference_tv->axis(tidx_pos + 1);
      }
    }
    unswitch_pos = 2;
  }

  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree spanning_tree(reference_tv);
  spanning_tree.traverse(&propagator);
  scheduler_utils::parallelizeAllLike(reference_tv);

  // We first vectorize the quantized outputs of the block quantization ops.
  // We then convert the vectorized ID to group ID.
  // We do so as the runtime function for block quantization expects 2/4/8
  // elements per thread.
  auto bq_ops = ir_utils::getOpsOfType<BlockQuantizationOp>(fusion);
  auto gbq_ops = ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion);
  std::vector<TensorView*> nvfp4_quantized_outputs = {};
  for (auto bq_op : bq_ops) {
    nvfp4_quantized_outputs.push_back(
        bq_op->quantizedOutput()->as<TensorView>());
  }
  for (auto gbq_op : gbq_ops) {
    nvfp4_quantized_outputs.push_back(
        gbq_op->quantizedOutput()->as<TensorView>());
  }

  if (pparams->vectorization_factor > 1) {
    // Grab all tensor views that should be vectorized
    auto inputs_outputs =
        scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true, true);
    std::vector<TensorView*> vectorized_tvs;
    bool should_vectorize_reference_tv = false;
    for (auto tv : inputs_outputs) {
      if (tv == reference_tv) {
        should_vectorize_reference_tv = true;
      }
      if (!tv->isFusionInput()) {
        vectorized_tvs.emplace_back(tv);
        continue;
      }
      // move inputs to consumers of inputs
      auto consumer_tvs = ir_utils::consumerTvsOf(tv);
      vectorized_tvs.insert(
          vectorized_tvs.end(), consumer_tvs.begin(), consumer_tvs.end());
    }
    // Vectorize all casts
    if (pparams->vectorize_casts) {
      for (auto tv : fusion->allTvs()) {
        if (auto uop = dynamic_cast<UnaryOp*>(tv->definition())) {
          if (uop->getUnaryOpType() == UnaryOpType::Cast &&
              (dataTypeSizeBit(tv->dtype()) < 8 ||
               dataTypeSizeBit(uop->in()->dtype()) < 8)) {
            vectorized_tvs.emplace_back(tv);
          }
        }
      }
    }

    // Vectorize nvfp4 quantized outputs.
    // We will later change the vectorized ID to group ID
    for (auto quantized_output : nvfp4_quantized_outputs) {
      vectorized_tvs.emplace_back(quantized_output);
    }

    if (!vectorized_tvs.empty()) {
      // Aggressively mark with vectorized and cleanup later. That way we
      // don't have to manually specify parallelization outside the reference.
      vectorize_id->parallelize(ParallelType::Vectorize);
      scheduler_utils::parallelizeAllLike(
          reference_tv, vectorized_tvs, {ParallelType::Vectorize});
      if (!should_vectorize_reference_tv) {
        vectorize_id->parallelize(ParallelType::Serial);
      }

      // Change vectorized IDs to group IDs for quantized outputs
      for (auto quantized_output : nvfp4_quantized_outputs) {
        for (auto id : quantized_output->getLoopDomain()) {
          if (id->getParallelType() == ParallelType::Vectorize) {
            id->parallelize(ParallelType::Group);
          }
        }
      }
    }
  }

  // Begin by inlining at the unswitch position for the entire DAG. The cached
  // inputs, and outputs will keep this inline position, but other tensors will
  // get a higher position in later inline propagation. We need this separate
  // step because we were not using ParallelType::Unroll, so we have to do
  // unrolling manually.
  inlineAllAt(reference_tv, unswitch_pos, true);

  auto all_tvs = fusion->allTvs();

  // Inline at the inner most position. The CA position of all tensors except
  // inputs, cached inputs and outputs will be updated.
  std::unordered_set<TensorView*> inner_most_tensors(
      all_tvs.begin(), all_tvs.end());
  for (const auto& [cached_input, input_idx] : cached_inputs) {
    inner_most_tensors.erase(cached_input);
  }
  for (const auto& [cached_output, output_idx] : schedule_info.cached_outputs) {
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
    inner_most_tensors.erase(output);
  }
  // IndexSelectOp reads lookup tv without cache. Because pointwise scheduler
  // doesn't use ParallelType::Unroll, we need to exclude consumer of fusion
  // inputs to be inlineMost. This allows us to aggregate the allocation of
  // manual unroll ID and its inner ID.
  for (auto idx_sel : ir_utils::getOpsOfType<IndexSelectOp>(fusion)) {
    inner_most_tensors.erase(idx_sel->output(0)->as<TensorView>());
  }

  inlineMost(inner_most_tensors);

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);

  markAliases(fusion);
}
} // namespace non_tma
} // namespace pointwise
} // namespace nvfuser
