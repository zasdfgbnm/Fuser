// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "scheduler/reduction.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <string>
#include <vector>

#include "debug.h"
#include "instrumentation.h"
#include "ir/utils.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/reduction_non_tma.h"
#include "scheduler/reduction_outer_tma.h"
#include "scheduler/reduction_tma.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"

namespace nvfuser {

//! Check if the reduction heuristics apply in given fusion
bool ReductionScheduler::canScheduleCompileTime(Fusion* fusion) {
  FUSER_PERF_SCOPE("ReductionScheduler::canScheduleCompileTime");

  for (auto tv : fusion->allTvs()) {
    if (tv->dtype() != DataType::Index &&
        dataTypeSizeBit(tv->dtype()) % 8 != 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Does not support sub-byte data types.");
      return false;
    }
  }

  if (scheduler_utils::isResharding(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Needs at least one reduction to consider.
  if (!ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "No reduction op to schedule");
    return false;
  }

  if (ir_utils::filterByType<TensorView>(fusion->inputs()).empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Scheduling not supported with no input");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);

  if (reduction_tvs.empty()) {
    // Use pointwise logic
    return false;
  }

  // Reject when output IDs are not covered by reference tv. Assuming reduction
  // scheduler simply uses reduction_tvs[0] as the reference, if that changes,
  // this needs to be changed. see issue
  // https://github.com/NVIDIA/Fuser/issues/3811
  scheduler_tools::DomainMap domain_map(fusion);
  if (!domain_map.isValidReference(reduction_tvs[0], /*check_inputs=*/true)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Output contains ID that's not scheduled by reference tv.");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }

    // Reduction scheduler simply uses reduction_tvs[0] as the reference, if
    // that changes, this needs to be changed.
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reduction_tvs[0])) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "View may interfere with reduction scheduling.");
      return false;
    }
  }

  // Make sure reduction axes are consistent through the fusion
  auto reduction_ops = ir_utils::getAllTypesOfReductionOps(fusion);
  if (reduction_ops.size() > 1) {
    // Before examining the reduction axes want to quickly
    //   check the reductions have the same axis width
    //   to avoid building root domain map in easier cases
    bool valid_axis_count = false;
    size_t axis_count = 0;
    auto reduction_root_size = [](TensorView* red_tv) {
      size_t count = 0;
      for (auto id : red_tv->getMaybeRootDomain()) {
        if (!id->isBroadcast()) {
          count++;
        }
      }
      return count;
    };

    for (auto red : reduction_tvs) {
      if (!valid_axis_count) {
        valid_axis_count = true;
        axis_count = reduction_root_size(red);
      } else {
        if (reduction_root_size(red) != axis_count) {
          scheduler_debug_utils::canScheduleRejectReason(
              schedulerType(),
              "Inconsistent reduction root size: ",
              red->toString(),
              ", expected: ",
              axis_count);
          return false;
        }
      }
    }

    // Use root domain map to check the reduction ops have the same axes
    FusionGuard fg(fusion);
    ComputeAtLogicalDomainMap logical_map;
    logical_map.build(true);

    // red_ops.size()>1 checked before
    for (size_t it = 1; it < reduction_tvs.size(); it++) {
      if (!registry_utils::checkPatternEquivalence(
              reduction_tvs[it - 1], reduction_tvs[it], logical_map)) {
        scheduler_debug_utils::canScheduleRejectReason(
            schedulerType(),
            "Un-mapped multi-reduction: ",
            reduction_tvs[it - 1]->toString(),
            " and ",
            reduction_tvs[it]->toString());
        return false;
      }
    }
  }

  // Doesn't allow persistent kernels in this scheduler
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (!persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "need persistent buffers that reduction scheduler doesn't handle");
    return false;
  }

  if (!registry_utils::SchedulerTopologyChecker::supportedPostReductionFusion(
          fusion, reduction_tvs) ||
      registry_utils::SchedulerTopologyChecker::hasPostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "has unsupported post reduction fusion");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "has unsupported gather-like ops before reduction");
    return false;
  }

  return true;
}

bool ReductionScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ReductionScheduler::canScheduleRunTime");
  return true;
}

namespace {

bool hasIterationAxis(Fusion* fusion) {
  const auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.empty()) {
    return false;
  }
  for (const auto* axis : reduction_tvs.front()->getLoopDomain()) {
    if (!axis->isReduction() && !axis->isDeviceDim()) {
      return true;
    }
  }
  return false;
}

// Only hard TMA requirements belong here. The 16 KiB transfer preference is
// applied after both complete scheduler parameter plans are available.
bool canUseTmaInner(
    Fusion* fusion,
    const reduction_scheduler_utils::FusionRuntimeProperties& props,
    bool require_iteration_axis) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();

  if (dev_prop->major < 9) {
    return false;
  }

  if (!scheduler_utils::inputsHaveContiguousInnerDim(fusion)) {
    return false;
  }

  // Require the reduction shape is 2D inner reduction: [I, R]
  if (!props.fastest_dim_reduction ||
      (require_iteration_axis && !hasIterationAxis(fusion)) ||
      props.total_reduction_numel != props.inner_most_dimension_numel) {
    return false;
  }

  // Require reduction dim fits into smem, until we add iteration over large
  // reduction dim.
  const int64_t smem_elems =
      (static_cast<int64_t>(dev_prop->sharedMemPerBlockOptin) * 8) /
      props.max_dtype_size_bit_for_vectorization;

  if (props.inner_most_dimension_numel > smem_elems) {
    return false;
  }

  // Smem check assumes only one input tensor.
  if (props.n_tensor_inputs != 1) {
    return false;
  }

  // Require that the innermost dim is contiguous.
  if (props.vectorize_factor <= 1) {
    return false;
  }

  uint64_t vect_bits =
      props.vectorize_factor * props.max_dtype_size_bit_for_vectorization;

  // TMA requires 16-byte alignment (128 bits) for memory transactions
  if (vect_bits % 128 != 0) {
    return false;
  }

  // Welford reductions not yet supported by TMA schedulers.
  if (ir_utils::hasOpsOfType<WelfordOp>(fusion)) {
    return false;
  }

  return true;
}

bool canUseTmaOuter(
    Fusion* fusion,
    const reduction_scheduler_utils::FusionRuntimeProperties& props,
    bool require_iteration_axis) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();

  if (dev_prop->major < 9) {
    return false;
  }

  if (!scheduler_utils::inputsHaveContiguousInnerDim(fusion)) {
    return false;
  }

  // Require outer reduction
  if (props.fastest_dim_reduction ||
      (require_iteration_axis && !hasIterationAxis(fusion))) {
    return false;
  }

  int64_t dtype_bytes = props.max_dtype_size_bit_for_vectorization / 8;

  // Check that TMA tiles for all inputs fit in shared memory.
  // The heuristic will shrink tiles to fit, but the minimum tile size is
  // bdimy × bdimx (16 × 32). Reject if even minimum tiles don't fit.
  // Reserve space for block reduction workspace and static smem overhead.
  const int64_t min_tile_r = 16; // bdimy
  const int64_t min_tile_i = 32; // bdimx
  const int64_t threads_per_block = min_tile_i * min_tile_r;
  int64_t smem_overhead =
      alignSharedMemoryBytes(threads_per_block * dtype_bytes) +
      kSharedMemoryAlignmentBytes;
  int64_t min_smem_per_input = min_tile_r * min_tile_i * dtype_bytes;
  int64_t min_total_smem =
      min_smem_per_input * props.n_tensor_inputs + smem_overhead;
  if (min_total_smem > (int64_t)dev_prop->sharedMemPerBlockOptin) {
    return false;
  }

  // Require that the innermost dim is contiguous.
  if (props.vectorize_factor <= 1) {
    return false;
  }

  uint64_t vect_bits =
      props.vectorize_factor * props.max_dtype_size_bit_for_vectorization;

  // TMA requires 16-byte alignment (128 bits) for memory transactions
  if (vect_bits % 128 != 0) {
    return false;
  }

  // Welford reductions not yet supported by the TMA outer scheduler.
  if (ir_utils::hasOpsOfType<WelfordOp>(fusion)) {
    return false;
  }

  return true;
}

bool nativePrefersTma(
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  constexpr uint64_t kMinTmaBytes = 16384;
  const auto dtype_bytes =
      static_cast<uint64_t>(props.max_dtype_size_bit_for_vectorization / 8);
  NVF_ERROR(dtype_bytes > 0);
  return static_cast<uint64_t>(props.total_reduction_numel) >=
      (kMinTmaBytes + dtype_bytes - 1) / dtype_bytes;
}
} // namespace

std::unique_ptr<HeuristicParams> ReductionScheduler::computeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("ReductionScheduler::computeHeuristics");

  auto props = reduction_scheduler_utils::getFusionRuntimeProperties(
      fusion, runtime_info, data_cache);

  bool tma_enabled = isOptionEnabled(EnableOption::TmaReduction);

  const auto jev_mode = scheduler::jev::mode();
  // The legacy off path has no iteration-axis guard. Expanded candidates in
  // record/remote must have the axis required by both TMA schedulers.
  const bool require_iteration_axis =
      jev_mode != scheduler::jev::Mode::Off;
  const bool outer_tma_legal =
      tma_enabled && canUseTmaOuter(fusion, props, require_iteration_axis);
  const bool inner_tma_legal =
      !outer_tma_legal && tma_enabled &&
      canUseTmaInner(fusion, props, require_iteration_axis);

  // Preserve the original fast path and its fallback behavior in off mode.
  if (jev_mode == scheduler::jev::Mode::Off ||
      (!outer_tma_legal && !inner_tma_legal)) {
    const bool prefer_tma =
        (outer_tma_legal || inner_tma_legal) && nativePrefersTma(props);
    std::unique_ptr<HeuristicParams> params;
    if (prefer_tma && outer_tma_legal) {
      params = reduction::outer_tma::getReductionHeuristics(
          fusion, runtime_info, data_cache, props);
    } else if (prefer_tma && inner_tma_legal) {
      params = reduction::tma::getReductionHeuristics(
          fusion, runtime_info, data_cache, props);
    }
    if (params == nullptr) {
      params = reduction::non_tma::getReductionHeuristics(
          fusion, runtime_info, data_cache, props);
    }
    NVF_ERROR(params != nullptr);
    return params;
  }

  // Build complete plans before asking about performance. A failed TMA plan
  // keeps the ordinary fallback; a missing ordinary plan cannot be offered.
  std::unique_ptr<HeuristicParams> tma_params;
  if (outer_tma_legal) {
    tma_params = reduction::outer_tma::getReductionHeuristics(
        fusion, runtime_info, data_cache, props);
  } else {
    tma_params = reduction::tma::getReductionHeuristics(
        fusion, runtime_info, data_cache, props);
  }
  if (tma_params == nullptr) {
    auto ordinary = reduction::non_tma::getReductionHeuristics(
        fusion, runtime_info, data_cache, props);
    NVF_ERROR(ordinary != nullptr);
    return ordinary;
  }
  auto ordinary_params = reduction::non_tma::getReductionHeuristics(
      fusion, runtime_info, data_cache, props);
  if (ordinary_params == nullptr) {
    if (jev_mode == scheduler::jev::Mode::Record) {
      NVF_ERROR(
          nativePrefersTma(props),
          "No ordinary reduction plan for native fallback");
    }
    return tma_params;
  }

  const std::string site = outer_tma_legal ? "RD6" : "RD1";
  const std::string state =
      "reduction.tma_preference.v1\naxis=" +
      std::string(outer_tma_legal ? "outer" : "inner") +
      "\nreduction_numel=" + std::to_string(props.total_reduction_numel) +
      "\ndtype_bits=" +
      std::to_string(props.max_dtype_size_bit_for_vectorization);
  const std::vector<scheduler::jev::ChoiceOption> options = {
      {"tma", "Use the complete TMA reduction plan"},
      {"ordinary", "Use the complete ordinary reduction plan"}};
  const std::string native_choice =
      jev_mode == scheduler::jev::Mode::Record
      ? (nativePrefersTma(props) ? "tma" : "ordinary")
      : "";
  const auto selected = scheduler::jev::chooseDecision(
      site, state, options, native_choice);
  NVF_ERROR(
      selected == "tma" || selected == "ordinary",
      "TMA preference must select a complete native reduction plan");
  return selected == "tma" ? std::move(tma_params)
                           : std::move(ordinary_params);
}

void ReductionScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("ReductionScheduler::schedule");
  if (auto* outer_tma_params =
          dynamic_cast<const TmaOuterReductionParams*>(params)) {
    reduction::outer_tma::scheduleReduction(fusion, outer_tma_params);
  } else if (
      auto* tma_params = dynamic_cast<const TmaInnerReductionParams*>(params)) {
    reduction::tma::scheduleReduction(fusion, tma_params);
  } else {
    auto rparams = dynamic_cast<const ReductionParams*>(params);
    NVF_ERROR(
        rparams != nullptr,
        "Incorrect parameters sent to ReductionScheduler::schedule",
        params);
    reduction::non_tma::scheduleReduction(fusion, rparams);
  }
}
} // namespace nvfuser
