// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/reduction_tma.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <string>
#include <vector>

#include "scheduler/cache_policy_refiner.h"
#include "scheduler/jev_client.h"
#include "scheduler/reduction_tma_unroll_choice.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"

namespace nvfuser {
namespace reduction {
namespace tma {

std::unique_ptr<TmaInnerReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props) {
  FusionGuard fg(fusion);

  auto dev_prop = at::cuda::getCurrentDeviceProperties();

  // Vectorization is not as useful for TMA since we're not doing global loads.
  // For now we don't do it.
  int64_t vectorization_factor = 1;

  // Benchmarking shows some benefit to 512 block size for has_mufu_computation,
  // but it's not across the board. Keep 256 as the native preference.
  int64_t threads_per_block = 256;

  const auto jev_mode = scheduler::jev::mode();
  if (jev_mode != scheduler::jev::Mode::Off) {
    const int64_t max_threads_per_block = dev_prop->maxThreadsPerBlock;
    const int64_t max_threads_dim_x = dev_prop->maxThreadsDim[0];
    const int64_t shared_mem_per_block = dev_prop->sharedMemPerBlockOptin;
    const int64_t dtype_bytes =
        props.max_dtype_size_bit_for_vectorization / 8;
    NVF_ERROR(
        props.total_reduction_numel > 0 && dtype_bytes > 0 &&
            max_threads_per_block >= 256 && max_threads_dim_x >= 256,
        "RD2.cta requires a valid native TMA reduction plan");

    // The TMA input and block-reduction workspace share dynamic memory.
    // The original 256-thread plan remains available even at the edge of the
    // legacy eligibility check. Admit alternatives only when a conservative
    // aligned upper bound fits the device's opt-in shared-memory budget.
    std::vector<scheduler::jev::ChoiceOption> options;
    for (const int64_t cta : {128, 256, 512}) {
      if (cta > max_threads_per_block || cta > max_threads_dim_x) {
        continue;
      }
      const int64_t reduction_bytes =
          props.total_reduction_numel * dtype_bytes;
      const int64_t workspace_bytes = alignSharedMemoryBytes(cta * dtype_bytes);
      if (cta != 256 &&
          alignSharedMemoryBytes(reduction_bytes) + workspace_bytes +
                  kSharedMemoryAlignmentBytes >
              shared_mem_per_block) {
        continue;
      }
      options.push_back(
          {"t" + std::to_string(cta),
           "TIDx reduction CTA " + std::to_string(cta) + " threads"});
    }
    std::string state = "reduction.tma_inner_cta.v1\n";
    state += "reduction_numel=" +
        std::to_string(props.total_reduction_numel) + '\n';
    state += "max_dtype_bits=" +
        std::to_string(props.max_dtype_size_bit_for_vectorization) + '\n';
    state += "smem_per_block_bytes=" +
        std::to_string(shared_mem_per_block) + '\n';
    state += "max_threads_per_block=" +
        std::to_string(max_threads_per_block) + '\n';
    state += "max_threads_dim_x=" +
        std::to_string(max_threads_dim_x) + '\n';
    state += "warp_size=" + std::to_string(dev_prop->warpSize) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "t256"
        : std::string();
    const auto selected = scheduler::jev::chooseDecision(
        "RD2.cta", state, options, native_choice);
    bool matched = false;
    for (const auto& option : options) {
      if (selected == option.id) {
        threads_per_block = std::stoll(selected.substr(1));
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "RD2.cta response exceeds hard CTA bounds");
  }

  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  const int64_t target_threads_per_sm = max_threads_per_sm / 2;

  int64_t target_vect_unroll = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    target_vect_unroll = reduction_scheduler_utils::getVectUnroll(
        props.max_dtype_size_bit_for_vectorization,
        props.vectorize_factor,
        props.n_tensor_inputs,
        target_threads_per_sm,
        props.has_mufu_computation);
  }

  // TMA uses a serial innermost axis, so the entire preferred vector/unroll
  // product becomes an unroll factor. Its structural maximum is the number of
  // reduction elements that one of the selected threads can own. Larger
  // factors add only predicated, empty unroll lanes.
  int64_t unroll_factor = jev_mode == scheduler::jev::Mode::Remote
      ? 0
      : scheduler_utils::lastPow2(target_vect_unroll);
  if (jev_mode != scheduler::jev::Mode::Off) {
    NVF_ERROR(props.total_reduction_numel > 0);
    const int64_t hard_max = scheduler_utils::lastPow2(
        ceilDiv(props.total_reduction_numel, threads_per_block));
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t factor = 1;; factor *= 2) {
      options.push_back(
          {"u" + std::to_string(factor),
           "Unroll " + std::to_string(factor) +
               " reduction element(s) per thread"});
      if (factor > hard_max / 2) {
        break;
      }
    }

    int memory_clock_khz = 0;
    const int64_t memory_bus_bits = props.has_mufu_computation
        ? 0
        : static_cast<int64_t>(dev_prop->memoryBusWidth);
    if (!props.has_mufu_computation) {
      NVF_ERROR(
          cudaDeviceGetAttribute(
              &memory_clock_khz,
              cudaDevAttrMemoryClockRate,
              at::cuda::current_device()) == cudaSuccess,
          "RD2.unroll could not read CUDA memory clock");
    }
    std::string state = jev_mode == scheduler::jev::Mode::Record
        ? "reduction.tma_inner_unroll.v1\n"
        : "reduction.tma_inner_unroll.v2\n";
    state += "reduction_numel=" +
        std::to_string(props.total_reduction_numel) + '\n';
    state += "threads_per_block=" + std::to_string(threads_per_block) + '\n';
    if (jev_mode == scheduler::jev::Mode::Remote) {
      state += "max_threads_per_block=" +
          std::to_string(dev_prop->maxThreadsPerBlock) + '\n';
      state += "max_threads_dim_x=" +
          std::to_string(dev_prop->maxThreadsDim[0]) + '\n';
      state += "smem_per_block_bytes=" +
          std::to_string(dev_prop->sharedMemPerBlockOptin) + '\n';
    }
    state += "layout_max=" + std::to_string(props.vectorize_factor) + '\n';
    state += "max_dtype_bits=" +
        std::to_string(props.max_dtype_size_bit_for_vectorization) + '\n';
    state += "n_tensor_inputs=" + std::to_string(props.n_tensor_inputs) + '\n';
    state += "has_mufu=" +
        std::to_string(props.has_mufu_computation ? 1 : 0) + '\n';
    state += "sm_count=" + std::to_string(dev_prop->multiProcessorCount) + '\n';
    state += "max_threads_per_sm=" +
        std::to_string(max_threads_per_sm) + '\n';
    state += "memory_bus_bits=" + std::to_string(memory_bus_bits) + '\n';
    state += "memory_clock_khz=" + std::to_string(memory_clock_khz) + '\n';
    if (const auto selected =
            chooseInnerUnrollIfRepresentable(state, options, unroll_factor)) {
      bool matched = false;
      for (int64_t factor = 1;; factor *= 2) {
        if (*selected == "u" + std::to_string(factor)) {
          unroll_factor = factor;
          matched = true;
          break;
        }
        if (factor > hard_max / 2) {
          break;
        }
      }
      NVF_ERROR(matched, "RD2.unroll response exceeds hard reduction bounds");
    }
  }

  auto params = std::make_unique<TmaInnerReductionParams>();
  params->vectorization_factor = vectorization_factor;
  params->threads_per_block = threads_per_block;
  params->unroll_factor = unroll_factor;

  params->tag = "Reduction TMA heuristics";
  params->cparams.index_type = runtime_info.getIndexType();

  return params;
}

void scheduleReduction(Fusion* fusion, const TmaInnerReductionParams* rparams) {
  FusionGuard fg(fusion);

  // Always cache inputs for TMA
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  std::vector<TensorView*> tma_tvs;
  for (auto [tv, input_idx] : cached_inputs) {
    if (auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition())) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulk);
      tv->setMemoryType(MemoryType::Shared);
      tma_tvs.push_back(tv);
    }
  }

  NVF_ERROR(!tma_tvs.empty());

  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  NVF_ERROR(!reduction_tvs.empty());
  TensorView* reduction_tv = reduction_tvs.at(0);

  // Merge all iteration and reduction dimensions into canonical form [I, R]
  auto dim_analysis =
      scheduler_utils::canonicalizeReduction(fusion, reduction_tv, false);
  bool has_iter_axis = dim_analysis.first;
  bool has_red_axis = dim_analysis.second;
  NVF_ERROR(has_iter_axis && has_red_axis);

  // Propagate the merges to all TMA TVs
  TransformPropagator tma_propagator(reduction_tv);
  SetSelector tma_selector({tma_tvs.begin(), tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reduction_tv, &tma_selector)
      .traverse(&tma_propagator);

  int64_t iter_axis = 0;
  int64_t inner_reduce_axis = 1;

  // Schedule TMA tvs as [BIDx, Bulk]
  tma_tvs[0]->axis(iter_axis)->parallelize(ParallelType::BIDx);
  tma_tvs[0]->axis(inner_reduce_axis)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(tma_tvs[0], tma_tvs);

  // Non-TMA scheduling
  //
  // Apply splits following the pattern:
  // [I, R] -> [I, R/vect, vect]
  //        -> [I, R/vect/tidx, tidx, vect]
  //        -> [I, R/vect/tidx/unroll, unroll, tidx, vect]

  // Split 1: Vectorization factor (innermost serial split for TMA)
  if (rparams->vectorization_factor > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->vectorization_factor);
    reduction_tv->axis(inner_reduce_axis + 1)
        ->parallelize(ParallelType::Serial);
  }

  // Split 2: TIDx (always applied)
  reduction_tv->split(inner_reduce_axis, rparams->threads_per_block);
  reduction_tv->axis(inner_reduce_axis + 1)->parallelize(ParallelType::TIDx);

  // Split 3: Inner unroll (outside of TIDx)
  if (rparams->unroll_factor > 1) {
    reduction_tv->split(inner_reduce_axis, rparams->unroll_factor);
    reduction_tv->axis(inner_reduce_axis + 1)
        ->parallelize(ParallelType::Unroll);
  }

  // Split 4: Unswitch (always applied)
  reduction_tv->split(inner_reduce_axis, 1);
  reduction_tv->axis(inner_reduce_axis + 1)
      ->parallelize(ParallelType::Unswitch);

  // Serial outer loop (remainder after all splits)
  reduction_tv->axis(inner_reduce_axis)->parallelize(ParallelType::Serial);

  // Parallelize iteration axis
  reduction_tv->axis(iter_axis)->parallelize(ParallelType::BIDx);

  // Collect rFactor axes: all reduction axes that are not parallelized with
  // threads (i.e., not TIDx). This includes serial, unswitch, unroll, and vect
  // axes.
  std::vector<int64_t> rfactor_axes;
  for (int64_t i = 0; i < reduction_tv->nDims(); i++) {
    if (reduction_tv->axis(i)->isReduction() &&
        !reduction_tv->axis(i)->isThread()) {
      rfactor_axes.push_back(i);
    }
  }

  TensorView* reference_tv = reduction_tv;
  if (!rfactor_axes.empty()) {
    reference_tv = reduction_tv->rFactor(rfactor_axes);
  }

  // Schedule non-TMA tvs based on reference tv
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector non_tma_selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &non_tma_selector)
      .traverse(&non_tma_propagator);

  if (reference_tv != reduction_tv) {
    reduction_scheduler_utils::propagateRFactor(
        reference_tv, reduction_tv, reduction_tvs);
    non_tma_tvs =
        ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  }

  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  inlineMost();

  refineCachePolicy(fusion);
}
} // namespace tma
} // namespace reduction
} // namespace nvfuser
