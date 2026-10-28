// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "scheduler/normalization_inner_non_tma.h"

#include <algorithm>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

#include "instrumentation.h"
#include "ir/utils.h"
#include "options.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/normalization_inner_smem_cta_choice.h"
#include "scheduler/normalization_utils.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/utils.h"

namespace nvfuser {
namespace normalization_inner {
namespace non_tma {

using PersistentKernelProperties =
    normalization_scheduler_utils::PersistentKernelProperties;

namespace {

std::pair<int64_t, int64_t> getPersistentBufferSizeBit(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& reduction_tvs,
    const bool can_use_smem_persistent,
    const bool is_3d_reduction) {
  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();

  auto persistent_buffer_size_info = scheduler_utils::persistentBufferSizeBit(
      fusion, runtime_info, persistent_buffer_info, data_cache);

  normalization_scheduler_utils::BufferProjectionStrategy project_strategy =
      normalization_scheduler_utils::isProjectBufferToInputs(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          persistent_buffer_size_info,
          InnerPersistentKernelScheduler::schedulerType(),
          can_use_smem_persistent);
  bool project_persistent_buffers =
      (project_strategy ==
       normalization_scheduler_utils::BufferProjectionStrategy::
           ProjectToInputs);
  auto persistent_buffer_size_bit = project_persistent_buffers
      ? persistent_buffer_size_info.projected_persistent_buffer_size_bit
      : persistent_buffer_size_info.persistent_buffer_size_bit;

  int64_t available_persistent_buffer_size_bit = normalization_scheduler_utils::
      getMaxRegOrSharedMemorySizeBitForPersistentBuffer(
          fusion,
          runtime_info,
          reduction_tvs,
          persistent_buffer_info,
          can_use_smem_persistent,
          project_persistent_buffers);
  // If one SM doesn't have enough persistent buffer size, try multiple SMs.
  // Current implementation doesn't support 3D reduction with multiple SMs.
  if (!is_3d_reduction &&
      available_persistent_buffer_size_bit < persistent_buffer_size_bit) {
    available_persistent_buffer_size_bit *=
        scheduler_utils::getMaxClusterSize();
  }
  return std::make_pair(
      persistent_buffer_size_bit, available_persistent_buffer_size_bit);
}

// Return the maximum register count each thread can use and achieved occupancy.
// We always guarantee the returned register count is at least as large as the
// buffer+overhead estimate. We meet the desired occupancy but don't try to
// maximize it further. As long as it's as large as a given target occupancy, we
// consider it's good enough. The idea is that as long as we have a good enough
// occupancy, we should be able to saturate the memory bandwidth. Within these
// constraints, we try to maximize the number of registers each thread can use.
// Para [target_warps_per_sm]: required occupancy to saturate memory bandwidth.
// Para [register_overhead]: registers except those for persistent buffers.
std::pair<int64_t, int64_t> getMaxRegisterCountPerThreadAndOccupancy(
    const int64_t buffer_size_per_thread_bit,
    const int64_t threads_per_block,
    const int64_t target_warps_per_sm,
    const int64_t register_overhead) {
  // convert [target_warps_per_sm] to [target_blocks_per_sm]
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = dev_prop->warpSize;
  const int64_t max_threads_per_sm = dev_prop->maxThreadsPerMultiProcessor;
  // ensure higher than target by round up
  int64_t target_blocks_per_sm =
      ceilDiv(target_warps_per_sm * threads_per_warp, threads_per_block);
  // ensure lower than hardware limit by round down
  if (threads_per_block * target_blocks_per_sm > max_threads_per_sm) {
    target_blocks_per_sm = max_threads_per_sm / threads_per_block;
  }
  // minimum register each thread should use to avoid spills
  const int64_t register_per_thread_min =
      buffer_size_per_thread_bit / scheduler_utils::bits_per_register +
      register_overhead;

  // (1) use register calculated from target occupancy
  int64_t threads_per_sm = threads_per_block * target_blocks_per_sm;
  int64_t register_per_thread_target =
      getRegPerThreadGivenThreadsPerSM(threads_per_sm);

  if (register_per_thread_target >= register_per_thread_min) {
    return {register_per_thread_target, threads_per_sm / threads_per_warp};
  }

  //(2) can't achieve target occupancy. Estimate occupancy from minimum register
  // each thread should use, then derive register per thread from occupancy.
  int64_t blocks_per_sm_max = scheduler_utils::safeDiv(
      getThreadsPerSMGivenRegPerThread(register_per_thread_min),
      threads_per_block);
  threads_per_sm =
      std::min(blocks_per_sm_max * threads_per_block, max_threads_per_sm);

  return {
      getRegPerThreadGivenThreadsPerSM(threads_per_sm),
      threads_per_sm / threads_per_warp};
}

// Returns the maximum persistent batch size.
// For example: assuming we have 64K registers per SM and 28 warps (864 threads)
// per SM. Each thread can use up to 72 registers. Then minus the register
// overhead 16, there are 56 registers or 224 * 8 = 1792 bits to store the
// persistent buffer.
// (1) If each reduction element has 1 fp32 buffer and vectorized by 8,
//     [buffer_bits_per_batch] = 4 * 8 * 8 = 256. Then the maximum persistent
//      batch size is 1792 / 256 = 7
// (2) If each reduction element has 1 fp16 buffer and vectorized by 8,
//     [buffer_bits_per_batch] = 2 * 8 * 8 = 128. Then the maximum persistent
//      batch size is 1792 / 128 = 14, which is then capped to
//      [max_batches_per_block] whose value is 10.
int64_t getMaxPersistentBatch(
    const int64_t buffer_bits_per_batch,
    const int64_t target_threads_per_sm,
    const int64_t register_overhead,
    const bool is_high_bandwidth_flops_ratio = false) {
  // (1) calculate the maximum register count given the target occupancy.
  int64_t total_register =
      getRegPerThreadGivenThreadsPerSM(target_threads_per_sm);
  int64_t register_for_buffer = total_register - register_overhead;

  // (2) calculate the maximum persistent batch size using the register count.
  int64_t batch_size = scheduler_utils::safeDiv(
      register_for_buffer * scheduler_utils::bits_per_register,
      buffer_bits_per_batch);

  // (3) Avoid using very large persistent buffer size, which may lead to low
  // occupancy due to the limitation of the current heuristics. TODO: remove
  // this parameter when we have a better heuristic to select the best
  // persistent batch size.
  int64_t max_batches_per_block =
      normalization_scheduler_utils::getInnerPersistentMaxBatchSize(
          is_high_bandwidth_flops_ratio);
  return std::min(max_batches_per_block, batch_size);
}

// calculate bdimx, bdimy, occupancy, given a persistent batch size
struct NormInnerParams {
  int64_t bdimx = -1;
  int64_t bdimy = -1;
  int64_t padded_bdimx = -1;
  int64_t persistent_batch_size = -1;
  int64_t register_per_thread = -1;
  int64_t non_buffer_registers = -1;
  int64_t occupancy = -1;
  int64_t n_wave = -1;
  int64_t n_persistent_tails = -1;
  bool is_pad_bdimx = false;
  void print() const {
    std::cout << "bdimx: " << bdimx << ", bdimy: " << bdimy
              << ", padded_bdimx: " << padded_bdimx
              << ", persistent_batch_size: " << persistent_batch_size
              << ", register_per_thread: " << register_per_thread
              << ", non_buffer_registers: " << non_buffer_registers
              << ", occupancy: " << occupancy << ", n_wave: " << n_wave
              << ", n_persistent_tails: " << n_persistent_tails
              << ", is_pad_bdimx: " << is_pad_bdimx << std::endl;
  }
};

NormInnerParams getNormInnerParamsGivenPerisisentBatchSize(
    const int64_t reduction_count_after_vectorize,
    const int64_t total_iteration_numel,
    const int64_t max_multi_reduction_factor,
    const int64_t min_threads_per_block,
    const int64_t buffer_bits_per_batch,
    const int64_t target_warps_per_sm,
    const int64_t register_overhead,
    const int64_t persistent_batch_size) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  auto device_warp_size = dev_prop->warpSize;
  auto max_threads_per_block = dev_prop->maxThreadsPerBlock;
  auto sm_count = dev_prop->multiProcessorCount;
  NormInnerParams params;
  params.persistent_batch_size = persistent_batch_size;

  // set bdimx and bdimy
  params.bdimx = scheduler_utils::safeDiv(
      reduction_count_after_vectorize, persistent_batch_size);
  NVF_ERROR(
      params.bdimx <= max_threads_per_block,
      "persistent batch size too small! bdimx should be less than ",
      max_threads_per_block,
      ", but got ",
      params.bdimx);
  params.bdimy = std::min(
      scheduler_utils::safeDiv(min_threads_per_block, params.bdimx),
      max_multi_reduction_factor);
  params.padded_bdimx = params.bdimx % device_warp_size == 0
      ? params.bdimx
      : params.bdimx + (device_warp_size - params.bdimx % device_warp_size);
  params.is_pad_bdimx = params.bdimx > 16 &&
      params.padded_bdimx * params.bdimy <= max_threads_per_block;

  // calculate register per thread and achieved occupancy
  int64_t threads_per_block = params.is_pad_bdimx
      ? params.padded_bdimx * params.bdimy
      : params.bdimx * params.bdimy;
  int64_t persistent_buffer_size_bit =
      buffer_bits_per_batch * persistent_batch_size;
  auto reg_occ = getMaxRegisterCountPerThreadAndOccupancy(
      persistent_buffer_size_bit,
      threads_per_block,
      target_warps_per_sm,
      register_overhead);
  params.register_per_thread = reg_occ.first;
  params.occupancy = reg_occ.second;
  int64_t blocks_per_sm = scheduler_utils::safeDiv(
      params.occupancy * device_warp_size, threads_per_block);
  params.n_wave = ceilDiv(total_iteration_numel, sm_count * blocks_per_sm);
  params.non_buffer_registers = params.register_per_thread -
      persistent_buffer_size_bit / scheduler_utils::bits_per_register;
  // (4) Calculate other quantities reflecting the quality of the heuristic.
  // when [reduction_count_after_vectorize] is not divisible by
  // [persistent_val], the last batch is not be fully utilized, the wasted
  // threads in the last batch is quantified as [n_persistent_tails].
  params.n_persistent_tails =
      ceilDiv(reduction_count_after_vectorize, persistent_batch_size) *
          persistent_batch_size -
      reduction_count_after_vectorize;
  return params;
}

// Return true if ha is better than hb

// This sorting function ensures the selected heuristic meeting target
// occupancy, promotes even workload distribution, enhances register
// optimization, and prefers higher occupancy.
// TODO: It leads to 10% regression for softmax around 2K to 6K and 16K.
// See https://github.com/NVIDIA/Fuser/issues/1876
bool compareTwoHeuristics(
    const NormInnerParams& pa,
    const NormInnerParams& pb,
    const int64_t min_non_buffer_registers,
    const int64_t target_warps_per_sm,
    const bool is_high_bandwidth_flops_ratio,
    const bool has_exp_ops) {
  // This lambda compares a parameter between two `NormInnerParams`
  // configurations. If the parameters for configuration A and B are the same,
  // continue along and compare other parameters. Otherwise, short-circuit
  // compareTwoHeuristics.
  auto compare = [](int64_t a, int64_t b) -> int {
    return a > b ? 1 : (a < b ? -1 : 0);
  };
  int score = 0;

  // prefer occupancy larger than target
  score = compare(
      pa.occupancy >= target_warps_per_sm, pb.occupancy >= target_warps_per_sm);
  if (score != 0) {
    return score > 0;
  }

  // prefer reduction count after vectorization is divisible by persistent
  // batch size. Skip this check when the bandwidth to flops ratio is high and
  // has expensive ops, under this condition, using a larger persistent batch
  // is more beneficial than using a smaller persistent batch that is divisible.
  if (!(is_high_bandwidth_flops_ratio && has_exp_ops)) {
    score = compare(pa.n_persistent_tails == 0, pb.n_persistent_tails == 0);
    if (score != 0) {
      return score > 0;
    }
  }

  // Ensure the count of non buffer registers is larger than (or equal to if the
  // bandwidth to flops ratio is high and fusion has expensive ops) the min
  // overhead. But don't want to achieve this goal at the cost of using a very
  // large block size, it avoids using a small persistent batch with a large
  // block size, which usually leads to 10% lower in performance.
  constexpr int64_t opt_max_threads_per_block = 512;
  if (is_high_bandwidth_flops_ratio && has_exp_ops) {
    score = compare(
        pa.non_buffer_registers >= min_non_buffer_registers &&
            pa.padded_bdimx <= opt_max_threads_per_block,
        pb.non_buffer_registers >= min_non_buffer_registers &&
            pb.padded_bdimx <= opt_max_threads_per_block);
  } else {
    score = compare(
        pa.non_buffer_registers > min_non_buffer_registers &&
            pa.padded_bdimx <= opt_max_threads_per_block,
        pb.non_buffer_registers > min_non_buffer_registers &&
            pb.padded_bdimx <= opt_max_threads_per_block);
  }

  if (score != 0) {
    return score > 0;
  }

  // when there are enough waves, prefer the one with less waves, less waves
  // means higher occupancy. We don't want to directly use occupancy as two
  // different occupancies may lead to the same number of waves.
  if (is_high_bandwidth_flops_ratio && (pa.n_wave >= 8 || pb.n_wave >= 8)) {
    score = compare(pa.n_wave, pb.n_wave);
    if (score != 0) {
      return score < 0;
    }
  }

  // Prefer large occupancy
  score = compare(pa.occupancy, pb.occupancy);
  if (score != 0) {
    return score > 0;
  }
  // Tiebreaker, use large persistent batch size so more registers are used
  // for the persistent buffer.
  return pa.persistent_batch_size > pb.persistent_batch_size;
}

// Generate a heuristic for each possible persistent batch size.
// (1) If the maximum occupancy is less than the target occupancy, use the batch
//     leads to the largest occupancy.
// (2) sort the heuristics as follows:
//     (a) Prefer occupancy exceeding target.
//         Ensures minimum required occupancy is surpassed.
//     (b) Prefer divisible by persistent batch size.
//         Aims for even workload distribution.
//     (c) Prefer non buffer register exceeds min overhead.
//         Maximizes compiler optimization potential.
//     (d) Seek larger occupancy.
//         Exceeds the target minimum for better performance.
//     (e) Use large persistent batch size as a tiebreaker.
//         Use more registers for persistent buffers.
// This sequence ensures meeting target occupancy, promotes even workload
// distribution, enhances register optimization, and prefers higher occupancy.
struct Inner2DPolicyResources {
  int64_t batch_min = 0;
  int64_t batch_max = 0;
  bool supported_menu = false;
  // Ascending, unranked score plans copied before native preference sorting.
  // Used only for the semantic-v3 width frontier; no selected winner is sent.
  std::vector<NormInnerParams> candidate_plans;
};

void innerPersistentHeuristic2DPlan(
    const PersistentKernelProperties& properties,
    ReductionParams* rparams,
    int64_t register_overhead,
    int64_t target_warps_per_sm,
    bool enable_batch_choice,
    Inner2DPolicyResources* policy_resources = nullptr,
    int64_t min_threads_override = 0,
    bool expand_preview = false) {
  bool is_high_bandwidth_flops_ratio =
      scheduler_utils::isHighBandwidthFlopsRatio();
  // Currently, we only considered the influence of exp op which is used in
  // softmax should extend to other MUFU units. Note that, rng op is an
  // expensive op however, test shows it can't be processed similarly to exp op
  // since it doesn't use the MUFU units.
  bool has_exp_op = properties.has_exp_op;
  bool disable_project_to_avoid_recompute =
      properties.disable_project_to_avoid_recompute;
  // device properties
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  const int64_t threads_per_warp = (int64_t)dev_prop->warpSize;
  const int64_t max_threads_in_block = (int64_t)dev_prop->maxThreadsPerBlock;
  const int64_t max_threads_per_sm =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  // alwasy use [vectorize_factor]
  const int64_t parallel_after_vectorize =
      properties.inner_most_dimension_numel / properties.vectorize_factor;

  // The heuristic targets multi-wave kernels with far more blocks than SMs.
  // If the iteration dimension is small, not all SMs can be utilized. In such
  // cases, using the maximum threads per block helps increase the number of
  // warps and improves performance.
  bool can_use_all_sms =
      properties.total_iteration_numel >= device_multiprocessor_count;
  const int64_t native_min_threads_per_block =
      can_use_all_sms ? 4l * threads_per_warp : max_threads_in_block;
  const int64_t min_threads_per_block = min_threads_override > 0
      ? min_threads_override
      : native_min_threads_per_block;

  // set the min persistent buffer size to avoid requesting
  // a block size larger than device limit
  int64_t batches_per_block_inner_reduction_min =
      ceilDiv(parallel_after_vectorize, max_threads_in_block);

  // set the max persistent batch size to avoid low occupancy
  // (1) limitation set by min_threads_per_block
  const int64_t pbs_max_1 =
      ceilDiv(parallel_after_vectorize, min_threads_per_block);
  // (2) derived the maximum persistent batch size from the target occupancy
  const int64_t buffer_bits_per_batch =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * properties.vectorize_factor;
  const int64_t target_threads_per_sm =
      std::min(target_warps_per_sm * threads_per_warp, max_threads_per_sm);
  const int64_t pbs_max_2 = getMaxPersistentBatch(
      buffer_bits_per_batch,
      target_threads_per_sm,
      register_overhead,
      is_high_bandwidth_flops_ratio);
  int64_t batches_per_block_inner_reduction_max = std::max(
      batches_per_block_inner_reduction_min, std::min(pbs_max_1, pbs_max_2));

  // An alternate policy is only a candidate when even its largest proposed
  // batch can fit the architectural per-thread register cap. Check this before
  // the occupancy helper, whose inverse estimate assumes a feasible minimum.
  if (policy_resources != nullptr &&
      (buffer_bits_per_batch <= 0 ||
       batches_per_block_inner_reduction_max >
           (scheduler_utils::max_registers_per_thread - register_overhead) *
               scheduler_utils::bits_per_register / buffer_bits_per_batch)) {
    policy_resources->batch_min = batches_per_block_inner_reduction_min;
    policy_resources->batch_max = batches_per_block_inner_reduction_max;
    return;
  }

  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size. Bounded by the wave count for utilization of
  // SMs.
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size_bit,
          properties.max_persistent_buffer_size_bit),
      ceilDiv(properties.total_iteration_numel, device_multiprocessor_count));

  // Generate a heuristic for each possible persistent batch size.
  // record which persistent batch size has the highest occupancy.
  int64_t idx_max_occupancy = -1;
  int64_t current_max_occupancy = -1;
  std::vector<NormInnerParams> all_heuristics;

  all_heuristics.reserve(
      batches_per_block_inner_reduction_max -
      batches_per_block_inner_reduction_min + 1);
  for (int64_t pbs = batches_per_block_inner_reduction_min;
       pbs <= batches_per_block_inner_reduction_max;
       pbs++) {
    all_heuristics.push_back(getNormInnerParamsGivenPerisisentBatchSize(
        parallel_after_vectorize,
        properties.total_iteration_numel,
        max_multi_reduction_factor,
        min_threads_per_block,
        buffer_bits_per_batch,
        target_warps_per_sm,
        register_overhead,
        pbs));
    if (all_heuristics.back().occupancy > current_max_occupancy) {
      current_max_occupancy = all_heuristics.back().occupancy;
      idx_max_occupancy = (int64_t)all_heuristics.size() - 1;
    }
  }

  // Select the best plan. When none reaches the target occupancy, retain the
  // first plan with maximum occupancy; otherwise apply the empirical sort.
  const auto jev_mode =
      enable_batch_choice ? scheduler::jev::mode() : scheduler::jev::Mode::Off;
  // Keep the original, ascending-batch candidate order for the Choice
  // request. The native stable sort below would otherwise put its winner
  // first and reveal the preference to the service.
  std::vector<NormInnerParams> candidate_heuristics =
      jev_mode == scheduler::jev::Mode::Off && policy_resources == nullptr
      ? std::vector<NormInnerParams>{}
      : all_heuristics;
  NormInnerParams best_heuristic;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    if (current_max_occupancy < target_warps_per_sm) {
      best_heuristic = all_heuristics.at(idx_max_occupancy);
    } else {
      std::stable_sort(
          all_heuristics.begin(),
          all_heuristics.end(),
          [&register_overhead,
           &target_warps_per_sm,
           &is_high_bandwidth_flops_ratio,
           &has_exp_op](const NormInnerParams& a, const NormInnerParams& b) {
            return compareTwoHeuristics(
                a,
                b,
                register_overhead,
                target_warps_per_sm,
                is_high_bandwidth_flops_ratio,
                has_exp_op);
          });
      best_heuristic = all_heuristics.at(0);
    }
  }

  if (jev_mode != scheduler::jev::Mode::Off || policy_resources != nullptr) {
    // Validate a complete 2D schedule, including its physical CTA, register
    // allocation and grid, before it can appear in a Choice request.
    const auto supported_plan = [&](const NormInnerParams& candidate) {
      if (candidate.persistent_batch_size <= 0 || candidate.bdimx <= 0 ||
          candidate.bdimx > max_threads_in_block || candidate.bdimy <= 0 ||
          candidate.bdimy > max_threads_in_block ||
          candidate.padded_bdimx < candidate.bdimx ||
          candidate.padded_bdimx > max_threads_in_block + threads_per_warp) {
        return false;
      }
      const int64_t padded_threads = candidate.padded_bdimx * candidate.bdimy;
      const bool should_pad =
          candidate.bdimx > 16 && padded_threads <= max_threads_in_block;
      const int64_t threads =
          should_pad ? padded_threads : candidate.bdimx * candidate.bdimy;
      const int64_t buffer_registers = buffer_bits_per_batch *
          candidate.persistent_batch_size / scheduler_utils::bits_per_register;
      const int64_t godim =
          ceilDiv(properties.total_iteration_numel, candidate.bdimy);
      const int64_t grid_x_limit = scheduler_utils::x_grid_limit;
      const int64_t grid_y_limit = std::min<int64_t>(
          scheduler_utils::y_grid_limit, dev_prop->maxGridSize[1]);
      const int64_t physical_x =
          ceilDiv(parallel_after_vectorize, candidate.persistent_batch_size);
      const int64_t physical_padded_x =
          ceilDiv(physical_x, threads_per_warp) * threads_per_warp;
      const int64_t actual_x =
          candidate.is_pad_bdimx ? physical_padded_x : physical_x;
      const int64_t actual_threads = actual_x * candidate.bdimy;
      // The score uses floor(parallel / batch), but the scheduled TIDx uses
      // ceil(parallel / batch). A policy can fit the score CTA yet exceed the
      // physical register budget, even with the native CTA seed. Check the
      // physical CTA for every offered policy, not just alternate seeds.
      return actual_x > 0 && actual_x <= dev_prop->maxThreadsDim[0] &&
          actual_threads > 0 && actual_threads <= max_threads_in_block &&
          candidate.register_per_thread <=
              dev_prop->regsPerBlock / actual_threads &&
          candidate.is_pad_bdimx == should_pad && threads > 0 &&
          threads <= max_threads_in_block &&
          candidate.bdimx <= dev_prop->maxThreadsDim[0] &&
          candidate.bdimy <= dev_prop->maxThreadsDim[1] &&
          candidate.register_per_thread > 0 &&
          candidate.register_per_thread <= 255 &&
          candidate.register_per_thread <= dev_prop->regsPerBlock / threads &&
          candidate.register_per_thread >=
          buffer_registers + register_overhead &&
          godim > 0 && grid_x_limit > 0 && grid_y_limit > 0 &&
          (godim <= dev_prop->maxGridSize[0] ||
           (godim > grid_x_limit &&
            grid_x_limit <= dev_prop->maxGridSize[0])) &&
          ceilDiv(godim, grid_x_limit) <= grid_y_limit;
    };
    // Keep the original capped range as the native decision domain. The
    // empirical 10/12 limit is not a resource bound: in Jev modes, extend the
    // offered menu through the target-occupancy register budget when complete
    // plans pass the same hard checks. Off and Record still pick the original
    // native winner; the API may choose a proved larger batch.
    const int64_t native_batch_max = batches_per_block_inner_reduction_max;
    const bool supported_native_menu = !candidate_heuristics.empty() &&
        candidate_heuristics.size() <= 12 &&
        std::all_of(
            candidate_heuristics.begin(),
            candidate_heuristics.end(),
            supported_plan);
    if ((jev_mode != scheduler::jev::Mode::Off || expand_preview) &&
        supported_native_menu &&
        buffer_bits_per_batch > 0) {
      const int64_t register_budget =
          getRegPerThreadGivenThreadsPerSM(target_threads_per_sm) -
          register_overhead;
      if (register_budget > 0) {
        const int64_t uncapped_max = std::min(
            pbs_max_1,
            register_budget * scheduler_utils::bits_per_register /
                buffer_bits_per_batch);
        for (int64_t pbs = native_batch_max + 1;
             pbs <= uncapped_max && candidate_heuristics.size() < 255;
             ++pbs) {
          auto plan = getNormInnerParamsGivenPerisisentBatchSize(
              parallel_after_vectorize,
              properties.total_iteration_numel,
              max_multi_reduction_factor,
              min_threads_per_block,
              buffer_bits_per_batch,
              target_warps_per_sm,
              register_overhead,
              pbs);
          // The later persistent split uses ceil(reduction / batch) as its
          // physical TIDx extent. The legacy score estimates floor instead;
          // require both to produce the same physical CTA for a new plan.
          const int64_t actual_bdimx = ceilDiv(parallel_after_vectorize, pbs);
          const int64_t actual_padded_bdimx =
              ceilDiv(actual_bdimx, threads_per_warp) * threads_per_warp;
          const bool same_physical_cta = plan.is_pad_bdimx
              ? actual_padded_bdimx == plan.padded_bdimx
              : actual_bdimx == plan.bdimx;
          // A contiguous menu lets the service verify that no cheaper batch
          // was silently omitted. Stop at the first unsupported plan.
          if (!same_physical_cta || actual_bdimx > dev_prop->maxThreadsDim[0] ||
              !supported_plan(plan)) {
            break;
          }
          candidate_heuristics.push_back(plan);
        }
      }
    }
    const bool supported_menu = supported_native_menu &&
        candidate_heuristics.size() <= 255 &&
        std::all_of(candidate_heuristics.begin(),
                    candidate_heuristics.end(),
                    supported_plan);
    if (policy_resources != nullptr) {
      policy_resources->batch_min = batches_per_block_inner_reduction_min;
      policy_resources->batch_max = candidate_heuristics.empty()
          ? native_batch_max
          : candidate_heuristics.back().persistent_batch_size;
      policy_resources->supported_menu = supported_menu;
      if (supported_menu) {
        policy_resources->candidate_plans = candidate_heuristics;
      }
    }
    // Candidate preflight has no nested Choice or log side effect.
    if (jev_mode != scheduler::jev::Mode::Off) {
      // A future change to candidate generation must not silently send an
      // oversized or physically invalid menu. Record retains the native plan;
      // remote fails closed because it has no trusted native winner.
      NVF_ERROR(
          supported_menu || jev_mode == scheduler::jev::Mode::Record,
          "Unsupported register-backed inner persistent batch menu");
      if (supported_menu) {
        // Native analysis owns the candidate plans and their CTA/register
        // limits. The service gets unranked measurements and independently
        // applies the empirical comparison; it never receives the native
        // winner.
        const bool expanded =
            candidate_heuristics.back().persistent_batch_size >
            native_batch_max;
        const bool selected_seed =
            min_threads_per_block != native_min_threads_per_block;
        std::string state = std::string(
                                selected_seed
                                    ? "inner_persistent.reg_batch.v4\n"
                                    : expanded
                                    ? "inner_persistent.reg_batch.v3\n"
                                    : "inner_persistent.reg_batch.v2\n") +
            "parallel_after_vectorize=" +
            std::to_string(parallel_after_vectorize) +
            "\ntotal_iteration_numel=" +
            std::to_string(properties.total_iteration_numel) +
            "\nmin_threads_per_block=" + std::to_string(min_threads_per_block) +
            "\nmax_multi_reduction_factor=" +
            std::to_string(max_multi_reduction_factor) +
            "\nbuffer_bits_per_batch=" + std::to_string(buffer_bits_per_batch) +
            "\nhigh_bandwidth_flops_ratio=" +
            std::to_string(is_high_bandwidth_flops_ratio) +
            "\nhas_exp_op=" + std::to_string(has_exp_op) +
            "\ndisable_projection_avoid_recompute=" +
            std::to_string(disable_project_to_avoid_recompute) +
            "\nwarp_size=" + std::to_string(threads_per_warp) +
            "\nmax_threads_per_block=" + std::to_string(max_threads_in_block) +
            "\nmax_threads_per_sm=" + std::to_string(max_threads_per_sm) +
            "\nsm_count=" + std::to_string(device_multiprocessor_count) +
            "\npolicy_register_overhead=" + std::to_string(register_overhead) +
            "\npolicy_target_warps_per_sm=" +
            std::to_string(target_warps_per_sm) + "\nbits_per_register=" +
            std::to_string(scheduler_utils::bits_per_register) +
            "\nbatch_min=" +
            std::to_string(batches_per_block_inner_reduction_min) +
            "\nbatch_max=" +
            std::to_string(candidate_heuristics.back().persistent_batch_size);
        if (expanded || selected_seed) {
          state += "\nnative_batch_max=" + std::to_string(native_batch_max);
        }
        if (selected_seed) {
          state += "\nnative_min_threads_per_block=" +
              std::to_string(native_min_threads_per_block);
        }
        std::vector<scheduler::jev::ChoiceOption> options;
        options.reserve(candidate_heuristics.size());
        for (const auto& candidate : candidate_heuristics) {
          state += "\nplan=" + std::to_string(candidate.persistent_batch_size) +
              ":" + std::to_string(candidate.bdimx) + ":" +
              std::to_string(candidate.bdimy) + ":" +
              std::to_string(candidate.padded_bdimx) + ":" +
              std::to_string(candidate.register_per_thread) + ":" +
              std::to_string(candidate.non_buffer_registers) + ":" +
              std::to_string(candidate.occupancy) + ":" +
              std::to_string(candidate.n_wave) + ":" +
              std::to_string(candidate.n_persistent_tails);
          options.push_back(
              {"b" + std::to_string(candidate.persistent_batch_size),
               "Persistent batch " +
                   std::to_string(candidate.persistent_batch_size) + ", CTA " +
                   std::to_string(candidate.bdimx) + "x" +
                   std::to_string(candidate.bdimy) + ", maxrregcount " +
                   std::to_string(candidate.register_per_thread)});
        }
        // The state uses only ASCII digits, separators and newlines, so JSON
        // quoting at most doubles its size. Option text needs no escaping.
        // These limits leave ample room for the fixed envelope under 64 KiB.
        size_t option_text_bytes = 0;
        for (const auto& option : options) {
          option_text_bytes += option.id.size() + option.description.size();
        }
        const bool request_fits =
            state.size() <= 16 * 1024 && option_text_bytes <= 16 * 1024;
        NVF_ERROR(
            request_fits || jev_mode == scheduler::jev::Mode::Record,
            "Register-backed inner persistent Choice request exceeds safe "
            "size");
        if (request_fits) {
          const std::string native_choice =
              jev_mode == scheduler::jev::Mode::Record
              ? "b" + std::to_string(best_heuristic.persistent_batch_size)
              : "";
          const std::string selected = scheduler::jev::chooseDecision(
              "IP5.batch", state, options, native_choice);
          const auto selected_it = std::find_if(
              candidate_heuristics.begin(),
              candidate_heuristics.end(),
              [&](const auto& candidate) {
                return selected ==
                    "b" + std::to_string(candidate.persistent_batch_size);
              });
          NVF_ERROR(
              selected_it != candidate_heuristics.end(),
              "Unknown register-backed inner persistent batch plan");
          const int64_t selected_threads =
              (selected_it->bdimx > 16 &&
               selected_it->padded_bdimx * selected_it->bdimy <=
                   max_threads_in_block)
              ? selected_it->padded_bdimx * selected_it->bdimy
              : selected_it->bdimx * selected_it->bdimy;
          NVF_ERROR(
              selected_threads > 0 &&
                  selected_threads <= max_threads_in_block &&
                  selected_it->register_per_thread > 0 &&
                  selected_it->register_per_thread <= 255,
              "Selected inner persistent batch exceeds CTA/register limits");
          best_heuristic = *selected_it;
        }
      }
    }
  }

  // Fill in the reduction params
  rparams->cparams.maxrregcount = best_heuristic.register_per_thread;

  // Disable magic zero to further reduce computation cost.
  // Magic zero reduces register usage, so only disble it when the register
  // usage is so low that we can disable project to avoid recompute.
  if (is_high_bandwidth_flops_ratio && disable_project_to_avoid_recompute) {
    rparams->cparams.enable_magic_zero = false;
  }
  // Inner reduction domain
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = best_heuristic.is_pad_bdimx;
  rparams->batches_per_block_inner_reduction =
      best_heuristic.persistent_batch_size;

  // For persistent schedules always have to mark the reduction unrolled
  // otherwise rfactor can fail
  rparams->unroll_factor_inner_reduction = properties.vectorize_factor;
  rparams->vectorize_inner_reduction = properties.vectorize_factor > 1;

  // Iter domain
  rparams->multiple_reds_per_blk = best_heuristic.bdimy > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t godim =
      ceilDiv(properties.total_iteration_numel, best_heuristic.bdimy);
  if (godim > 1) {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  rparams->lparams = LaunchParams(
      gdimx,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      best_heuristic.bdimy,
      LaunchParams::UNINITIALIZED_VAL);
}

// A smaller power-of-two width preserves the alignment, contiguity, and
// vectorization breakpoint proved by getVectorizationFactor. Recompute the
// complete register-backed batch/CTA/register/grid menu before offering it;
// the width Choice must not rely on a later policy to make a plan legal.
std::string innerWidthFrontierConsequence(
    const PersistentKernelProperties& properties,
    int64_t width,
    const Inner2DPolicyResources& resources) {
  const int64_t parallel = properties.inner_most_dimension_numel / width;
  const int64_t buffer_bits =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * width;
  std::string result =
      "{\"schema\":\"nvfuser.ip5.width.batch_frontier.v1\","
      "\"scope\":\"source_unranked_batch_score_frontier_under_initial_targets_and_cta_seed\","
      "\"conditioning\":\"One source-initial resource target and CTA seed; other target and seed branches are pending.\","
      "\"vector_width\":" +
      std::to_string(width) +
      ",\"reduction_elements_after_vectorize\":" + std::to_string(parallel) +
      ",\"buffer_bits_per_batch\":" + std::to_string(buffer_bits) +
      ",\"batch_min\":" + std::to_string(resources.batch_min) +
      ",\"batch_max\":" + std::to_string(resources.batch_max) +
      ",\"batch_plans\":[";
  for (const auto& plan : resources.candidate_plans) {
    const int64_t score_x =
        plan.is_pad_bdimx ? plan.padded_bdimx : plan.bdimx;
    const int64_t required_x = ceilDiv(parallel, plan.persistent_batch_size);
    const int64_t grid_x =
        ceilDiv(properties.total_iteration_numel, plan.bdimy);
    if (result.back() != '[') {
      result += ',';
    }
    result +=
        "{\"persistent_batch\":" +
        std::to_string(plan.persistent_batch_size) +
        ",\"score_block\":{\"x\":" + std::to_string(score_x) +
        ",\"y\":" + std::to_string(plan.bdimy) +
        ",\"threads\":" + std::to_string(score_x * plan.bdimy) +
        ",\"padded_x\":" + std::to_string(plan.padded_bdimx) +
        ",\"pad_x_to_warp\":" +
        (plan.is_pad_bdimx ? "true" : "false") +
        "},\"required_reduction_threads_x\":" +
        std::to_string(required_x) +
        ",\"nvrtc_register_cap_per_thread\":" +
        std::to_string(plan.register_per_thread) +
        ",\"estimated_occupancy_warps_per_sm\":" +
        std::to_string(plan.occupancy) +
        ",\"estimated_waves\":" + std::to_string(plan.n_wave) +
        ",\"persistent_tail_elements\":" +
        std::to_string(plan.n_persistent_tails) +
        ",\"logical_iteration_grid_blocks_before_split\":" +
        std::to_string(grid_x) + '}';
  }
  return result +
      "],\"pending_choices\":[\"IP5.targets\",\"IP5.cta_seed\",\"IP5.batch\"]}";
}

PersistentKernelProperties chooseInner2DVectorWidth(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t native_overhead,
    int64_t native_warps) {
  const auto jev_mode = scheduler::jev::mode();
  if (jev_mode == scheduler::jev::Mode::Off) {
    return properties;
  }

  const int64_t max_width = properties.vectorize_factor;
  const bool valid_max = max_width > 0 && max_width <= 128 &&
      (max_width & (max_width - 1)) == 0 &&
      properties.inner_most_dimension_numel > 0 &&
      properties.total_reduction_numel == properties.inner_most_dimension_numel &&
      properties.inner_most_dimension_numel % max_width == 0 &&
      properties.max_persistent_buffer_size_bit > 0;
  if (!valid_max) {
    if (jev_mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "IP5.width", "No proved 2D register vectorization maximum");
    }
    return properties;
  }

  struct WidthPlan {
    int64_t width;
    Inner2DPolicyResources resources;
  };
  std::vector<WidthPlan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  // Build the native plan first, but serialize and offer candidates in
  // ascending width order so the request does not encode its preference.
  Inner2DPolicyResources native_resources;
  ReductionParams native_preview(base);
  innerPersistentHeuristic2DPlan(
      properties,
      &native_preview,
      native_overhead,
      native_warps,
      false,
      &native_resources);
  if (!native_resources.supported_menu) {
    // A failed Choice-menu proof does not invalidate the native scheduler.
    // Remote cannot emit a Record-only coverage marker, but must still leave
    // the width unchanged and let the native-only 2D path complete.
    if (jev_mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "IP5.width", "Native width has no proved 2D register plan");
    }
    return properties;
  }
  const bool native_axis_fits = native_resources.batch_max > 0 &&
      native_resources.batch_max <=
          std::numeric_limits<int64_t>::max() / max_width;
  if (!native_axis_fits) {
    if (jev_mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "IP5.width", "Native vector extent exceeds arithmetic bounds");
    }
    return properties;
  }

  const bool semantic_frontier = scheduler::jev::semanticV3Enabled();
  const auto* device = at::cuda::getCurrentDeviceProperties();
  const int64_t min_threads_per_block =
      properties.total_iteration_numel >= device->multiProcessorCount
      ? 4l * device->warpSize
      : device->maxThreadsPerBlock;
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size_bit,
          properties.max_persistent_buffer_size_bit),
      ceilDiv(properties.total_iteration_numel,
              (int64_t)device->multiProcessorCount));
  std::string state =
      std::string(semantic_frontier ? "inner_persistent.reg_width.v2\n"
                                    : "inner_persistent.reg_width.v1\n") +
      "inner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\nreduction_numel=" +
      std::to_string(properties.total_reduction_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_legal_width=" + std::to_string(max_width) +
      "\nnative_batch_max=" +
      std::to_string(native_resources.batch_max) +
      "\nmax_threads_per_block=" +
      std::to_string(device->maxThreadsPerBlock);
  if (semantic_frontier) {
    state += "\nmin_threads_per_block=" +
        std::to_string(min_threads_per_block) +
        "\nmax_multi_reduction_factor=" +
        std::to_string(max_multi_reduction_factor) +
        "\nsm_count=" + std::to_string(device->multiProcessorCount) +
        "\nwarp_size=" + std::to_string(device->warpSize) +
        "\nmax_threads_per_sm=" +
        std::to_string(device->maxThreadsPerMultiProcessor) +
        "\nmax_threads_x=" + std::to_string(device->maxThreadsDim[0]) +
        "\nmax_threads_y=" + std::to_string(device->maxThreadsDim[1]) +
        "\nregisters_per_block=" + std::to_string(device->regsPerBlock) +
        "\nmax_grid_x=" + std::to_string(device->maxGridSize[0]) +
        "\nmax_grid_y=" + std::to_string(device->maxGridSize[1]);
  }
  for (int64_t width = 1; width <= max_width; width *= 2) {
    auto candidate_properties = properties;
    candidate_properties.vectorize_factor = width;
    Inner2DPolicyResources resources;
    if (width == max_width) {
      resources = native_resources;
    } else {
      ReductionParams preview(base);
      innerPersistentHeuristic2DPlan(
          candidate_properties,
          &preview,
          native_overhead,
          native_warps,
          false,
          &resources);
    }
    // A smaller width cannot allocate more persistent values per thread than
    // the proved native maximum. Batch-menu validation already checks every
    // completed CTA, register count, and grid coordinate against hardware.
    if (!resources.supported_menu ||
        resources.batch_max >
            native_resources.batch_max * max_width / width) {
      continue;
    }
    plans.push_back({width, resources});
    scheduler::jev::ChoiceOption option{
        "w" + std::to_string(width),
        "Vector width " + std::to_string(width) +
            ", checked persistent batches " +
            std::to_string(resources.batch_min) + " through " +
            std::to_string(resources.batch_max)};
    if (semantic_frontier) {
      option.consequence_json =
          innerWidthFrontierConsequence(properties, width, resources);
    }
    options.push_back(std::move(option));
    state += "\nplan=" + std::to_string(width) + ":" +
        std::to_string(resources.batch_min) + ":" +
        std::to_string(resources.batch_max);
    if (semantic_frontier) {
      for (const auto& plan : resources.candidate_plans) {
        state += "\nbatch_plan=" + std::to_string(width) + ":" +
            std::to_string(plan.persistent_batch_size) + ":" +
            std::to_string(plan.bdimx) + ":" +
            std::to_string(plan.bdimy) + ":" +
            std::to_string(plan.padded_bdimx) + ":" +
            std::to_string(plan.register_per_thread) + ":" +
            std::to_string(plan.occupancy) + ":" +
            std::to_string(plan.n_wave) + ":" +
            std::to_string(plan.n_persistent_tails);
      }
    }
  }
  NVF_ERROR(!plans.empty(), "No legal 2D register vector width");
  NVF_ERROR(
      state.size() <= (semantic_frontier ? 64 * 1024 : 16 * 1024),
      "Inner persistent width request exceeds safe size");
  const std::string native_choice =
      jev_mode == scheduler::jev::Mode::Record
      ? "w" + std::to_string(max_width)
      : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP5.width", state, options, native_choice);
  const auto plan_it = std::find_if(
      plans.begin(), plans.end(), [&](const auto& plan) {
        return selected == "w" + std::to_string(plan.width);
      });
  NVF_ERROR(plan_it != plans.end(), "Unknown inner persistent vector width");
  auto chosen = properties;
  chosen.vectorize_factor = plan_it->width;
  return chosen;
}

struct Inner2DSeedPlan {
  int64_t seed;
  int64_t batch_min = 0;
  int64_t batch_max = 0;
  int64_t batch = 0;
  int64_t physical_x = 0;
  int64_t physical_y = 0;
  bool pad_to_warp = false;
  int64_t register_cap = 0;
  int64_t grid_x = 0;
  bool valid = false;
  std::vector<NormInnerParams> batch_plans;
};

// Pure preflight shared by the target frontier and the later seed Choice.
// expand_preview collects the later IP5.batch domain without selecting a
// batch or emitting a nested Choice.
std::vector<Inner2DSeedPlan> enumerateInner2DCTASeeds(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t overhead,
    int64_t warps,
    bool expand_preview) {
  const auto* device = at::cuda::getCurrentDeviceProperties();
  const int64_t warp = device->warpSize;
  const int64_t max_threads = device->maxThreadsPerBlock;
  const int64_t first_seed = 4 * warp;
  const int64_t parallel =
      properties.inner_most_dimension_numel / properties.vectorize_factor;
  const int64_t buffer_bits =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * properties.vectorize_factor;
  std::vector<Inner2DSeedPlan> plans;
  if (first_seed > 0 && first_seed <= max_threads && parallel > 0 &&
      buffer_bits > 0 && properties.total_iteration_numel > 0) {
    for (int64_t seed = first_seed;; seed = std::min(seed * 2, max_threads)) {
      Inner2DSeedPlan plan{seed};
      try {
        ReductionParams preview(base);
        Inner2DPolicyResources resources;
        innerPersistentHeuristic2DPlan(
            properties, &preview, overhead, warps, false, &resources, seed,
            expand_preview);
        plan.batch_min = resources.batch_min;
        plan.batch_max = resources.batch_max;
        plan.batch = preview.batches_per_block_inner_reduction;
        plan.pad_to_warp = preview.pad_inner_reduction_to_warp;
        plan.physical_y = preview.lparams.bdimy() ==
                LaunchParams::UNINITIALIZED_VAL
            ? 1
            : preview.lparams.bdimy();
        plan.register_cap = preview.cparams.maxrregcount;
        if (resources.supported_menu && plan.batch > 0) {
          const int64_t x = ceilDiv(parallel, plan.batch);
          plan.physical_x = plan.pad_to_warp
              ? ceilDiv(x, warp) * warp
              : x;
          plan.grid_x =
              ceilDiv(properties.total_iteration_numel, plan.physical_y);
          const int64_t threads = plan.physical_x * plan.physical_y;
          const bool grid_fits = plan.grid_x > 0 &&
              (plan.grid_x <= device->maxGridSize[0] ||
               (scheduler_utils::x_grid_limit <= device->maxGridSize[0] &&
                ceilDiv(plan.grid_x, scheduler_utils::x_grid_limit) <=
                    std::min<int64_t>(
                        scheduler_utils::y_grid_limit,
                        device->maxGridSize[1])));
          plan.valid = plan.batch >= plan.batch_min &&
              plan.batch <= plan.batch_max && plan.physical_x > 0 &&
              plan.physical_x <= device->maxThreadsDim[0] &&
              plan.physical_y > 0 &&
              plan.physical_y <= device->maxThreadsDim[1] && threads > 0 &&
              threads <= max_threads && plan.register_cap > 0 &&
              plan.register_cap <= 255 &&
              plan.register_cap <= device->regsPerBlock / threads && grid_fits;
          if (plan.valid && expand_preview) {
            plan.batch_plans = std::move(resources.candidate_plans);
          }
        }
      } catch (const std::exception&) {
        // A failed alternate preflight never becomes an API option.
      }
      if (!plan.valid) {
        plan = Inner2DSeedPlan{seed};
      }
      plans.push_back(plan);
      if (seed == max_threads) {
        break;
      }
    }
  }
  return plans;
}

// The 128-thread versus device-maximum seed is an empirical occupancy target.
// Keep the register, CTA and grid checks in the native planner and offer only
// seeds for which that planner produces a complete physical schedule.
int64_t chooseInner2DCTASeed(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t overhead,
    int64_t warps) {
  const auto mode = scheduler::jev::mode();
  const auto* device = at::cuda::getCurrentDeviceProperties();
  const int64_t warp = device->warpSize;
  const int64_t max_threads = device->maxThreadsPerBlock;
  const int64_t first_seed = 4 * warp;
  const int64_t native_seed =
      properties.total_iteration_numel >= device->multiProcessorCount
      ? first_seed
      : max_threads;
  if (mode == scheduler::jev::Mode::Off) {
    return native_seed;
  }
  const int64_t parallel =
      properties.inner_most_dimension_numel / properties.vectorize_factor;
  const int64_t buffer_bits =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * properties.vectorize_factor;
  const auto plans = enumerateInner2DCTASeeds(
      properties, base, overhead, warps, false);
  const bool native_valid = std::any_of(
      plans.begin(), plans.end(), [&](const Inner2DSeedPlan& plan) {
        return plan.seed == native_seed && plan.valid;
      });
  NVF_ERROR(
      native_valid || mode == scheduler::jev::Mode::Record,
      "Native 2D register CTA seed lacks a proved physical schedule");
  if (!native_valid) {
    scheduler::jev::recordCoverageSkip(
        "IP5.cta_seed", "Native CTA seed lacks a complete physical plan");
    return native_seed;
  }
  const auto valid_count = std::count_if(
      plans.begin(), plans.end(),
      [](const Inner2DSeedPlan& plan) { return plan.valid; });
  // Two seeds can have the same preview winner but expose different later
  // IP5.batch menus and per-batch CTA/register plans. Only a single hard-valid
  // seed is proved to lack a model decision.
  if (valid_count == 1) {
    scheduler::jev::recordNoChoiceDecision(
        "IP5.cta_seed",
        "t" + std::to_string(native_seed),
        "Only one hard-valid CTA seed",
        "native_cta_seed");
    return native_seed;
  }
  std::vector<scheduler::jev::ChoiceOption> options;
  // Keep ascending seed order on the wire; never send a native-winner hint.
  for (const auto& plan : plans) {
    if (!plan.valid) {
      continue;
    }
    options.push_back({
        "t" + std::to_string(plan.seed),
        "Minimum CTA target " + std::to_string(plan.seed) +
            "; checked batch " + std::to_string(plan.batch) +
            ", physical CTA " + std::to_string(plan.physical_x) + "x" +
            std::to_string(plan.physical_y) + ", register cap " +
            std::to_string(plan.register_cap)});
  }

  std::string state =
      "inner_persistent.reg_cta_seed.v1\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nsm_count=" + std::to_string(device->multiProcessorCount) +
      "\nwarp_size=" + std::to_string(warp) +
      "\nmax_threads_per_block=" + std::to_string(max_threads) +
      "\nmax_threads_x=" + std::to_string(device->maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device->maxThreadsDim[1]) +
      "\nregisters_per_block=" + std::to_string(device->regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device->maxGridSize[0]) +
      "\nmax_grid_y=" + std::to_string(device->maxGridSize[1]) +
      "\nparallel_after_vectorize=" + std::to_string(parallel) +
      "\nbuffer_bits_per_batch=" + std::to_string(buffer_bits) +
      "\nhigh_bandwidth_flops_ratio=" +
      std::to_string(scheduler_utils::isHighBandwidthFlopsRatio()) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_multi_reduction_factor=" +
      std::to_string(std::min(
          scheduler_utils::safeDiv(
              scheduler_utils::register_file_size_bit,
              properties.max_persistent_buffer_size_bit),
          ceilDiv(
              properties.total_iteration_numel,
              device->multiProcessorCount))) +
      "\nselected_register_overhead=" + std::to_string(overhead) +
      "\nselected_target_warps_per_sm=" + std::to_string(warps);
  for (const auto& plan : plans) {
    state += "\nseed=" + std::to_string(plan.seed) + ":" +
        std::to_string(plan.valid) + ":" +
        std::to_string(plan.batch_min) + ":" +
        std::to_string(plan.batch_max) + ":" +
        std::to_string(plan.batch) + ":" +
        std::to_string(plan.physical_x) + ":" +
        std::to_string(plan.physical_y) + ":" +
        std::to_string(plan.pad_to_warp) + ":" +
        std::to_string(plan.register_cap) + ":" +
        std::to_string(plan.grid_x);
  }
  NVF_ERROR(state.size() <= 16 * 1024, "2D register CTA seed request too large");
  const std::string selected = scheduler::jev::chooseDecision(
      "IP5.cta_seed", state, options,
      mode == scheduler::jev::Mode::Record
          ? "t" + std::to_string(native_seed)
          : "");
  const auto chosen = std::find_if(
      plans.begin(), plans.end(), [&](const Inner2DSeedPlan& plan) {
        return plan.valid && selected == "t" + std::to_string(plan.seed);
      });
  NVF_ERROR(chosen != plans.end(), "Unknown 2D register CTA seed");
  return chosen->seed;
}

void innerPersistentHeuristic2D(
    const PersistentKernelProperties& properties,
    ReductionParams* rparams) {
  // These two values are performance targets. The original exp overhead is a
  // conservative floor for every offered policy; no alternative may enlarge
  // the native maximum persistent batch or bypass a resource check.
  const int64_t native_overhead = properties.has_exp_op ? 32l : 16l;
  const bool high_bandwidth = scheduler_utils::isHighBandwidthFlopsRatio();
  const int64_t native_warps =
      high_bandwidth && properties.disable_project_to_avoid_recompute ? 16l
                                                                      : 28l;
  const auto jev_mode = scheduler::jev::mode();
  if (jev_mode == scheduler::jev::Mode::Off) {
    innerPersistentHeuristic2DPlan(
        properties, rparams, native_overhead, native_warps, false);
    return;
  }

  Inner2DPolicyResources native_resources;
  ReductionParams native_preview(*rparams);
  innerPersistentHeuristic2DPlan(
      properties,
      &native_preview,
      native_overhead,
      native_warps,
      false,
      &native_resources);
  if (!native_resources.supported_menu) {
    if (jev_mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "IP5.targets", "Native 2D register policy lacks a proved batch menu");
      scheduler::jev::recordCoverageSkip(
          "IP5.cta_seed", "Native 2D register policy lacks a proved batch menu");
    }
    // The validator is deliberately stricter than the original heuristic.
    // An unproved menu has no API option, so Remote executes the same native
    // planner as Off. Record retains its existing coverage-skip trace.
    innerPersistentHeuristic2DPlan(
        properties,
        rparams,
        native_overhead,
        native_warps,
        jev_mode == scheduler::jev::Mode::Record);
    return;
  }

  struct Policy {
    int64_t overhead;
    int64_t warps;
    Inner2DPolicyResources resources;
    std::vector<Inner2DSeedPlan> seeds;
    std::string id() const {
      return "o" + std::to_string(overhead) + "_w" + std::to_string(warps);
    }
  };
  std::vector<Policy> policies;
  std::vector<scheduler::jev::ChoiceOption> options;
  const bool semantic_frontier = scheduler::jev::semanticV3Enabled();
  const auto* device = at::cuda::getCurrentDeviceProperties();
  std::string state =
      std::string(semantic_frontier ? "inner_persistent.reg_targets.v2\n"
                                    : "inner_persistent.reg_targets.v1\n") +
      "has_exp_op=" +
      std::to_string(properties.has_exp_op) +
      "\nhigh_bandwidth_flops_ratio=" + std::to_string(high_bandwidth) +
      "\ndisable_projection_avoid_recompute=" +
      std::to_string(properties.disable_project_to_avoid_recompute) +
      "\nparallel_after_vectorize=" +
      std::to_string(
          properties.inner_most_dimension_numel / properties.vectorize_factor) +
      "\ntotal_iteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nbuffer_bits_per_batch=" +
      std::to_string(
          properties.max_persistent_buffer_size_bit /
          properties.total_reduction_numel * properties.vectorize_factor) +
      "\nwarp_size=" +
      std::to_string(device->warpSize) +
      "\nmax_threads_per_block=" +
      std::to_string(device->maxThreadsPerBlock);
  if (semantic_frontier) {
    state +=
        "\ninner_numel=" +
        std::to_string(properties.inner_most_dimension_numel) +
        "\nreduction_numel=" + std::to_string(properties.total_reduction_numel) +
        "\npersistent_buffer_bits=" +
        std::to_string(properties.max_persistent_buffer_size_bit) +
        "\nvector_width=" + std::to_string(properties.vectorize_factor) +
        "\nsm_count=" + std::to_string(device->multiProcessorCount) +
        "\nmax_threads_per_sm=" +
        std::to_string(device->maxThreadsPerMultiProcessor) +
        "\nmax_threads_x=" + std::to_string(device->maxThreadsDim[0]) +
        "\nmax_threads_y=" + std::to_string(device->maxThreadsDim[1]) +
        "\nregisters_per_block=" + std::to_string(device->regsPerBlock) +
        "\nmax_grid_x=" + std::to_string(device->maxGridSize[0]) +
        "\nmax_grid_y=" + std::to_string(device->maxGridSize[1]) +
        "\nmax_multi_reduction_factor=" +
        std::to_string(std::min(
            scheduler_utils::safeDiv(
                scheduler_utils::register_file_size_bit,
                properties.max_persistent_buffer_size_bit),
            ceilDiv(properties.total_iteration_numel,
                    (int64_t)device->multiProcessorCount)));
  }

  // Canonical order does not expose the native policy. Preflight computes all
  // batch/CTA/register plans and suppresses the nested IP5.batch Choice.
  for (const int64_t overhead : {16l, 32l}) {
    if (overhead < native_overhead) {
      continue;
    }
    for (const int64_t warps : {16l, 28l}) {
      Inner2DPolicyResources resources;
      if (overhead == native_overhead && warps == native_warps) {
        resources = native_resources;
      } else {
        ReductionParams preview(*rparams);
        innerPersistentHeuristic2DPlan(
            properties, &preview, overhead, warps, false, &resources);
      }
      if (!resources.supported_menu ||
          resources.batch_max > native_resources.batch_max) {
        continue;
      }
      policies.push_back({overhead, warps, resources, {}});
      const auto& policy = policies.back();
      options.push_back(
          {policy.id(),
           "Register overhead " + std::to_string(overhead) +
               ", occupancy target " + std::to_string(warps) +
               " warps per SM"});
      state += "\npolicy=" + std::to_string(overhead) + ":" +
          std::to_string(warps) + ":" + std::to_string(resources.batch_min) +
          ":" + std::to_string(resources.batch_max);
    }
  }
  NVF_ERROR(!policies.empty(), "No legal 2D register policy");
  bool complete_frontier = true;
  if (semantic_frontier) {
    const int64_t parallel =
        properties.inner_most_dimension_numel / properties.vectorize_factor;
    for (auto& policy : policies) {
      policy.seeds = enumerateInner2DCTASeeds(
          properties, *rparams, policy.overhead, policy.warps, true);
      const int64_t native_seed =
          properties.total_iteration_numel >= device->multiProcessorCount
          ? 4 * device->warpSize
          : device->maxThreadsPerBlock;
      const bool native_valid = std::any_of(
          policy.seeds.begin(), policy.seeds.end(), [&](const auto& seed) {
            return seed.seed == native_seed && seed.valid;
          });
      if (!native_valid) {
        complete_frontier = false;
        break;
      }
      for (const auto& seed : policy.seeds) {
        state += "\nseed=" + std::to_string(policy.overhead) + ":" +
            std::to_string(policy.warps) + ":" +
            std::to_string(seed.seed) + ":" +
            std::to_string(seed.valid) + ":" +
            std::to_string(seed.batch_min) + ":" +
            std::to_string(seed.batch_max);
        if (!seed.valid) {
          continue;
        }
        if (seed.batch_plans.empty() ||
            seed.batch_plans.front().persistent_batch_size != seed.batch_min ||
            seed.batch_plans.back().persistent_batch_size != seed.batch_max ||
            !std::all_of(
                seed.batch_plans.begin(),
                seed.batch_plans.end(),
                [&](const auto& batch) {
                  return batch.persistent_batch_size ==
                      seed.batch_min + (&batch - seed.batch_plans.data());
                })) {
          complete_frontier = false;
          break;
        }
        for (const auto& batch : seed.batch_plans) {
          state += "\nbatch_plan=" + std::to_string(policy.overhead) + ":" +
              std::to_string(policy.warps) + ":" +
              std::to_string(seed.seed) + ":" +
              std::to_string(batch.persistent_batch_size) + ":" +
              std::to_string(batch.bdimx) + ":" +
              std::to_string(batch.bdimy) + ":" +
              std::to_string(batch.padded_bdimx) + ":" +
              std::to_string(batch.register_per_thread) + ":" +
              std::to_string(batch.occupancy) + ":" +
              std::to_string(batch.n_wave) + ":" +
              std::to_string(batch.n_persistent_tails);
        }
      }
      if (!complete_frontier) {
        break;
      }
      auto& option = options.at(&policy - policies.data());
      std::string effect =
          "{\"schema\":\"nvfuser.ip5.targets.seed_batch_frontier.v1\","
          "\"scope\":\"source_unranked_seed_batch_frontier_under_selected_width\","
          "\"register_overhead_target\":" +
          std::to_string(policy.overhead) +
          ",\"resident_warps_per_sm_target\":" +
          std::to_string(policy.warps) +
          ",\"policy_batch_offer_min\":" +
          std::to_string(policy.resources.batch_min) +
          ",\"policy_batch_offer_max\":" +
          std::to_string(policy.resources.batch_max) +
          ",\"seed_frontier\":[";
      for (const auto& seed : policy.seeds) {
        if (!seed.valid) {
          continue;
        }
        if (effect.back() != '[') {
          effect += ',';
        }
        auto range = [&](const auto& measure) {
          int64_t low = std::numeric_limits<int64_t>::max();
          int64_t high = std::numeric_limits<int64_t>::min();
          for (const auto& batch : seed.batch_plans) {
            const int64_t value = measure(batch);
            low = std::min(low, value);
            high = std::max(high, value);
          }
          return "[" + std::to_string(low) + "," +
              std::to_string(high) + "]";
        };
        effect += "{\"minimum_cta_target_threads\":" +
            std::to_string(seed.seed) +
            ",\"batch_min\":" + std::to_string(seed.batch_min) +
            ",\"batch_max\":" + std::to_string(seed.batch_max) +
            ",\"batch_count\":" +
            std::to_string(seed.batch_plans.size()) +
            ",\"score_cta_threads_range\":" + range([&](const auto& b) {
              const bool pad = b.bdimx > 16 &&
                  b.padded_bdimx * b.bdimy <= device->maxThreadsPerBlock;
              return (pad ? b.padded_bdimx : b.bdimx) * b.bdimy;
            }) +
            ",\"physical_cta_threads_range\":" + range([&](const auto& b) {
              const int64_t x = ceilDiv(parallel, b.persistent_batch_size);
              const int64_t padded = ceilDiv(x, device->warpSize) *
                  device->warpSize;
              const bool pad = b.bdimx > 16 &&
                  b.padded_bdimx * b.bdimy <= device->maxThreadsPerBlock;
              return (pad ? padded : x) * b.bdimy;
            }) +
            ",\"nvrtc_register_cap_per_thread_range\":" +
            range([](const auto& b) { return b.register_per_thread; }) +
            ",\"estimated_occupancy_warps_per_sm_range\":" +
            range([](const auto& b) { return b.occupancy; }) +
            ",\"estimated_waves_range\":" +
            range([](const auto& b) { return b.n_wave; }) +
            ",\"persistent_tail_elements_range\":" +
            range([](const auto& b) { return b.n_persistent_tails; }) +
            ",\"logical_iteration_grid_blocks_before_split_range\":" +
            range([&](const auto& b) {
              return ceilDiv(properties.total_iteration_numel, b.bdimy);
            }) + '}';
      }
      option.consequence_json = effect +
          "],\"pending_choices\":[\"IP5.cta_seed\",\"IP5.batch\"]}";
    }
  }
  size_t option_bytes = 0;
  for (const auto& option : options) {
    option_bytes += option.id.size() + option.description.size() +
        option.consequence_json.size();
  }
  const bool request_fits = semantic_frontier
      ? state.size() * 2 + option_bytes <= 64 * 1024
      : state.size() <= 16 * 1024;
  if (semantic_frontier && (!complete_frontier || !request_fits)) {
    const std::string native_id = "o" + std::to_string(native_overhead) +
        "_w" + std::to_string(native_warps);
    scheduler::jev::recordHardSingletonDecision(
        "IP5.targets", native_id,
        complete_frontier ? "target_preview_too_large"
                          : "target_preview_incomplete",
        "native");
    if (jev_mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "IP5.targets",
          complete_frontier ? "Complete target seed/batch frontier exceeds 64 KiB"
                            : "Target seed/batch frontier preflight is incomplete");
    }
    innerPersistentHeuristic2DPlan(
        properties, rparams, native_overhead, native_warps, false);
    return;
  }
  NVF_ERROR(request_fits, "Inner persistent policy request too large");
  const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
      ? "o" + std::to_string(native_overhead) + "_w" +
          std::to_string(native_warps)
      : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP5.targets", state, options, native_choice);
  const auto policy_it =
      std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
        return policy.id() == selected;
      });
  NVF_ERROR(policy_it != policies.end(), "Unknown 2D register policy");
  const int64_t selected_seed = chooseInner2DCTASeed(
      properties, *rparams, policy_it->overhead, policy_it->warps);
  innerPersistentHeuristic2DPlan(
      properties, rparams, policy_it->overhead, policy_it->warps, true,
      nullptr, selected_seed);
}

// This heuristic is only tuned for cross entropy loss since it is the only
// layer that uses cluster reduction due to the large vocabulary size, e.g.
// 256000 in Gemma2. For other layers, the reduction domain is the hidden size
// dimension, which is usually smaller than 64K and block reduction is used. The
// reduction domain is parallelized by vectorization, bdimx, gdimx, and
// persistent batch size. The logic is based on empirical tests:
// (1) Fixed vectorization of 128 bit
// (2) Fixed bdimx of 256
// (3) Cluster size is calculated based on buffer size and available SM register
// buffers. Assuming each SM uses 32KB of registers and 2 blocks per SM, thus
// each block uses 16KB registers. Therefore, cluster size is total buffer size
// divided by 16KB per block.
// (4) Estimate persistent batch size given vectorization, bdimx, and cluster
// size

void innerPersistentHeuristicCluster(
    const PersistentKernelProperties& properties,
    ReductionParams* rparams) {
  const auto jev_mode = scheduler::jev::mode();
  int64_t vectorize_factor = properties.vectorize_factor;
  int64_t after_vect = properties.total_reduction_numel / vectorize_factor;
  int64_t bdimx = 256; // empirical value
  int64_t after_vect_bdimx = ceilDiv(after_vect, bdimx);

  // The legacy batch is a safety envelope as well as the off/record answer.
  // Each persistent tensor can have a different shape (including broadcasts),
  // so dividing the aggregate buffer bits by the cluster size is not a valid
  // per-CTA capacity proof. A no-larger batch gives every thread no larger a
  // persistent extent than in the existing schedule for every tensor shape.
  const int64_t register_per_block =
      scheduler_utils::register_file_size_bit / 2;
  int64_t blocks_per_cluster =
      ceilDiv(properties.max_persistent_buffer_size_bit, register_per_block);
  blocks_per_cluster = scheduler_utils::roundUpPow2(blocks_per_cluster);
  blocks_per_cluster =
      std::min(blocks_per_cluster, scheduler_utils::getMaxClusterSize());
  const int64_t safe_batch = ceilDiv(after_vect_bdimx, blocks_per_cluster);
  int64_t persistent_batch =
      jev_mode == scheduler::jev::Mode::Remote ? 0 : safe_batch;

  if (jev_mode != scheduler::jev::Mode::Off) {
    const auto device = at::cuda::getCurrentDeviceProperties();
    const int64_t max_cluster = scheduler_utils::getMaxClusterSize();
    const int64_t max_grid_y =
        std::min<int64_t>(scheduler_utils::y_grid_limit, device->maxGridSize[1]);
    struct ClusterBatchPlan {
      int64_t batch;
      int64_t cluster;
    };
    std::vector<ClusterBatchPlan> plans;
    std::vector<scheduler::jev::ChoiceOption> options;
    // BIDx's actual extent is derived by the reduction split. Only batches no
    // larger than the proven legacy plan are safe for every underlying buffer.
    if (after_vect_bdimx > 0 && max_cluster >= 2 && max_cluster <= 16 &&
        bdimx <= device->maxThreadsPerBlock &&
        bdimx <= device->maxThreadsDim[0] &&
        device->regsPerBlock >=
            scheduler_utils::register_file_size_bit_full /
                scheduler_utils::bits_per_register &&
        properties.total_iteration_numel <= max_grid_y &&
        properties.max_persistent_buffer_size_bit > 0) {
      for (int64_t target_cluster = 2; target_cluster <= max_cluster;
           ++target_cluster) {
        const int64_t batch = ceilDiv(after_vect_bdimx, target_cluster);
        const int64_t actual_cluster = ceilDiv(after_vect_bdimx, batch);
        if (batch > safe_batch || actual_cluster < 2 ||
            actual_cluster > max_cluster ||
            std::any_of(plans.begin(), plans.end(), [&](const auto& plan) {
              return plan.batch == batch;
            })) {
          continue;
        }
        plans.push_back({batch, actual_cluster});
        options.push_back({
            "b" + std::to_string(batch),
            "Persistent batch " + std::to_string(batch) + ", cluster " +
                std::to_string(actual_cluster) + ", CTA 256"});
      }
    }

    const auto native_it = std::find_if(
        plans.begin(), plans.end(), [&](const auto& plan) {
          return plan.batch == persistent_batch;
        });
    if (jev_mode == scheduler::jev::Mode::Record &&
        native_it == plans.end()) {
      scheduler::jev::recordCoverageSkip(
          "IP7.cluster", "Native cluster batch falls outside proved menu");
    } else {
      NVF_ERROR(!options.empty(), "No legal inner persistent cluster plan");
      const std::string state =
          "inner_persistent.cluster.v1\nreduction_numel=" +
          std::to_string(properties.total_reduction_numel) +
          "\niteration_numel=" +
          std::to_string(properties.total_iteration_numel) +
          "\nvector_width=" + std::to_string(vectorize_factor) +
          "\npersistent_buffer_bits=" +
          std::to_string(properties.max_persistent_buffer_size_bit) +
          "\nregister_file_bits=" +
          std::to_string(scheduler_utils::register_file_size_bit) +
          "\nmax_cluster_size=" + std::to_string(max_cluster) +
          "\nmax_threads_per_block=" +
          std::to_string(device->maxThreadsPerBlock) +
          "\nmax_threads_dim_x=" +
          std::to_string(device->maxThreadsDim[0]) +
          "\nregs_per_block=" + std::to_string(device->regsPerBlock) +
          "\nmax_grid_y=" + std::to_string(max_grid_y);
      const std::string native_choice =
          jev_mode == scheduler::jev::Mode::Record
          ? "b" + std::to_string(persistent_batch)
          : "";
      const std::string selected = scheduler::jev::chooseDecision(
          "IP7.cluster", state, options, native_choice);
      const auto selected_it = std::find_if(
          plans.begin(), plans.end(), [&](const auto& plan) {
            return selected == "b" + std::to_string(plan.batch);
          });
      NVF_ERROR(
          selected_it != plans.end(),
          "Unknown inner persistent cluster batch plan");
      persistent_batch = selected_it->batch;
    }
  }

  rparams->cross_block_inner_reduction = true;
  rparams->cross_grid_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = true;
  rparams->batches_per_block_inner_reduction = persistent_batch;
  rparams->unroll_factor_inner_reduction = vectorize_factor;
  rparams->vectorize_inner_reduction = vectorize_factor > 1;

  // cluster reduction is true, the blocks in grid_dim_inner_reduction are
  // grouped into a cluster.
  rparams->cross_cluster_reduction = true;
  rparams->grid_dim_inner_reduction = ParallelType::BIDx;

  // Iter domain parameters
  rparams->grid_dim_iter_dom = ParallelType::BIDy;
  rparams->multiple_reds_per_blk = false;
  rparams->unroll_factor_iter_dom = 1;

  // launch parameters
  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      bdimx,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);
}

void innerPersistentHeuristicSharedMemory(
    Fusion* fusion,
    const PersistentKernelProperties& properties,
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info,
    ReductionParams* rparams) {
  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // Inner reduction domain
  // This heuristic is only used for cases with large total_reduction_numel.
  // e.g. layer_norm with hidden size larger than 64K for fp16 or 32K for fp32.
  // fully vectorized, use maxThreadsPerBlock to reduce workload per threads
  int64_t vectorize_factor = properties.vectorize_factor;
  int64_t bdimx = dev_prop->maxThreadsPerBlock;
  NVF_ERROR(
      properties.total_reduction_numel >= vectorize_factor * bdimx,
      "total_reduction_numel should be larger than or equal to "
      "vectorize_factor * bdimx.\n",
      "total_reduction_numel= ",
      properties.total_reduction_numel,
      ", vectorize_factor= ",
      vectorize_factor,
      ", bdimx= ",
      bdimx);
  const auto jev_mode = scheduler::jev::mode();
  int64_t persistent_batch = jev_mode == scheduler::jev::Mode::Remote
      ? -1
      : ceilDiv(properties.total_reduction_numel, vectorize_factor * bdimx);

  if (jev_mode != scheduler::jev::Mode::Off) {
    // The batch sets the residual TIDx extent in scheduleReductionTV. Its
    // launch parameter is inferred there, then padded to a whole warp. The
    // maximum-thread seed is an empirical preference, not a fixed launch dim.
    const int64_t warp_size = dev_prop->warpSize;
    const bool supported_topology =
        dev_prop->major < 9 &&
        properties.total_reduction_numel ==
            properties.inner_most_dimension_numel &&
        scheduler_utils::getMaxClusterSize() == 1;
    const bool supported_width =
        vectorize_factor > 0 &&
        properties.total_reduction_numel % vectorize_factor == 0;
    const bool supported_threads = warp_size > 0 && bdimx >= warp_size &&
        (bdimx & (bdimx - 1)) == 0 && (warp_size & (warp_size - 1)) == 0;
    if (supported_topology && supported_width && supported_threads) {
      auto persistent_buffer_info_entry =
          HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
              data_cache, [&fusion]() {
                return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                    scheduler_utils::persistentBuffers(fusion));
              });
      const auto& persistent_buffer_info = persistent_buffer_info_entry.get();
      auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
      const auto& buffers = properties.project_persistent_buffers
          ? persistent_buffer_info.projectable_buffer_inputs
          : persistent_buffer_info.persistent_buffers;
      std::vector<smem_cta_choice::Buffer> buffer_facts;
      buffer_facts.reserve(buffers.size());
      for (auto* buffer : buffers) {
        buffer_facts.push_back(
            {"T" + std::to_string(buffer->name()),
             scheduler_utils::getPersistentBufferSizeBitOfTensor(
                 buffer, runtime_info, persistent_buffer_info),
             dataTypeSizeBit(buffer->getDataType())});
      }
      const smem_cta_choice::Input resource_input{
          properties.total_reduction_numel,
          vectorize_factor,
          bdimx,
          warp_size,
          properties.max_persistent_buffer_size_bit,
          static_cast<int64_t>(dev_prop->sharedMemPerBlockOptin) * 8,
          scheduler_utils::getReductionSmemWorkspaceBit(fusion, reduction_tvs),
          scheduler_utils::static_smem_usage_in_bits,
          kSharedMemoryAlignmentBytes * 8,
          getMaxVectorizationSizeInBit(),
          std::move(buffer_facts)};
      const auto menu = smem_cta_choice::enumerate(resource_input);
      for (const auto& candidate : menu.geometry) {
        // The pure preview is the production admission calculation. Keep an
        // exact cross-check against the existing fusion-aware analyzer while
        // moving the resource operands into the versioned text contract.
        const int64_t source_available_bits = normalization_scheduler_utils::
            getMaxSharedMemorySizeBitForPersistentBuffer(
                fusion,
                runtime_info,
                reduction_tvs,
                persistent_buffer_info,
                properties.project_persistent_buffers,
                candidate.batch,
                candidate.padded_threads);
        NVF_ERROR(
            candidate.available_persistent_bits == source_available_bits,
            "IP8.cta pure capacity preview disagrees with source analysis");
      }
      std::vector<scheduler::jev::ChoiceOption> options;
      for (const auto& candidate : menu.candidates) {
        const std::string id = "b" + std::to_string(candidate.batch);
        options.push_back({
            id,
            "Use persistent batch " + std::to_string(candidate.batch) +
                " from CTA target " + std::to_string(candidate.seed) +
                "; derived warp-padded TIDx=" +
                std::to_string(candidate.padded_threads)});
      }
      // Preserve the native answer for Record when its seed fails capacity;
      // Remote may still choose a separately proved smaller CTA target.
      const bool native_is_legal = menu.hasNativeSeed(bdimx);
      if (native_is_legal) {
        const std::string state =
            "inner_persistent.smem_cta.v1\nreduction_numel=" +
            std::to_string(properties.total_reduction_numel) +
            "\nvector_width=" + std::to_string(vectorize_factor) +
            "\nmax_threads_per_block=" + std::to_string(bdimx) +
            "\nwarp_size=" + std::to_string(warp_size) +
            "\npersistent_buffer_bits=" +
            std::to_string(properties.max_persistent_buffer_size_bit) +
            "\nsmem_per_block_bytes=" +
            std::to_string(dev_prop->sharedMemPerBlockOptin);
        const std::string native_choice =
            jev_mode == scheduler::jev::Mode::Record ? options.front().id : "";
        const std::string selected = scheduler::jev::chooseDecision(
            "IP8.cta", state, options, native_choice);
        const auto option_it = std::find_if(
            options.begin(),
            options.end(),
            [&](const auto& option) { return option.id == selected; });
        NVF_ERROR(
            option_it != options.end(), "Unknown inner persistent CTA plan");
        persistent_batch =
            menu.candidates.at(std::distance(options.begin(), option_it)).batch;
      } else if (
          jev_mode == scheduler::jev::Mode::Remote &&
          !menu.candidates.empty()) {
        std::string state =
            "inner_persistent.smem_cta.v2\nreduction_numel=" +
            std::to_string(properties.total_reduction_numel) +
            "\nvector_width=" + std::to_string(vectorize_factor) +
            "\nmax_threads_per_block=" + std::to_string(bdimx) +
            "\nwarp_size=" + std::to_string(warp_size) +
            "\npersistent_buffer_bits=" +
            std::to_string(properties.max_persistent_buffer_size_bit) +
            "\nsmem_per_block_bytes=" +
            std::to_string(dev_prop->sharedMemPerBlockOptin) +
            "\nworkspace_bits=" +
            std::to_string(resource_input.workspace_bits) +
            "\nstatic_reserve_bits=" +
            std::to_string(resource_input.static_reserve_bits) +
            "\nalignment_bits=" +
            std::to_string(resource_input.alignment_bits) +
            "\nmax_vector_bits=" +
            std::to_string(resource_input.max_vector_bits) +
            "\nproject_to_inputs=" +
            std::to_string(properties.project_persistent_buffers) +
            "\ninner_numel=" +
            std::to_string(properties.inner_most_dimension_numel) +
            "\ndevice_major=" + std::to_string(dev_prop->major) +
            "\nmax_cluster_size=" +
            std::to_string(scheduler_utils::getMaxClusterSize());
        for (const auto& buffer : resource_input.buffers) {
          state += "\nbuffer=" + buffer.tv + ':' +
              std::to_string(buffer.logical_bits) + ':' +
              std::to_string(buffer.dtype_bits);
        }
        std::vector<scheduler::jev::ChoiceOption> v2_options;
        v2_options.reserve(menu.candidates.size());
        for (const auto& plan : menu.candidates) {
          v2_options.push_back({
              "b" + std::to_string(plan.batch),
              "Use persistent batch " + std::to_string(plan.batch) +
                  " from CTA target " + std::to_string(plan.seed) +
                  "; derived warp-padded TIDx=" +
                  std::to_string(plan.padded_threads) +
                  "; required shared memory " +
                  std::to_string(plan.required_smem_bits) + " bits of " +
                  std::to_string(resource_input.smem_cap_bits) + " bits",
              "{\"cta_seed\":" + std::to_string(plan.seed) +
                  ",\"persistent_batch\":" +
                  std::to_string(plan.batch) +
                  ",\"actual_threads\":" +
                  std::to_string(plan.actual_threads) +
                  ",\"padded_threads\":" +
                  std::to_string(plan.padded_threads) +
                  ",\"roundup_overhead_bits\":" +
                  std::to_string(plan.roundup_overhead_bits) +
                  ",\"available_persistent_bits\":" +
                  std::to_string(plan.available_persistent_bits) +
                  ",\"required_smem_bits\":" +
                  std::to_string(plan.required_smem_bits) +
                  ",\"pending_choices\":[]}"});
        }
        const std::string selected = scheduler::jev::chooseDecision(
            "IP8.cta", state, v2_options, "");
        const auto selected_it = std::find_if(
            v2_options.begin(),
            v2_options.end(),
            [&](const auto& option) { return option.id == selected; });
        NVF_ERROR(
            selected_it != v2_options.end(),
            "Unknown inner persistent CTA plan");
        persistent_batch =
            menu.candidates.at(std::distance(v2_options.begin(), selected_it))
                .batch;
      } else if (jev_mode == scheduler::jev::Mode::Record) {
        scheduler::jev::recordCoverageSkip(
            "IP8.cta", "Native CTA seed exceeds proved shared-memory budget");
      } else {
        NVF_ERROR(false, "No hard-valid shared-memory persistent CTA plan");
      }
    } else {
      NVF_ERROR(
          jev_mode != scheduler::jev::Mode::Remote,
          "Unsupported shared-memory persistent CTA topology");
    }
  }
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = true;
  rparams->batches_per_block_inner_reduction = persistent_batch;
  rparams->unroll_factor_inner_reduction = vectorize_factor;
  rparams->vectorize_inner_reduction = vectorize_factor > 1;

  // Iter
  rparams->multiple_reds_per_blk = false;
  rparams->grid_dim_iter_dom = ParallelType::BIDx;
  rparams->unroll_factor_iter_dom = 1;
  rparams->lparams = LaunchParams(
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL,
      LaunchParams::UNINITIALIZED_VAL);
}

struct Inner3DPlanResources {
  int64_t padded_threads_x = 0;
  int64_t threads_y = 0;
  int64_t threads_z = 0;
  int64_t padded_threads_per_block = 0;
  int64_t registers_per_thread = 0;
  int64_t grid_blocks = 0;
  int64_t pre_outer_threads_z = 0;
  int64_t target_unroll = 0;
  int64_t target_iterations = 0;
};

// A Choice is made before later IP6 controls can rebuild the plan. Describe
// the complete plan materialized for this candidate at the current site, not
// a final launch or measured occupancy. This is only sent by semantic-v3.
std::string inner3DLocalPlanConsequence(
    const ReductionParams& params,
    const Inner3DPlanResources& resources) {
  return "{\"schema\":\"nvfuser.inner_persistent.3d_local_plan.v1\","
         "\"scope\":\"local_preview_before_later_choices\","
         "\"inner\":{\"batches_per_block\":" +
      std::to_string(params.batches_per_block_inner_reduction) +
      ",\"unroll\":" +
      std::to_string(params.unroll_factor_inner_reduction) +
      ",\"vectorized\":" +
      (params.vectorize_inner_reduction ? "true" : "false") +
      "},\"outer\":{\"batches_per_block\":" +
      std::to_string(params.batches_per_block_outer_reduction) +
      ",\"unroll\":" +
      std::to_string(params.unroll_factor_outer_reduction) +
      "},\"block\":{\"padded_x\":" +
      std::to_string(resources.padded_threads_x) +
      ",\"y\":" + std::to_string(resources.threads_y) +
      ",\"z\":" + std::to_string(resources.threads_z) +
      ",\"threads\":" +
      std::to_string(resources.padded_threads_per_block) +
      "},\"iteration_grid_blocks\":" +
      std::to_string(resources.grid_blocks) +
      ",\"nvrtc_maxrregcount\":" +
      std::to_string(resources.registers_per_thread) +
      ",\"planner_work\":{\"target_unroll\":" +
      std::to_string(resources.target_unroll) +
      ",\"target_iterations\":" +
      std::to_string(resources.target_iterations) + "}}";
}

// Retain the exact L1/L2 estimate used by the native 3D planner.
int64_t getInner3DNativeWarpTarget(
    const PersistentKernelProperties& properties,
    const cudaDeviceProp& device) {
  const int64_t n_elems =
      properties.total_reduction_numel * properties.total_iteration_numel;
  constexpr int64_t l1_cache_bit = 32l * 1024l * 8;
  constexpr int64_t active_threads = 1024;
  const bool fits_in_l2 =
      n_elems * properties.max_dtype_size_bit * properties.n_tensor_inputs <
      device.l2CacheSize * 8;
  const int64_t warp_size_based_on_l2 =
      fits_in_l2 ? 256l / properties.max_dtype_size_bit : 16l;
  const int64_t warp_size_based_on_l1 = std::min(
      ceilDiv(
          properties.total_reduction_numel,
          scheduler_utils::safeDiv(
              l1_cache_bit,
              properties.n_tensor_inputs * properties.max_dtype_size_bit *
                  active_threads)),
      16l);
  // This empirical target can be odd; preserve that exact native result.
  return std::min(warp_size_based_on_l1, warp_size_based_on_l2);
}

// TODO: clean and revise the heuristics
void innerPersistentHeuristic3DPlan(
    const PersistentKernelProperties& properties,
    int64_t wave_cap,
    ReductionParams* rparams,
    Inner3DPlanResources* resources = nullptr,
    int64_t warp_target = -1,
    int64_t occupancy_target = 28,
    int64_t outer_unroll_cap = std::numeric_limits<int64_t>::max(),
    int64_t register_overhead_override = -1,
    bool unroll_first = true,
    int64_t min_threads_override = -1) {
  // Define two free parameters used in this heuristic.
  // register_overhead is all registers except those for the persistent
  // buffers. The register in each thread = register_overhead +
  // persistent_buffer_size_bit / bits_per_register
  // Current values are based on tests of sofmax, layer_norm, softmax_dropout,
  // dropout_layer_norm on A100 & H100. It directly affects maxregcount passed
  // to NVRTC and influences the occupancy.
  const int64_t register_overhead = register_overhead_override > 0
      ? register_overhead_override
      : (properties.has_exp_op ? 32l : 16l);

  // The original target is 28 warps. Non-native targets reach this planner
  // only after a complete-plan Choice; Off mode retains the original target.
  const int64_t target_warps_per_sm = occupancy_target;

  // Set some targets for parallelization
  const int64_t n_elems =
      properties.total_reduction_numel * properties.total_iteration_numel;

  const int64_t outer_reduction_numel =
      properties.total_reduction_numel / properties.inner_most_dimension_numel;

  const auto dev_prop = at::cuda::getCurrentDeviceProperties();
  // WARNING: At some point we may want to generate heuristics for another
  // device that is not the current device.
  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;

  const int64_t device_multiprocessor_count =
      (int64_t)dev_prop->multiProcessorCount;

  auto const max_unroll = ceilDiv(
      // Available unrolling based on size of data type
      128l / properties.max_dtype_size_bit,
      // Reduce unrolling if we have many inputs, start reduction at 4 inputs
      scheduler_utils::lastPow2(std::max(properties.n_tensor_inputs >> 2, 1l)));

  // A caller may substitute a legal target after materializing and checking
  // the complete plan. Off mode uses the unchanged L1/L2 estimate.
  const int64_t warp_size = warp_target > 0
      ? warp_target
      : getInner3DNativeWarpTarget(properties, *dev_prop);

  // Initialization
  int64_t target_blocks = 1;
  int64_t target_unroll = 1;
  int64_t target_iterations = 1;

  // Try to set a minmum amount of work for each thread, as cross thread
  // communication is slow so it shouldn't be done for every element in the
  // reduction.
  int64_t min_target_iterations =
      scheduler_utils::safeDiv(256, properties.max_dtype_size_bit);

  // Start trying to break parallelization up across threads,
  // unrolling/iterations, and blocks.

  // max_threads_in_block is the cap on a thread block, the minimum is based on
  // warp_size
  int64_t max_threads_in_block = std::max(
      warp_size,
      ceilDiv(properties.total_reduction_numel, min_target_iterations));

  // If we have one warp per block, check if that's enough to saturate the SMs
  target_blocks = ceilDiv(n_elems, warp_size);

  // If we have more than a wave of blocks, put parallelism into unrolling and
  // target iterations
  if (target_blocks > device_multiprocessor_count) {
    auto available_unroll = scheduler_utils::safeDiv(
        n_elems, warp_size * device_multiprocessor_count);

    // Spread across unrolling and iterations, want a balance of the two so flip
    // back and forth to alternate adding to them.
    bool flip = unroll_first;

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

      available_unroll = scheduler_utils::safeDiv(
          n_elems,
          warp_size * device_multiprocessor_count * target_unroll *
              target_iterations);
      flip = !flip;
    }

    // Recompute target blocks
    target_blocks =
        ceilDiv(n_elems, warp_size * target_unroll * target_iterations);
  }

  // The wave cap is an empirical target. All subsequent dimension, register,
  // and launch calculations are recomputed for each offered complete plan.
  target_blocks = std::min(target_blocks, device_multiprocessor_count * wave_cap);

  if (target_blocks * target_unroll * target_iterations < n_elems) {
    if (outer_reduction_numel == 1) {
      // set to hardware limit to use small persistent buffer for large
      // reductions
      max_threads_in_block = std::min(
          ceilDiv(n_elems, target_blocks * target_unroll),
          (int64_t)dev_prop->maxThreadsPerBlock);
    } else {
      // targetting 4 waves, so try to use a quarter of available threads
      max_threads_in_block = std::min(
          ceilDiv(n_elems, target_blocks * target_unroll),
          ceilDiv(device_max_threads_per_multiprocessor, (int64_t)4));
    }
  }

  // Round up to nearest warp.
  if (max_threads_in_block % warp_size != 0) {
    max_threads_in_block += warp_size - max_threads_in_block % warp_size;
    max_threads_in_block =
        std::min(max_threads_in_block, (int64_t)dev_prop->maxThreadsPerBlock);
  }
  // Compute maximum number of reductions we could do in the same kernel based
  // on persistent buffer size. Bounded by the wave count for utilization of
  // SMs.
  const int64_t max_multi_reduction_factor = std::min(
      scheduler_utils::safeDiv(
          scheduler_utils::register_file_size_bit,
          properties.max_persistent_buffer_size_bit),
      ceilDiv(properties.total_iteration_numel, device_multiprocessor_count));
  // To get to target threads:
  // Prioritize
  // (1) x dim in reduction
  // (2) unrolling in reduction
  // (3) y in output
  // To get target blocks:
  // Prioritize
  // (1) x dim in multiple outputs
  // (2) y dim in multiple reductions

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
      properties.vectorize_factor > 1 ? properties.vectorize_factor : 1;

  // Grab what we can out of reduction domain, but don't go over a warp size yet
  bdimx = std::min(
      std::max(
          ceilDiv(
              properties.inner_most_dimension_numel,
              inner_reduction_unroll_factor),
          (int64_t)warp_size),
      max_threads_in_block);

  // If we're not just barely covering the dimension, round to a more friendly
  // number
  if (bdimx * inner_reduction_unroll_factor !=
      properties.inner_most_dimension_numel) {
    bdimx = bdimx > warp_size ? bdimx - bdimx % warp_size
                              : scheduler_utils::lastPow2(bdimx);

    // Round bdimx down to multiple of warp size or power 2
    if (bdimx < warp_size) {
      bdimx = scheduler_utils::lastPow2(bdimx);
    } else {
      bdimx = bdimx - bdimx % warp_size;
    }
  }

  // Put everything else in bdimy for now
  bdimy = std::min(
      scheduler_utils::safeDiv(warp_size, bdimx), max_multi_reduction_factor);
  // If 3D fill the rest of the threads into bdimz
  bdimz = std::min(
      std::min(
          scheduler_utils::safeDiv(max_threads_in_block, bdimx * bdimy),
          outer_reduction_numel),
      scheduler_utils::z_block_limit);

  bool vectorize = false;

  // Move unrolling factor into vectorization upto vectorization limit.
  if (properties.vectorize_factor > 1 && inner_reduction_unroll_factor > 1) {
    vectorize = true;
    inner_reduction_unroll_factor = std::min(
        scheduler_utils::lastPow2(inner_reduction_unroll_factor),
        properties.vectorize_factor);
  }

  // calculate the maximum persistent buffer size
  const int64_t buffer_bits_per_batch =
      properties.max_persistent_buffer_size_bit /
      properties.total_reduction_numel * inner_reduction_unroll_factor;
  const int64_t batches_per_block_inner_reduction_max = getMaxPersistentBatch(
      buffer_bits_per_batch,
      target_warps_per_sm * dev_prop->warpSize,
      register_overhead);

  // start from small block size to minimize expensive inter-threads reduction
  const int64_t threads_after_vectorize =
      properties.inner_most_dimension_numel / inner_reduction_unroll_factor;

  // Test min_threads_per_block using 3 values:
  // (1) One warp, so we can use single warp reduction and sync.
  // (2) Two warps, so we can achieve 100% occupancy since most GPUs allow 32
  //     blocks per SM.
  // (3) Four warps, number recommended by the cuda-c-best-practices-guide.
  const int64_t min_threads_per_block = min_threads_override > 0
      ? min_threads_override
      : 4l * dev_prop->warpSize;

  // start bdimx with min_threads_per_block then increase if we have too many
  // persistent buffer batches per block
  if (outer_reduction_numel == 1 && vectorize) {
    bdimx = std::min(min_threads_per_block, threads_after_vectorize);
  }

  // If we don't have enough threads, let's do multiple reductions per block.
  // Multiple reductions per block shows better performance than unroll
  // iterations. Still keep vectorization as it is important for performance
  // since V100.
  if (bdimx * bdimy * bdimz < min_threads_per_block) {
    bdimy = std::min(
        scheduler_utils::safeDiv(min_threads_per_block, bdimx * bdimz),
        max_multi_reduction_factor);
  }

  // Set size of persistent per thread buffer on inner reduction buffer
  // if too large, will be reduced later to reduce register usage
  int64_t batches_per_block_inner_reduction = ceilDiv(
      properties.inner_most_dimension_numel,
      bdimx * inner_reduction_unroll_factor);
  const int64_t pre_outer_threads_z = bdimz;

  // Attempt to put some unrolling into the outer reduction if inner hasn't
  // taken the max unrolling
  if (inner_reduction_unroll_factor < max_unroll) {
    outer_reduction_unroll_factor = std::min(
        std::min(
            ceilDiv(max_unroll, inner_reduction_unroll_factor),
            ceilDiv(outer_reduction_numel, bdimz)),
        outer_unroll_cap);
  }

  godim = ceilDiv(properties.total_iteration_numel, bdimy);

  // Prefer putting iterations into unrolling over having a very large
  // persistent buffer.
  while (!vectorize && inner_reduction_unroll_factor < max_unroll &&
         batches_per_block_inner_reduction >= 2) {
    inner_reduction_unroll_factor *= 2;
    batches_per_block_inner_reduction = scheduler_utils::roundUpPow2Or8(ceilDiv(
        properties.inner_most_dimension_numel,
        bdimx * inner_reduction_unroll_factor));
  }

  // Set size of persistent per thread buffer on outer reduction buffer
  int64_t batches_per_block_outer_reduction =
      scheduler_utils::roundUpPow2Or8(ceilDiv(
          ceilDiv(
              properties.total_reduction_numel,
              properties.inner_most_dimension_numel),
          bdimz * outer_reduction_unroll_factor));

  // Prefer putting iterations into unrolling over having a very large
  // persistent buffer.
  while (outer_reduction_unroll_factor < max_unroll &&
         outer_reduction_unroll_factor <= outer_unroll_cap / 2 &&
         batches_per_block_outer_reduction >= 2) {
    outer_reduction_unroll_factor *= 2;
    batches_per_block_outer_reduction = scheduler_utils::roundUpPow2Or8(
        ceilDiv(outer_reduction_numel, bdimz * outer_reduction_unroll_factor));
  }

  // Adjust bdimx based on batches_per_block and unroll factor set as they could
  // have moved a bit since they're the free variables, not the buffers
  bdimx = ceilDiv(
      properties.inner_most_dimension_numel,
      inner_reduction_unroll_factor * batches_per_block_inner_reduction);
  bdimz = ceilDiv(
      outer_reduction_numel,
      outer_reduction_unroll_factor * batches_per_block_outer_reduction);

  // Try moving persistent buffer factors into threads until we have too many
  // threads.

  while (
      // If block size can be doubled
      bdimx * bdimy * bdimz * 2 <= max_threads_in_block &&
      // And batches_per_block_inner_reduction can be divided by two
      (batches_per_block_inner_reduction >
           batches_per_block_inner_reduction_max ||
       batches_per_block_outer_reduction >= 2)) {
    // Try to decrease per thread register allocation persistence size on inner
    // reduction by double bdimx.
    if (batches_per_block_inner_reduction >
        batches_per_block_inner_reduction_max) {
      bdimx *= 2;
      batches_per_block_inner_reduction = ceilDiv(
          properties.inner_most_dimension_numel,
          inner_reduction_unroll_factor * bdimx);
      continue;
    }

    // Try to decrease per thread register allocation persistence size on outer
    // reduction
    if (batches_per_block_outer_reduction >= 2 &&
        batches_per_block_outer_reduction !=
            scheduler_utils::roundUpPow2Or8(
                batches_per_block_outer_reduction / 2) &&
        bdimz * 2 <= scheduler_utils::z_block_limit) {
      batches_per_block_outer_reduction = scheduler_utils::roundUpPow2Or8(
          batches_per_block_outer_reduction / 2);
      bdimz = ceilDiv(
          outer_reduction_numel,
          batches_per_block_outer_reduction * outer_reduction_unroll_factor);
      continue;
    }
    break;
  }

  // Register pressure is really high per thread, which could lead to local
  // memory leaks, if using less than maximum threads, decrease batches per
  // block by a factor of 2
  if (batches_per_block_outer_reduction * batches_per_block_inner_reduction *
              inner_reduction_unroll_factor * outer_reduction_unroll_factor *
              4l >
          scheduler_utils::max_registers_per_thread * 3l &&
      bdimx * bdimy * bdimz * 2l <= max_threads_in_block &&
      batches_per_block_inner_reduction >
          batches_per_block_inner_reduction_max) {
    batches_per_block_inner_reduction = batches_per_block_inner_reduction / 2;
  }

  // Do the same on the outer reduction dimension
  if (batches_per_block_outer_reduction * batches_per_block_inner_reduction *
              inner_reduction_unroll_factor * outer_reduction_unroll_factor *
              4l >
          scheduler_utils::max_registers_per_thread * 3l &&
      bdimx * bdimy * bdimz * 2l <= device_max_threads_per_multiprocessor &&
      batches_per_block_outer_reduction >= 2l) {
    batches_per_block_outer_reduction /= 2l;
  }

  auto device_warp_size = (int64_t)at::cuda::warp_size();
  auto padded_bdimx = bdimx % device_warp_size == 0
      ? bdimx
      : bdimx + (device_warp_size - bdimx % device_warp_size);

  bool pad_bdimx = bdimx > 16 &&
      padded_bdimx * bdimy * bdimz <= (int64_t)dev_prop->maxThreadsPerBlock;

  // estimate register usage and occupancy raito.
  // If occupancy raito is less than a preset occupancy_ratio, reduce register
  // usage register per thread is estimated as overhead + buffer_size_bit /
  // bits_per_register
  int64_t nvrtc_register_per_thread = scheduler_utils::max_registers_per_thread;
  const int64_t blocksPerKernel = godim;
  // register estimation is only valid for vectorized gmem access
  // we've seen unexpectedly high register counts with vectorization factor less
  // than 4, which would make the below estimate inaccurate.
  // TODO: support the non vectorized case. consider shmem.
  // only need to balance register and occupancy ratio if there are enough
  // blocks and buffers
  if (vectorize && blocksPerKernel > device_multiprocessor_count &&
      batches_per_block_inner_reduction > 1) {
    // Estimate register per thread based on buffer size, since inner reduction
    // dim is fully parallelized, the buffer size of each element equals the
    // total buffer size divide by inner_most_dimension_numel. Each thread will
    // hold batches_per_block_inner_reduction * inner_reduction_unroll_factor
    // elements.
    const int64_t persistent_buffer_size_bit =
        properties.max_persistent_buffer_size_bit /
        properties.inner_most_dimension_numel *
        batches_per_block_inner_reduction * inner_reduction_unroll_factor;
    const int64_t threads_per_block =
        pad_bdimx ? padded_bdimx * bdimy * bdimz : bdimx * bdimy * bdimz;

    // Calculate the max register count each thread can use.
    nvrtc_register_per_thread = getMaxRegisterCountPerThreadAndOccupancy(
                                    persistent_buffer_size_bit,
                                    threads_per_block,
                                    target_warps_per_sm,
                                    register_overhead)
                                    .first;
  }

  // Will be used once supporting inter-block persistence
  int64_t gdimx = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimy = LaunchParams::UNINITIALIZED_VAL;
  int64_t gdimz = LaunchParams::UNINITIALIZED_VAL;

  rparams->cparams.maxrregcount = nvrtc_register_per_thread;

  // Inner reduction domain
  rparams->cross_block_inner_reduction = true;
  rparams->block_dim_inner_reduction = ParallelType::TIDx;
  rparams->pad_inner_reduction_to_warp = pad_bdimx;
  rparams->batches_per_block_inner_reduction =
      batches_per_block_inner_reduction;

  // For persistent schedules always have to mark the reduction unrolled
  // otherwise rfactor can fail
  rparams->unroll_factor_inner_reduction = inner_reduction_unroll_factor;
  rparams->vectorize_inner_reduction = vectorize;

  // Iter domain
  rparams->multiple_reds_per_blk = bdimy > 1;
  if (rparams->multiple_reds_per_blk) {
    rparams->block_dim_iter_dom = ParallelType::TIDy;
  }

  if (godim > 1) {
    rparams->grid_dim_iter_dom = ParallelType::BIDx;
    if (godim > scheduler_utils::x_grid_limit) {
      rparams->split_grid_dim_iter_dom_outer = true;
      gdimx = scheduler_utils::x_grid_limit;
    }
  }

  if (iter_unroll_factor > 1) {
    rparams->unroll_factor_iter_dom = iter_unroll_factor;
  }

  // Outer reduction domain
  rparams->schedule_3d =
      properties.total_reduction_numel != properties.inner_most_dimension_numel;
  if (rparams->schedule_3d) {
    rparams->batches_per_block_outer_reduction =
        batches_per_block_outer_reduction;
    rparams->block_dim_outer_reduction = ParallelType::TIDz;
    rparams->cross_block_outer_reduction = true;
    rparams->unroll_factor_outer_reduction = outer_reduction_unroll_factor;
  }

  rparams->lparams = LaunchParams(
      gdimx,
      gdimy,
      gdimz,
      LaunchParams::UNINITIALIZED_VAL,
      bdimy,
      LaunchParams::UNINITIALIZED_VAL);

  if (resources != nullptr) {
    *resources = {
        pad_bdimx ? padded_bdimx : bdimx,
        bdimy,
        bdimz,
        (pad_bdimx ? padded_bdimx : bdimx) * bdimy * bdimz,
        nvrtc_register_per_thread,
        godim,
        pre_outer_threads_z,
        target_unroll,
        target_iterations};
  }
}

bool noLargerPersistentAxis(
    int64_t candidate_batch,
    int64_t candidate_unroll,
    int64_t native_batch,
    int64_t native_unroll) {
  if (candidate_batch <= 0 || candidate_unroll <= 0 || native_batch <= 0 ||
      native_unroll <= 0 ||
      candidate_batch >
          std::numeric_limits<int64_t>::max() / candidate_unroll ||
      native_batch > std::numeric_limits<int64_t>::max() / native_unroll) {
    return false;
  }
  return candidate_batch * candidate_unroll <= native_batch * native_unroll;
}

bool isNativeValidInner3DAlternative(
    const PersistentKernelProperties& properties,
    const ReductionParams& candidate,
    const ReductionParams& native,
    const Inner3DPlanResources& resources,
    const cudaDeviceProp& device) {
  // The original four-wave plan remains available. A different wave target is
  // offered only when its *completed* CTA, register cap and grid all fit native
  // hardware limits without relying on a later grid split. Native admission
  // already bounds total persistent bits by the register-file budget. Requiring
  // no larger per-thread persistent extents on either reduction axis preserves
  // that capacity for every underlying buffer shape, including broadcasts.
  return properties.max_persistent_buffer_size_bit > 0 &&
      properties.max_persistent_buffer_size_bit <=
          scheduler_utils::register_file_size_bit &&
      candidate.schedule_3d && candidate.persistent_kernel &&
      candidate.cross_block_inner_reduction &&
      candidate.cross_block_outer_reduction &&
      !candidate.cross_grid_inner_reduction &&
      !candidate.cross_grid_outer_reduction &&
      candidate.block_dim_inner_reduction == ParallelType::TIDx &&
      candidate.block_dim_outer_reduction == ParallelType::TIDz &&
      noLargerPersistentAxis(
          candidate.batches_per_block_inner_reduction,
          candidate.unroll_factor_inner_reduction,
          native.batches_per_block_inner_reduction,
          native.unroll_factor_inner_reduction) &&
      noLargerPersistentAxis(
          candidate.batches_per_block_outer_reduction,
          candidate.unroll_factor_outer_reduction,
          native.batches_per_block_outer_reduction,
          native.unroll_factor_outer_reduction) &&
      resources.padded_threads_x > 0 && resources.threads_y > 0 &&
      resources.threads_z > 0 &&
      resources.padded_threads_x <= device.maxThreadsDim[0] &&
      resources.threads_y <= device.maxThreadsDim[1] &&
      resources.threads_z <=
          std::min<int64_t>(scheduler_utils::z_block_limit, device.maxThreadsDim[2]) &&
      resources.padded_threads_per_block > 0 &&
      resources.padded_threads_per_block <= device.maxThreadsPerBlock &&
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <= scheduler_utils::max_registers_per_thread &&
      resources.registers_per_thread <=
          device.regsPerBlock / resources.padded_threads_per_block &&
      resources.grid_blocks > 0 &&
      resources.grid_blocks <=
          std::min<int64_t>(scheduler_utils::x_grid_limit, device.maxGridSize[0]);
}

// A failed *conservative* menu preflight does not invalidate the original
// scheduler plan. Record its actual native selections on both traced modes,
// then leave that plan unchanged without asking the API about unproved plans.
void recordInner3DNativeOnlyTail(
    const PersistentKernelProperties& properties,
    const ReductionParams& native,
    const cudaDeviceProp& device,
    std::string_view reason) {
  constexpr std::string_view source = "native_3d_preflight_fallback";
  scheduler::jev::recordNoChoiceDecision(
      "IP6.work_split", "unroll_first", reason, source);
  scheduler::jev::recordNoChoiceDecision(
      "IP6.outer_unroll",
      "u" + std::to_string(native.unroll_factor_outer_reduction),
      reason,
      source);
  scheduler::jev::recordNoChoiceDecision(
      "IP6.reg_overhead",
      properties.has_exp_op ? "r32" : "r16",
      reason,
      source);
  scheduler::jev::recordNoChoiceDecision(
      "IP6.cta_floor",
      "floor" + std::to_string(4 * device.warpSize),
      reason,
      source);
}

void recordInner3DNativeOnlyDecisions(
    const PersistentKernelProperties& properties,
    const ReductionParams& native,
    const cudaDeviceProp& device,
    std::string_view reason) {
  constexpr std::string_view source = "native_3d_preflight_fallback";
  scheduler::jev::recordNoChoiceDecision(
      "IP6.width",
      "w" + std::to_string(properties.vectorize_factor),
      reason,
      source);
  scheduler::jev::recordNoChoiceDecision(
      "IP6.waves", "waves4", reason, source);
  scheduler::jev::recordNoChoiceDecision(
      "IP6.occupancy", "occ28", reason, source);
  scheduler::jev::recordNoChoiceDecision(
      "IP6.warp",
      "warp" + std::to_string(getInner3DNativeWarpTarget(properties, device)),
      reason,
      source);
  recordInner3DNativeOnlyTail(properties, native, device, reason);
}

PersistentKernelProperties chooseInner3DVectorWidth(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    const cudaDeviceProp& device) {
  struct Plan {
    int64_t width;
    ReductionParams params;
    Inner3DPlanResources resources;
  };
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::string plan_state;
  const int64_t layout_max = properties.vectorize_factor;
  if (layout_max <= 0 || (layout_max & (layout_max - 1)) != 0 ||
      properties.inner_most_dimension_numel % layout_max != 0) {
    scheduler::jev::recordNoChoiceDecision(
        "IP6.width",
        "w" + std::to_string(layout_max),
        "native vector width is not a divisible power of two",
        "native_vector_width");
    return properties;
  }

  // The source analysis already found the maximum safe alignment width. Try
  // only divisors of it, keeping the original four-wave policy fixed during
  // preflight. This makes every menu member a complete native-valid plan and
  // avoids nested API calls for candidates that are never selected.
  for (int64_t width = layout_max; width >= 1; width /= 2) {
    auto candidate_properties = properties;
    candidate_properties.vectorize_factor = width;
    auto candidate = base;
    Inner3DPlanResources resources;
    innerPersistentHeuristic3DPlan(
        candidate_properties, 4, &candidate, &resources);
    plan_state += "\nplan" + std::to_string(width) + "=" +
        std::to_string(candidate.batches_per_block_inner_reduction) + "," +
        std::to_string(candidate.unroll_factor_inner_reduction) + "," +
        std::to_string(candidate.batches_per_block_outer_reduction) + "," +
        std::to_string(candidate.unroll_factor_outer_reduction) + "," +
        std::to_string(resources.padded_threads_x) + "," +
        std::to_string(resources.threads_y) + "," +
        std::to_string(resources.threads_z) + "," +
        std::to_string(resources.padded_threads_per_block) + "," +
        std::to_string(resources.registers_per_thread) + "," +
        std::to_string(resources.grid_blocks) + "," +
        std::to_string(candidate.schedule_3d) + "," +
        std::to_string(candidate.persistent_kernel) + "," +
        std::to_string(candidate.cross_block_inner_reduction) + "," +
        std::to_string(candidate.cross_block_outer_reduction) + "," +
        std::to_string(candidate.cross_grid_inner_reduction) + "," +
        std::to_string(candidate.cross_grid_outer_reduction) + "," +
        std::to_string(candidate.block_dim_inner_reduction == ParallelType::TIDx) +
        "," +
        std::to_string(candidate.block_dim_outer_reduction == ParallelType::TIDz);
    if (width == layout_max || isNativeValidInner3DAlternative(
                                   properties,
                                   candidate,
                                   plans.front().params,
                                   resources,
                                   device)) {
      options.push_back({
          "w" + std::to_string(width),
          "Vectorize " + std::to_string(width) +
              " inner element(s) per thread; completed CTA=" +
              std::to_string(resources.padded_threads_per_block) +
              ", register cap=" +
              std::to_string(resources.registers_per_thread) +
              ", grid blocks=" + std::to_string(resources.grid_blocks),
          inner3DLocalPlanConsequence(candidate, resources)});
    }
    plans.push_back({width, std::move(candidate), resources});
  }

  const std::string state =
      "inner_persistent.3d_vector_width.v1\nreduction_numel=" +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nlayout_max=" + std::to_string(layout_max) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_threads_x=" + std::to_string(device.maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device.maxThreadsDim[1]) +
      "\nmax_threads_z=" + std::to_string(device.maxThreadsDim[2]) +
      "\nmax_threads_per_block=" +
      std::to_string(device.maxThreadsPerBlock) +
      "\nregisters_per_block=" + std::to_string(device.regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device.maxGridSize[0]) + plan_state;
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record
      ? "w" + std::to_string(layout_max)
      : "";
  std::string selected;
  if (options.size() == 1) {
    NVF_ERROR(
        options.front().id == "w" + std::to_string(layout_max),
        "The only hard-valid inner 3D width must be native");
    selected = options.front().id;
    scheduler::jev::recordHardSingletonDecision(
        "IP6.width",
        selected,
        "only one hard-valid vector width",
        "hard_legal_singleton");
  } else {
    selected = scheduler::jev::chooseDecision(
        "IP6.width", state, options, native_choice);
  }
  const auto option_it = std::find_if(
      options.begin(),
      options.end(),
      [&](const auto& option) { return option.id == selected; });
  NVF_ERROR(option_it != options.end(), "Unknown inner 3D vector width");
  const int64_t width = std::stoll(selected.substr(1));
  auto selected_properties = properties;
  selected_properties.vectorize_factor = width;
  return selected_properties;
}

struct Inner3DChosenOccupancy {
  int64_t target;
  std::unique_ptr<ReductionParams> params;
  Inner3DPlanResources resources;
};

Inner3DChosenOccupancy chooseInner3DOccupancyTarget(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t wave_cap,
    std::unique_ptr<ReductionParams> native,
    const Inner3DPlanResources& native_resources,
    const cudaDeviceProp& device,
    int64_t layout_max) {
  struct Plan {
    int64_t target;
    std::unique_ptr<ReductionParams> params;
    Inner3DPlanResources resources;
  };
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::string plan_state;
  for (const int64_t target : {28, 16, 32}) {
    auto params = target == 28 ? std::move(native)
                               : std::make_unique<ReductionParams>(base);
    Inner3DPlanResources resources = native_resources;
    if (target != 28) {
      innerPersistentHeuristic3DPlan(
          properties, wave_cap, params.get(), &resources, -1, target);
    }
    // Send measurements for *all* plans. The service independently applies
    // the native hard-resource predicate to recover the exact legal menu.
    // No native winner is present in the Remote request.
    plan_state += "\nplan" + std::to_string(target) + "=" +
        std::to_string(params->batches_per_block_inner_reduction) + "," +
        std::to_string(params->unroll_factor_inner_reduction) + "," +
        std::to_string(params->batches_per_block_outer_reduction) + "," +
        std::to_string(params->unroll_factor_outer_reduction) + "," +
        std::to_string(resources.padded_threads_x) + "," +
        std::to_string(resources.threads_y) + "," +
        std::to_string(resources.threads_z) + "," +
        std::to_string(resources.padded_threads_per_block) + "," +
        std::to_string(resources.registers_per_thread) + "," +
        std::to_string(resources.grid_blocks) + "," +
        std::to_string(params->schedule_3d) + "," +
        std::to_string(params->persistent_kernel) + "," +
        std::to_string(params->cross_block_inner_reduction) + "," +
        std::to_string(params->cross_block_outer_reduction) + "," +
        std::to_string(params->cross_grid_inner_reduction) + "," +
        std::to_string(params->cross_grid_outer_reduction) + "," +
        std::to_string(
            params->block_dim_inner_reduction == ParallelType::TIDx) + "," +
        std::to_string(
            params->block_dim_outer_reduction == ParallelType::TIDz);
    if (target == 28 || isNativeValidInner3DAlternative(
                            properties,
                            *params,
                            *plans.front().params,
                            resources,
                            device)) {
      options.push_back({
          "occ" + std::to_string(target),
          "Target " + std::to_string(target) +
              " resident warps per SM; completed CTA=" +
              std::to_string(resources.padded_threads_per_block) +
              ", register cap=" +
              std::to_string(resources.registers_per_thread) +
              ", grid blocks=" + std::to_string(resources.grid_blocks),
          inner3DLocalPlanConsequence(*params, resources)});
    }
    plans.push_back({target, std::move(params), resources});
  }

  const std::string state =
      (properties.vectorize_factor == layout_max
           ? "inner_persistent.3d_occupancy_target.v1\nreduction_numel="
           : "inner_persistent.3d_occupancy_target.v2\nreduction_numel=") +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nselected_wave_cap=" + std::to_string(wave_cap) +
      (properties.vectorize_factor == layout_max
           ? std::string()
           : "\nselected_vector_width=" +
                 std::to_string(properties.vectorize_factor)) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_threads_x=" + std::to_string(device.maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device.maxThreadsDim[1]) +
      "\nmax_threads_z=" + std::to_string(device.maxThreadsDim[2]) +
      "\nmax_threads_per_block=" +
      std::to_string(device.maxThreadsPerBlock) +
      "\nregisters_per_block=" + std::to_string(device.regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device.maxGridSize[0]) + plan_state;
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record ? "occ28" : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP6.occupancy", state, options, native_choice);
  const auto option_it = std::find_if(
      options.begin(),
      options.end(),
      [&](const auto& option) { return option.id == selected; });
  NVF_ERROR(option_it != options.end(), "Unknown inner 3D occupancy target");
  const auto plan_it = std::find_if(
      plans.begin(),
      plans.end(),
      [&](const auto& plan) {
        return "occ" + std::to_string(plan.target) == selected;
      });
  NVF_ERROR(plan_it != plans.end(), "Missing inner 3D occupancy plan");
  return {plan_it->target, std::move(plan_it->params), plan_it->resources};
}

std::unique_ptr<ReductionParams> chooseInner3DCtaFloor(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t wave_cap,
    int64_t warp_target,
    int64_t occupancy_target,
    int64_t outer_unroll_cap,
    int64_t register_overhead,
    bool unroll_first,
    std::unique_ptr<ReductionParams> native,
    const Inner3DPlanResources& native_resources,
    const cudaDeviceProp& device) {
  const int64_t native_floor = 4 * device.warpSize;
  if (!isNativeValidInner3DAlternative(
          properties, *native, *native, native_resources, device)) {
    scheduler::jev::recordNoChoiceDecision(
        "IP6.cta_floor",
        "floor" + std::to_string(native_floor),
        "native 3D CTA floor failed complete resource preflight",
        "native_3d_preflight_fallback");
    return native;
  }

  // This 1/2/4-warp floor is an empirical performance policy. Recompute the
  // final plan after all earlier 3D choices and keep correctness, capacity and
  // launch checks in C++. Its 18 measured fields also let the local service
  // reconstruct the same complete-plan menu without a native winner hint.
  auto complete_plan = [](const ReductionParams& params,
                          const Inner3DPlanResources& resources) {
    return std::to_string(params.batches_per_block_inner_reduction) + "," +
        std::to_string(params.unroll_factor_inner_reduction) + "," +
        std::to_string(params.batches_per_block_outer_reduction) + "," +
        std::to_string(params.unroll_factor_outer_reduction) + "," +
        std::to_string(resources.padded_threads_x) + "," +
        std::to_string(resources.threads_y) + "," +
        std::to_string(resources.threads_z) + "," +
        std::to_string(resources.padded_threads_per_block) + "," +
        std::to_string(resources.registers_per_thread) + "," +
        std::to_string(resources.grid_blocks) + "," +
        std::to_string(params.schedule_3d) + "," +
        std::to_string(params.persistent_kernel) + "," +
        std::to_string(params.cross_block_inner_reduction) + "," +
        std::to_string(params.cross_block_outer_reduction) + "," +
        std::to_string(params.cross_grid_inner_reduction) + "," +
        std::to_string(params.cross_grid_outer_reduction) + "," +
        std::to_string(params.block_dim_inner_reduction == ParallelType::TIDx) +
        "," +
        std::to_string(params.block_dim_outer_reduction == ParallelType::TIDz);
  };
  struct Plan {
    int64_t floor;
    std::unique_ptr<ReductionParams> params;
  };
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::string plan_state;
  const std::string native_measurements =
      complete_plan(*native, native_resources);
  std::vector<std::string> offered_measurements = {native_measurements};
  plan_state += "\nplan" + std::to_string(native_floor) + "=" +
      native_measurements;
  options.push_back({
      "floor" + std::to_string(native_floor),
      "Set planner CTA floor to " + std::to_string(native_floor) +
          " threads; completed CTA=" +
          std::to_string(native_resources.padded_threads_per_block) +
          ", register cap=" +
          std::to_string(native_resources.registers_per_thread) +
          ", grid blocks=" + std::to_string(native_resources.grid_blocks)});
  plans.push_back({native_floor, std::move(native)});

  for (const int64_t floor : {2 * device.warpSize, device.warpSize}) {
    auto params = std::make_unique<ReductionParams>(base);
    Inner3DPlanResources resources;
    try {
      innerPersistentHeuristic3DPlan(
          properties,
          wave_cap,
          params.get(),
          &resources,
          warp_target,
          occupancy_target,
          outer_unroll_cap,
          register_overhead,
          unroll_first,
          floor);
    } catch (const std::exception&) {
      scheduler::jev::recordNoChoiceDecision(
          "IP6.cta_floor",
          "floor" + std::to_string(native_floor),
          "alternate 3D CTA floor failed planning",
          "native_cta_floor");
      return std::move(plans.front().params);
    }
    const std::string measurements = complete_plan(*params, resources);
    plan_state += "\nplan" + std::to_string(floor) + "=" + measurements;
    if (std::find(
            offered_measurements.begin(),
            offered_measurements.end(),
            measurements) == offered_measurements.end() &&
        isNativeValidInner3DAlternative(
            properties,
            *params,
            *plans.front().params,
            resources,
            device)) {
      options.push_back({
          "floor" + std::to_string(floor),
          "Set planner CTA floor to " + std::to_string(floor) +
              " threads; completed CTA=" +
              std::to_string(resources.padded_threads_per_block) +
              ", register cap=" +
              std::to_string(resources.registers_per_thread) +
              ", grid blocks=" + std::to_string(resources.grid_blocks)});
      offered_measurements.push_back(measurements);
      plans.push_back({floor, std::move(params)});
    }
  }
  if (options.size() == 1) {
    scheduler::jev::recordNoChoiceDecision(
        "IP6.cta_floor",
        "floor" + std::to_string(native_floor),
        "no distinct, resource-valid 3D CTA floor",
        "native_cta_floor");
    return std::move(plans.front().params);
  }

  const std::string state =
      "inner_persistent.3d_cta_floor.v1\nreduction_numel=" +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nselected_vector_width=" +
      std::to_string(properties.vectorize_factor) +
      "\nselected_wave_cap=" + std::to_string(wave_cap) +
      "\nselected_occupancy_target=" +
      std::to_string(occupancy_target) +
      "\nselected_warp_target=" + std::to_string(warp_target) +
      "\nselected_outer_unroll=" + std::to_string(outer_unroll_cap) +
      "\nselected_register_overhead=" +
      std::to_string(register_overhead) +
      "\nselected_unroll_first=" + std::to_string(unroll_first) +
      "\nwarp_size=" + std::to_string(device.warpSize) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_threads_x=" + std::to_string(device.maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device.maxThreadsDim[1]) +
      "\nmax_threads_z=" + std::to_string(device.maxThreadsDim[2]) +
      "\nmax_threads_per_block=" +
      std::to_string(device.maxThreadsPerBlock) +
      "\nregisters_per_block=" + std::to_string(device.regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device.maxGridSize[0]) + plan_state;
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record
      ? "floor" + std::to_string(native_floor)
      : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP6.cta_floor", state, options, native_choice);
  const auto plan_it = std::find_if(
      plans.begin(),
      plans.end(),
      [&](const auto& plan) {
        return "floor" + std::to_string(plan.floor) == selected;
      });
  NVF_ERROR(plan_it != plans.end(), "Unknown inner 3D CTA floor");
  return std::move(plan_it->params);
}

std::unique_ptr<ReductionParams> chooseInner3DRegisterOverhead(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t wave_cap,
    int64_t warp_target,
    int64_t occupancy_target,
    int64_t outer_unroll_cap,
    std::unique_ptr<ReductionParams> native,
    const Inner3DPlanResources& native_resources,
    const cudaDeviceProp& device,
    bool unroll_first) {
  const int64_t native_overhead = properties.has_exp_op ? 32 : 16;
  const int64_t alternate_overhead = native_overhead + 8;
  struct Plan {
    int64_t overhead;
    std::unique_ptr<ReductionParams> params;
    Inner3DPlanResources resources;
  };
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::string plan_state;
  auto add_plan = [&](int64_t overhead,
                      std::unique_ptr<ReductionParams> params,
                      const Inner3DPlanResources& resources,
                      bool eligible) {
    plan_state += "\nplan" + std::to_string(overhead) + "=" +
        std::to_string(params->batches_per_block_inner_reduction) + "," +
        std::to_string(params->unroll_factor_inner_reduction) + "," +
        std::to_string(params->batches_per_block_outer_reduction) + "," +
        std::to_string(params->unroll_factor_outer_reduction) + "," +
        std::to_string(resources.padded_threads_x) + "," +
        std::to_string(resources.threads_y) + "," +
        std::to_string(resources.threads_z) + "," +
        std::to_string(resources.padded_threads_per_block) + "," +
        std::to_string(resources.registers_per_thread) + "," +
        std::to_string(resources.grid_blocks) + "," +
        std::to_string(params->schedule_3d) + "," +
        std::to_string(params->persistent_kernel) + "," +
        std::to_string(params->cross_block_inner_reduction) + "," +
        std::to_string(params->cross_block_outer_reduction) + "," +
        std::to_string(params->cross_grid_inner_reduction) + "," +
        std::to_string(params->cross_grid_outer_reduction) + "," +
        std::to_string(params->block_dim_inner_reduction == ParallelType::TIDx) +
        "," +
        std::to_string(params->block_dim_outer_reduction == ParallelType::TIDz);
    if (eligible) {
      options.push_back({
          "r" + std::to_string(overhead),
          "Reserve " + std::to_string(overhead) +
              " nonbuffer registers per thread; completed CTA=" +
              std::to_string(resources.padded_threads_per_block) +
              ", register cap=" +
              std::to_string(resources.registers_per_thread) +
              ", grid blocks=" + std::to_string(resources.grid_blocks)});
    }
    plans.push_back({overhead, std::move(params), resources});
  };
  add_plan(native_overhead, std::move(native), native_resources, true);

  // The 16/32-register allowance is empirical. Only offer a larger allowance
  // when the native vectorized multi-block plan actually uses the register
  // estimate and its buffer plus allowance fits the per-thread hardware cap.
  const int64_t inner_batch =
      plans.front().params->batches_per_block_inner_reduction;
  const int64_t inner_unroll =
      plans.front().params->unroll_factor_inner_reduction;
  const int64_t buffer_bits_per_element =
      properties.max_persistent_buffer_size_bit /
      properties.inner_most_dimension_numel;
  const int64_t max_buffer_bits =
      (scheduler_utils::max_registers_per_thread - alternate_overhead) *
      scheduler_utils::bits_per_register;
  if (properties.vectorize_factor > 1 &&
      native_resources.grid_blocks > device.multiProcessorCount &&
      inner_batch > 1 &&
      inner_unroll > 0 &&
      inner_batch <= std::numeric_limits<int64_t>::max() / inner_unroll &&
      buffer_bits_per_element <=
          max_buffer_bits / (inner_batch * inner_unroll)) {
    auto params = std::make_unique<ReductionParams>(base);
    Inner3DPlanResources resources;
    innerPersistentHeuristic3DPlan(
        properties,
        wave_cap,
        params.get(),
        &resources,
        warp_target,
        occupancy_target,
        outer_unroll_cap,
        alternate_overhead,
        unroll_first);
    const bool eligible =
        resources.registers_per_thread >
            native_resources.registers_per_thread &&
        isNativeValidInner3DAlternative(
            properties,
            *params,
            *plans.front().params,
            resources,
            device);
    add_plan(alternate_overhead, std::move(params), resources, eligible);
  }

  const std::string state =
      "inner_persistent.3d_register_overhead.v1\nreduction_numel=" +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nselected_vector_width=" +
      std::to_string(properties.vectorize_factor) +
      "\nselected_wave_cap=" + std::to_string(wave_cap) +
      "\nselected_occupancy_target=" +
      std::to_string(occupancy_target) +
      "\nselected_warp_target=" + std::to_string(warp_target) +
      "\nselected_outer_unroll=" + std::to_string(outer_unroll_cap) +
      "\nhas_exp_op=" + std::to_string(properties.has_exp_op) +
      "\nsm_count=" + std::to_string(device.multiProcessorCount) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_threads_x=" + std::to_string(device.maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device.maxThreadsDim[1]) +
      "\nmax_threads_z=" + std::to_string(device.maxThreadsDim[2]) +
      "\nmax_threads_per_block=" +
      std::to_string(device.maxThreadsPerBlock) +
      "\nregisters_per_block=" + std::to_string(device.regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device.maxGridSize[0]) + plan_state;
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record
      ? "r" + std::to_string(native_overhead)
      : "";
  std::string selected;
  if (options.size() == 1) {
    NVF_ERROR(
        options.front().id == "r" + std::to_string(native_overhead),
        "The only hard-valid inner 3D register allowance must be native");
    selected = options.front().id;
    scheduler::jev::recordHardSingletonDecision(
        "IP6.reg_overhead",
        selected,
        "only one hard-valid register allowance",
        "hard_legal_singleton");
  } else {
    selected = scheduler::jev::chooseDecision(
        "IP6.reg_overhead", state, options, native_choice);
  }
  const auto option_it = std::find_if(
      options.begin(),
      options.end(),
      [&](const auto& option) { return option.id == selected; });
  NVF_ERROR(option_it != options.end(), "Unknown inner 3D register allowance");
  const auto plan_it = std::find_if(
      plans.begin(),
      plans.end(),
      [&](const auto& plan) {
        return "r" + std::to_string(plan.overhead) == selected;
      });
  NVF_ERROR(plan_it != plans.end(), "Missing inner 3D register plan");
  return chooseInner3DCtaFloor(
      properties,
      base,
      wave_cap,
      warp_target,
      occupancy_target,
      outer_unroll_cap,
      plan_it->overhead,
      unroll_first,
      std::move(plan_it->params),
      plan_it->resources,
      device);
}

std::unique_ptr<ReductionParams> chooseInner3DOuterUnroll(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t wave_cap,
    int64_t warp_target,
    int64_t occupancy_target,
    std::unique_ptr<ReductionParams> native,
    const Inner3DPlanResources& native_resources,
    const cudaDeviceProp& device,
    bool unroll_first) {
  struct Plan {
    int64_t target;
    std::unique_ptr<ReductionParams> params;
    Inner3DPlanResources resources;
  };
  const int64_t native_target = native->unroll_factor_outer_reduction;
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::string plan_state;
  auto add_plan = [&](int64_t target,
                      std::unique_ptr<ReductionParams> params,
                      const Inner3DPlanResources& resources,
                      bool eligible) {
    plan_state += "\nplan" + std::to_string(target) + "=" +
        std::to_string(params->batches_per_block_inner_reduction) + "," +
        std::to_string(params->unroll_factor_inner_reduction) + "," +
        std::to_string(params->batches_per_block_outer_reduction) + "," +
        std::to_string(params->unroll_factor_outer_reduction) + "," +
        std::to_string(resources.padded_threads_x) + "," +
        std::to_string(resources.threads_y) + "," +
        std::to_string(resources.threads_z) + "," +
        std::to_string(resources.padded_threads_per_block) + "," +
        std::to_string(resources.registers_per_thread) + "," +
        std::to_string(resources.grid_blocks) + "," +
        std::to_string(params->schedule_3d) + "," +
        std::to_string(params->persistent_kernel) + "," +
        std::to_string(params->cross_block_inner_reduction) + "," +
        std::to_string(params->cross_block_outer_reduction) + "," +
        std::to_string(params->cross_grid_inner_reduction) + "," +
        std::to_string(params->cross_grid_outer_reduction) + "," +
        std::to_string(params->block_dim_inner_reduction == ParallelType::TIDx) +
        "," +
        std::to_string(params->block_dim_outer_reduction == ParallelType::TIDz);
    if (eligible) {
      options.push_back({
          "u" + std::to_string(target),
          "Unroll " + std::to_string(target) +
              " outer reduction element(s) per thread; completed CTA=" +
              std::to_string(resources.padded_threads_per_block) +
              ", register cap=" +
              std::to_string(resources.registers_per_thread) +
              ", grid blocks=" + std::to_string(resources.grid_blocks),
          inner3DLocalPlanConsequence(*params, resources)});
    }
    plans.push_back({target, std::move(params), resources});
  };
  add_plan(native_target, std::move(native), native_resources, true);

  // Caps only reduce the planner's empirical outer unroll. Recompute every
  // selected-width/wave/occupancy/warp plan and retain alternatives whose
  // complete resource footprint is no larger than the native one.
  for (int64_t target = native_target > 1
           ? scheduler_utils::lastPow2(native_target - 1)
           : 0;
       target >= 1;
       target /= 2) {
    auto params = std::make_unique<ReductionParams>(base);
    Inner3DPlanResources resources;
    innerPersistentHeuristic3DPlan(
        properties,
        wave_cap,
        params.get(),
        &resources,
        warp_target,
        occupancy_target,
        target,
        -1,
        unroll_first);
    const bool eligible =
        params->unroll_factor_outer_reduction == target &&
        isNativeValidInner3DAlternative(
            properties,
            *params,
            *plans.front().params,
            resources,
            device);
    add_plan(target, std::move(params), resources, eligible);
    if (target == 1) {
      break;
    }
  }

  const std::string state =
      "inner_persistent.3d_outer_unroll.v1\nreduction_numel=" +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nselected_vector_width=" +
      std::to_string(properties.vectorize_factor) +
      "\nselected_wave_cap=" + std::to_string(wave_cap) +
      "\nselected_occupancy_target=" + std::to_string(occupancy_target) +
      "\nselected_warp_target=" + std::to_string(warp_target) +
      "\npre_outer_threads_z=" +
      std::to_string(native_resources.pre_outer_threads_z) +
      "\nmax_dtype_bits=" + std::to_string(properties.max_dtype_size_bit) +
      "\nn_tensor_inputs=" + std::to_string(properties.n_tensor_inputs) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_threads_x=" + std::to_string(device.maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device.maxThreadsDim[1]) +
      "\nmax_threads_z=" + std::to_string(device.maxThreadsDim[2]) +
      "\nmax_threads_per_block=" +
      std::to_string(device.maxThreadsPerBlock) +
      "\nregisters_per_block=" + std::to_string(device.regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device.maxGridSize[0]) + plan_state;
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record
      ? "u" + std::to_string(native_target)
      : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP6.outer_unroll", state, options, native_choice);
  const auto option_it = std::find_if(
      options.begin(),
      options.end(),
      [&](const auto& option) { return option.id == selected; });
  NVF_ERROR(option_it != options.end(), "Unknown inner 3D outer unroll");
  const auto plan_it = std::find_if(
      plans.begin(),
      plans.end(),
      [&](const auto& plan) {
        return "u" + std::to_string(plan.target) == selected;
      });
  NVF_ERROR(plan_it != plans.end(), "Missing inner 3D outer unroll plan");
  return chooseInner3DRegisterOverhead(
      properties,
      base,
      wave_cap,
      warp_target,
      occupancy_target,
      plan_it->target,
      std::move(plan_it->params),
      plan_it->resources,
      device,
      unroll_first);
}

std::unique_ptr<ReductionParams> chooseInner3DWorkSplit(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t wave_cap,
    int64_t warp_target,
    int64_t occupancy_target,
    std::unique_ptr<ReductionParams> native,
    const Inner3DPlanResources& native_resources,
    const cudaDeviceProp& device) {
  // The target-unroll/iteration flip-flop is a performance preference. Both
  // orders must finish the complete register-backed 3D plan before selection.
  if (!isNativeValidInner3DAlternative(
          properties, *native, *native, native_resources, device)) {
    recordInner3DNativeOnlyTail(
        properties,
        *native,
        device,
        "native 3D plan failed complete resource preflight");
    return native;
  }

  auto alternate = std::make_unique<ReductionParams>(base);
  Inner3DPlanResources alternate_resources;
  try {
    innerPersistentHeuristic3DPlan(
        properties,
        wave_cap,
        alternate.get(),
        &alternate_resources,
        warp_target,
        occupancy_target,
        std::numeric_limits<int64_t>::max(),
        -1,
        false);
  } catch (const std::exception&) {
    // Failure to materialize an optional plan cannot displace the proved
    // native plan. In particular, never send an unproved singleton remotely.
    scheduler::jev::recordHardSingletonDecision(
        "IP6.work_split",
        "unroll_first",
        "alternate 3D work split failed planning",
        "native_unroll_first_branch");
    return chooseInner3DOuterUnroll(
        properties,
        base,
        wave_cap,
        warp_target,
        occupancy_target,
        std::move(native),
        native_resources,
        device,
        true);
  }

  auto complete_plan = [](const ReductionParams& params,
                          const Inner3DPlanResources& resources) {
    return std::to_string(params.batches_per_block_inner_reduction) + "," +
        std::to_string(params.unroll_factor_inner_reduction) + "," +
        std::to_string(params.batches_per_block_outer_reduction) + "," +
        std::to_string(params.unroll_factor_outer_reduction) + "," +
        std::to_string(resources.padded_threads_x) + "," +
        std::to_string(resources.threads_y) + "," +
        std::to_string(resources.threads_z) + "," +
        std::to_string(resources.padded_threads_per_block) + "," +
        std::to_string(resources.registers_per_thread) + "," +
        std::to_string(resources.grid_blocks) + "," +
        std::to_string(params.schedule_3d) + "," +
        std::to_string(params.persistent_kernel) + "," +
        std::to_string(params.cross_block_inner_reduction) + "," +
        std::to_string(params.cross_block_outer_reduction) + "," +
        std::to_string(params.cross_grid_inner_reduction) + "," +
        std::to_string(params.cross_grid_outer_reduction) + "," +
        std::to_string(params.block_dim_inner_reduction == ParallelType::TIDx) +
        "," +
        std::to_string(params.block_dim_outer_reduction == ParallelType::TIDz);
  };
  const std::string native_plan = complete_plan(*native, native_resources);
  const std::string alternate_plan =
      complete_plan(*alternate, alternate_resources);
  const bool alternate_valid =
      (native_resources.target_unroll != alternate_resources.target_unroll ||
       native_resources.target_iterations !=
           alternate_resources.target_iterations) &&
      alternate_plan != native_plan &&
      isNativeValidInner3DAlternative(
          properties,
          *alternate,
          *native,
          alternate_resources,
          device);
  if (!alternate_valid) {
    scheduler::jev::recordHardSingletonDecision(
        "IP6.work_split",
        "unroll_first",
        "no distinct, resource-valid 3D work split",
        "native_unroll_first_branch");
    return chooseInner3DOuterUnroll(
        properties,
        base,
        wave_cap,
        warp_target,
        occupancy_target,
        std::move(native),
        native_resources,
        device,
        true);
  }

  std::vector<scheduler::jev::ChoiceOption> options = {{
      "unroll_first",
      "Prioritize unrolling; completed CTA=" +
          std::to_string(native_resources.padded_threads_per_block) +
          ", register cap=" +
          std::to_string(native_resources.registers_per_thread) +
          ", grid blocks=" + std::to_string(native_resources.grid_blocks)}};
  options.push_back({
      "iter_first",
      "Prioritize serial iterations; completed CTA=" +
          std::to_string(alternate_resources.padded_threads_per_block) +
          ", register cap=" +
          std::to_string(alternate_resources.registers_per_thread) +
          ", grid blocks=" +
          std::to_string(alternate_resources.grid_blocks)});
  const std::string state =
      "inner_persistent.3d_work_split.v1\nreduction_numel=" +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nselected_vector_width=" +
      std::to_string(properties.vectorize_factor) +
      "\nselected_wave_cap=" + std::to_string(wave_cap) +
      "\nselected_occupancy_target=" +
      std::to_string(occupancy_target) +
      "\nselected_warp_target=" + std::to_string(warp_target) +
      "\nmax_dtype_bits=" + std::to_string(properties.max_dtype_size_bit) +
      "\nn_tensor_inputs=" + std::to_string(properties.n_tensor_inputs) +
      "\nsm_count=" + std::to_string(device.multiProcessorCount) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit) +
      "\nmax_threads_x=" + std::to_string(device.maxThreadsDim[0]) +
      "\nmax_threads_y=" + std::to_string(device.maxThreadsDim[1]) +
      "\nmax_threads_z=" + std::to_string(device.maxThreadsDim[2]) +
      "\nmax_threads_per_block=" +
      std::to_string(device.maxThreadsPerBlock) +
      "\nregisters_per_block=" + std::to_string(device.regsPerBlock) +
      "\nmax_grid_x=" + std::to_string(device.maxGridSize[0]) +
      "\nplan_unroll_first=" + native_plan + "," +
      std::to_string(native_resources.target_unroll) + "," +
      std::to_string(native_resources.target_iterations) +
      "\nplan_iter_first=" + alternate_plan + "," +
      std::to_string(alternate_resources.target_unroll) + "," +
      std::to_string(alternate_resources.target_iterations);
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record
      ? "unroll_first"
      : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP6.work_split", state, options, native_choice);
  const bool unroll_first = selected == "unroll_first";
  NVF_ERROR(
      unroll_first || (selected == "iter_first" && alternate_valid),
      "Unknown inner 3D work split");
  return chooseInner3DOuterUnroll(
      properties,
      base,
      wave_cap,
      warp_target,
      occupancy_target,
      unroll_first ? std::move(native) : std::move(alternate),
      unroll_first ? native_resources : alternate_resources,
      device,
      unroll_first);
}

std::unique_ptr<ReductionParams> chooseInner3DWarpTarget(
    const PersistentKernelProperties& properties,
    const ReductionParams& base,
    int64_t wave_cap,
    std::unique_ptr<ReductionParams> native,
    const Inner3DPlanResources& native_resources,
    const cudaDeviceProp& device,
    int64_t occupancy_target,
    int64_t layout_max) {
  const int64_t native_warp = getInner3DNativeWarpTarget(properties, device);
  struct Plan {
    int64_t target;
    std::unique_ptr<ReductionParams> params;
    Inner3DPlanResources resources;
  };
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  auto add_plan = [&](int64_t target,
                      std::unique_ptr<ReductionParams> params,
                      const Inner3DPlanResources& resources) {
    const std::string id = "warp" + std::to_string(target);
    options.push_back({
        id,
        "Target " + std::to_string(target) +
            " threads; completed CTA=" +
            std::to_string(resources.padded_threads_per_block) +
            ", register cap=" + std::to_string(resources.registers_per_thread) +
            ", grid blocks=" + std::to_string(resources.grid_blocks),
        inner3DLocalPlanConsequence(*params, resources)});
    plans.push_back({target, std::move(params), resources});
  };
  add_plan(native_warp, std::move(native), native_resources);

  // The L1/L2 estimate is empirical. Each alternate target recomputes the
  // complete selected-wave plan and must retain native capacity and launch
  // limits. The native plan remains available even if no alternate survives.
  for (const int64_t target : {4, 8, 16, 32}) {
    if (target == native_warp || target > device.warpSize) {
      continue;
    }
    auto params = std::make_unique<ReductionParams>(base);
    Inner3DPlanResources resources;
    innerPersistentHeuristic3DPlan(
        properties, wave_cap, params.get(), &resources, target, occupancy_target);
    if (isNativeValidInner3DAlternative(
            properties, *params, *plans.front().params, resources, device)) {
      add_plan(target, std::move(params), resources);
    }
  }

  const std::string state =
      (properties.vectorize_factor == layout_max
           ? (occupancy_target == 28
                  ? "inner_persistent.3d_warp_target.v1\nreduction_numel="
                  : "inner_persistent.3d_warp_target.v2\nreduction_numel=")
           : (occupancy_target == 28
                  ? "inner_persistent.3d_warp_target.v3\nreduction_numel="
                  : "inner_persistent.3d_warp_target.v4\nreduction_numel=")) +
      std::to_string(properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(properties.total_iteration_numel) +
      "\nselected_wave_cap=" + std::to_string(wave_cap) +
      (properties.vectorize_factor == layout_max
           ? std::string()
           : "\nselected_vector_width=" +
                 std::to_string(properties.vectorize_factor)) +
      (occupancy_target == 28
           ? std::string()
           : "\nselected_occupancy_target=" + std::to_string(occupancy_target)) +
      "\nmax_dtype_bits=" + std::to_string(properties.max_dtype_size_bit) +
      "\nn_tensor_inputs=" + std::to_string(properties.n_tensor_inputs) +
      "\nl2_cache_bytes=" + std::to_string(device.l2CacheSize) +
      "\ndevice_warp_size=" + std::to_string(device.warpSize) +
      "\npersistent_buffer_bits=" +
      std::to_string(properties.max_persistent_buffer_size_bit);
  const std::string native_choice =
      scheduler::jev::mode() == scheduler::jev::Mode::Record
      ? "warp" + std::to_string(native_warp)
      : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP6.warp", state, options, native_choice);
  const auto option_it = std::find_if(
      options.begin(),
      options.end(),
      [&](const auto& option) { return option.id == selected; });
  NVF_ERROR(option_it != options.end(), "Unknown inner 3D warp target");
  auto& chosen_plan = plans.at(std::distance(options.begin(), option_it));
  return chooseInner3DWorkSplit(
      properties,
      base,
      wave_cap,
      chosen_plan.target,
      occupancy_target,
      std::move(chosen_plan.params),
      chosen_plan.resources,
      device);
}

std::unique_ptr<ReductionParams> innerPersistentHeuristic3D(
    const PersistentKernelProperties& properties,
    const ReductionParams& base) {
  const auto jev_mode = scheduler::jev::mode();
  auto native = std::make_unique<ReductionParams>(base);
  if (jev_mode == scheduler::jev::Mode::Off) {
    innerPersistentHeuristic3DPlan(properties, 4, native.get());
    return native;
  }

  const auto device = at::cuda::getCurrentDeviceProperties();
  auto native_preview = std::make_unique<ReductionParams>(base);
  Inner3DPlanResources native_resources;
  innerPersistentHeuristic3DPlan(
      properties, 4, native_preview.get(), &native_resources);
  // The original plan may run successfully even when the stricter Choice
  // validator rejects its estimated register footprint. If the native anchor
  // is not admitted, no alternate can be proved against it. Keep the exact
  // Off-mode result and make the native-only path explicit in both traces.
  const int64_t layout_max = properties.vectorize_factor;
  const bool layout_valid =
      layout_max > 0 && (layout_max & (layout_max - 1)) == 0 &&
      properties.inner_most_dimension_numel % layout_max == 0;
  if (!layout_valid || !isNativeValidInner3DAlternative(
                           properties,
                           *native_preview,
                           *native_preview,
                           native_resources,
                           *device)) {
    recordInner3DNativeOnlyDecisions(
        properties,
        *native_preview,
        *device,
        layout_valid ? "native 3D plan failed complete resource preflight"
                     : "native vector width is not a divisible power of two");
    return native_preview;
  }
  const auto selected_properties =
      chooseInner3DVectorWidth(properties, base, *device);
  struct Plan {
    int64_t waves;
    std::unique_ptr<ReductionParams> params;
    Inner3DPlanResources resources;
  };
  std::vector<Plan> plans;
  std::vector<scheduler::jev::ChoiceOption> options;
  // Build the original plan first so every alternative can be checked against
  // its per-thread persistent allocation before it enters the Choice menu.
  for (const int64_t waves : {4, 2, 8}) {
    auto params = waves == 4 ? std::move(native)
                             : std::make_unique<ReductionParams>(base);
    Inner3DPlanResources resources;
    innerPersistentHeuristic3DPlan(
        selected_properties, waves, params.get(), &resources);
    if (waves != 4 &&
        !isNativeValidInner3DAlternative(
            selected_properties,
            *params,
            *plans.front().params,
            resources,
            *device)) {
      continue;
    }
    const std::string id = "waves" + std::to_string(waves);
    options.push_back({
        id,
        "Target at most " + std::to_string(waves) +
            " SM waves; completed CTA=" +
            std::to_string(resources.padded_threads_per_block) +
            ", register cap=" + std::to_string(resources.registers_per_thread) +
            ", grid blocks=" + std::to_string(resources.grid_blocks),
        inner3DLocalPlanConsequence(*params, resources)});
    plans.push_back({waves, std::move(params), resources});
  }

  const std::string state =
      "inner_persistent.3d_wave_cap.v1\nreduction_numel=" +
      std::to_string(selected_properties.total_reduction_numel) +
      "\ninner_numel=" +
      std::to_string(selected_properties.inner_most_dimension_numel) +
      "\niteration_numel=" +
      std::to_string(selected_properties.total_iteration_numel) +
      "\nsm_count=" + std::to_string(device->multiProcessorCount) +
      "\nwarp_size=" + std::to_string(device->warpSize) +
      "\nmax_threads_per_block=" +
      std::to_string(device->maxThreadsPerBlock) +
      "\nmax_threads_per_sm=" +
      std::to_string(device->maxThreadsPerMultiProcessor) +
      "\nregisters_per_block=" + std::to_string(device->regsPerBlock) +
      "\nmax_dtype_bits=" + std::to_string(selected_properties.max_dtype_size_bit) +
      "\nn_tensor_inputs=" + std::to_string(selected_properties.n_tensor_inputs) +
      "\nvector_width=" + std::to_string(selected_properties.vectorize_factor) +
      "\npersistent_buffer_bits=" +
      std::to_string(selected_properties.max_persistent_buffer_size_bit);
  const std::string native_choice =
      jev_mode == scheduler::jev::Mode::Record ? "waves4" : "";
  const std::string selected = scheduler::jev::chooseDecision(
      "IP6.waves", state, options, native_choice);
  const auto option_it = std::find_if(
      options.begin(),
      options.end(),
      [&](const auto& option) { return option.id == selected; });
  NVF_ERROR(option_it != options.end(), "Unknown inner 3D wave target");
  const auto plan_index = std::distance(options.begin(), option_it);
  auto& chosen_plan = plans.at(plan_index);
  auto chosen_occupancy = chooseInner3DOccupancyTarget(
      selected_properties,
      base,
      chosen_plan.waves,
      std::move(chosen_plan.params),
      chosen_plan.resources,
      *device,
      properties.vectorize_factor);
  return chooseInner3DWarpTarget(
      selected_properties,
      base,
      chosen_plan.waves,
      std::move(chosen_occupancy.params),
      chosen_occupancy.resources,
      *device,
      chosen_occupancy.target,
      properties.vectorize_factor);
}

} // namespace

std::unique_ptr<ReductionParams> getInnerPersistentHeuristics(
    Fusion* fusion,
    const PersistentKernelProperties& prop,
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info) {
  FusionGuard fg(fusion);

  std::unique_ptr<ReductionParams> rparams = std::make_unique<ReductionParams>(
      InnerPersistentKernelScheduler::schedulerType());

  // shared heuristics for all cases
  rparams->persistent_kernel = true;
  rparams->fastest_dim = true;
  rparams->project_persistent_buffers = prop.project_persistent_buffers;
  rparams->cparams.index_type = prop.index_type;
  // The register budget is a hard capacity gate, but below it both physical
  // storage modes may be legal. Build the native register plan without a
  // nested Choice, and check the complete shared-memory allocation at the
  // shared planner's native batch/CTA before offering either mode.
  bool choose_shared_storage = false;
  const auto jev_mode = scheduler::jev::mode();
  const auto* device = at::cuda::getCurrentDeviceProperties();
  const int64_t register_file_bits = scheduler_utils::register_file_size_bit;
  if (jev_mode != scheduler::jev::Mode::Off &&
      prop.max_persistent_buffer_size_bit > 0 &&
      prop.max_persistent_buffer_size_bit <= register_file_bits &&
      prop.total_reduction_numel == prop.inner_most_dimension_numel &&
      !prop.project_persistent_buffers && device->major < 9 &&
      scheduler_utils::getMaxClusterSize() == 1 &&
      prop.vectorize_factor > 0 &&
      prop.total_reduction_numel % prop.vectorize_factor == 0 &&
      device->warpSize > 0 && device->maxThreadsPerBlock > 0 &&
      device->maxThreadsPerBlock >= device->warpSize &&
      (device->warpSize & (device->warpSize - 1)) == 0 &&
      (device->maxThreadsPerBlock & (device->maxThreadsPerBlock - 1)) == 0 &&
      device->maxThreadsPerBlock <= device->maxThreadsDim[0] &&
      prop.total_reduction_numel /
              prop.vectorize_factor >= device->maxThreadsPerBlock &&
      prop.total_iteration_numel > 0 &&
      prop.total_iteration_numel <= device->maxGridSize[0]) {
    const int64_t register_overhead = prop.has_exp_op ? 32l : 16l;
    const bool high_bandwidth =
        scheduler_utils::isHighBandwidthFlopsRatio();
    const int64_t target_warps =
        high_bandwidth && prop.disable_project_to_avoid_recompute ? 16l : 28l;
    Inner2DPolicyResources register_resources;
    ReductionParams register_preview(*rparams);
    innerPersistentHeuristic2DPlan(
        prop,
        &register_preview,
        register_overhead,
        target_warps,
        false,
        &register_resources);
    if (register_resources.supported_menu) {
      auto buffer_info_entry =
          HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
              data_cache, [&fusion]() {
                return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                    scheduler_utils::persistentBuffers(fusion));
              });
      const auto& buffer_info = buffer_info_entry.get();
      const auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
      const int64_t after_vectorize =
          prop.total_reduction_numel / prop.vectorize_factor;
      const int64_t shared_batch =
          ceilDiv(after_vectorize, (int64_t)device->maxThreadsPerBlock);
      const int64_t actual_threads = ceilDiv(after_vectorize, shared_batch);
      const int64_t padded_threads =
          ceilDiv(actual_threads, (int64_t)device->warpSize) * device->warpSize;
      const int64_t shared_capacity = normalization_scheduler_utils::
          getMaxSharedMemorySizeBitForPersistentBuffer(
              fusion,
              runtime_info,
              reduction_tvs,
              buffer_info,
              false,
              shared_batch,
              padded_threads);
      if (shared_capacity >= prop.max_persistent_buffer_size_bit &&
          padded_threads <= device->maxThreadsPerBlock &&
          !buffer_info.persistent_buffers.empty() &&
          buffer_info.persistent_buffers.size() <= 32) {
        int64_t reduction_dtype_bits = 1;
        for (auto* tv : reduction_tvs) {
          reduction_dtype_bits =
              std::max(reduction_dtype_bits, dataTypeSizeBit(tv->getDataType()));
        }
        const bool has_welford = ir_utils::hasOpsOfType<WelfordOp>(fusion);
        std::string state =
            "inner_persistent.storage.v1\nreduction_numel=" +
            std::to_string(prop.total_reduction_numel) +
            "\niteration_numel=" + std::to_string(prop.total_iteration_numel) +
            "\nvector_width=" + std::to_string(prop.vectorize_factor) +
            "\npersistent_buffer_bits=" +
            std::to_string(prop.max_persistent_buffer_size_bit) +
            "\nregister_file_bits=" + std::to_string(register_file_bits) +
            "\nsmem_limit_bits=" +
            std::to_string((int64_t)device->sharedMemPerBlockOptin * 8) +
            "\nmax_threads_per_block=" +
            std::to_string(device->maxThreadsPerBlock) +
            "\nwarp_size=" + std::to_string(device->warpSize) +
            "\nreduction_dtype_bits=" +
            std::to_string(reduction_dtype_bits) +
            "\nhas_welford=" + std::to_string(has_welford) +
            "\nmax_vector_bits=" +
            std::to_string(getMaxVectorizationSizeInBit());
        for (auto* buffer : buffer_info.persistent_buffers) {
          state += "\nbuffer=" +
              std::to_string(scheduler_utils::getPersistentBufferSizeBitOfTensor(
                  buffer, runtime_info, buffer_info)) +
              ":" + std::to_string(dataTypeSizeBit(buffer->getDataType()));
        }
        if (state.size() <= 16 * 1024) {
          const std::vector<scheduler::jev::ChoiceOption> options = {
              {"register", "Keep the proved 2D register persistent plan"},
              {"shared", "Use the proved 2D shared-memory persistent plan"}};
          const std::string native_choice =
              jev_mode == scheduler::jev::Mode::Record ? "register" : "";
          choose_shared_storage = scheduler::jev::chooseDecision(
                                      "IP4.storage",
                                      state,
                                      options,
                                      native_choice) == "shared";
        } else {
          NVF_ERROR(
              jev_mode == scheduler::jev::Mode::Record,
              "Inner persistent storage request exceeds safe size");
          scheduler::jev::recordCoverageSkip(
              "IP4.storage", "Storage state exceeds 16 KiB");
        }
      }
    }
  }
  // specific heuristics for different cases
  if (prop.max_persistent_buffer_size_bit >
          scheduler_utils::register_file_size_bit &&
      scheduler_utils::getMaxClusterSize() > 1) {
    innerPersistentHeuristicCluster(prop, rparams.get());
  } else if (
      prop.max_persistent_buffer_size_bit >
          scheduler_utils::register_file_size_bit ||
      choose_shared_storage) {
    rparams->tag = "Shared Memory Inner Persistent Heuristic.\n";
    // all persistent buffers are moved to shared memory
    // TODO: allow only part of the buffers to be moved to shared memory
    rparams->smem_persistent_buffers = prop.persistent_buffers;
    innerPersistentHeuristicSharedMemory(
        fusion, prop, data_cache, runtime_info, rparams.get());
  } else if (prop.total_reduction_numel == prop.inner_most_dimension_numel) {
    rparams->tag = "2D Register Inner Persistent Heuristic.\n";
    const int64_t native_overhead = prop.has_exp_op ? 32l : 16l;
    const bool high_bandwidth = scheduler_utils::isHighBandwidthFlopsRatio();
    const int64_t native_warps =
        high_bandwidth && prop.disable_project_to_avoid_recompute ? 16l : 28l;
    const auto width_properties = chooseInner2DVectorWidth(
        prop, *rparams, native_overhead, native_warps);
    innerPersistentHeuristic2D(width_properties, rparams.get());
    // Magic zero is a value-preserving index/predicate protection. The native
    // policy omits it only for a cheap, unprojected persistent buffer. Keep
    // that resource condition, the selected register-backed plan, and the
    // runtime's TMA/global-disable constraints local. The bandwidth ratio is
    // then only the native performance preference between two legal forms.
    const auto jev_mode = scheduler::jev::mode();
    if (jev_mode != scheduler::jev::Mode::Off) {
      const auto exprs = fusion->exprs();
      const bool has_tma = std::any_of(
          exprs.begin(), exprs.end(),
          [](Expr* expr) { return ir_utils::isCpAsyncBulk(expr); });
      const bool eligible =
          width_properties.disable_project_to_avoid_recompute &&
          rparams->persistent_kernel && !rparams->schedule_3d &&
          rparams->smem_persistent_buffers.empty() &&
          rparams->cparams.maxrregcount > 0 &&
          rparams->cparams.maxrregcount <= 255 &&
          rparams->batches_per_block_inner_reduction > 0 &&
          rparams->unroll_factor_inner_reduction > 0 && !has_tma &&
          !isOptionDisabled(DisableOption::MagicZero);
      if (!eligible) {
        scheduler::jev::recordNoChoiceDecision(
            "IP5.magic_zero",
            rparams->cparams.enable_magic_zero ? "on" : "off",
            "No proved register-backed magic-zero pair",
            "native_magic_zero_policy");
      } else {
        // Send the same primitive device facts used by the original
        // bandwidth/flops heuristic. Sending its boolean result here would
        // reveal the native magic-zero answer for this eligible domain.
        const auto* device = at::cuda::getCurrentDeviceProperties();
        const int device_index = at::cuda::current_device();
        int memory_clock_khz = 0;
        int gpu_clock_khz = 0;
        cudaDeviceGetAttribute(
            &memory_clock_khz, cudaDevAttrMemoryClockRate, device_index);
        cudaDeviceGetAttribute(
            &gpu_clock_khz, cudaDevAttrClockRate, device_index);
        const bool valid_hardware =
            device->multiProcessorCount > 0 &&
            device->multiProcessorCount <= 65536 && device->major >= 3 &&
            device->major <= 20 &&
            device->minor >= 0 && device->minor <= 9 &&
            device->memoryBusWidth > 0 && memory_clock_khz > 0 &&
            gpu_clock_khz > 0;
        if (!valid_hardware) {
          NVF_ERROR(
              jev_mode == scheduler::jev::Mode::Record,
              "Cannot reconstruct magic-zero bandwidth preference");
          scheduler::jev::recordCoverageSkip(
              "IP5.magic_zero", "Device bandwidth inputs unavailable");
        } else {
          const std::string state =
              "inner_persistent.reg_magic_zero.v1\n"
              "sm_count=" + std::to_string(device->multiProcessorCount) +
              "\nsm_major=" + std::to_string(device->major) +
              "\nsm_minor=" + std::to_string(device->minor) +
              "\nmemory_bus_bits=" + std::to_string(device->memoryBusWidth) +
              "\nmemory_clock_khz=" + std::to_string(memory_clock_khz) +
              "\ngpu_clock_khz=" + std::to_string(gpu_clock_khz) +
              "\ndisable_projection_avoid_recompute=" +
              std::to_string(
                  width_properties.disable_project_to_avoid_recompute) +
              "\nregister_cap=" +
              std::to_string(rparams->cparams.maxrregcount) +
              "\npersistent_batch=" +
              std::to_string(rparams->batches_per_block_inner_reduction) +
              "\nvector_width=" +
              std::to_string(rparams->unroll_factor_inner_reduction) +
              "\nno_tma=" + std::to_string(!has_tma) +
              "\nmagic_zero_available=" +
              std::to_string(!isOptionDisabled(DisableOption::MagicZero));
          const std::vector<scheduler::jev::ChoiceOption> options = {
              {"off", "Omit zero-valued index and predicate protection"},
              {"on", "Use zero-valued index and predicate protection"}};
          const std::string native_choice =
              jev_mode == scheduler::jev::Mode::Record
              ? (rparams->cparams.enable_magic_zero ? "on" : "off")
              : "";
          const std::string selected = scheduler::jev::chooseDecision(
              "IP5.magic_zero", state, options, native_choice);
          NVF_ERROR(
              selected == "on" || selected == "off",
              "Unknown 2D register magic-zero choice");
          rparams->cparams.enable_magic_zero = selected == "on";
        }
      }
    }
  } else {
    rparams->tag = "3D Register Inner Persistent Heuristic.\n";
    rparams = innerPersistentHeuristic3D(prop, *rparams);
  }

  // debug print
  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << prop.toString() << std::endl;
    debug() << rparams->toString() << std::endl;
  }
  return rparams;
}

void scheduleInnerPersistent(Fusion* fusion, const ReductionParams* rparams) {
  FusionGuard fg(fusion);
  NVF_ERROR(
      rparams != nullptr &&
          rparams->scheduler_type ==
              InnerPersistentKernelScheduler::schedulerType(),
      "Incorrect parameters sent to inner persistent non-TMA scheduler");
  normalization_scheduler_utils::schedulePersistentKernel(
      fusion, rparams, rparams->scheduler_type);
}

} // namespace non_tma
} // namespace normalization_inner
} // namespace nvfuser
