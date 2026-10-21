// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <exception>
#include <optional>
#include <ranges>

#include <ATen/cuda/CUDAContext.h>

#include "instrumentation.h"
#include "options.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/normalization_inner_outer_multi_wave.h"
#include "scheduler/normalization_inner_outer_tma_ws.h"
#include "scheduler/normalization_inner_outer_utils.h"
#include "scheduler/normalization_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/utils.h"

namespace nvfuser {
namespace {

// Architecture and TMA shape requirements are hard eligibility checks.
bool canTryWarpSpecialized(Fusion* fusion) {
  // False, for pre-Blackwell GPUs
  if (at::cuda::getCurrentDeviceProperties()->major < 10) {
    return false;
  }
  // TMA requires compile-time known contiguous innermost dimension on inputs
  if (!scheduler_utils::inputsHaveContiguousInnerDim(fusion)) {
    return false;
  }
  // False, if any of the inputs is dynamically shaped
  // TODO: extend to support dynamic inputs, warp specialization requires
  // static CTA size
  auto inp_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
  if (std::ranges::any_of(inp_tvs, [](TensorView* tv) {
        return scheduler_utils::isSymbolicTensor(tv);
      })) {
    return false;
  }

  return true;
}

// Prioritize warp specialized approach if possible.
bool preferWarpSpecialized(
    Fusion* fusion,
    int64_t total_iteration_numel,
    int64_t n_inner_reduction_tvs) {
  if (!canTryWarpSpecialized(fusion)) {
    return false;
  }
  // False, when iteration dimension is too small.
  // (1) Benefit from amortizing weight tensor loading overhead is minimal.
  // (2) Can't create deep circular buffering.
  // (3) Empirically determined thresholds on B200:
  // - RMS Norm Bwd: ≤16 rows per SM, Layer Norm Bwd: ≤4 rows per SM
  int64_t sm_count =
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  int64_t rows_per_sm = ceilDiv(total_iteration_numel, sm_count);
  bool is_layer_norm = n_inner_reduction_tvs > 1;
  int64_t prefered_rows_per_sm = is_layer_norm ? 16 : 4;
  if (rows_per_sm <= prefered_rows_per_sm) {
    return false;
  }
  // Try to use warp specialized, but if the heuristic fails, fall back to
  // multi-wave
  return true;
}

struct WarpSpecializedPlanMeasurements {
  int64_t compute_bdimx;
  int64_t computation_groups;
  int64_t stages;
  int64_t total_threads;
  int64_t available_registers_per_thread;
};

// Check the finalized WS plan, including its first stage. The constructor
// checks resource usage when growing a plan but can start at one stage even
// when that first-stage estimate is too large.
std::optional<WarpSpecializedPlanMeasurements> hardValidWarpSpecializedPlan(
    const ReductionParams& params,
    const inner_outer_utils::PersistentBufferStorageParams& buffers,
    int64_t inner_dim_numel,
    int64_t computation_dtype_size_bit,
    int64_t vector_width) {
  const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
  const auto& launch = params.lparams;
  const auto& circular = params.circular_buffer_options;
  if (!buffers.has_enough_regs_and_smem || !buffers.project_to_input ||
      !std::holds_alternative<WarpSpecialized>(circular.type) ||
      circular.stage < 2 || inner_dim_numel < 1 || vector_width < 1 ||
      inner_dim_numel % vector_width != 0 ||
      launch.nThreads() > dev_prop->maxThreadsPerBlock ||
      launch.bdimx() > dev_prop->maxThreadsDim[0] ||
      launch.bdimy() > dev_prop->maxThreadsDim[1] ||
      launch.gdimy() > dev_prop->maxGridSize[1]) {
    return std::nullopt;
  }
  const auto& ws = std::get<WarpSpecialized>(circular.type);
  if (ws.on != ParallelType::TIDx && ws.on != ParallelType::TIDy) {
    return std::nullopt;
  }
  const int64_t bdimx = ws.on == ParallelType::TIDx
      ? launch.bdimx() - kWarpSpecializationPaddedThreads
      : launch.bdimx();
  const int64_t groups = params.computation_warp_groups;
  const int64_t compute_threads = bdimx * groups;
  const int64_t total_threads =
      compute_threads + kWarpSpecializationPaddedThreads;
  if (bdimx < 128 || bdimx % 128 != 0 || groups < 1 ||
      compute_threads < 1 || total_threads != launch.nThreads() ||
      buffers.circular_buffered_smem_size_bit < 0 ||
      buffers.non_circular_buffered_smem_size_bit < 0 ||
      buffers.regs_buffer_size_bit < 0) {
    return std::nullopt;
  }

  const int64_t unroll = params.unroll_factor_iter_dom;
  const auto smem_bits =
      static_cast<__int128>(buffers.circular_buffered_smem_size_bit) *
          unroll * circular.stage +
      (params.is_non_circular_buffer_gmem_to_regs
           ? 0
           : buffers.non_circular_buffered_smem_size_bit) +
      alignSharedMemoryBits(64 * circular.stage) +
      alignSharedMemoryBits(
          unroll * bdimx * groups * computation_dtype_size_bit);
  if (unroll < 1 || smem_bits >
          static_cast<__int128>(dev_prop->sharedMemPerBlockOptin) * 8) {
    return std::nullopt;
  }

  const int64_t after_vectorize = inner_dim_numel / vector_width;
  const int64_t batch = ceilDiv(after_vectorize, bdimx);
  auto buffer_registers = [&](int64_t logical_bits) {
    const int64_t bits_per_element = logical_bits / inner_dim_numel;
    return bits_per_element * batch * vector_width /
        scheduler_utils::bits_per_register;
  };
  const int64_t available = getRegPerThreadGivenThreadsPerSM(total_threads);
  const auto [_, compute_branch_registers] =
      scheduler_utils::getRegisterSharing(
          available, compute_threads, kWarpSpecializationPaddedThreads);
  const auto required_registers =
      static_cast<__int128>(112) +
      (params.is_circular_buffer_regs_cached
           ? buffer_registers(buffers.circular_buffered_smem_size_bit) * unroll
           : 0) +
      (params.is_non_circular_buffer_gmem_to_regs
           ? buffer_registers(buffers.non_circular_buffered_smem_size_bit)
           : 0) +
      buffer_registers(buffers.regs_buffer_size_bit);
  if (required_registers > compute_branch_registers) {
    return std::nullopt;
  }
  return WarpSpecializedPlanMeasurements{
      bdimx, groups, circular.stage, total_threads, available};
}

bool hardValidMultiWavePlan(
    const ReductionParams& params,
    const inner_outer_utils::PersistentBufferStorageParams& buffers) {
  const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
  const auto& launch = params.lparams;
  const auto smem_bits =
      static_cast<__int128>(buffers.smem_buffer_size_bit) +
      buffers.smem_overhead_bit;
  return buffers.has_enough_regs_and_smem &&
      buffers.smem_buffer_size_bit >= 0 && buffers.smem_overhead_bit >= 0 &&
      smem_bits <= static_cast<__int128>(dev_prop->sharedMemPerBlockOptin) * 8 &&
      launch.nThreads() <= dev_prop->maxThreadsPerBlock &&
      launch.bdimx() <= dev_prop->maxThreadsDim[0] &&
      launch.bdimy() <= dev_prop->maxThreadsDim[1] &&
      launch.gdimy() <= dev_prop->maxGridSize[1] &&
      params.cparams.maxrregcount > 0 &&
      params.cparams.maxrregcount <=
          scheduler_utils::max_registers_per_thread;
}

std::unique_ptr<ReductionParams> getInnerOuterPersistentHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const scheduler_utils::SchedulerHyperParameters* hp = nullptr) {
  FusionGuard fg(fusion);

  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();

  NVF_ERROR(!reduction_tvs.empty(), "Need reduction tensor views to schedule.");

  // Get dtype used to store partial outer reduction
  // Get the first inner reduction tv and use it as the reference tv
  int64_t max_outer_reduction_dtype_size_bit = 1;
  int64_t n_inner_reduction_tvs = 0;
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      n_inner_reduction_tvs++;
    } else {
      max_outer_reduction_dtype_size_bit = std::max(
          max_outer_reduction_dtype_size_bit,
          dataTypeSizeBit(tv->getDataType()));
    }
  }
  auto ref_red_tv = first_inner_reduction_tv;

  // Verify the presence of a reduction TensorView connected to a Fusion input
  normalization_scheduler_utils::checkReductionTvForScheduling(
      fusion, ref_red_tv);

  auto properties =
      scheduler_utils::getReductionProperties(fusion, runtime_info, ref_red_tv);
  auto reduced_tv = ir_utils::getSoleProducerTv(ref_red_tv);

  // Although properties contains runtime information
  // "inner_most_dimension_ndims" is a compile time value
  auto vec_break_point = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizationBreakPointOfReductionProducer>(
      data_cache, [&ref_red_tv, &reduced_tv, &properties]() {
        return std::make_unique<int64_t>(
            vectorize_helper::getVectorizationBreakPointOfReductionProducer(
                ref_red_tv, reduced_tv, properties.inner_most_dimension_ndims));
      });

  auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info, reduced_tv, data_cache, vec_break_point.get());

  auto persistent_buffer_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::PersistentBufferInfo>(
          data_cache, [&fusion]() {
            return std::make_unique<scheduler_utils::PersistentBufferInfo>(
                scheduler_utils::persistentBuffers(fusion));
          });

  int64_t threads_per_block_min =
      InnerOuterPersistentKernelScheduler::threads_per_block_min;
  int64_t threads_per_block_max =
      InnerOuterPersistentKernelScheduler::threads_per_block_max;
  bool user_enforced_warp_specialization =
      isOptionEnabled(EnableOption::WarpSpecializedNormalization);
  if (hp) {
    NVF_ERROR(
        (hp->vectorize_factor & (hp->vectorize_factor - 1)) == 0,
        "Vectorization factor must be a power of 2");
    NVF_ERROR(
        hp->vectorize_factor <= vectorize_factor,
        "Vectorization factor must be less than or equal to the max allowed "
        "vectorization factor");
    threads_per_block_min = hp->threads_per_block_min;
    threads_per_block_max = hp->threads_per_block_max;
    user_enforced_warp_specialization = hp->is_warp_specialized;
  }

  auto& persistent_buffer_info = persistent_buffer_info_entry.get();
  NVF_ERROR(
      !persistent_buffer_info.persistent_buffers.empty(),
      "Persistent scheduler requires persistent buffers.");

  auto makeRParams = [&]() {
    return std::make_unique<ReductionParams>(
        InnerOuterPersistentKernelScheduler::schedulerType());
  };
  std::optional<inner_outer_utils::PersistentBufferStorageParams>
      multiwave_buffer_params;
  auto multiwaveBuffers = [&]()
      -> const inner_outer_utils::PersistentBufferStorageParams& {
    if (!multiwave_buffer_params) {
      multiwave_buffer_params.emplace(
          inner_outer_utils::getPersistentBufferStorageParams(
              fusion,
              runtime_info,
              data_cache,
              reduction_tvs,
              vectorize_factor,
              threads_per_block_min,
              threads_per_block_max,
              /*is_warp_specialized=*/false));
    }
    return *multiwave_buffer_params;
  };
  auto makeMultiWave = [&](const auto& buffers,
                           int64_t inner_vector_width,
                           bool enable_jev_choice) {
    auto params = makeRParams();
    params->smem_persistent_buffers = buffers.smem_persistent_buffers;
    inner_outer_multi_wave::getHeuristics(
        params.get(),
        properties.total_iteration_numel,
        properties.total_reduction_numel,
        buffers.regs_buffer_size_bit,
        buffers.smem_buffer_size_bit,
        buffers.smem_overhead_bit,
        max_outer_reduction_dtype_size_bit,
        inner_vector_width,
        threads_per_block_min,
        threads_per_block_max,
        buffers.project_to_input,
        runtime_info.getIndexType(),
        enable_jev_choice);
    return params;
  };
  auto makeWarpSpecialized =
      [&](const auto& buffers,
          bool enable_debug_dump,
          bool enable_jev_choice) {
    auto params = makeRParams();
    params->smem_persistent_buffers = buffers.smem_persistent_buffers;
    inner_outer_tma_warp_specialized::getHeuristics(
        params.get(),
        properties.total_iteration_numel,
        properties.total_reduction_numel,
        buffers.regs_buffer_size_bit,
        buffers.circular_buffered_smem_size_bit,
        buffers.non_circular_buffered_smem_size_bit,
        max_outer_reduction_dtype_size_bit,
        vectorize_factor,
        threads_per_block_min,
        threads_per_block_max,
        buffers.project_to_input,
        runtime_info.getIndexType(),
        enable_debug_dump,
        enable_jev_choice);
    return params;
  };

  const auto jev_mode = scheduler::jev::mode();
  auto chooseMultiWaveStorage = [&](const auto& native_buffers,
                                    int64_t inner_vector_width,
                                    bool enable_jev_choice) {
    if (!enable_jev_choice || jev_mode == scheduler::jev::Mode::Off ||
        hp != nullptr || !native_buffers.has_enough_regs_and_smem ||
        native_buffers.smem_persistent_buffers.empty()) {
      return makeMultiWave(
          native_buffers, inner_vector_width, enable_jev_choice);
    }

    // The native helper tries all shared memory first. Its register total is
    // then only the outer partial reduction. Offer all registers only when
    // that exact all-shared placement and the alternative both fit.
    const int64_t outer_regs_bit =
        inner_outer_utils::partialOuterReductionBufferSizeBit(
            reduction_tvs, runtime_info);
    const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
    const int64_t register_limit_bit =
        scheduler_utils::register_file_size_bit_56k;
    const int64_t smem_limit_bit =
        (int64_t)dev_prop->sharedMemPerBlockOptin * 8;
    auto all_buffers = native_buffers.project_to_input
        ? persistent_buffer_info.projectable_buffer_inputs
        : persistent_buffer_info.persistent_buffers;
    if (native_buffers.project_to_input) {
      for (auto* buffer : persistent_buffer_info.persistent_buffers) {
        if (buffer->isFusionInput()) {
          all_buffers.push_back(buffer);
        }
      }
    }
    if (outer_regs_bit < 0 ||
        native_buffers.smem_persistent_buffers != all_buffers ||
        native_buffers.regs_buffer_size_bit != outer_regs_bit ||
        native_buffers.smem_buffer_size_bit <= 0 ||
        native_buffers.smem_overhead_bit < 0 ||
        native_buffers.smem_overhead_bit > smem_limit_bit ||
        native_buffers.smem_buffer_size_bit >
            smem_limit_bit - native_buffers.smem_overhead_bit ||
        native_buffers.smem_persistent_buffers.size() > 64) {
      return makeMultiWave(
          native_buffers, inner_vector_width, enable_jev_choice);
    }

    int64_t all_register_bits = outer_regs_bit;
    std::string buffer_bits_state;
    int64_t all_shared_bits = 0;
    bool all_registers_fit = all_register_bits <= register_limit_bit;
    for (auto* buffer : native_buffers.smem_persistent_buffers) {
      const int64_t bits = scheduler_utils::getPersistentBufferSizeBitOfTensor(
          buffer, runtime_info, persistent_buffer_info);
      const int64_t dtype_bits = dataTypeSizeBit(buffer->getDataType());
      if (bits <= 0 || !all_registers_fit ||
          bits > register_limit_bit - all_register_bits) {
        all_registers_fit = false;
        break;
      }
      all_register_bits += bits;
      const int64_t shared_bits = alignSharedMemoryBits(
          inner_outer_utils::roundUpSharedMemory(
              bits,
              dtype_bits,
              inner_vector_width,
              threads_per_block_min,
              threads_per_block_max,
              dev_prop->warpSize));
      if (shared_bits < 0 ||
          shared_bits > native_buffers.smem_buffer_size_bit - all_shared_bits) {
        all_registers_fit = false;
        break;
      }
      all_shared_bits += shared_bits;
      buffer_bits_state += "\nbuffer=" + std::to_string(bits) + ":" +
          std::to_string(dtype_bits);
    }
    if (!all_registers_fit ||
        all_shared_bits != native_buffers.smem_buffer_size_bit) {
      return makeMultiWave(
          native_buffers, inner_vector_width, enable_jev_choice);
    }

    auto register_buffers = native_buffers;
    register_buffers.smem_persistent_buffers.clear();
    register_buffers.regs_buffer_size_bit = all_register_bits;
    register_buffers.smem_buffer_size_bit = 0;
    register_buffers.circular_buffered_smem_size_bit = 0;
    register_buffers.non_circular_buffered_smem_size_bit = 0;

    std::unique_ptr<ReductionParams> shared_preview;
    std::unique_ptr<ReductionParams> register_preview;
    try {
      shared_preview = makeMultiWave(
          native_buffers, inner_vector_width, /*enable_jev_choice=*/false);
      register_preview = makeMultiWave(
          register_buffers, inner_vector_width, /*enable_jev_choice=*/false);
    } catch (const std::exception&) {
      return makeMultiWave(
          native_buffers, inner_vector_width, enable_jev_choice);
    }
    if (!hardValidMultiWavePlan(*shared_preview, native_buffers) ||
        !hardValidMultiWavePlan(*register_preview, register_buffers) ||
        shared_preview->unroll_factor_inner_reduction != inner_vector_width ||
        register_preview->unroll_factor_inner_reduction != inner_vector_width) {
      return makeMultiWave(
          native_buffers, inner_vector_width, enable_jev_choice);
    }

    auto planState = [](const ReductionParams& params) {
      const auto& launch = params.lparams;
      return std::to_string(launch.bdimx()) + ":" +
          std::to_string(launch.bdimy()) + ":" +
          std::to_string(launch.gdimy()) + ":" +
          std::to_string(params.cparams.maxrregcount);
    };
    const std::string state =
        "inner_outer_multi_wave.storage.v1\nselected_inner_width=" +
        std::to_string(inner_vector_width) + "\nouter_accumulator_bits=" +
        std::to_string(outer_regs_bit) + "\nregister_limit_bits=" +
        std::to_string(register_limit_bit) + "\nshared_buffer_bits=" +
        std::to_string(native_buffers.smem_buffer_size_bit) +
        "\nshared_overhead_bits=" +
        std::to_string(native_buffers.smem_overhead_bit) +
        "\nshared_limit_bits=" + std::to_string(smem_limit_bit) +
        "\nmin_threads=" + std::to_string(threads_per_block_min) +
        "\nmax_threads=" + std::to_string(threads_per_block_max) +
        "\nwarp_size=" + std::to_string(dev_prop->warpSize) +
        "\nmax_threads_per_block=" +
        std::to_string(dev_prop->maxThreadsPerBlock) +
        "\nmax_thread_x=" + std::to_string(dev_prop->maxThreadsDim[0]) +
        "\nmax_thread_y=" + std::to_string(dev_prop->maxThreadsDim[1]) +
        "\nmax_grid_y=" + std::to_string(dev_prop->maxGridSize[1]) +
        buffer_bits_state + "\nshared_plan=" +
        planState(*shared_preview) + "\nregister_plan=" +
        planState(*register_preview);
    const std::vector<scheduler::jev::ChoiceOption> options = {
        {"shared", "Keep all persistent buffers in shared memory"},
        {"register", "Place all persistent buffers in registers"}};
    const auto selected = scheduler::jev::chooseDecision(
        "IOP3.storage",
        state,
        options,
        jev_mode == scheduler::jev::Mode::Record ? "shared" : "");
    NVF_ERROR(
        selected == "shared" || selected == "register",
        "Unknown IOP3 storage placement");
    return makeMultiWave(
        selected == "shared" ? native_buffers : register_buffers,
        inner_vector_width,
        enable_jev_choice);
  };
  auto chooseMultiWave = [&](const auto& native_buffers,
                             bool enable_jev_choice) {
    // The layout helper supplies the alignment maximum. Recompute storage for
    // every narrower width: rounding a persistent buffer into shared memory
    // depends on the width, even when its placement does not change.
    if (!enable_jev_choice || jev_mode == scheduler::jev::Mode::Off ||
        hp != nullptr) {
      return makeMultiWave(
          native_buffers, vectorize_factor, enable_jev_choice);
    }

    const auto* dev_prop = at::cuda::getCurrentDeviceProperties();
    const bool valid_bounds =
        vectorize_factor >= 1 && vectorize_factor <= 128 &&
        (vectorize_factor & (vectorize_factor - 1)) == 0 &&
        properties.total_reduction_numel > 0 &&
        properties.total_reduction_numel % vectorize_factor == 0 &&
        properties.total_iteration_numel > 0 &&
        max_outer_reduction_dtype_size_bit >= 8 &&
        max_outer_reduction_dtype_size_bit <= 128 &&
        (max_outer_reduction_dtype_size_bit &
         (max_outer_reduction_dtype_size_bit - 1)) == 0 &&
        native_buffers.has_enough_regs_and_smem;
    std::vector<scheduler::jev::ChoiceOption> options;
    std::string state =
        "inner_outer_multi_wave.inner_width.v1\ninner_dim=" +
        std::to_string(properties.total_reduction_numel) + "\nouter_dim=" +
        std::to_string(properties.total_iteration_numel) +
        "\nmax_layout_width=" + std::to_string(vectorize_factor) +
        "\ntmp_dtype_bits=" +
        std::to_string(max_outer_reduction_dtype_size_bit) +
        "\nmax_threads_per_block=" +
        std::to_string(dev_prop->maxThreadsPerBlock) +
        "\nmax_thread_x=" + std::to_string(dev_prop->maxThreadsDim[0]) +
        "\nmax_thread_y=" + std::to_string(dev_prop->maxThreadsDim[1]) +
        "\nmax_grid_y=" + std::to_string(dev_prop->maxGridSize[1]) +
        "\nsmem_limit_bits=" +
        std::to_string((int64_t)dev_prop->sharedMemPerBlockOptin * 8);
    if (valid_bounds) {
      for (int64_t width = 1; width <= vectorize_factor; width *= 2) {
        auto buffers = width == vectorize_factor
            ? native_buffers
            : inner_outer_utils::getPersistentBufferStorageParams(
                  fusion,
                  runtime_info,
                  data_cache,
                  reduction_tvs,
                  width,
                  threads_per_block_min,
                  threads_per_block_max,
                  /*is_warp_specialized=*/false);
        // Hold the storage strategy fixed for this first width experiment.
        // The candidate still uses its own recomputed resource sizes.
        if (!buffers.has_enough_regs_and_smem ||
            buffers.project_to_input != native_buffers.project_to_input ||
            buffers.smem_persistent_buffers !=
                native_buffers.smem_persistent_buffers ||
            buffers.regs_buffer_size_bit !=
                native_buffers.regs_buffer_size_bit ||
            buffers.smem_overhead_bit !=
                native_buffers.smem_overhead_bit) {
          continue;
        }
        try {
          auto preview = makeMultiWave(
              buffers, width, /*enable_jev_choice=*/false);
          if (!hardValidMultiWavePlan(*preview, buffers) ||
              preview->unroll_factor_inner_reduction != width ||
              preview->vectorization_factor_outer < 1 ||
              preview->vectorization_factor_tmp_gmem_write <
                  preview->vectorization_factor_outer ||
              preview->vectorization_factor_tmp_gmem_write > width) {
            continue;
          }
          const auto& launch = preview->lparams;
          options.push_back(
              {"w" + std::to_string(width),
               "Use inner reduction vector width " + std::to_string(width)});
          state += "\nplan=" + std::to_string(width) + ":" +
              std::to_string(buffers.regs_buffer_size_bit) + ":" +
              std::to_string(buffers.smem_buffer_size_bit) + ":" +
              std::to_string(buffers.smem_overhead_bit) + ":" +
              std::to_string(launch.bdimx()) + ":" +
              std::to_string(launch.bdimy()) + ":" +
              std::to_string(launch.gdimy()) + ":" +
              std::to_string(preview->batches_per_block_inner_reduction) +
              ":" +
              std::to_string(preview->vectorization_factor_outer) + ":" +
              std::to_string(preview->vectorization_factor_tmp_gmem_write) +
              ":" + std::to_string(preview->cparams.maxrregcount);
        } catch (const std::exception&) {
          // An unconstructible native preview cannot be offered.
        }
      }
    }
    const std::string native_id = "w" + std::to_string(vectorize_factor);
    const bool native_offered = std::any_of(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == native_id;
        });
    if (!native_offered) {
      NVF_ERROR(
          jev_mode != scheduler::jev::Mode::Remote,
          "IOP3.inner_width native width has no complete hard-valid plan");
      scheduler::jev::recordCoverageSkip(
          "IOP3.inner_width", "native width lacks a complete hard-valid plan");
      return makeMultiWave(
          native_buffers, vectorize_factor, enable_jev_choice);
    }
    if (options.size() == 1) {
      // A proved native singleton has no performance choice to send remotely.
      scheduler::jev::recordNoChoiceDecision(
          "IOP3.inner_width",
          native_id,
          "only the native width is hard-valid",
          "native_inner_width");
      return chooseMultiWaveStorage(
          native_buffers, vectorize_factor, enable_jev_choice);
    }
    if (state.size() >= 48 * 1024) {
      NVF_ERROR(
          jev_mode != scheduler::jev::Mode::Remote,
          "IOP3.inner_width choice state exceeds the bounded request size");
      scheduler::jev::recordCoverageSkip(
          "IOP3.inner_width", "choice state exceeds bounded request size");
      return makeMultiWave(
          native_buffers, vectorize_factor, enable_jev_choice);
    }
    const std::string selected = scheduler::jev::chooseDecision(
        "IOP3.inner_width",
        state,
        options,
        jev_mode == scheduler::jev::Mode::Record ? native_id : "");
    const auto choice = std::find_if(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == selected;
        });
    NVF_ERROR(choice != options.end(), "Unknown IOP3 inner vector width");
    const int64_t selected_width = std::stoll(selected.substr(1));
    NVF_ERROR(selected_width > 0, "Invalid IOP3 inner vector width");
    if (selected_width == vectorize_factor) {
      return chooseMultiWaveStorage(
          native_buffers, vectorize_factor, enable_jev_choice);
    }
    auto buffers = inner_outer_utils::getPersistentBufferStorageParams(
        fusion,
        runtime_info,
        data_cache,
        reduction_tvs,
        selected_width,
        threads_per_block_min,
        threads_per_block_max,
        /*is_warp_specialized=*/false);
    NVF_ERROR(
        buffers.has_enough_regs_and_smem &&
            buffers.project_to_input == native_buffers.project_to_input &&
            buffers.smem_persistent_buffers ==
                native_buffers.smem_persistent_buffers &&
            buffers.regs_buffer_size_bit == native_buffers.regs_buffer_size_bit &&
            buffers.smem_overhead_bit == native_buffers.smem_overhead_bit,
        "Selected IOP3 width changed persistent buffer placement");
    return chooseMultiWaveStorage(
        buffers, selected_width, enable_jev_choice);
  };
  const bool attempt_ws = user_enforced_warp_specialization ||
      (jev_mode == scheduler::jev::Mode::Off
           ? preferWarpSpecialized(
                 fusion,
                 properties.total_iteration_numel,
                 n_inner_reduction_tvs)
           : canTryWarpSpecialized(fusion));
  if (attempt_ws) {
    auto buffer_params = inner_outer_utils::getPersistentBufferStorageParams(
        fusion,
        runtime_info,
        data_cache,
        reduction_tvs,
        vectorize_factor,
        threads_per_block_min,
        threads_per_block_max,
        /*is_warp_specialized=*/true);

    // Current implementation assumes persistent buffers are projected to
    // inputs, making TMA loading beneficial. If not, shared memory persistent
    // buffers cannot use TMA since their producers are not inputs.
    if (buffer_params.project_to_input) {
      // TMA load requires size to be multiple of 16 bytes (128 bits).
      // In inner-outer scheduler, only shared memory persistent buffers are
      // TMA loaded and tma load size equals to the buffer size.
      if (std::ranges::all_of(
              buffer_params.smem_persistent_buffers,
              [&runtime_info, &persistent_buffer_info](TensorView* buffer) {
                int64_t buffer_size_regs_bit =
                    scheduler_utils::getPersistentBufferSizeBitOfTensor(
                        buffer, runtime_info, persistent_buffer_info);
                return buffer_size_regs_bit % 128 == 0;
              })) {
        const bool execute_ws_now =
            jev_mode == scheduler::jev::Mode::Off ||
            user_enforced_warp_specialization;
        auto rparams = makeWarpSpecialized(
            buffer_params, execute_ws_now, execute_ws_now);

        // A user override is an explicit instruction, not a model choice.
        if (user_enforced_warp_specialization) {
          return rparams;
        }
        if (jev_mode == scheduler::jev::Mode::Off) {
          if (rparams->is_good_ws_heuristic) {
            return rparams;
          }
        } else {
          const bool native_ws =
              jev_mode == scheduler::jev::Mode::Record &&
              preferWarpSpecialized(
                  fusion,
                  properties.total_iteration_numel,
                  n_inner_reduction_tvs) &&
              rparams->is_good_ws_heuristic;
          const auto ws_measurements = hardValidWarpSpecializedPlan(
              *rparams,
              buffer_params,
              properties.total_reduction_numel,
              max_outer_reduction_dtype_size_bit,
              vectorize_factor);
          if (!ws_measurements) {
            if (native_ws) {
              scheduler::jev::recordCoverageSkip(
                  "IOP1.mode", "native WS plan fails conservative hard bounds");
              return makeWarpSpecialized(
                  buffer_params, /*debug=*/true, /*choice=*/true);
            }
            NVF_ERROR(
                jev_mode != scheduler::jev::Mode::Remote,
                "No hard-valid IOP1 warp-specialized alternative");
          } else {
            // Build a complete native multiwave preview. Its constructor does
            // not touch Fusion; disable only its nested IOP3 Choice and debug
            // output. Once IOP1 chooses multiwave, run it normally so IOP3 is
            // logged only for the executed branch.
            const auto& multiwave_buffers = multiwaveBuffers();
            bool multiwave_valid = false;
            if (multiwave_buffers.has_enough_regs_and_smem) {
              try {
                auto preview = makeMultiWave(
                    multiwave_buffers,
                    vectorize_factor,
                    /*enable_jev_choice=*/false);
                multiwave_valid =
                    hardValidMultiWavePlan(*preview, multiwave_buffers);
              } catch (const std::exception&) {
                // An unconstructible plan is never offered to the model.
              }
            }
            if (!multiwave_valid) {
              if (jev_mode == scheduler::jev::Mode::Record) {
                scheduler::jev::recordCoverageSkip(
                    "IOP1.mode", "multiwave preflight fails hard bounds");
                if (native_ws) {
                  return makeWarpSpecialized(
                      buffer_params, /*debug=*/true, /*choice=*/true);
                }
              }
              NVF_ERROR(false, "No hard-valid IOP1 multiwave alternative");
            }

            const auto& ws = *ws_measurements;
            const std::string state =
                "inner_outer.mode.v1\niteration_numel=" +
                std::to_string(properties.total_iteration_numel) +
                "\ninner_reduction_tvs=" +
                std::to_string(n_inner_reduction_tvs) + "\nsm_count=" +
                std::to_string(
                    at::cuda::getCurrentDeviceProperties()->multiProcessorCount) +
                "\ninner_dim_numel=" +
                std::to_string(properties.total_reduction_numel) +
                "\nvector_width=" + std::to_string(vectorize_factor) +
                "\nws_bdimx=" + std::to_string(ws.compute_bdimx) +
                "\nws_computation_groups=" +
                std::to_string(ws.computation_groups) + "\nws_stages=" +
                std::to_string(ws.stages) + "\nws_total_threads=" +
                std::to_string(ws.total_threads) +
                "\nws_available_registers_per_thread=" +
                std::to_string(ws.available_registers_per_thread) +
                "\nws_non_circular_gmem_to_regs=" +
                std::to_string(rparams->is_non_circular_buffer_gmem_to_regs) +
                "\nws_non_circular_buffer_bits=" +
                std::to_string(
                    buffer_params.non_circular_buffered_smem_size_bit) +
                "\nws_register_buffer_bits=" +
                std::to_string(buffer_params.regs_buffer_size_bit);
            const std::vector<scheduler::jev::ChoiceOption> options = {
                {"multiwave", "Use the complete multiwave persistent plan"},
                {"ws", "Use the complete TMA warp-specialized persistent plan"}};
            const std::string selected = scheduler::jev::chooseDecision(
                "IOP1.mode",
                state,
                options,
                jev_mode == scheduler::jev::Mode::Record
                    ? (native_ws ? "ws" : "multiwave")
                    : "");
            if (selected == "ws") {
              auto selected_ws = makeWarpSpecialized(
                  buffer_params, /*debug=*/true, /*choice=*/true);
              if (jev_mode == scheduler::jev::Mode::Remote &&
                  !selected_ws->is_good_ws_heuristic) {
                selected_ws->tag =
                    "TMA Warp Specialized Persistent Heuristic "
                    "(IOP1 Choice selected despite native quality estimate).\n";
              }
              return selected_ws;
            }
            return chooseMultiWave(
                multiwave_buffers, /*enable_jev_choice=*/true);
          }
        }
      }
    }
  }

  // Fallback to multi-wave
  return chooseMultiWave(multiwaveBuffers(), /*enable_jev_choice=*/true);
}

} // namespace

bool InnerOuterPersistentKernelScheduler::canScheduleCompileTime(
    Fusion* fusion) {
  FUSER_PERF_SCOPE(
      "InnerOuterPersistentKernelScheduler::canScheduleCompileTime");

  for (auto tv : fusion->allTvs()) {
    if (tv->dtype() != DataType::Index &&
        dataTypeSizeBit(tv->dtype()) % 8 != 0) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Does not support sub-byte data types.");
      return false;
    }
  }

  // common checks for all persistent heuristics
  if (!normalization_scheduler_utils::checkOpsAndInputs(
          fusion, schedulerType())) {
    return false;
  }

  // check reduction type
  auto reduction_tvs = scheduler_utils::getReductionTvs(fusion);
  if (reduction_tvs.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no reduction tv");
    return false;
  }
  auto reduction_type =
      reduction_scheduler_utils::getReductionType(reduction_tvs);
  const SchedulerType persistent_heuristic =
      normalization_scheduler_utils::getPersistentHeuristicFor(reduction_type);
  if (persistent_heuristic != schedulerType()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "schedulerType() doesn't match with reduction type `",
        persistent_heuristic,
        "`.");
    return false;
  }
  std::vector<TensorView*> inner_reduction_tvs;
  std::vector<TensorView*> outer_reduction_tvs;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      inner_reduction_tvs.emplace_back(tv);
    } else {
      outer_reduction_tvs.emplace_back(tv);
    }
  }

  // check connections between inner reduction and outer reduction tvs.
  if (!normalization_scheduler_utils::checkIfReductionsAreInnerOuter(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction tensor should be "
        "[I,I,...,R,R] and outer reduction tensor should be [R,R,...,I,I]");
    return false;
  }

  if (!normalization_scheduler_utils::hasSharedInput(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should "
        "have shared input.");
    return false;
  }

  if (!normalization_scheduler_utils::isConnectedOnlyThroughReductionProducer(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, inner reduction and outer reduction should "
        "not have shared consumer, their consumers should not have shared "
        "non-outer-reduction producer.");
    return false;
  }

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }
    // Persistent scheduler simply uses reference_tv as the reference, if
    // that changes, this needs to be changed.
    auto reference_tv = inner_reduction_tvs[0];
    if (registry_utils::reductionInterferingView(
            fusion, ca_map, reference_tv)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "View may interfere with normalization scheduling.");
      return false;
    }
  }

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
            "inconsistent reduction root size: ",
            red->toString(),
            ", expected: ",
            axis_count);
        return false;
      }
    }
  }

  // the reduction axis of outer reduction tv should match to the iteration axis
  // of the inner reduction tv.
  if (!normalization_scheduler_utils::isReductionIterationAxisMatched(
          inner_reduction_tvs, outer_reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "to use combined reduction, every iteration axis in inner reduction tv "
        "should match to a reduction domain in outer reduction tv.");
    return false;
  }

  if (!normalization_scheduler_utils::checkReductionPattern(
          fusion, schedulerType(), inner_reduction_tvs, outer_reduction_tvs)) {
    return false;
  }

  // Only accept persistent kernels
  auto persistent_buffer_info = scheduler_utils::persistentBuffers(fusion);
  if (persistent_buffer_info.persistent_buffers.empty()) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no persistent buffer identified");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasNonNormalizePostReductionBCast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "unsupported post reduction normalization");
    return false;
  }

  if (registry_utils::SchedulerTopologyChecker::
          hasGatherToBroadcastBeforeReduction(fusion, reduction_tvs)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "has unsupported gather-like ops before normalization");
    return false;
  }

  return true;
}

bool InnerOuterPersistentKernelScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::canScheduleRunTime");
  auto reduction_tv_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReductionTVs>(
          data_cache, [&fusion]() {
            return std::make_unique<std::vector<TensorView*>>(
                scheduler_utils::getReductionTvs(fusion));
          });

  auto& reduction_tvs = reduction_tv_entry.get();
  TensorView* first_inner_reduction_tv = nullptr;
  for (auto tv : reduction_tvs) {
    if (scheduler_utils::isFastestDimReduction(tv)) {
      first_inner_reduction_tv = tv;
      break;
    }
  }
  auto reference_tv = first_inner_reduction_tv;

  auto properties = scheduler_utils::getReductionProperties(
      fusion, runtime_info, reference_tv);

  const int64_t warp_size = at::cuda::getCurrentDeviceProperties()->warpSize;

  auto reduced_tv = ir_utils::getSoleProducerTv(reference_tv);
  const auto vectorize_factor = vectorize_helper::getVectorizationFactor(
      runtime_info,
      reduced_tv,
      data_cache,
      (int)(reduced_tv->nDims() - properties.inner_most_dimension_ndims));

  // check if there is enough register and shared memory for persistence
  const auto buffer_params =
      inner_outer_utils::getPersistentBufferStorageParams(
          fusion,
          runtime_info,
          data_cache,
          reduction_tvs,
          vectorize_factor,
          InnerOuterPersistentKernelScheduler::threads_per_block_min,
          InnerOuterPersistentKernelScheduler::threads_per_block_max,
          isOptionEnabled(EnableOption::WarpSpecializedNormalization));

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (!buffer_params.has_enough_regs_and_smem) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "not enough registers or shared memory for persistence");
    return false;
  }

  const int64_t device_max_threads_per_multiprocessor =
      (int64_t)at::cuda::getCurrentDeviceProperties()
          ->maxThreadsPerMultiProcessor;

  const int64_t required_sm_per_norm = ceilDiv(
      buffer_params.regs_buffer_size_bit,
      scheduler_utils::register_file_size_bit);

  // If the persistence requires over half the device don't do grid
  // persistence as we can't overlap the grid comms.
  if (required_sm_per_norm >
      scheduler_utils::safeDiv(device_multiprocessor_count, 2)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "requires over half GPU persistence.");
    return false;
  }

  // Don't go persistent if we can't use a small fraction of the
  // available SMs yet have a large reduction size.
  if ( // Large reduction dim
      properties.total_reduction_numel >=
          device_max_threads_per_multiprocessor * 4 &&
      properties.total_iteration_numel <
          (properties.fastest_dim_reduction
               ? scheduler_utils::safeDiv(device_multiprocessor_count, 8)
               // Make sure we at least use a quarter of the device * a
               // half warp
               : (warp_size / 8) * device_multiprocessor_count)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "not enough blocks");
    return false;
  }

  return true;
}

std::unique_ptr<HeuristicParams> InnerOuterPersistentKernelScheduler::
    computeHeuristicsImpl(
        Fusion* fusion,
        SchedulerRuntimeInfo& runtime_info,
        HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::computeHeuristics");
  auto rparams = getInnerOuterPersistentHeuristics(
      fusion, runtime_info, data_cache, getSchedulerHyperParameters());
  NVF_ERROR(rparams != nullptr);
  return rparams;
}

void InnerOuterPersistentKernelScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("InnerOuterPersistentKernelScheduler::schedule");
  auto rparams = dynamic_cast<const ReductionParams*>(params);
  NVF_ERROR(
      rparams != nullptr && rparams->scheduler_type == schedulerType(),
      "Incorrect parameters sent to "
      "InnerOuterPersistentKernelScheduler::schedule",
      params);
  if (rparams->tma_warp_specialized) {
    inner_outer_tma_warp_specialized::scheduleFusion(fusion, rparams);
  } else {
    inner_outer_multi_wave::scheduleFusion(fusion, rparams);
  }
}
} // namespace nvfuser
