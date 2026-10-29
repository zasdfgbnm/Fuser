// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "fusion.h"
#include "scheduler/reduction.h"
#include "scheduler/reduction_utils.h"

namespace nvfuser {
namespace reduction {
namespace non_tma {

// LaunchParams::operator== omits the z dimensions. Native preview parity must
// compare the complete launch, including shared-memory bytes.
std::array<int64_t, 7> nativeLaunchFields(const LaunchParams& params);

// Inputs to the pure outer-grid CTA admission check. The same complete
// baseline and stride-capped plans are checked by production and CPU tests.
struct OuterCtaState {
  int64_t reduction_numel;
  int64_t iteration_numel;
  int64_t max_dtype_bits;
  int64_t sm_count;
  int64_t max_threads_per_sm;
  int64_t max_threads_per_block;
  int64_t max_unroll;
  int64_t vector_width;
  int64_t waves;
  int64_t logical_warp;
};

std::vector<int64_t> hardValidOuterCtaShares(const OuterCtaState& state);

// Preview the two wave-dependent targets consumed by the remaining bounded
// inner 3D planner. Equal targets give identical descendant CTA and grid menus.
struct Inner3DWaveTarget {
  int64_t capped_target_blocks;
  int64_t rounded_target_threads_in_block;

  bool operator==(const Inner3DWaveTarget&) const = default;
};

struct Inner3DWaveCandidate {
  int64_t waves;
  Inner3DWaveTarget target;
};

Inner3DWaveTarget previewInner3DWaveTarget(
    int64_t n_elems,
    int64_t uncapped_target_blocks,
    int64_t initial_target_threads,
    int64_t target_unroll,
    int64_t target_iterations,
    int64_t logical_warp,
    int64_t sm_count,
    int64_t max_threads_per_sm,
    int64_t target_waves);

std::vector<Inner3DWaveCandidate> distinctInner3DWaveCandidates(
    const std::vector<Inner3DWaveCandidate>& hard_valid_candidates);

std::unique_ptr<ReductionParams> getReductionHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const reduction_scheduler_utils::FusionRuntimeProperties& props);

void scheduleReduction(Fusion* fusion, const ReductionParams* pparams);
} // namespace non_tma
} // namespace reduction
} // namespace nvfuser
