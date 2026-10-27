// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "fusion.h"
#include "scheduler/jev_client.h"
#include "scheduler/pointwise.h"
#include "scheduler/pointwise_utils.h"

namespace nvfuser {
namespace pointwise {
namespace tma {
// Conservative PW4.breakpoint alternatives that pass native Off
// materialization. This preview performs no Choices, uses no caller cache,
// and emits no SchedulerDebug. The aligned reasons distinguish materialized
// plans from plans whose PW4 Remote child protocol can be entered.
struct TmaBreakPointAvailability {
  std::string state;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::vector<pointwise_utils::BreakPointInfo> plans;
  // Empty means the current PW4 child menus have a source-preflight route;
  // it is not a guarantee that a Hopper kernel will compile or run.
  std::vector<std::string> remote_unavailability_reasons;
};

// Called only for an alternative that passed native Off materialization.
// This pure gate is also used by the production availability builder.
std::string_view tmaBreakPointRemoteRouteabilityReason(
    std::size_t loop_axes,
    int64_t n_elems,
    int64_t break_point,
    int64_t layout_width,
    int64_t min_dtype_bits);

TmaBreakPointAvailability getTmaBreakPointAvailability(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const pointwise_utils::FusionRuntimeProperties& prop);

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop);

void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams);
} // namespace tma
} // namespace pointwise
} // namespace nvfuser
