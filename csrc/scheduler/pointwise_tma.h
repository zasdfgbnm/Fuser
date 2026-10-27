// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <string>
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
// and emits no SchedulerDebug. It does not certify that later Remote child
// menus are available for each alternative.
struct TmaBreakPointAvailability {
  std::string state;
  std::vector<scheduler::jev::ChoiceOption> options;
  std::vector<pointwise_utils::BreakPointInfo> plans;
};

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
