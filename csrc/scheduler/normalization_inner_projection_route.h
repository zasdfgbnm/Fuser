// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "scheduler/jev_client.h"

namespace nvfuser::normalization_inner {

struct ProjectionPropertyRoute {
  bool may_attempt_tma;
  bool enable_projection_on_initial_properties;
  bool recompute_projection_on_non_tma_fallback;
};

// The TMA gate must see native properties. Only the final non-TMA branch may
// make IP1.projection, including on SM90+ when native TMA admission fails.
constexpr ProjectionPropertyRoute projectionPropertyRoute(
    int device_major,
    bool tma_option_enabled,
    scheduler::jev::Mode mode,
    bool non_tma_selected) {
  const bool may_attempt_tma = device_major >= 9 && tma_option_enabled;
  const bool model_enabled = mode != scheduler::jev::Mode::Off;
  return {
      may_attempt_tma,
      model_enabled && !may_attempt_tma,
      model_enabled && may_attempt_tma && non_tma_selected};
}

} // namespace nvfuser::normalization_inner
