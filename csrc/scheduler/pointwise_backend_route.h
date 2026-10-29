// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <string_view>

#include "scheduler/jev_client.h"

namespace nvfuser {
class PointwiseParams;
namespace pointwise_utils {
struct FusionRuntimeProperties;
}
namespace pointwise::tma {
struct TmaBreakPointAvailability;
}
} // namespace nvfuser

namespace nvfuser::pointwise {

// A PW1 route is based on complete cache-free source previews and on the
// current Remote PW4 continuation menu. It does not rank either backend.
enum class BackendRouteAction {
  PreserveNativeTma,
  PreserveNativeOrdinary,
  OfferChoice,
  TmaOnly,
  OrdinaryOnly,
  NoPlan,
};

constexpr BackendRouteAction pointwiseBackendRoute(
    scheduler::jev::Mode mode,
    bool native_tma_available,
    bool native_tma_breakpoint_routeable,
    bool remote_tma_available,
    bool ordinary_available) {
  if (mode == scheduler::jev::Mode::Record && native_tma_available &&
      !native_tma_breakpoint_routeable) {
    return BackendRouteAction::PreserveNativeTma;
  }
  // When native TMA fails but another TMA breakpoint is routeable, the
  // source baseline is ordinary. That fact is not independently derivable
  // from the PW1 wire, so Record skips PW1 rather than leaking a winner.
  if (mode == scheduler::jev::Mode::Record && !native_tma_available) {
    if (!ordinary_available) {
      return BackendRouteAction::NoPlan;
    }
    if (remote_tma_available) {
      return BackendRouteAction::PreserveNativeOrdinary;
    }
  }
  if (remote_tma_available && ordinary_available) {
    return BackendRouteAction::OfferChoice;
  }
  if (remote_tma_available) {
    return BackendRouteAction::TmaOnly;
  }
  if (ordinary_available) {
    return BackendRouteAction::OrdinaryOnly;
  }
  return BackendRouteAction::NoPlan;
}

// Emit the versioned PW1 request from source-computed branch projections.
// The exact same builder is used by the live parent path and source-level
// fixtures on devices where TMA kernel execution is unavailable.
std::string choosePointwiseBackend(
    const pointwise_utils::FusionRuntimeProperties& prop,
    const tma::TmaBreakPointAvailability& remote_tma,
    const PointwiseParams& tma_baseline,
    std::string_view tma_baseline_kind,
    const PointwiseParams& ordinary_baseline,
    std::string_view native_choice);

} // namespace nvfuser::pointwise
