// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "scheduler/jev_client.h"

namespace nvfuser::inner_outer_mode_preflight {

enum class Action {
  OfferChoice,
  PreserveNativeWs,
  PreserveNativeMultiwave,
  RejectRemote,
};

// Called only after the WS plan passed its conservative hard check. If its
// multiwave alternative cannot be proved, Record must retain whichever plan
// Off would have selected; Remote cannot expose an unproved two-plan menu.
constexpr Action actionForValidWs(
    scheduler::jev::Mode mode,
    bool multiwave_hard_valid,
    bool native_ws) noexcept {
  if (multiwave_hard_valid) {
    return Action::OfferChoice;
  }
  if (mode == scheduler::jev::Mode::Record) {
    return native_ws ? Action::PreserveNativeWs
                     : Action::PreserveNativeMultiwave;
  }
  return Action::RejectRemote;
}

} // namespace nvfuser::inner_outer_mode_preflight
