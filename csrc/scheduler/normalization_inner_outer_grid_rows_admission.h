// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>

#include "scheduler/jev_client.h"

namespace nvfuser::inner_outer_multi_wave {

enum class GridRowsAdmission {
  OfferWithNative,
  OfferWithoutNative,
  PreserveNative,
  RejectRemote,
};

// The caller counts only thresholds whose entire downstream CTA search has
// passed hard preflight. The tuned eight-row target is a Record preference,
// not a legality requirement for a Remote alternative.
constexpr GridRowsAdmission gridRowsAdmission(
    scheduler::jev::Mode mode,
    std::size_t offered_count,
    bool native_offered,
    bool request_fits) noexcept {
  if (mode == scheduler::jev::Mode::Record) {
    return native_offered && request_fits
        ? GridRowsAdmission::OfferWithNative
        : GridRowsAdmission::PreserveNative;
  }
  if (mode == scheduler::jev::Mode::Remote) {
    if (offered_count == 0 || !request_fits) {
      return GridRowsAdmission::RejectRemote;
    }
    return native_offered ? GridRowsAdmission::OfferWithNative
                          : GridRowsAdmission::OfferWithoutNative;
  }
  return GridRowsAdmission::PreserveNative;
}

} // namespace nvfuser::inner_outer_multi_wave
