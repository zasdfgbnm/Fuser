// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "scheduler/jev_client.h"

namespace nvfuser::reduction::non_tma::inner_width_choice {

// The non-power-of-two layout cap is hard-valid, but its description must not
// reveal that the source heuristic chooses it. Legacy text requests retain
// their original wording for replay compatibility.
inline scheduler::jev::ChoiceOption layoutCapOption(
    int64_t width,
    bool semantic_v3) {
  if (width < 1) {
    throw std::invalid_argument("RD3 layout width cap must be positive");
  }
  return {
      "w" + std::to_string(width),
      semantic_v3
          ? "Vectorize " + std::to_string(width) +
              " element(s) per thread at the layout width cap"
          : "Use the native layout width cap"};
}

} // namespace nvfuser::reduction::non_tma::inner_width_choice
