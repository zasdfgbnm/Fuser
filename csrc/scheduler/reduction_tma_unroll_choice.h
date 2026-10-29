// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "scheduler/jev_client.h"

namespace nvfuser::reduction::tma {

// An eligible native TMA reduction can still prefer more unrolling than the
// per-thread Choice menu exposes. Record that gap without changing its plan.
inline std::optional<std::string> chooseInnerUnrollIfRepresentable(
    std::string_view state,
    const std::vector<scheduler::jev::ChoiceOption>& options,
    int64_t native_unroll_factor) {
  const auto jev_mode = scheduler::jev::mode();
  const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
      ? "u" + std::to_string(native_unroll_factor)
      : std::string();
  if (jev_mode == scheduler::jev::Mode::Record &&
      std::none_of(options.begin(), options.end(), [&](const auto& option) {
        return option.id == native_choice;
      })) {
    scheduler::jev::recordCoverageSkip(
        "RD2.unroll", "native unroll exceeds per-thread reduction menu");
    return std::nullopt;
  }
  return scheduler::jev::chooseDecision(
      "RD2.unroll", state, options, native_choice);
}

} // namespace nvfuser::reduction::tma
