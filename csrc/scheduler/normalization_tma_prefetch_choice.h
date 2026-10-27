// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "scheduler/jev_client.h"

namespace nvfuser::normalization_inner::tma {

struct PrefetchDistanceRange {
  int64_t low;
  int64_t high;

  std::string id() const {
    if (low == high) {
      return "p" + std::to_string(low);
    }
    return "r" + std::to_string(low) + "_" + std::to_string(high);
  }

  std::string description() const {
    if (low == high) {
      return "Prefetch TMA " + std::to_string(low) + " iteration(s) ahead";
    }
    return "Prefetch TMA in [" + std::to_string(low) + ", " +
        std::to_string(high) + "] iteration(s) ahead";
  }
};

// Partition an inclusive distance range into at most 255 contiguous choices.
// Division before addition keeps the stride calculation within int64_t.
inline std::vector<PrefetchDistanceRange> prefetchDistanceRanges(
    int64_t low,
    int64_t high) {
  if (low < 1 || high < low) {
    throw std::invalid_argument("invalid IP3.prefetch distance range");
  }
  constexpr int64_t max_options = 255;
  const int64_t span = high - low + 1;
  const int64_t stride = span / max_options + (span % max_options != 0);
  std::vector<PrefetchDistanceRange> ranges;
  for (int64_t first = low;;) {
    const int64_t last = first + std::min(stride - 1, high - first);
    ranges.push_back({first, last});
    if (last == high) {
      break;
    }
    first = last + 1;
  }
  return ranges;
}

// The stage menu can choose values above 256, while one Choice supports at
// most 255 options. Each request names its current inclusive range, and every
// positive prefetch distance remains reachable through the hierarchy.
inline int64_t chooseLargePrefetchDistance(int64_t stage_count) {
  if (stage_count <= 256) {
    throw std::invalid_argument("IP3.prefetch v2 needs more than 256 stages");
  }
  const auto jev_mode = scheduler::jev::mode();
  if (jev_mode == scheduler::jev::Mode::Off) {
    throw std::invalid_argument("IP3.prefetch v2 requires Record or Remote");
  }
  const int64_t native_distance = stage_count - 1;
  int64_t low = 1;
  int64_t high = native_distance;
  for (;;) {
    const auto ranges = prefetchDistanceRanges(low, high);
    std::vector<scheduler::jev::ChoiceOption> options;
    options.reserve(ranges.size());
    std::string native_choice;
    for (const auto& range : ranges) {
      options.push_back({range.id(), range.description()});
      if (jev_mode == scheduler::jev::Mode::Record &&
          range.low <= native_distance && native_distance <= range.high) {
        native_choice = range.id();
      }
    }
    const std::string state =
        "inner_persistent.tma_prefetch.v2\nstage_count=" +
        std::to_string(stage_count) + "\nrange_low=" + std::to_string(low) +
        "\nrange_high=" + std::to_string(high);
    const auto selected = scheduler::jev::chooseDecision(
        "IP3.prefetch", state, options, native_choice);
    const auto found = std::find_if(
        ranges.begin(), ranges.end(), [&](const auto& range) {
          return range.id() == selected;
        });
    if (found == ranges.end()) {
      throw std::runtime_error("IP3.prefetch response is outside the range");
    }
    if (found->low == found->high) {
      return found->low;
    }
    low = found->low;
    high = found->high;
  }
}

} // namespace nvfuser::normalization_inner::tma
