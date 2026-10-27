// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "scheduler/jev_client.h"

namespace nvfuser::reduction::non_tma {

struct InnerUnrollRange {
  int64_t low;
  int64_t high;
  std::string id;
};

// Partition the complete positive int64 domain into at most 255 options per
// request. Division before addition avoids overflow at INT64_MAX.
inline std::vector<InnerUnrollRange> innerUnrollRanges(
    int64_t low,
    int64_t high) {
  if (low < 1 || high < low) {
    throw std::invalid_argument("invalid RD3 inner range");
  }
  constexpr int64_t max_choices = 255;
  const int64_t span = high - low + 1;
  const int64_t stride = span / max_choices + (span % max_choices != 0);
  std::vector<InnerUnrollRange> ranges;
  for (int64_t first = low;;) {
    const int64_t last = first + std::min(stride - 1, high - first);
    const std::string id = first == last
        ? "u" + std::to_string(first)
        : "r" + std::to_string(first) + "_" + std::to_string(last);
    ranges.push_back({first, last, id});
    if (last == high) {
      break;
    }
    first = last + 1;
  }
  return ranges;
}

// The selected RD3.unroll target is passed to CTA, warp, and remainder
// planning. This hierarchy retains the full structural range even when its
// upper endpoint is too large for a single Choice request.
inline int64_t chooseInnerUnrollTarget(
    const std::string& state_base,
    int64_t after_vect,
    int64_t native_target,
    scheduler::jev::Mode jev_mode) {
  if (jev_mode == scheduler::jev::Mode::Off || after_vect < 1 ||
      (jev_mode == scheduler::jev::Mode::Record &&
       (native_target < 1 || native_target > after_vect))) {
    throw std::invalid_argument("invalid RD3.unroll Choice range");
  }
  int64_t low = 1;
  int64_t high = after_vect;
  for (;;) {
    const auto ranges = innerUnrollRanges(low, high);
    std::vector<scheduler::jev::ChoiceOption> options;
    options.reserve(ranges.size());
    std::string native_choice;
    for (const auto& range : ranges) {
      options.push_back(
          {range.id,
           "Inner unroll target in [" + std::to_string(range.low) + ", " +
               std::to_string(range.high) + "]"});
      if (jev_mode == scheduler::jev::Mode::Record &&
          range.low <= native_target && native_target <= range.high) {
        native_choice = range.id;
      }
    }
    const std::string state = state_base +
        "range_low=" + std::to_string(low) + '\n' +
        "range_high=" + std::to_string(high) + '\n';
    const auto selected = scheduler::jev::chooseDecision(
        "RD3.unroll", state, options, native_choice);
    const auto found = std::find_if(
        ranges.begin(), ranges.end(), [&](const InnerUnrollRange& range) {
          return range.id == selected;
        });
    if (found == ranges.end()) {
      throw std::runtime_error("RD3.unroll response is outside reduction bounds");
    }
    if (found->low == found->high) {
      return found->low;
    }
    low = found->low;
    high = found->high;
  }
}

// RD3.remainder_x and RD3.remainder_u share this bounded Choice hierarchy.
// The native value remains inside every Record node and never reaches HTTP.
inline int64_t chooseInnerRemainderRange(
    std::string_view site,
    const std::string& state_base,
    int64_t low,
    int64_t high,
    int64_t native_value,
    scheduler::jev::Mode jev_mode) {
  if (jev_mode == scheduler::jev::Mode::Off || low < 1 || high < low ||
      (jev_mode == scheduler::jev::Mode::Record &&
       (native_value < low || native_value > high))) {
    throw std::invalid_argument("invalid RD3 remainder Choice range");
  }
  for (;;) {
    const auto ranges = innerUnrollRanges(low, high);
    std::vector<scheduler::jev::ChoiceOption> options;
    options.reserve(ranges.size());
    std::string native_choice;
    for (const auto& range : ranges) {
      options.push_back(
          {range.id,
           "Choose integer in [" + std::to_string(range.low) + ", " +
               std::to_string(range.high) + "]"});
      if (jev_mode == scheduler::jev::Mode::Record &&
          range.low <= native_value && native_value <= range.high) {
        native_choice = range.id;
      }
    }
    const std::string state = state_base +
        "range_low=" + std::to_string(low) + '\n' +
        "range_high=" + std::to_string(high) + '\n';
    const auto selected = scheduler::jev::chooseDecision(
        site, state, options, native_choice);
    const auto found = std::find_if(
        ranges.begin(), ranges.end(), [&](const InnerUnrollRange& range) {
          return range.id == selected;
        });
    if (found == ranges.end()) {
      throw std::runtime_error(
          "RD3 remainder response is outside the candidate range");
    }
    if (found->low == found->high) {
      return jev_mode == scheduler::jev::Mode::Record
          ? native_value
          : found->low;
    }
    low = found->low;
    high = found->high;
  }
}

// Unlike the generic (a + b - 1) form, this stays in range for positive
// int64 dimensions close to INT64_MAX.
inline int64_t positiveCeilDiv(int64_t dividend, int64_t divisor) {
  return dividend / divisor + (dividend % divisor != 0);
}

struct InnerRemainderScanState {
  int64_t block_x;
  int64_t unroll;
  int64_t current_remainder;
};

// Replay one native CTA width's descending factor scan exactly. For a fixed
// quotient q = floor(after_block / factor), the remainder decreases as factor
// increases. The largest factor in that quotient plateau is therefore the
// only one that can improve the native strict-minimum comparison or hit zero.
// A zero updates the selected pair but deliberately leaves current_remainder
// unchanged, matching the original scan's behavior at later CTA widths.
inline void scanInnerRemainderWidthByPlateaus(
    int64_t after_block,
    int64_t target_unroll,
    int64_t candidate_x,
    InnerRemainderScanState& state) {
  for (int64_t factor = target_unroll; factor > 1;) {
    const int64_t quotient = after_block / factor;
    const int64_t remainder = after_block % factor;
    if (remainder == 0) {
      state.block_x = candidate_x;
      state.unroll = factor;
      break;
    }
    if (remainder < state.current_remainder) {
      state.current_remainder = remainder;
      state.block_x = candidate_x;
      state.unroll = factor;
    }
    factor = after_block / (quotient + 1);
  }
}

} // namespace nvfuser::reduction::non_tma
