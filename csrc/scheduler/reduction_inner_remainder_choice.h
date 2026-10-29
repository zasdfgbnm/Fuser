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

// RD3.unroll chooses a concrete cap before the CTA, warp, and remainder
// planners. The menu depends only on the hard reduction extent; the native
// empirical target is never used to construct a Remote menu.
inline std::vector<scheduler::jev::ChoiceOption> innerUnrollTargetChoices(
    int64_t after_vect) {
  if (after_vect < 1) {
    throw std::invalid_argument("invalid RD3 inner extent");
  }
  std::vector<scheduler::jev::ChoiceOption> options;
  for (int64_t target = 1; target <= after_vect && target <= 16;
       target *= 2) {
    options.push_back({
        "u" + std::to_string(target),
        "Cap inner unroll target at " + std::to_string(target) +
            " element(s) per thread before CTA and remainder planning"});
  }
  return options;
}

// A non-power-of-two or larger native target remains unchanged in Record.
// Such a case is explicitly outside this model experiment, so Remote cannot
// silently approximate the original scheduler's result.
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
  const auto options = innerUnrollTargetChoices(after_vect);
  const std::string native_choice =
      jev_mode == scheduler::jev::Mode::Record
      ? "u" + std::to_string(native_target)
      : "";
  if (jev_mode == scheduler::jev::Mode::Record &&
      std::none_of(options.begin(), options.end(), [&](const auto& option) {
        return option.id == native_choice;
      })) {
    scheduler::jev::recordCoverageSkip(
        "RD3.unroll", "native_target_outside_compact_menu");
    return native_target;
  }
  const auto selected = scheduler::jev::chooseDecision(
      "RD3.unroll", state_base + "candidate_cap=16\n", options, native_choice);
  for (int64_t target = 1; target <= after_vect && target <= 16;
       target *= 2) {
    if (selected == "u" + std::to_string(target)) {
      return target;
    }
  }
  throw std::runtime_error("RD3.unroll response is outside concrete target menu");
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

// The width menu contains the seed plus one CTA width per warp step. When it
// has exactly one physical width, the skipped RD3.remainder_x Choice would
// have selected index u1. Retain that canonical index in semantic-v3 prior
// choices and send the physical width and its enumeration basis to later tail
// requests. The no_choice diagnostic keeps its physical xN spelling.
struct FixedInnerRemainderWidthWitness {
  int64_t bdimx;
  int64_t seed_bdimx;
  int64_t after_vect;
  int64_t initial_bdimx;
  int64_t initial_inner_unroll;
  int64_t target_unroll;
  int64_t target_cta;
  int64_t min_warp;
  int64_t scan_first;

  std::string stateFields() const {
    return "fixed_remainder_x_index=1\nfixed_remainder_x_bdimx=" +
        std::to_string(bdimx) + "\nfixed_remainder_x_seed_bdimx=" +
        std::to_string(seed_bdimx) + "\nfixed_remainder_x_after_vect=" +
        std::to_string(after_vect) + "\nfixed_remainder_x_initial_bdimx=" +
        std::to_string(initial_bdimx) +
        "\nfixed_remainder_x_initial_inner_unroll=" +
        std::to_string(initial_inner_unroll) +
        "\nfixed_remainder_x_target_unroll=" +
        std::to_string(target_unroll) + "\nfixed_remainder_x_target_cta=" +
        std::to_string(target_cta) + "\nfixed_remainder_x_min_warp=" +
        std::to_string(min_warp) + "\nfixed_remainder_x_scan_first=" +
        std::to_string(scan_first) + '\n';
  }
};

inline std::string_view innerGridStateTag(bool fixed_remainder_x) {
  return fixed_remainder_x ? "reduction.inner_grid.v2\n"
                           : "reduction.inner_grid.v1\n";
}

inline std::string_view innerIterUnrollStateTag(bool fixed_remainder_x) {
  return fixed_remainder_x ? "reduction.inner_iter_unroll.v2\n"
                           : "reduction.inner_iter_unroll.v1\n";
}

inline FixedInnerRemainderWidthWitness proveFixedInnerRemainderWidth(
    int64_t after_vect,
    int64_t initial_bdimx,
    int64_t initial_inner_unroll,
    int64_t target_unroll,
    int64_t target_cta,
    int64_t min_warp,
    int64_t scan_first,
    int64_t warp_size,
    int64_t max_threads_per_block,
    int64_t seed_bdimx,
    int64_t selected_bdimx) {
  if (after_vect < 1 || target_unroll < 2 || min_warp < 1 ||
      min_warp > 16 || warp_size < 1 || warp_size > 1024 ||
      min_warp > warp_size || max_threads_per_block < 1 ||
      scan_first != 4 * warp_size || target_cta < scan_first ||
      target_cta > max_threads_per_block || initial_bdimx < 1 ||
      initial_inner_unroll < 1 || seed_bdimx < 1) {
    throw std::invalid_argument(
        "RD3 fixed remainder width lacks a singleton enumeration proof");
  }
  const auto ceil_div = [](int64_t numerator, int64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
  };
  const int64_t expected_initial_bdimx =
      std::min(std::max(after_vect, min_warp), scan_first);
  const int64_t expected_initial_unroll =
      std::min(ceil_div(after_vect, expected_initial_bdimx), target_unroll);
  const int64_t expected_seed =
      std::min(ceil_div(after_vect, expected_initial_unroll), target_cta);
  const int64_t expected_remainder =
      ceil_div(ceil_div(after_vect, expected_seed), expected_initial_unroll);
  if (initial_bdimx != expected_initial_bdimx ||
      initial_inner_unroll != expected_initial_unroll ||
      seed_bdimx != expected_seed || seed_bdimx != scan_first ||
      target_cta - scan_first >= min_warp ||
      selected_bdimx != seed_bdimx || expected_remainder <= 1) {
    throw std::invalid_argument(
        "RD3 fixed remainder width lacks a singleton enumeration proof");
  }
  return {selected_bdimx, seed_bdimx, after_vect,
          initial_bdimx, initial_inner_unroll, target_unroll,
          target_cta, min_warp, scan_first};
}

inline void recordFixedInnerRemainderWidth(
    const FixedInnerRemainderWidthWitness& witness,
    scheduler::jev::Mode jev_mode) {
  if (jev_mode == scheduler::jev::Mode::Off) {
    return;
  }
  scheduler::jev::recordHardSingletonDecision(
      "RD3.remainder_x",
      "u1",
      "single_cta_width",
      jev_mode == scheduler::jev::Mode::Record
          ? "native_remainder_scan"
          : "remote_remainder_seed",
      "x" + std::to_string(witness.bdimx));
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
