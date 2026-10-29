// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "scheduler/jev_client.h"

namespace nvfuser::pointwise::non_tma {

struct PointwiseUnrollEnvelope {
  int64_t model_limit;
  int64_t native_factor;
  bool offer_choice;
};

// Caller has already proved 1 <= native_factor <= max(1, total_blocks).
// This budget limits model-selected code expansion without changing a native
// schedule that exceeds it. The expression count approximates cloned work;
// the bound is a resource policy, not a register allocation proof.
inline PointwiseUnrollEnvelope getPointwiseUnrollEnvelope(
    int64_t total_blocks,
    std::size_t expression_count,
    int64_t vector_width,
    int64_t native_factor) {
  constexpr int64_t kMaxUnroll = 64;
  constexpr int64_t kExpressionWidthBudget = 8192;
  const int64_t bounded_expression_count = std::max<int64_t>(
      1,
      expression_count >= static_cast<std::size_t>(kExpressionWidthBudget)
          ? kExpressionWidthBudget
          : static_cast<int64_t>(expression_count));
  // Divide in stages so the estimate cannot overflow.
  const int64_t budget_limit = std::max<int64_t>(
      1,
      kExpressionWidthBudget / bounded_expression_count /
          std::max<int64_t>(1, vector_width));
  const int64_t model_limit =
      std::min({std::max<int64_t>(1, total_blocks), kMaxUnroll, budget_limit});
  const bool offer_choice = native_factor <= model_limit && model_limit > 1;
  if (!offer_choice) {
    // This branch runs in both Record and Remote. A one-element menu cannot
    // change the answer; an over-budget native factor must remain unchanged.
    scheduler::jev::recordNoChoiceDecision(
        "PW3.unroll",
        "u" + std::to_string(native_factor),
        native_factor > model_limit
            ? "native factor exceeds bounded model resource envelope"
            : "bounded model menu contains only u1",
        native_factor > model_limit ? "native_unroll" : "bounded_model_menu");
  }
  return {model_limit, native_factor, offer_choice};
}

} // namespace nvfuser::pointwise::non_tma
