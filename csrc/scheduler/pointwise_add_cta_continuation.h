// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>

namespace nvfuser::pointwise::non_tma {

struct AddCtaContinuation {
  int64_t vector_iterations;
  int64_t source_total_blocks;
  int64_t unroll_bound;
  int64_t effective_grid_x;
  int64_t covered_right_positions;
  int64_t tail_right_positions;
};

struct AddLaunch2DContinuation {
  int64_t threads_y;
  int64_t vector_iterations_right;
  int64_t preliminary_grid_left;
  int64_t preliminary_grid_right;
  int64_t unroll_bound;

  constexpr int64_t rightGridAfterUnroll(int64_t unroll) const {
    return preliminary_grid_right / unroll +
        (preliminary_grid_right % unroll != 0);
  }

  constexpr int64_t coveredRightPositions(
      int64_t threads_x,
      int64_t width,
      int64_t unroll) const {
    return rightGridAfterUnroll(unroll) * threads_x * width * unroll;
  }
};

// Exact Add[2,16] 2D CTA continuation. The source's native unroll is u1:
// at most 16 preliminary CTAs fit inside a single SM89 scheduling wave.
// Model unroll may choose every U in 1..unroll_bound; the terminal launch
// follows rightGridAfterUnroll(U) without materializing all leaf plans here.
constexpr std::optional<AddLaunch2DContinuation> addLaunch2DContinuation(
    int64_t left_elems,
    int64_t right_elems,
    int64_t width,
    int64_t threads_x,
    int64_t expression_count) {
  if (left_elems != 2 || right_elems != 16 || expression_count != 1 ||
      (width != 1 && width != 2 && width != 4) || threads_x < 1 ||
      threads_x > 128) {
    return std::nullopt;
  }
  const auto ceil_positive = [](int64_t numerator, int64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
  };
  const int64_t threads_y = 128 / threads_x;
  const int64_t right_vectors = ceil_positive(right_elems, width);
  const int64_t grid_left = ceil_positive(left_elems, threads_y);
  const int64_t grid_right = ceil_positive(right_vectors, threads_x);
  const int64_t total_blocks = grid_left * grid_right;
  const int64_t budget_limit =
      std::max<int64_t>(1, 8192 / expression_count / width);
  const int64_t model_limit =
      std::min({std::max<int64_t>(1, total_blocks), int64_t{64}, budget_limit});
  if (total_blocks < 1 || total_blocks > 16 || model_limit != total_blocks) {
    return std::nullopt;
  }
  return AddLaunch2DContinuation{
      threads_y, right_vectors, grid_left, grid_right, model_limit};
}

// A compact continuation for the exact 32-element FP32 Add carrier. Its
// source total-block count is one for every offered width and 1D CTA, so the
// later PW3.unroll site has only its no-choice u1 result.
constexpr std::optional<AddCtaContinuation> addCtaContinuation(
    int64_t n_elems,
    int64_t width,
    int64_t cta_threads,
    int64_t expression_count) {
  if (n_elems != 32 || expression_count != 1 ||
      (width != 1 && width != 2 && width != 4) ||
      (cta_threads != 32 && cta_threads != 64 && cta_threads != 128)) {
    return std::nullopt;
  }
  const auto ceil_positive = [](int64_t numerator, int64_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
  };
  const int64_t vector_iterations = ceil_positive(n_elems, width);
  // Match the source's floor-divided pre-unroll block estimate exactly.
  const int64_t total_blocks = ceil_positive(n_elems / width, cta_threads);
  const int64_t budget_limit = 8192 / expression_count / width;
  const int64_t unroll_bound =
      std::min({std::max<int64_t>(1, total_blocks), int64_t{64}, budget_limit});
  if (unroll_bound != 1) {
    return std::nullopt;
  }
  const int64_t grid_x =
      ceil_positive(ceil_positive(vector_iterations, cta_threads), 1);
  const int64_t covered = grid_x * cta_threads * width;
  return AddCtaContinuation{
      vector_iterations,
      total_blocks,
      unroll_bound,
      grid_x,
      covered,
      covered - n_elems};
}

struct AddWidthContinuation {
  int64_t cta_count;
  int64_t terminal_leaf_count;
  int64_t max_unroll_bound;
};

struct AddBreakpointContinuation {
  int64_t width_count;
  int64_t cta_count_per_width;
  int64_t terminal_leaf_count;
  int64_t max_unroll_bound;
};

// Count the complete bounded descendants of one width without constructing
// or scheduling any losing CTA branch. The exact carrier has three 1D CTA
// leaves, or all 128 source-hard-valid 2D CTA splits with 1..B unroll leaves.
constexpr std::optional<AddWidthContinuation> addWidthContinuation(
    int64_t break_point,
    int64_t width) {
  if (break_point == 0) {
    int64_t leaves = 0;
    for (const int64_t x : {32L, 64L, 128L}) {
      const auto c = addCtaContinuation(32, width, x, 1);
      if (!c.has_value()) {
        return std::nullopt;
      }
      leaves += c->unroll_bound;
    }
    return AddWidthContinuation{3, leaves, 1};
  }
  if (break_point == 1) {
    int64_t leaves = 0;
    int64_t max_bound = 0;
    for (int64_t x = 1; x <= 128; ++x) {
      const auto c = addLaunch2DContinuation(2, 16, width, x, 1);
      if (!c.has_value()) {
        return std::nullopt;
      }
      leaves += c->unroll_bound;
      max_bound = std::max(max_bound, c->unroll_bound);
    }
    return AddWidthContinuation{128, leaves, max_bound};
  }
  return std::nullopt;
}

// Summarize all descendants of one legal breakpoint without invoking their
// analyzers. The same three hard widths are valid for all exact Add branches.
constexpr std::optional<AddBreakpointContinuation> addBreakpointContinuation(
    int64_t break_point) {
  if (break_point != 0 && break_point != 1) {
    return std::nullopt;
  }
  int64_t leaves = 0;
  int64_t max_bound = 0;
  for (const int64_t width : {1L, 2L, 4L}) {
    const auto c = addWidthContinuation(break_point, width);
    if (!c.has_value()) {
      return std::nullopt;
    }
    leaves += c->terminal_leaf_count;
    max_bound = std::max(max_bound, c->max_unroll_bound);
  }
  return AddBreakpointContinuation{
      3, break_point == 0 ? 3 : 128, leaves, max_bound};
}

} // namespace nvfuser::pointwise::non_tma
