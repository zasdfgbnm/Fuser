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

} // namespace nvfuser::pointwise::non_tma
