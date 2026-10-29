// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>

namespace nvfuser::inner_outer_tma_warp_specialized::tmp_width {

struct Bounds {
  int64_t hard_max = 0;
  int64_t native_read = 0;
};

inline Bounds bounds(
    int64_t inner_dim,
    int64_t inner_vector_width,
    int64_t tmp_dtype_bits) {
  if (inner_dim <= 0 || inner_vector_width <= 0 || tmp_dtype_bits <= 0) {
    return {};
  }
  const int64_t hard_max =
      std::min<int64_t>(128 / tmp_dtype_bits, inner_vector_width);
  return {
      hard_max,
      std::min<int64_t>(inner_dim >= 4096 ? 4 : 2, hard_max)};
}

struct ReadPlan {
  bool hard_valid = false;
  bool combined_split_grid_inner_dim = false;
  int64_t tmp_write_width_seed = 0;
  bool tmp_write_choice_pending = false;
};

inline ReadPlan readPlan(
    int64_t width,
    int64_t inner_dim,
    int64_t inner_vector_width,
    int64_t tmp_dtype_bits,
    int64_t bdimx,
    int64_t gdimy,
    bool fixed_ws_plan_hard_valid) {
  const Bounds limit = bounds(inner_dim, inner_vector_width, tmp_dtype_bits);
  if (!fixed_ws_plan_hard_valid || width <= 0 ||
      (width & (width - 1)) != 0 || limit.hard_max <= 0 ||
      (limit.hard_max & (limit.hard_max - 1)) != 0 ||
      width > limit.hard_max || limit.hard_max % width != 0 ||
      inner_vector_width % width != 0 || inner_dim % width != 0 ||
      width > 128 / tmp_dtype_bits || bdimx <= 0 || gdimy <= 0) {
    return {};
  }
  // Avoid overflow in the continuation's original width * x * grid test.
  const bool split = width <= (inner_dim - 1) / bdimx / gdimy;
  return {true, split, limit.hard_max, width < limit.hard_max};
}

} // namespace nvfuser::inner_outer_tma_warp_specialized::tmp_width
