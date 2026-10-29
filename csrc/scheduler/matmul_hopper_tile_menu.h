// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace nvfuser::matmul_utils::hopper_tile_menu {

// Numeric Hopper tile admission, independent of a device or a Fusion. The
// three-stage condition is the legacy scoring policy; a low-stage candidate
// must also fit its operands and a Float accumulator before it is offered.
struct Candidate {
  int64_t cta_m;
  int64_t cta_n;
  int64_t cta_k;
  int64_t instr_n;
  int64_t stages_per_sm;
  bool native_valid;
};

struct Menu {
  std::vector<Candidate> candidates;
  bool has_native_config = false;
};

inline int64_t ceilDivPositive(int64_t numerator, int64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

inline Menu enumerate(
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t num_sms,
    int64_t smem_bytes,
    bool offer_low_stage_tiles) {
  Menu menu;
  // Keep the original traversal order: equal native scores prefer the first
  // larger-K candidate, and option ordering is part of the Choice contract.
  for (int64_t cta_k = 256; cta_k > 0; cta_k -= 64) {
    for (const auto [m_ratio, n_ratio] :
         std::array<std::pair<int64_t, int64_t>, 3>{
             {{2, 1}, {1, 2}, {1, 1}}}) {
      for (int64_t instr_n = 256; instr_n > 8; instr_n -= 64) {
        const int64_t cta_m = m_ratio * 64;
        const int64_t cta_n = n_ratio * instr_n;
        const int64_t epilogue_bytes = cta_m * cta_n * 2;
        const int64_t bytes_per_stage = (cta_m + cta_n) * cta_k * 2 + 16;
        const int64_t tiles_m = ceilDivPositive(m, cta_m);
        const int64_t tiles_n = ceilDivPositive(n, cta_n);
        const int64_t stages_per_tile = ceilDivPositive(k, cta_k);
        const int64_t tiles_per_sm =
            ceilDivPositive(tiles_m * tiles_n, num_sms);
        const int64_t stages_per_sm = stages_per_tile * tiles_per_sm;
        const int64_t min_stages = std::max<int64_t>(
            1, std::min<int64_t>(3, stages_per_sm));
        const bool native_valid =
            bytes_per_stage * min_stages + epilogue_bytes <= smem_bytes;
        const bool low_stage_valid =
            offer_low_stage_tiles && !native_valid &&
            bytes_per_stage + cta_m * cta_n * 4 <= smem_bytes;
        if ((!native_valid && !low_stage_valid) || cta_m > 256 ||
            cta_n > 256 || cta_k > 256) {
          continue;
        }
        menu.has_native_config |= native_valid;
        menu.candidates.push_back(
            {cta_m, cta_n, cta_k, instr_n, stages_per_sm, native_valid});
      }
    }
  }
  return menu;
}

struct StageBudget {
  int64_t output_smem_reserve_bytes;
  int64_t operand_stage_bytes;
  int64_t max_operand_stages;
};

inline StageBudget stageBudget(
    int64_t cta_m,
    int64_t cta_n,
    int64_t cta_k,
    int64_t smem_bytes,
    int64_t output_dtype_bytes_sum,
    size_t num_problems) {
  constexpr int64_t max_int64 = std::numeric_limits<int64_t>::max();
  const auto add = [](int64_t lhs, int64_t rhs) {
    return lhs > max_int64 - rhs ? max_int64 : lhs + rhs;
  };
  const auto multiply = [](int64_t lhs, int64_t rhs) {
    return lhs > max_int64 / rhs ? max_int64 : lhs * rhs;
  };
  if (num_problems > static_cast<size_t>(max_int64)) {
    return {max_int64, max_int64, 0};
  }
  const int64_t output_reserve =
      multiply(multiply(output_dtype_bytes_sum, cta_m), cta_n);
  const int64_t operand_stage = add(
      multiply(
          multiply(
              multiply(static_cast<int64_t>(num_problems), 2),
              add(cta_m, cta_n)),
          cta_k),
      16);
  const int64_t stages = output_reserve <= smem_bytes
      ? (smem_bytes - output_reserve) / operand_stage
      : 0;
  return {output_reserve, operand_stage, stages};
}

} // namespace nvfuser::matmul_utils::hopper_tile_menu
