// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace nvfuser::reduction::non_tma::inner3d_iter_continuation {

struct Input {
  int64_t iteration_numel;
  int64_t block_y;
  int64_t requested_unroll;
  int64_t target_blocks;
  int64_t remainder_in_reduction;
  int64_t remainder_in_inner_dim;
  int64_t block_x;
  int64_t sm_count;
  int64_t max_grid_y;
};

struct Plan {
  int64_t requested_unroll;
  int64_t pre_rollback_godim;
  int64_t raw_gridim;
  int64_t clean_gridim;
  int64_t grodim;
  int64_t final_unroll;
  int64_t final_godim;
  bool rd5_grid_eligible;

  // RD5.grid can choose the raw grid only for a requested u1. Include the
  // pending alternative in equality: otherwise an upstream factor could
  // silently remove a later performance decision.
  auto realizedSignature() const {
    return std::tuple{
        final_unroll,
        final_godim,
        clean_gridim,
        grodim,
        rd5_grid_eligible,
        rd5_grid_eligible ? raw_gridim : int64_t{0}};
  }
};

inline int64_t ceilDivPositive(int64_t numerator, int64_t denominator) {
  return numerator / denominator + (numerator % denominator != 0);
}

inline Plan preview(const Input& input) {
  const int64_t provisional_godim = ceilDivPositive(
      input.iteration_numel, input.block_y * input.requested_unroll);
  int64_t gridim = 1;
  int64_t grodim = 1;
  if ((provisional_godim < input.target_blocks &&
       input.remainder_in_reduction >= 0) ||
      input.remainder_in_reduction >= 8) {
    const int64_t grdim = std::min(
        input.remainder_in_reduction, input.block_x * input.block_y * 8);
    gridim = input.remainder_in_inner_dim;
    grodim = std::max(grdim / gridim, int64_t{1});
    grodim = std::max(
        std::min(input.remainder_in_reduction /
                     input.remainder_in_inner_dim,
                 grodim),
        int64_t{1});
  }
  const int64_t raw_gridim = gridim;
  // Preserve the native ragged-wave cleanup exactly, including its signed
  // difference tests. The later RD5.grid Choice may restore raw_gridim.
  if (grodim == 1 && gridim * provisional_godim < input.sm_count * 8 &&
      (gridim * provisional_godim) % input.sm_count != 0 &&
      gridim * provisional_godim > input.sm_count) {
    const int64_t waves = std::max(
        (provisional_godim * gridim) / input.sm_count, int64_t{1});
    const int64_t new_gridim = std::max(
        (waves * input.sm_count) / provisional_godim, int64_t{1});
    if ((new_gridim - gridim) * 4 < gridim &&
        ((new_gridim * provisional_godim) -
         (gridim * provisional_godim)) * 4 < gridim * provisional_godim) {
      gridim = new_gridim;
    }
  }
  const bool rd5_grid_eligible =
      raw_gridim > gridim && gridim >= 1 && raw_gridim <= 1024 &&
      input.block_y == 1 && grodim == 1 && input.requested_unroll == 1 &&
      input.iteration_numel == provisional_godim && provisional_godim >= 1 &&
      provisional_godim <= input.max_grid_y && input.sm_count >= 1 &&
      input.sm_count <= 256;
  const bool rollback = grodim > 1 || gridim > 1;
  return {
      input.requested_unroll,
      provisional_godim,
      raw_gridim,
      gridim,
      grodim,
      rollback ? int64_t{1} : input.requested_unroll,
      rollback ? ceilDivPositive(input.iteration_numel, input.block_y)
               : provisional_godim,
      rd5_grid_eligible};
}

struct Candidate {
  int64_t factor;
  Plan plan;
};

inline std::vector<Candidate> distinctCandidates(
    const std::vector<Candidate>& candidates) {
  std::vector<Candidate> distinct;
  for (const Candidate& candidate : candidates) {
    const auto found = std::find_if(
        distinct.begin(), distinct.end(), [&](const Candidate& existing) {
          return existing.plan.realizedSignature() ==
              candidate.plan.realizedSignature();
        });
    if (found == distinct.end()) {
      distinct.push_back(candidate);
    } else if (candidate.factor < found->factor) {
      *found = candidate;
    }
  }
  std::sort(
      distinct.begin(),
      distinct.end(),
      [](const Candidate& left, const Candidate& right) {
        return left.factor < right.factor;
      });
  return distinct;
}

struct StateFields {
  int64_t iteration_numel;
  int64_t block_y;
  int64_t pre_unroll_godim;
  int64_t max_dtype_bits;
  int64_t n_tensor_inputs;
  int64_t inner_unroll;
  int64_t outer_unroll;
  int64_t sm_count;
  int64_t reduction_numel;
  int64_t inner_numel;
  int64_t block_x;
  int64_t block_z;
  int64_t target_iterations;
  int64_t target_blocks;
  int64_t remainder_in_reduction;
  int64_t remainder_in_inner_dim;
  int64_t max_grid_x;
  int64_t max_grid_y;
  int64_t max_grid_z;

  std::string text() const {
    return "reduction.inner3d_iter_unroll.v2\niteration_numel=" +
        std::to_string(iteration_numel) + "\nbdimy=" +
        std::to_string(block_y) + "\npre_unroll_godim=" +
        std::to_string(pre_unroll_godim) + "\nmax_dtype_bits=" +
        std::to_string(max_dtype_bits) + "\nn_tensor_inputs=" +
        std::to_string(n_tensor_inputs) + "\ninner_unroll=" +
        std::to_string(inner_unroll) + "\nouter_unroll=" +
        std::to_string(outer_unroll) + "\nsm_count=" +
        std::to_string(sm_count) + "\nreduction_numel=" +
        std::to_string(reduction_numel) + "\ninner_numel=" +
        std::to_string(inner_numel) + "\nblock_x=" +
        std::to_string(block_x) + "\nblock_z=" +
        std::to_string(block_z) + "\ntarget_iterations=" +
        std::to_string(target_iterations) + "\ntarget_blocks=" +
        std::to_string(target_blocks) + "\nremainder_in_reduction=" +
        std::to_string(remainder_in_reduction) +
        "\nremainder_in_inner_dim=" +
        std::to_string(remainder_in_inner_dim) + "\nmax_grid_x=" +
        std::to_string(max_grid_x) + "\nmax_grid_y=" +
        std::to_string(max_grid_y) + "\nmax_grid_z=" +
        std::to_string(max_grid_z);
  }
};

struct RequestCandidate {
  int64_t factor;
  Plan plan;
  std::string id;
  std::string description;
  std::string consequence_json;
};

struct RequestContract {
  std::string state;
  std::vector<RequestCandidate> candidates;
  std::string native_choice;
};

// The model-facing state and candidate bytes are independent of the native
// preference. Record only adds a native answer mapped to its realized class.
inline RequestContract buildRequest(
    const StateFields& fields,
    int64_t hard_limit,
    std::optional<int64_t> native_factor) {
  std::vector<Candidate> previewed;
  previewed.reserve(hard_limit);
  for (int64_t factor = 1; factor <= hard_limit; ++factor) {
    previewed.push_back(
        {factor,
         preview({fields.iteration_numel,
                  fields.block_y,
                  factor,
                  fields.target_blocks,
                  fields.remainder_in_reduction,
                  fields.remainder_in_inner_dim,
                  fields.block_x,
                  fields.sm_count,
                  fields.max_grid_y})});
  }
  const auto distinct = distinctCandidates(previewed);
  RequestContract request{fields.text(), {}, ""};
  request.candidates.reserve(distinct.size());
  for (const auto& candidate : distinct) {
    const int64_t factor = candidate.factor;
    const auto& plan = candidate.plan;
    request.candidates.push_back(
        {factor,
         plan,
         "u" + std::to_string(factor),
         "Request output unroll " + std::to_string(factor) +
             "; computed continuation: unroll " +
             std::to_string(plan.final_unroll) + ", output grid " +
             std::to_string(plan.final_godim) + ", inner grid " +
             std::to_string(plan.clean_gridim) + ", outer grid " +
             std::to_string(plan.grodim),
         "{\"requested_unroll\":" + std::to_string(factor) +
             ",\"pre_rollback_godim\":" +
             std::to_string(plan.pre_rollback_godim) +
             ",\"raw_gridim\":" + std::to_string(plan.raw_gridim) +
             ",\"clean_gridim\":" + std::to_string(plan.clean_gridim) +
             ",\"grodim\":" + std::to_string(plan.grodim) +
             ",\"final_unroll_if_clean\":" +
             std::to_string(plan.final_unroll) +
             ",\"final_godim_if_clean\":" +
             std::to_string(plan.final_godim) +
             ",\"rd5_grid_eligible\":" +
             (plan.rd5_grid_eligible ? "true" : "false") +
             ",\"pending_choices\":" +
             (plan.rd5_grid_eligible ? "[\"RD5.grid\"]" : "[]") + "}"});
  }
  if (native_factor.has_value()) {
    const auto native_signature =
        previewed.at(native_factor.value() - 1).plan.realizedSignature();
    const auto native_class = std::find_if(
        request.candidates.begin(),
        request.candidates.end(),
        [&](const auto& candidate) {
          return candidate.plan.realizedSignature() == native_signature;
        });
    if (native_class != request.candidates.end()) {
      request.native_choice = native_class->id;
    }
  }
  return request;
}

} // namespace nvfuser::reduction::non_tma::inner3d_iter_continuation
