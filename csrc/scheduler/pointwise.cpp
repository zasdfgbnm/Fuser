// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/pointwise.h"

#include <algorithm>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <ATen/cuda/CUDAContext.h>

#include "instrumentation.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/pointwise_non_tma.h"
#include "scheduler/pointwise_tma.h"
#include "scheduler/pointwise_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/utils.h"

namespace nvfuser {

namespace {

// This propagator checks that TransformPropagator will be able to properly
// schedule the entire Fusion. To do so, it tracks logical IterDomains that are
// introduced along the propagation path.
//
//  Example 1:
//
//    addInput(tv0)         // [ i0 ]
//    addInput(tv1)         // [ i0, i1 ]
//    tv2 = neg(tv0)        // [ i0 ]
//    tv3 = broadcast(tv2)  // [ i0, 1 ]
//    tv4 = mul(tv3, tv1)   // [ i0, i1 ]
//    addOutput(tv4)
//
//  In this example, the broadcast of tv2 is concretized when multiplying by
//  tv1, which is 2D. If we propagate from tv2, then we introduce that
//  broadcast dimension which is then concretized as i1, so we detect an
//  unscheduled concrete ID. Note that this should not happen as tv2 will not be
//  selected as a potential reference tensor.
//
//  Example 2:
//
//    addInput(tv0)         // [ i0, 1 ]
//    addInput(tv1)         // [ i0, i1 ]
//    tv2 = squeeze(tv0)    // [ i0 ]
//    tv3 = neg(tv2)        // [ i0 ]
//    tv4 = mul(tv0, tv1)   // [ i0, i1 ]
//    addOutput(tv3)
//    addOutput(tv4)
//
//  In this example, if we propagate from the output tv3, the backward
//  propagation from tv2 to tv0 picks up a broadcast ID. The broadcast is
//  concretized in the multiplication with tv1, so we have an unscheduled i1 ID
//  in the output tv4. Note that tv2 should not be chosen as a reference tensor
//  anyway as it does not have all concrete IDs.
//
//  Example 3:
//
//    addInput(tv0)         // [ i0 ]
//    addInput(tv1)         // [ i1 ]
//    tv2 = broadcast(tv0)  // [ i0, 1 ]
//    tv3 = broadcast(tv1)  // [ 1, i1 ]
//    tv4 = mul(tv2, tv3)   // [ i0, i1 ]
//    tv5 = broadcast(tv0)  // [ i0, 1 ]
//    tv6 = broadcast(tv1)  // [ 1, i1 ]
//    tv7 = add(tv5, tv6)   // [ i0, i1 ]
//    addOutput(tv4)
//    addOutput(tv7)
//
//  If we choose tv4 as the reference then a possible propagation path is
//      4->2->0->5->7->6
//      4->3->1
//  The 2->0 propagation loses the broadcast dimension and then subsequently
//  propagating 0->5->7 means we have concrete dimension i1 unscheduled in tv7,
//  even though there is a scheduled i1 axis in the reference tensor tv4.
//
//  Example 4:
//
//    Suppose instead of creating new broadcasts, we reuse the broadcasts tv2
//    and tv3:
//
//    addInput(tv0)         // [ i0 ]
//    addInput(tv1)         // [ i1 ]
//    tv2 = broadcast(tv0)  // [ i0, 1 ]
//    tv3 = broadcast(tv1)  // [ 1, i1 ]
//    tv4 = mul(tv2, tv3)   // [ i0, i1 ]
//    tv7 = add(tv2, tv3)   // [ i0, i1 ]
//    addOutput(tv4)
//    addOutput(tv7)
//
//  Now the propagation may look like this instead:
//      4->2->0
//         2->7
//      4->3->1
//  None of these steps loses an ID that later is needed for scheduling, so
//  hasUnscheduledConcreteIDs() returns false.
class CoveredDomainPropagator : public MaxInfoSpanningTree::Propagator {
 public:
  void propagateC2P(TensorView* from, TensorView* to) override {
    if (to->isFusionInput()) {
      // We are not concerned with scheduling fusion inputs during transform
      // propagation
      return;
    }
    std::unordered_map<IterDomain*, IterDomain*> c2p =
        PairwiseLogicalDomainMap(to, from)
            .mapBroadcast(true)
            .mapDifferentExtents(true)
            .mapIndexedDomains(true)
            .mapConsumerToProducer();
    check(from->getMaybeRootDomain(), to->getLogicalDomain(), c2p);
    if (to->hasRoot()) {
      // propagate untracked property through root->logical transforms
      for (Expr* e : std::ranges::views::reverse(StmtSort::getExprsBetween(
               {to->getMaybeRootDomain().begin(),
                to->getMaybeRootDomain().end()},
               {to->getLogicalDomain().begin(),
                to->getLogicalDomain().end()}))) {
        bool has_unscheduled_output = std::any_of(
            e->outputs().begin(), e->outputs().end(), [&](Val* out_val) {
              auto* id = dynamic_cast<IterDomain*>(out_val);
              return id && unscheduled_ids_.count(id);
            });
        if (has_unscheduled_output) {
          for (Val* in_val : e->inputs()) {
            unscheduled_ids_.insert(in_val->as<IterDomain>());
          }
        }
      }
    }
  }
  void propagateP2C(TensorView* from, TensorView* to) override {
    std::unordered_map<IterDomain*, IterDomain*> p2c =
        PairwiseLogicalDomainMap(from, to)
            .mapBroadcast(true)
            .mapDifferentExtents(true)
            .mapIndexedDomains(true)
            .mapProducerToConsumer();
    check(from->getLogicalDomain(), to->getMaybeRootDomain(), p2c);
    if (to->hasRoot()) {
      // propagate untracked property through root->logical transforms
      for (Expr* e : StmtSort::getExprsBetween(
               {to->getMaybeRootDomain().begin(),
                to->getMaybeRootDomain().end()},
               {to->getLogicalDomain().begin(),
                to->getLogicalDomain().end()})) {
        // TODO: should we exclude ID exprs other than Merge/Split here?
        bool has_unscheduled_input =
            std::ranges::any_of(e->inputs(), [&](Val* in_val) {
              auto* id = dynamic_cast<IterDomain*>(in_val);
              return id && unscheduled_ids_.count(id);
            });
        if (has_unscheduled_input) {
          for (Val* out_val : e->outputs()) {
            unscheduled_ids_.insert(out_val->as<IterDomain>());
          }
        }
      }
    }
  }
  void propagateSibling(TensorView* from, TensorView* to) override {
    // Siblings require no special consideration in this check
  }

  bool hasUnscheduledConcreteIDs() const {
    for (IterDomain* id : unscheduled_ids_) {
      if (!id->isBroadcast()) {
        return true;
      }
    }
    return false;
  }

 private:
  void check(
      const std::vector<IterDomain*>& from_domain,
      const std::vector<IterDomain*>& to_domain,
      const std::unordered_map<IterDomain*, IterDomain*>& f2t) {
    std::unordered_set<IterDomain*> seen_to;
    for (IterDomain* from_id : from_domain) {
      auto it = f2t.find(from_id);
      if (it != f2t.end()) {
        IterDomain* to_id = it->second;
        if (unscheduled_ids_.count(from_id)) {
          unscheduled_ids_.insert(to_id);
        }
        seen_to.insert(to_id);
      }
    }
    for (IterDomain* to_id : to_domain) {
      if (!seen_to.count(to_id)) {
        // This is a new ID introduced along the propagation path
        unscheduled_ids_.insert(to_id);
      }
    }
  }

 private:
  std::unordered_set<IterDomain*> unscheduled_ids_;
};

} // namespace

//! Utility for canSchedule interface to check if this fusion has
//!  a fully broadcasted reference tensor, which is necessary for
//!  the pointwise scheduler.
bool hasReferenceTensorView(Fusion* fusion) {
  TensorView* reference = pointwise_utils::getReferenceTensor(fusion);
  if (reference == nullptr) {
    return false;
  }

  // If we can find a reference TV, verify that propagation will not need to
  // schedule any IDs that were lost along the propagation path
  MaxLogicalDomainInfoSpanningTree tree(
      reference,
      /*selector=*/nullptr,
      /*propagate_through_resize=*/true);
  CoveredDomainPropagator propagator;
  tree.traverse(&propagator);
  return !propagator.hasUnscheduledConcreteIDs();
}

bool PointWiseScheduler::canScheduleCompileTime(Fusion* fusion) {
  if (scheduler_utils::isResharding(fusion)) {
    FUSER_PERF_SCOPE("PointWiseScheduler::canScheduleCompileTime");
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "Fusion is resharding.");
    return false;
  }

  // Currently using the same path as the scheduler
  // to eliminate mismatch between canSchedule and
  // schedule pointwise.
  if (!hasReferenceTensorView(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "cannot find reference tensor");
    return false;
  }

  // Check that inputs of all select/gather-like ops are fusion inputs
  if (registry_utils::rejectScheduleForMemoryPromotion(
          fusion, schedulerType())) {
    return false;
  }

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    ComputeAtMap ca_map(fusion);
    if (registry_utils::requiresForwardViewReplay(fusion, ca_map)) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(), "Fusion requires view being reversible.");
      return false;
    }
  }

  if (ir_utils::hasAnyReductionOps(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(), "no support for reduction ops");
    return false;
  }

  if (registry_utils::hasNonUniqueBcast(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "Broadcasting dimension might be broadcasting to multiple sizes.");
    return false;
  }

  // The block scales output of the Block Quantization Op
  // should be a segment output as it is written to the global
  // memory.
  if (registry_utils::hasNonTerminalBlockQuantizeOp(fusion)) {
    scheduler_debug_utils::canScheduleRejectReason(
        schedulerType(),
        "no support for block quantization where block scales is not a fusion "
        "output");
    return false;
  }

  return true;
}

bool PointWiseScheduler::canScheduleRunTime(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::canScheduleRunTime");
  // Check if the fusion has a Block Quantization Op
  // If so, ensure that the vectorization factor is at least 2
  // and that the grid y dimension is not split.
  // These are requirements of the current implementation of the
  // Block Quantization Op runtime function.

  auto block_quantization_ops =
      HeuristicDataCacheEntry<HeuristicCompileTime::BlockQuantizationOps>(
          data_cache,
          [fusion]() {
            auto ops = std::make_unique<std::vector<Expr*>>();
            // Get BlockQuantizationOp operations
            auto block_quant_ops =
                ir_utils::getOpsOfType<BlockQuantizationOp>(fusion);
            ops->insert(
                ops->end(), block_quant_ops.begin(), block_quant_ops.end());
            // Get GroupedBlockQuantizationOp operations
            auto grouped_ops =
                ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion);
            ops->insert(ops->end(), grouped_ops.begin(), grouped_ops.end());
            return ops;
          })
          .get();

  if (!block_quantization_ops.empty()) {
    auto heuristics = computeHeuristics(fusion, runtime_info, data_cache);
    auto pparams = static_cast<const PointwiseParams*>(heuristics.get());
    NVF_ERROR(pparams != nullptr);
    if (pparams->vectorization_factor < 2) {
      for (auto op : block_quantization_ops) {
        auto data_type_width = op->isA<BlockQuantizationOp>()
            ? dataTypeSizeBit(op->as<BlockQuantizationOp>()
                                  ->quantizedOutput()
                                  ->getDataType())
            : dataTypeSizeBit(op->as<GroupedBlockQuantizationOp>()
                                  ->quantizedOutput()
                                  ->getDataType());

        if (data_type_width % 8 != 0) {
          scheduler_debug_utils::canScheduleRejectReason(
              schedulerType(),
              "Block Quantization Op requires vectorization factor to be at "
              "least "
              "2 for quantizing to sub-byte dtypes such as NVFP4");
          return false;
        }
      }
    }

    if (pparams->split_grid_y_dim) {
      scheduler_debug_utils::canScheduleRejectReason(
          schedulerType(),
          "Block Quantization Op is not supported when splitting grid y "
          "dimension. This is because this will create a serial ID with an "
          "extent > 1. The runtime function implementing block quantization "
          "will currently not be able to handle that.");
      return false;
    }
  }
  return true;
}

namespace {

// TODO: Refine this function to check contiguity, broadcasts, reshapes, etc.
bool mayHaveTmaCompatibleInputs(
    const pointwise_utils::FusionRuntimeProperties& prop) {
  for (auto tv : prop.vectorizable_inputs_outputs) {
    if (!tv->isFusionInput()) {
      continue;
    }
    // If the minimum dtype size is suitable, then all other dtypes are
    // suitable.
    auto dtype_bits = prop.min_dtype_size_bit_for_vectorization;

    // Note: The actual element count should consider the breakpoint and be
    // computed individually for each input. Here, the largest output is used
    // as a conservative estimate. If the largest output fails these checks,
    // then no input is suitable for TMA since all inputs are smaller than or
    // equal to the largest output in a pointwise fusion.
    auto elem_count = prop.n_elems;

    // Condition 1: We only support 2D TMA, which requires at least 2 tiles in
    // the inner dimension, each with  at least 16 bytes. This imposes a minimum
    // inner TMA domain size of 2 * 16 bytes. Additionally, skip if the inner
    // TMA domain size equals the total element count, as this would mean the
    // outer TMA domain is 1, which is not a valid 2D TMA configuration.
    const int64_t tma_domain_inner_min =
        static_cast<int64_t>(2 * 128) / dtype_bits;
    if (elem_count % tma_domain_inner_min != 0 ||
        elem_count == tma_domain_inner_min) {
      continue;
    }

    // TODO: Add checks for reshape, contiguity, allocation domain, etc.
    // TODO: Add performance checks:
    //   - Skip if input size is too small
    //   - Skip if inner TMA domain size is too small

    // Passed all preliminary checks, may be suitable for TMA
    return true;
  }
  return false;
}

// Preliminary check to determine if TMA can be used for this fusion. This
// serves as a fast path to avoid computing full heuristics if TMA is clearly
// not applicable. Passing this check does not guarantee that TMA will be used;
// the final decision is made during heuristics computation.
bool mayUseTma(
    Fusion* fusion,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  // Hardware requirement: Don't use TMA for pre-Hopper GPUs
  if (at::cuda::getCurrentDeviceProperties()->major < 9) {
    return false;
  }

  // TMA requires compile-time known contiguous innermost dimension on inputs
  if (!scheduler_utils::inputsHaveContiguousInnerDim(fusion)) {
    return false;
  }
  // Check if there are TMA-compatible inputs
  if (!mayHaveTmaCompatibleInputs(prop)) {
    return false;
  }
  return true;
}

// These snapshots describe the returned plan, after the block-quantization
// cap. Candidate Choice rows alone cannot distinguish a discarded ordinary
// plan from the TMA plan actually returned by computeHeuristics.
class PointwiseRecordObject {
 public:
  void add(std::string_view key, std::string_view value) {
    addRaw(key, scheduler::jev::jsonQuote(value));
  }

  void add(std::string_view key, const char* value) {
    add(key, std::string_view(value));
  }

  void add(std::string_view key, int64_t value) {
    addRaw(key, std::to_string(value));
  }

  void add(std::string_view key, bool value) {
    addRaw(key, value ? "true" : "false");
  }

  void addRaw(std::string_view key, std::string_view value) {
    if (body_.size() != 1) body_ += ',';
    body_ += scheduler::jev::jsonQuote(key) + ':' + std::string(value);
  }

  std::string str() const {
    return body_ + '}';
  }

 private:
  std::string body_ = "{";
};

void recordFinalPointwisePlan(
    scheduler::jev::PointwiseRecordScope& scope,
    const PointwiseParams& params,
    bool has_runtime_properties) {
  if (!scope.recording()) return;

  PointwiseRecordObject final_params;
  final_params.add("scheduler_type", "PointWise");
  final_params.add("tag", params.tag);
  final_params.add("break_point", params.break_point);
  final_params.add("split_block", params.split_block);
  final_params.add("symbolic_1d_cta", params.symbolic_1d_cta);
  final_params.add("split_grid_y_dim", params.split_grid_y_dim);
  final_params.add("flip_grid_binding", params.flip_grid_binding);
  final_params.add("vectorization_factor", params.vectorization_factor);
  final_params.add("vectorize_casts", params.vectorize_casts);
  final_params.add("unroll_factor_outer", params.unroll_factor_outer);
  final_params.add("unroll_factor_inner", params.unroll_factor_inner);
  final_params.add("use_tma_load", params.use_tma_load);
  final_params.add("use_tma_store", params.use_tma_store);
  final_params.add("tma_domain_inner", params.tma_domain_inner);
  final_params.add("tma_tile_outer", params.tma_tile_outer);
  final_params.add("tma_tile_inner", params.tma_tile_inner);

  PointwiseRecordObject launch;
  launch.add("grid_x_raw", params.lparams.getRawVal(ParallelType::BIDx));
  launch.add("grid_y_raw", params.lparams.getRawVal(ParallelType::BIDy));
  launch.add("grid_z_raw", params.lparams.getRawVal(ParallelType::BIDz));
  launch.add("block_x_raw", params.lparams.getRawVal(ParallelType::TIDx));
  launch.add("block_y_raw", params.lparams.getRawVal(ParallelType::TIDy));
  launch.add("block_z_raw", params.lparams.getRawVal(ParallelType::TIDz));
  launch.add("dynamic_smem_bytes", params.lparams.smem());

  PointwiseRecordObject compile;
  const auto add_optional_number = [&](std::string_view key, const auto& value) {
    if (value.has_value()) {
      compile.add(key, static_cast<int64_t>(*value));
    } else {
      compile.addRaw(key, "null");
    }
  };
  add_optional_number("index_type", params.cparams.index_type);
  compile.add("maxrregcount", params.cparams.maxrregcount);
  compile.add("enable_magic_zero", params.cparams.enable_magic_zero);
  compile.add("enable_ptxas_verbose", params.cparams.enable_ptxas_verbose);
  if (params.cparams.device.has_value()) {
    compile.add("device", params.cparams.device->str());
  } else {
    compile.addRaw("device", "null");
  }
  std::string include_paths = "[";
  for (size_t i = 0; i < params.cparams.include_paths.size(); ++i) {
    if (i != 0) include_paths += ',';
    include_paths += scheduler::jev::jsonQuote(params.cparams.include_paths[i]);
  }
  include_paths += ']';
  compile.addRaw("include_paths", include_paths);
  add_optional_number("block_x", params.cparams.bdimx);
  add_optional_number("block_y", params.cparams.bdimy);
  add_optional_number("block_z", params.cparams.bdimz);
  add_optional_number("compute_block_x", params.cparams.compute_bdimx);
  add_optional_number("compute_block_y", params.cparams.compute_bdimy);
  add_optional_number("compute_block_z", params.cparams.compute_bdimz);

  scope.complete(
      !has_runtime_properties ? "empty"
                              : params.use_tma_load ? "tma" : "ordinary",
      final_params.str(),
      launch.str(),
      compile.str());
}
} // namespace

std::unique_ptr<HeuristicParams> PointWiseScheduler::computeHeuristicsImpl(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FUSER_PERF_SCOPE("PointWiseScheduler::computeHeuristics");
  scheduler::jev::PointwiseRecordScope record_scope;

  auto prop_opt = pointwise_utils::getFusionRuntimeProperties(
      fusion, runtime_info, data_cache);
  // Return default parameters if the fusion is zero-dimensional or zero-size
  if (!prop_opt.has_value()) {
    auto pwise_params = std::make_unique<PointwiseParams>();
    pwise_params->tag = "Pointwise heuristics";
    pwise_params->cparams.index_type = runtime_info.getIndexType();
    recordFinalPointwisePlan(record_scope, *pwise_params, false);
    return pwise_params;
  }
  const auto& prop = prop_opt.value();

  // A block-quantized schedule has downstream width and grid-Y correctness
  // requirements. canScheduleRunTime calls computeHeuristics separately from
  // the plan later used for code generation. Keep every PointWise preference
  // native in both calls so a variable stateless model cannot change the
  // result between the eligibility check and the final schedule.
  auto block_quantization_ops =
      HeuristicDataCacheEntry<HeuristicCompileTime::BlockQuantizationOps>(
          data_cache,
          [fusion]() {
            auto ops = std::make_unique<std::vector<Expr*>>();
            auto block_ops =
                ir_utils::getOpsOfType<BlockQuantizationOp>(fusion);
            ops->insert(ops->end(), block_ops.begin(), block_ops.end());
            auto grouped_ops =
                ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion);
            ops->insert(ops->end(), grouped_ops.begin(), grouped_ops.end());
            return ops;
          })
          .get();
  const bool block_quantization = !block_quantization_ops.empty();
  const auto configured_jev_mode = scheduler::jev::mode();
  std::optional<scheduler::jev::ScopedNativeDecisions> native_decisions;
  if (block_quantization && configured_jev_mode != scheduler::jev::Mode::Off) {
    native_decisions.emplace();
  }

  const bool use_tma =
      mayUseTma(fusion, prop) && isOptionEnabled(EnableOption::TmaPointwise);
  const auto jev_mode = scheduler::jev::mode();
  std::unique_ptr<HeuristicParams> pparams = nullptr;
  if (use_tma) {
    pparams = pointwise::tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache, prop);
  }
  if (pparams == nullptr) {
    // TMA eligibility and plan construction are native correctness gates.
    pparams = pointwise::non_tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache, prop);
  } else if (jev_mode != scheduler::jev::Mode::Off) {
    // Both completed plans must exist before the backend preference can be a
    // Choice. Off mode keeps the original TMA-first fast path unchanged.
    auto ordinary_params = pointwise::non_tma::getPointwiseHeuristics(
        fusion, runtime_info, data_cache, prop);
    if (ordinary_params != nullptr) {
      const std::string state = "pointwise.tma_preference.v1\nn_elems=" +
          std::to_string(prop.n_elems) +
          "\nsm_count=" + std::to_string(prop.device_multiprocessor_count) +
          "\nmin_dtype_bits=" +
          std::to_string(prop.min_dtype_size_bit_for_vectorization) +
          "\nmax_dtype_bits=" +
          std::to_string(prop.max_dtype_size_bit_for_vectorization);
      const std::vector<scheduler::jev::ChoiceOption> options = {
          {"tma", "Use the complete TMA pointwise plan"},
          {"ordinary", "Use the complete ordinary pointwise plan"}};
      // The original policy prefers every materialized TMA pointwise plan.
      const std::string native_choice =
          jev_mode == scheduler::jev::Mode::Record ? "tma" : "";
      const auto selected =
          scheduler::jev::chooseDecision("PW1", state, options, native_choice);
      NVF_ERROR(
          selected == "tma" || selected == "ordinary",
          "PW1 must select a complete native pointwise plan");
      if (selected == "ordinary") {
        pparams = std::move(ordinary_params);
      }
    }
  }
  NVF_ERROR(pparams != nullptr);

  // Keep the original final clamp and runtime eligibility check.
  if (block_quantization) {
    // FIXME: this needs to be done per input dtype. I'm capping it as 4 for
    // simplicity for now.
    pparams->as<PointwiseParams>()->vectorization_factor = std::min<int64_t>(
        4, pparams->as<PointwiseParams>()->vectorization_factor);
  }

  native_decisions.reset();
  if (block_quantization &&
      configured_jev_mode == scheduler::jev::Mode::Record) {
    const auto* pointwise = pparams->as<PointwiseParams>();
    const auto skip = [](std::string_view site) {
      scheduler::jev::recordCoverageSkip(
          site, "block_quantization_native_plan");
    };
    if (pointwise->use_tma_load) {
      skip("PW1");
      skip("PW4.domain_inner");
      skip("PW4.tile_inner");
      skip("PW4.tile_outer");
      skip("PW4.width");
      skip("PW4.breakpoint");
    } else {
      skip("PW2.entry");
      if (pointwise->break_point > 0) {
        skip("PW2");
      }
      skip("PW3.width");
      skip(pointwise->break_point > 0 ? "PW3.launch" : "PW3.cta_1d");
      skip("PW3.unroll");
    }
  }

  recordFinalPointwisePlan(record_scope, *pparams->as<PointwiseParams>(), true);
  return pparams;
}

void PointWiseScheduler::schedule(
    Fusion* fusion,
    const HeuristicParams* params) {
  FUSER_PERF_SCOPE("PointWiseScheduler::schedule");
  auto pparams = dynamic_cast<const PointwiseParams*>(params);
  NVF_ERROR(
      pparams != nullptr,
      "Incorrect parameters sent to PointWiseScheduler::schedule",
      params);
  bool has_block_quantization = false;
  bool subbyte_quantized_output = false;
  for (auto* op : ir_utils::getOpsOfType<BlockQuantizationOp>(fusion)) {
    has_block_quantization = true;
    subbyte_quantized_output |=
        dataTypeSizeBit(op->quantizedOutput()->getDataType()) % 8 != 0;
  }
  for (auto* op : ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion)) {
    has_block_quantization = true;
    subbyte_quantized_output |=
        dataTypeSizeBit(op->quantizedOutput()->getDataType()) % 8 != 0;
  }
  if (has_block_quantization) {
    // This checks the parameters actually being applied, independently of the
    // earlier canScheduleRunTime call and of any model or cache state.
    NVF_ERROR(
        pparams->vectorization_factor >= 1 &&
            pparams->vectorization_factor <= 4 &&
            (!subbyte_quantized_output || pparams->vectorization_factor >= 2) &&
            !pparams->split_grid_y_dim && pparams->unroll_factor_inner == 1 &&
            pparams->unroll_factor_outer == 1,
        "Block quantization PointWise schedule violates native width, "
        "unroll, or grid-Y requirements");
  }
  if (pparams->use_tma_load) {
    pointwise::tma::schedulePointwise(fusion, pparams);
  } else {
    NVF_ERROR(
        !pparams->use_tma_store,
        "Use TMA store without use TMA load is not supported");
    pointwise::non_tma::schedulePointwise(fusion, pparams);
  }
}

} // namespace nvfuser
