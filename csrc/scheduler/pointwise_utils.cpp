// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "scheduler/pointwise_utils.h"

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>
#include <string_view>
#include <utility>

#include "ir/printer.h"
#include "multidevice/utils.h"
#include "scheduler/cache_policy_refiner.h"
#include "scheduler/jev_client.h"
#include "scheduler/native_diagnostic_jsonl.h"
#include "scheduler/pointwise_add_cta_continuation.h"
#include "scheduler/registry.h"
#include "scheduler/runtime_info.h"

namespace nvfuser {
namespace pointwise_utils {

const char* semanticPointwiseEligibilityReason(
    Fusion* fusion,
    std::string_view snapshot,
    bool semantic_v3) {
  // v2 requests have a 64 KiB limit after candidates are added. v3 captures
  // at most 1 MiB of invocation input and allows an 8 MiB complete request.
  if (snapshot.empty()) {
    return "semantic_graph_snapshot_unavailable";
  }
  if (snapshot.size() > (semantic_v3 ? 1024 * 1024 : 8192)) {
    return "semantic_graph_snapshot_too_large";
  }
  if (fusion == nullptr || fusion->inputs().empty() ||
      fusion->outputs().empty()) {
    return "semantic_graph_missing_tensors";
  }
  for (Val* input : fusion->inputs()) {
    if (!input->isA<TensorView>()) {
      return "semantic_graph_scalar_input";
    }
  }
  for (Val* output : fusion->outputs()) {
    if (!output->isA<TensorView>() ||
        fusion->getOutputAlias(output).type != AllocationType::New) {
      return "semantic_graph_output_or_alias";
    }
  }
  // Fusion::print(false) renders loop domains, while ExpressionEvaluator
  // binds runtime tensor dimensions to logical domains. The equality check
  // must include intermediate TVs, not only fusion inputs and outputs.
  for (Val* val : fusion->vals()) {
    if (!val->isA<TensorView>()) {
      continue;
    }
    const auto* tv = val->as<TensorView>();
    const auto& logical = tv->getLogicalDomain();
    const auto& loop = tv->getLoopDomain();
    if (tv->hasDeviceMesh() || tv->isCpuScalar()) {
      return "semantic_graph_sharded_or_cpu_scalar";
    }
    if (tv->getMemoryType() != MemoryType::Global &&
        tv->getMemoryType() != MemoryType::Local) {
      return "semantic_graph_memory_type";
    }
    if (tv->dtype() != DataType::Float && tv->dtype() != DataType::Half) {
      return "semantic_graph_dtype";
    }
    if (logical.empty() || logical != loop ||
        tv->getMaybeAllocationDomain() != logical ||
        !tv->domain()->allExprs().empty()) {
      return "semantic_graph_hidden_domain_transform";
    }
    for (size_t i = 0; i < logical.size(); ++i) {
      const auto* id = logical[i];
      if (id->isReduction() || id->isStride() || id->isSymbolic() ||
          id->isDeviceDim() || id->hasExpandedExtent() ||
          id->isParallelized() ||
          (!id->isIteration() && !id->isBroadcast())) {
        return "semantic_graph_unsupported_axis";
      }
    }
    // Contiguity affects native vectorization limits, but does not change
    // logical extents. Runtime sizes and strides are part of the snapshot;
    // the semantic mock only derives extents and checks native summaries.
    // BroadcastOp may introduce a logical axis, so root==logical is not
    // required. Domain transforms are excluded by allExprs() above.
  }
  // IrPrinter emits exactly fusion->exprs(). Restrict this set to the mock's
  // proven unary/binary/broadcast/set grammar; other ops stay native.
  for (Expr* expr : fusion->exprs()) {
    if (expr->isA<BroadcastOp>()) {
      continue;
    }
    if (expr->isA<LoadStoreOp>()) {
      const auto* op = expr->as<LoadStoreOp>();
      if (op->opType() == LoadStoreOpType::Set &&
          op->cacheOp() == CacheOp::Unspecified) {
        continue;
      }
    }
    if (expr->isA<UnaryOp>()) {
      const auto* op = expr->as<UnaryOp>();
      if (op->getUnaryOpType() == UnaryOpType::Neg ||
          op->getUnaryOpType() == UnaryOpType::Erf ||
          op->getUnaryOpType() == UnaryOpType::Tanh) {
        if (op->out()->isA<TensorView>()) {
          continue;
        }
      }
      if (op->getUnaryOpType() == UnaryOpType::Cast &&
          ((op->in()->dtype() == DataType::Half &&
            op->out()->dtype() == DataType::Float) ||
           (op->in()->dtype() == DataType::Float &&
            op->out()->dtype() == DataType::Half) ||
           (op->in()->isConstScalar() &&
            op->out()->dtype() == DataType::Float))) {
        continue;
      }
    }
    if (expr->isA<BinaryOp>()) {
      const auto* op = expr->as<BinaryOp>();
      if (op->out()->isA<TensorView>() &&
          (op->getBinaryOpType() == BinaryOpType::Add ||
           op->getBinaryOpType() == BinaryOpType::Sub ||
           op->getBinaryOpType() == BinaryOpType::Mul ||
           op->getBinaryOpType() == BinaryOpType::Div)) {
        continue;
      }
    }
    return "semantic_graph_unsupported_operator";
  }
  return nullptr;
}

std::string semanticPointwiseWorkload(
    int64_t sm_count,
    const cudaDeviceProp* device) {
  const auto snapshot = scheduler::jev::computeInvocationInputSnapshot();
  NVF_ERROR(
      !snapshot.empty(),
      "PointWise semantic request has no complete computeHeuristics snapshot: ",
      scheduler::jev::computeInvocationInputUnavailableReason());
  std::string workload =
      "{\"fusion_and_runtime\":" + scheduler::jev::jsonQuote(snapshot) +
      ",\"snapshot_format\":\"nvfuser.compute_invocation_input.v1\"";
  workload += ",\"hardware\":{\"sm_count\":" + std::to_string(sm_count) +
      ",\"device_name\":" + scheduler::jev::jsonQuote(device->name) +
      ",\"compute_capability_major\":" + std::to_string(device->major) +
      ",\"compute_capability_minor\":" + std::to_string(device->minor) +
      ",\"warp_size\":" + std::to_string(device->warpSize) +
      ",\"max_threads_per_block\":" +
      std::to_string(device->maxThreadsPerBlock) + ",\"max_threads_per_sm\":" +
      std::to_string(device->maxThreadsPerMultiProcessor) +
      ",\"max_threads_x\":" + std::to_string(device->maxThreadsDim[0]) +
      ",\"max_threads_y\":" + std::to_string(device->maxThreadsDim[1]) +
      ",\"registers_per_sm\":" + std::to_string(device->regsPerMultiprocessor) +
      ",\"shared_memory_per_sm_bytes\":" +
      std::to_string(device->sharedMemPerMultiprocessor) +
      ",\"max_grid_x\":" + std::to_string(device->maxGridSize[0]) +
      ",\"max_grid_y\":" + std::to_string(device->maxGridSize[1]) + "}}";
  return workload;
}

TensorView* getReferenceTensor(Fusion* fusion) {
  FusionGuard fg(fusion);
  scheduler_tools::PointwiseDomainMap domain_map(fusion);
  auto reference_tv = domain_map.findReferenceTensor();
  return reference_tv;
}

TensorView* getLargestOutputTensor(
    Fusion* fusion,
    HeuristicDataCache* data_cache) {
  auto domain_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::DomainMap>(
          data_cache, [fusion]() {
            return std::make_unique<scheduler_tools::PointwiseDomainMap>(
                fusion);
          });
  const auto& domain_map = dynamic_cast<scheduler_tools::PointwiseDomainMap&>(
      domain_map_entry.get());

  auto largest_out_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensors>(
          data_cache, [&domain_map]() {
            std::vector<TensorView*> data{domain_map.findReferenceTensor()};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  TensorView* largest_out = largest_out_entry.get()[0];

  NVF_ERROR(largest_out != nullptr);
  return largest_out;
}

std::vector<IterDomain*> getReferenceLoop(
    TensorView* largest_out,
    bool has_reshapes,
    HeuristicDataCache* data_cache) {
  std::unordered_map<int64_t, int64_t> loop_reorder_map;
  if (!has_reshapes) {
    loop_reorder_map = scheduler_utils::reorderLoopAsAllocationMap(largest_out);
  }

  std::vector<IterDomain*> ref_loop = largest_out->getLoopDomain();
  // reorder of root to align with logical map should always help with indexing,
  // even when vectorization isn't used.
  if (!loop_reorder_map.empty()) {
    ref_loop = TensorDomain::orderedAs(ref_loop, loop_reorder_map);
  }
  // We always cacheBefore output at the beginning of the scheduling. And after
  // cacheBefore, the reference tensor will have all reduction IDs removed.
  ref_loop = TensorDomain::noDevices(TensorDomain::noReductions(ref_loop));

  return ref_loop;
}

std::pair<std::vector<int64_t>, int64_t> getElementCounts(
    const std::vector<IterDomain*>& ref_loop,
    SchedulerRuntimeInfo& runtime_info) {
  std::vector<int64_t> elem_counts;
  elem_counts.reserve(ref_loop.size());
  int64_t n_elems = 1;
  for (IterDomain* ref_id : ref_loop) {
    auto extent_pvalue =
        runtime_info.expressionEvaluator().evaluate(ref_id->extent());
    NVF_ERROR(
        extent_pvalue.hasValue(),
        "Error inferring size for pointwise scheduler: ",
        ref_id->extent()->toInlineString());
    auto extent = extent_pvalue.as<int64_t>();
    elem_counts.push_back(extent);
    n_elems *= extent;
  }
  return {elem_counts, n_elems};
}

std::unordered_map<int64_t, int64_t> getLogicalReorderMap(
    TensorView* largest_out,
    bool has_reshapes,
    HeuristicDataCache* data_cache) {
  auto logical_reorder_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::LogicalReorderMap>(
          data_cache, [largest_out, has_reshapes]() {
            // NOTE: reorder_map is only applied for fusion without view
            // op yet.
            if (has_reshapes) {
              return std::make_unique<std::unordered_map<int64_t, int64_t>>();
            }
            return std::make_unique<std::unordered_map<int64_t, int64_t>>(
                scheduler_utils::reorderLogicalAsAllocationMap(largest_out));
          });
  return logical_reorder_map_entry.get();
}

std::optional<FusionRuntimeProperties> getFusionRuntimeProperties(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionRuntimeProperties prop;

  prop.largest_out = getLargestOutputTensor(fusion, data_cache);

  prop.device_multiprocessor_count = static_cast<int64_t>(
      at::cuda::getCurrentDeviceProperties()->multiProcessorCount);

  prop.has_reshapes = !ir_utils::getReshapeOps(fusion).empty();

  prop.ref_loop =
      getReferenceLoop(prop.largest_out, prop.has_reshapes, data_cache);

  auto [elem_counts, n_elems] = getElementCounts(prop.ref_loop, runtime_info);
  prop.elem_counts = std::move(elem_counts);
  prop.n_elems = n_elems;

  // Check for zero dimensional or zero size tensors
  if (std::ranges::empty(
          prop.largest_out->getLoopDomain() | TensorDomain::kNoReductions |
          TensorDomain::kNoBroadcasts | TensorDomain::kNoDevices) ||
      prop.n_elems == 0) {
    return std::nullopt;
  }

  // Get vectorizable inputs/outputs using cache
  auto vectorizable_inputs_outputs_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::VectorizableInputsAndOutputs>(
      data_cache, [&prop]() {
        return std::make_unique<std::vector<TensorView*>>(
            scheduler_utils::getInputsOutputsWithInnerDim(
                prop.largest_out, true, true));
      });
  prop.vectorizable_inputs_outputs = vectorizable_inputs_outputs_entry.get();

  // Calculate max and min dtype size for vectorization
  const auto index_type = runtime_info.getIndexType();
  int64_t max_dtype_size_bit = 0;
  int64_t min_dtype_size_bit = std::numeric_limits<int64_t>::max();

  for (auto tv : prop.vectorizable_inputs_outputs) {
    int64_t dtype_size_bit = dataTypeSizeBit(tv->getDataType(), index_type);
    max_dtype_size_bit = std::max(max_dtype_size_bit, dtype_size_bit);
    min_dtype_size_bit = std::min(min_dtype_size_bit, dtype_size_bit);
  }

  // If no vectorizable inputs/outputs, set default values
  // This prevents having a too large vectorization factor.
  if (max_dtype_size_bit == 0) {
    max_dtype_size_bit = 8;
  }
  if (min_dtype_size_bit == std::numeric_limits<int64_t>::max()) {
    min_dtype_size_bit = 8;
  }

  prop.max_dtype_size_bit_for_vectorization = max_dtype_size_bit;
  prop.min_dtype_size_bit_for_vectorization = min_dtype_size_bit;
  prop.index_type = index_type;

  return prop;
}

TmaBreakPointSearch getTmaBreakPointSearch(
    const FusionRuntimeProperties& prop,
    HeuristicDataCache* data_cache) {
  const auto& largest_out = prop.largest_out;
  const auto index_type = prop.index_type;
  auto broadcast_info_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::BroadcastMultiples>(
      data_cache, [&largest_out, &index_type]() {
        return std::make_unique<scheduler_utils::BroadcastMultipleInformation>(
            scheduler_utils::getBroadcastMultiples(largest_out, index_type));
      });
  const auto& info = broadcast_info_entry.get();
  const auto& extents = prop.elem_counts;
  NVF_ERROR_EQ(extents.size(), info.broadcast_multiples.size());

  TmaBreakPointSearch search;
  search.state = "pointwise.tma_breakpoint.v1\n";
  auto append = [&](std::string_view key, const auto& value) {
    search.state += std::string(key) + '=';
    for (size_t i = 0; i < value.size(); ++i) {
      if (i != 0) {
        search.state += ',';
      }
      search.state += std::to_string(value[i]);
    }
    search.state += '\n';
  };
  std::vector<int64_t> disjoint;
  std::vector<int64_t> lhs;
  std::vector<int64_t> rhs;
  for (size_t i = 0; i < extents.size(); ++i) {
    disjoint.push_back(
        scheduler_utils::breakIsDisjoint(info.view_disjoint_set_ids, i));
    lhs.push_back(info.broadcast_multiples[i].lhs_multiple);
    rhs.push_back(info.broadcast_multiples[i].rhs_multiple);
  }
  append("elements", extents);
  append("disjoint", disjoint);
  append("lhs_bit_multiples", lhs);
  append("rhs_bit_multiples", rhs);

  // bp0 is the existing one-dimensional fallback, even when no split is
  // view-disjoint. Other candidates need a real left and right side.
  search.candidates.emplace_back();
  int64_t right_count = 1;
  for (int64_t i = static_cast<int64_t>(extents.size()) - 1; i >= 1; --i) {
    right_count *= extents[i];
    if (!disjoint[i] || prop.n_elems / right_count <= 1) {
      continue;
    }
    BreakPointInfo candidate;
    candidate.break_point = static_cast<int>(i);
    candidate.right_elem_count = right_count;
    candidate.is_outer_broadcast_dominated = lhs[i] < rhs[i];
    search.candidates.push_back(candidate);
  }
  return search;
}

BreakPointInfo getBreakPoint(
    Fusion* fusion,
    const FusionRuntimeProperties& prop,
    HeuristicDataCache* data_cache,
    bool is_tma,
    int64_t max_vect_factor,
    int64_t kThreadX,
    bool native_default_1d) {
  BreakPointInfo result;

  // Calculate dtype_sum_bit from fusion inputs/outputs
  int64_t dtype_sum_bit = 0;
  const auto index_type = prop.index_type;
  for (auto inp : ir_utils::filterByType<TensorView>(fusion->inputs())) {
    dtype_sum_bit += dataTypeSizeBit(inp->getDataType(), index_type);
  }
  for (auto out : ir_utils::filterByType<TensorView>(fusion->outputs())) {
    dtype_sum_bit += dataTypeSizeBit(out->getDataType(), index_type);
  }

  // Get broadcast information
  TensorView* largest_out = prop.largest_out;
  auto broadcast_info_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::BroadcastMultiples>(
      data_cache, [&largest_out, &index_type]() {
        return std::make_unique<scheduler_utils::BroadcastMultipleInformation>(
            scheduler_utils::getBroadcastMultiples(largest_out, index_type));
      });
  const auto& broadcast_info = broadcast_info_entry.get();

  const auto& ref_loop = prop.ref_loop;
  const auto& elem_counts = prop.elem_counts;
  const int64_t n_elems = prop.n_elems;
  const auto& view_disjoint_sets = broadcast_info.view_disjoint_set_ids;
  const auto& broadcast_bit_multiples = broadcast_info.broadcast_multiples;

  // Default values for 1D scheduling
  result.break_point = 0;
  result.flip_grid_binding = false;
  result.right_elem_count = 0;
  result.is_outer_broadcast_dominated = false;

  // PW2's choice includes both the split and the BID binding. Keep the view
  // and nonempty-left checks in nvFuser: they define which plans are legal.
  // The request carries measurements only, never the native winning plan.
  const auto jev_mode =
      is_tma ? scheduler::jev::Mode::Off : scheduler::jev::mode();
  struct JevBreakpointRequest {
    std::string state;
    std::vector<scheduler::jev::ChoiceOption> options;
    std::vector<BreakPointInfo> plans;
  };
  auto make_jev_request = [&]() {
    JevBreakpointRequest request;
    const bool semantic_v3 = scheduler::jev::semanticV3Enabled();
    request.state =
        semantic_v3 ? "pointwise.breakpoint.v3\n" : "pointwise.breakpoint.v2\n";
    std::vector<int64_t> disjoint;
    std::vector<int64_t> lhs;
    std::vector<int64_t> rhs;
    disjoint.reserve(ref_loop.size());
    lhs.reserve(ref_loop.size());
    rhs.reserve(ref_loop.size());
    for (const auto i : arange(ref_loop.size())) {
      disjoint.push_back(
          scheduler_utils::breakIsDisjoint(view_disjoint_sets, i));
      lhs.push_back(broadcast_bit_multiples[i].lhs_multiple);
      rhs.push_back(broadcast_bit_multiples[i].rhs_multiple);
    }
    auto append_vector = [&](std::string_view name, const auto& values) {
      request.state += std::string(name) + '=';
      for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
          request.state += ',';
        }
        request.state += std::to_string(values[i]);
      }
      request.state += '\n';
    };
    append_vector("elements", elem_counts);
    append_vector("disjoint", disjoint);
    append_vector("lhs_bit_multiples", lhs);
    append_vector("rhs_bit_multiples", rhs);
    request.state += "dtype_sum_bits=" + std::to_string(dtype_sum_bit) + '\n';
    request.state +=
        "max_vectorization=" + std::to_string(max_vect_factor) + '\n';
    request.state += "warp_size=" +
        std::to_string(at::cuda::getCurrentDeviceProperties()->warpSize) + '\n';
    request.state += "l2_cache_bytes=" +
        std::to_string(at::cuda::getCurrentDeviceProperties()->l2CacheSize) +
        '\n';
    // The ordinary scheduler's small-workload preference is part of PW2's
    // native answer after PW2.entry was removed. Expose every operand so a
    // stateless service can replay that preference from this request alone.
    request.state += "n_elems=" + std::to_string(n_elems) + '\n';
    request.state +=
        "sm_count=" + std::to_string(prop.device_multiprocessor_count) + '\n';
    request.state += "default_cta_threads=" + std::to_string(kThreadX);

    // A candidate consequence describes only the geometry selected at PW2.
    // The named PW3 sites are possible descendants, not a completed schedule.
    auto append_option = [&](std::string id,
                             std::string description,
                             const BreakPointInfo& plan) {
      std::string consequence;
      if (semantic_v3) {
        scheduler::native_diagnostic::Object effect;
        effect.add("break_point", static_cast<int64_t>(plan.break_point));
        // bp0 has no 2D split: all work belongs to its sole 1D domain.
        effect.add(
            "left_elems",
            plan.break_point == 0 ? 0 : n_elems / plan.right_elem_count);
        effect.add(
            "right_elems",
            plan.break_point == 0 ? n_elems : plan.right_elem_count);
        effect.add(
            "grid_binding",
            plan.break_point == 0
                ? "one_dimensional"
                : (plan.flip_grid_binding ? "flipped" : "normal"));
        effect.add(
            "outer_broadcast_dominated", plan.is_outer_broadcast_dominated);
        effect.addRaw(
            "pending_choices",
            plan.break_point == 0
                ? "[\"PW3.width\",\"PW3.cta_1d\",\"PW3.unroll\"]"
                : "[\"PW3.width\",\"PW3.launch\",\"PW3.unroll\"]");
        consequence = effect.str();
      }
      request.options.push_back(
          {std::move(id), std::move(description), std::move(consequence)});
      request.plans.push_back(plan);
    };
    append_option("bp0", "Use one-dimensional scheduling", BreakPointInfo{});
    int64_t right_count = 1;
    for (int64_t i = static_cast<int64_t>(ref_loop.size()) - 1; i >= 1; --i) {
      right_count *= elem_counts[i];
      if (!disjoint[i] || n_elems / right_count <= 1) {
        continue;
      }
      BreakPointInfo plan;
      plan.break_point = static_cast<int>(i);
      plan.right_elem_count = right_count;
      plan.is_outer_broadcast_dominated = lhs[i] < rhs[i];
      const std::string prefix = "bp" + std::to_string(i);
      append_option(
          prefix + "_normal",
          "Split at axis " + std::to_string(i) + ", normal BID binding",
          plan);
      plan.flip_grid_binding = true;
      append_option(
          prefix + "_flip",
          "Split at axis " + std::to_string(i) + ", flipped BID binding",
          plan);
    }
    return request;
  };

  if (jev_mode == scheduler::jev::Mode::Remote) {
    auto request = make_jev_request();
    const auto selected =
        scheduler::jev::chooseDecision("PW2", request.state, request.options);
    for (size_t i = 0; i < request.options.size(); ++i) {
      if (request.options[i].id != selected) {
        continue;
      }
      const auto& plan = request.plans[i];
      NVF_ERROR(
          plan.break_point == 0 ||
              (scheduler_utils::breakIsDisjoint(
                   view_disjoint_sets, plan.break_point) &&
               n_elems / plan.right_elem_count > 1),
          "Remote PW2 choice violated a native legality guard");
      return plan;
    }
    NVF_ERROR(false, "Remote PW2 choice is not in its candidate set");
  }

  // The original ordinary scheduler skips 2D analysis for small workloads.
  // Record its resulting bp0 directly, while still presenting the same legal
  // 1D and 2D alternatives that Remote can choose among. Off mode keeps the
  // call-site's original early return and TMA never uses this preference.
  if (jev_mode == scheduler::jev::Mode::Record && native_default_1d) {
    auto request = make_jev_request();
    (void)scheduler::jev::chooseDecision(
        "PW2", request.state, request.options, "bp0");
    return result;
  }

  // Figure out break point position
  // How much would this transfer cost if it was done as a 1-D schedule
  int64_t transfer_size_1d_bit = 1;

  for (const auto i : arange(ref_loop.size())) {
    transfer_size_1d_bit =
        transfer_size_1d_bit * elem_counts[i] * dtype_sum_bit;
  }

  // Calculate optimal break point for 2D scheduling
  int64_t min_total_transfer_bit = std::numeric_limits<int64_t>::max();
  // Don't check the inner most dimension, scheduler assumes there's always
  // an rhs
  for (const auto break_point_i : arange((int64_t)ref_loop.size())) {
    // If break point is incoherent with view, don't consider breaking here.
    if (!scheduler_utils::breakIsDisjoint(view_disjoint_sets, break_point_i)) {
      continue;
    }

    // Number of elements in the right side of reference tv with
    // break_point_i
    int64_t cur_right_elem_count = 1;
    for (const auto right_i : arange(break_point_i, ref_loop.size())) {
      cur_right_elem_count = cur_right_elem_count * elem_counts[right_i];
    }

    // For tma scheduling, allow no element in the left side of break point,
    // e.g. break at pos-0 for non-broadcasted case.
    auto cur_left_elem_count = n_elems / cur_right_elem_count;
    if (!is_tma && cur_left_elem_count <= 1) {
      continue;
    }

    auto lhs_bit_multiple = broadcast_bit_multiples[break_point_i].lhs_multiple;
    auto rhs_bit_multiple = broadcast_bit_multiples[break_point_i].rhs_multiple;

    // Estimate transfer cost with this break point
    int64_t cur_transfer_size_bit = 1;
    int64_t right_transfer_size_bit = 1;

    for (const auto left_i : arange(break_point_i)) {
      cur_transfer_size_bit =
          cur_transfer_size_bit * elem_counts[left_i] * lhs_bit_multiple;
    }

    for (const auto right_i : arange(break_point_i, ref_loop.size())) {
      right_transfer_size_bit =
          right_transfer_size_bit * elem_counts[right_i] * rhs_bit_multiple;
    }
    cur_transfer_size_bit *= right_transfer_size_bit;

    if (!is_tma) {
      //  Continue if this break point doesn't save at least 10% of 1D
      //  scheduling or isn't better than previous break_points found.
      if (cur_transfer_size_bit >= min_total_transfer_bit ||
          cur_transfer_size_bit * 10 >= transfer_size_1d_bit * 9) {
        continue;
      }

      // Need to be able to parallelize, don't use break if there's not
      // at least an unrolled warp.
      if (ceilDiv(cur_right_elem_count, max_vect_factor) <=
          at::cuda::getCurrentDeviceProperties()->warpSize) {
        continue;
      }
      // If outer broadcast, or balanced broadcast:
      if (lhs_bit_multiple <= rhs_bit_multiple &&
          // If right transfer size is bigger than half of L2
          static_cast<int64_t>(
              at::cuda::getCurrentDeviceProperties()->l2CacheSize) *
                  8 <
              right_transfer_size_bit * 2) {
        // flip BIDx and BIDy bindings
        result.flip_grid_binding = true;
      } else {
        result.flip_grid_binding = false;
      }
    } else {
      // If TMA is used, prioritize break if it saves transferred size
      // This ensures we break at broadcast dimensions, then we can optionally
      // load tvs with broadcasted dimensions.
      if (cur_transfer_size_bit >= min_total_transfer_bit) {
        continue;
      }
    }

    // Use this break point
    result.break_point = static_cast<int>(break_point_i);
    min_total_transfer_bit = cur_transfer_size_bit;
    result.right_elem_count = cur_right_elem_count;

    // when lhs byte multiple is smaller than rhs byte multiple,
    // there is broadcast in the lhs, which is outer broadcast.
    result.is_outer_broadcast_dominated = lhs_bit_multiple < rhs_bit_multiple;
  }

  if (jev_mode == scheduler::jev::Mode::Record) {
    auto request = make_jev_request();
    const auto native_choice = result.break_point == 0
        ? std::string("bp0")
        : "bp" + std::to_string(result.break_point) +
            (result.flip_grid_binding ? "_flip" : "_normal");
    (void)scheduler::jev::chooseDecision(
        "PW2", request.state, request.options, native_choice);
  }

  return result;
}

BlockGridConfig getBlockGridConfig(
    const FusionRuntimeProperties& prop,
    const BreakPointInfo& bp_info,
    int64_t max_vect_factor,
    int64_t kThreadX,
    bool block_quantization,
    bool exact_add2x16_continuation) {
  BlockGridConfig result;

  // Copy break point information
  result.break_point = bp_info.break_point;
  result.flip_grid_binding = bp_info.flip_grid_binding;
  result.right_elem_count = bp_info.right_elem_count;
  result.is_outer_broadcast_dominated = bp_info.is_outer_broadcast_dominated;

  // Default thread/grid dimensions for 1D scheduling
  result.bdimx = kThreadX;
  result.bdimy = 1;
  result.gdim_left = 1;
  result.gdim_right = 1;

  // Calculate thread/grid dimensions if we have a 2D break point
  if (result.break_point > 0 && result.right_elem_count > 0) {
    const int64_t n_elems = prop.n_elems;
    const auto* device = at::cuda::getCurrentDeviceProperties();
    int64_t threads_per_warp = (int64_t)device->warpSize;

    int64_t cur_left_elem_count = n_elems / result.right_elem_count;

    // Keep the native launch available for block quantization and for a
    // conservative fallback if its preliminary resource proof fails. It is
    // never sent to the model as a preference.
    int64_t after_vect = ceilDiv(result.right_elem_count, max_vect_factor);
    const auto jev_mode = scheduler::jev::mode();
    result.bdimx = std::min(after_vect, threads_per_warp);
    while (result.bdimx * 2 <= kThreadX && result.bdimx * 2 <= after_vect &&
           after_vect % (result.bdimx * 2) == 0) {
      result.bdimx *= 2;
    }
    if (jev_mode != scheduler::jev::Mode::Off && !block_quantization) {
      NVF_ERROR(
          n_elems > 0 && result.right_elem_count > 0 && max_vect_factor > 0 &&
              threads_per_warp > 0 && threads_per_warp <= kThreadX &&
              kThreadX == 128 && result.right_elem_count <= n_elems &&
              n_elems % result.right_elem_count == 0,
          "PW3.launch requires valid two-dimensional extents and CTA bounds");

      // Symbolic splits and predicated tails permit irregular TIDx values.
      // Filter the structural domain against the physical CTA and the
      // preliminary grid before offering a model any alternatives. Later
      // unroll can only shrink the grid; if this filter excludes the native
      // plan, retain the original native path instead of making an incomplete
      // pre-unroll proof into a rejection.
      std::vector<scheduler::jev::ChoiceOption> options;
      options.reserve(kThreadX);
      for (int64_t candidate = 1; candidate <= kThreadX; ++candidate) {
        const int64_t threads_y = kThreadX / candidate;
        const int64_t physical_threads = candidate * threads_y;
        const int64_t grid_left = ceilDiv(cur_left_elem_count, threads_y);
        const int64_t grid_right = ceilDiv(after_vect, candidate);
        const int64_t grid_x =
            result.flip_grid_binding ? grid_left : grid_right;
        const int64_t grid_y =
            result.flip_grid_binding ? grid_right : grid_left;
        if (candidate > device->maxThreadsDim[0] ||
            threads_y > device->maxThreadsDim[1] ||
            physical_threads > device->maxThreadsPerBlock ||
            physical_threads > device->maxThreadsPerMultiProcessor ||
            grid_x > device->maxGridSize[0] ||
            std::min<int64_t>(grid_y, 65535) > device->maxGridSize[1]) {
          continue;
        }
        options.push_back(
            {"x" + std::to_string(candidate),
             "Use " + std::to_string(candidate) + " TIDx threads"});
      }
      const bool semantic = scheduler::jev::semanticProtocolEnabled();
      const bool semantic_v3 = scheduler::jev::semanticV3Enabled();
      const bool native_in_menu =
          std::any_of(options.begin(), options.end(), [&](const auto& option) {
            return option.id == "x" + std::to_string(result.bdimx);
          });
      // The legacy text-v1 server has no device-axis fields for a partial
      // menu. The v3 state below carries the exact limits needed to replay
      // the hard filter; v2 already carries them in its structured workload.
      const char* skip_reason = options.empty() ? "empty_hard_cta_menu"
          : !native_in_menu                     ? "native_2d_cta_preflight"
          : !semantic && !semantic_v3 &&
                  options.size() != static_cast<size_t>(kThreadX)
          ? "text_v1_requires_full_hard_cta_menu"
          : nullptr;
      if (skip_reason != nullptr) {
        if (jev_mode == scheduler::jev::Mode::Record) {
          scheduler::jev::recordCoverageSkip("PW3.launch", skip_reason);
        }
      } else {
        if (semantic_v3) {
          // Preserve the v2 local preview. The exact Add carrier additionally
          // carries a bounded formula for every downstream unroll leaf.
          for (auto& option : options) {
            const int64_t candidate = std::stoll(option.id.substr(1));
            const int64_t threads_y = kThreadX / candidate;
            const int64_t grid_left = ceilDiv(cur_left_elem_count, threads_y);
            const int64_t grid_right = ceilDiv(after_vect, candidate);
            const int64_t idle_right =
                (candidate - after_vect % candidate) % candidate;
            const int64_t idle_left =
                (threads_y - cur_left_elem_count % threads_y) % threads_y;
            const int64_t tail_elements =
                (max_vect_factor - result.right_elem_count % max_vect_factor) %
                max_vect_factor;
            option.consequence_json =
                "{\"threads_x\":" + std::to_string(candidate) +
                ",\"threads_y\":" + std::to_string(threads_y) +
                ",\"actual_cta_threads\":" +
                std::to_string(candidate * threads_y) +
                ",\"right_vector_iterations\":" + std::to_string(after_vect) +
                ",\"preliminary_grid_left\":" + std::to_string(grid_left) +
                ",\"preliminary_grid_right\":" + std::to_string(grid_right) +
                ",\"preliminary_grid_x\":" +
                std::to_string(
                    result.flip_grid_binding ? grid_left : grid_right) +
                ",\"preliminary_grid_y\":" +
                std::to_string(
                    result.flip_grid_binding ? grid_right : grid_left) +
                ",\"preliminary_grid_y_exceeds_split_threshold\":" +
                std::string(
                    (result.flip_grid_binding ? grid_right : grid_left) > 65535
                        ? "true"
                        : "false") +
                ",\"idle_right_vector_lanes_last_cta\":" +
                std::to_string(idle_right) +
                ",\"idle_left_rows_last_cta\":" + std::to_string(idle_left) +
                ",\"right_tail_elements_last_vector\":" +
                std::to_string(tail_elements) +
                ",\"unroll_status\":\"" +
                (exact_add2x16_continuation ? "bounded_by_candidate"
                                             : "pending") +
                "\",\"grid_split_status\":\"" +
                (exact_add2x16_continuation ? "none_for_carrier" : "pending") +
                "\"}";
            if (exact_add2x16_continuation) {
              const auto continuation =
                  pointwise::non_tma::addLaunch2DContinuation(
                      cur_left_elem_count,
                      result.right_elem_count,
                      max_vect_factor,
                      candidate,
                      /*expression_count=*/1);
              NVF_ERROR(
                  continuation.has_value() &&
                      continuation->threads_y == threads_y &&
                      continuation->preliminary_grid_left == grid_left &&
                      continuation->preliminary_grid_right == grid_right,
                  "PW3.launch exact Add continuation differs from source "
                  "preliminary geometry");
              using scheduler::native_diagnostic::Object;
              Object grammar;
              grammar.add("schema", "nvfuser.pw3.launch.add2x16.v1");
              grammar.add("plan_scope", "ordinary_pointwise_branch");
              grammar.add("left_elems", cur_left_elem_count);
              grammar.add("right_elems", result.right_elem_count);
              grammar.add("vector_width", max_vect_factor);
              grammar.add("threads_x", candidate);
              grammar.add("threads_y", threads_y);
              grammar.add("preliminary_grid_left", grid_left);
              grammar.add("preliminary_grid_right", grid_right);
              grammar.add(
                  "unroll_total_blocks",
                  continuation->preliminary_grid_left *
                      continuation->preliminary_grid_right);
              grammar.add("unroll_bound", continuation->unroll_bound);
              grammar.add(
                  "unroll_choices_kind",
                  continuation->unroll_bound == 1 ? "no_choice"
                                                  : "bounded_choice");
              grammar.add("unroll_range_low", int64_t{1});
              grammar.add("unroll_range_high", continuation->unroll_bound);
              grammar.add(
                  "unroll_bound_formula",
                  "min(max(1,L*R),64,max(1,floor(8192/E/W)))");
              grammar.add("post_unroll_right_formula", "ceil(R/U)");
              grammar.add(
                  "effective_grid_x_formula",
                  result.flip_grid_binding ? "L" : "ceil(R/U)");
              grammar.add(
                  "effective_grid_y_formula",
                  result.flip_grid_binding ? "ceil(R/U)" : "L");
              grammar.add(
                  "right_coverage_formula", "ceil(R/U)*X*W*U");
              grammar.add("right_tail_formula", "right_coverage-16");
              grammar.add("grid_split_status", "none_for_carrier");
              option.consequence_json.pop_back();
              option.consequence_json +=
                  ",\"continuation\":" + grammar.str() +
                  ",\"pending_choices\":" +
                  (continuation->unroll_bound == 1
                       ? std::string("[]}")
                       : std::string("[\"PW3.unroll\"]}"));
            }
          }
        }
        std::string state;
        if (semantic) {
          state = "{\"schema\":\"nvfuser.systemone.semantic_decision.v2\"";
          state +=
              ",\"objective\":\"Minimize expected runtime by choosing "
              "a legal 2D CTA thread split for this fusion; balance right-axis "
              "vector work, left-axis grid coverage, CTA occupancy, and idle "
              "or "
              "tail lanes.\"";
          state += ",\"workload\":" +
              semanticPointwiseWorkload(
                       prop.device_multiprocessor_count, device);
          state +=
              ",\"partial_schedule\":{\"scheduler\":\"ordinary_pointwise\"";
          state += ",\"breakpoint_analysis_requested\":true,\"break_point\":" +
              std::to_string(result.break_point) + ",\"flip_grid_binding\":" +
              std::string(result.flip_grid_binding ? "true" : "false") +
              ",\"left_elems\":" + std::to_string(cur_left_elem_count) +
              ",\"right_elems\":" + std::to_string(result.right_elem_count) +
              ",\"outer_broadcast_dominated\":" +
              std::string(
                       result.is_outer_broadcast_dominated ? "true" : "false") +
              ",\"vector_width\":" + std::to_string(max_vect_factor) +
              ",\"vector_width_already_selected\":true,\"unroll\":\"pending\","
              "\"grid_splitting\":\"pending\",\"grid_x_axis\":\"" +
              std::string(result.flip_grid_binding ? "left" : "right") +
              "\",\"grid_y_axis\":\"" +
              std::string(result.flip_grid_binding ? "right" : "left") + "\"}";
          state += ",\"decision\":{\"kind\":\"cta_split_2d\"";
          state += ",\"n_elems\":" + std::to_string(n_elems) +
              ",\"cta_threads_cap\":" + std::to_string(kThreadX) +
              ",\"grid_y_split_threshold\":65535" + ",\"candidates\":[";
          for (size_t i = 0; i < options.size(); ++i) {
            const int64_t candidate = std::stoll(options[i].id.substr(1));
            const int64_t threads_y = kThreadX / candidate;
            const int64_t grid_left = ceilDiv(cur_left_elem_count, threads_y);
            const int64_t grid_right = ceilDiv(after_vect, candidate);
            const int64_t idle_right =
                (candidate - after_vect % candidate) % candidate;
            const int64_t idle_left =
                (threads_y - cur_left_elem_count % threads_y) % threads_y;
            const int64_t tail_elements =
                (max_vect_factor - result.right_elem_count % max_vect_factor) %
                max_vect_factor;
            if (i != 0)
              state += ',';
            state +=
                "{\"id\":\"x" + std::to_string(candidate) +
                "\",\"threads_x\":" + std::to_string(candidate) +
                ",\"threads_y\":" + std::to_string(threads_y) +
                ",\"actual_cta_threads\":" +
                std::to_string(candidate * threads_y) +
                ",\"right_vector_iterations\":" + std::to_string(after_vect) +
                ",\"preliminary_grid_left\":" + std::to_string(grid_left) +
                ",\"preliminary_grid_right\":" + std::to_string(grid_right) +
                ",\"preliminary_grid_x\":" +
                std::to_string(
                    result.flip_grid_binding ? grid_left : grid_right) +
                ",\"preliminary_grid_y\":" +
                std::to_string(
                    result.flip_grid_binding ? grid_right : grid_left) +
                ",\"preliminary_grid_y_exceeds_split_threshold\":" +
                std::string(
                    (result.flip_grid_binding ? grid_right : grid_left) > 65535
                        ? "true"
                        : "false") +
                ",\"idle_right_vector_lanes_last_cta\":" +
                std::to_string(idle_right) +
                ",\"idle_left_rows_last_cta\":" + std::to_string(idle_left) +
                ",\"right_tail_elements_last_vector\":" +
                std::to_string(tail_elements) + '}';
          }
          state += "]}}";
        } else if (semantic_v3) {
          state = exact_add2x16_continuation ? "pointwise.launch.v3\n"
                                             : "pointwise.launch.v2\n";
          state += "n_elems=" + std::to_string(n_elems) + '\n';
          state +=
              "right_elems=" + std::to_string(result.right_elem_count) + '\n';
          state += "vector_width=" + std::to_string(max_vect_factor) + '\n';
          state += "warp_size=" + std::to_string(threads_per_warp) + '\n';
          state += "cta_threads=" + std::to_string(kThreadX) + '\n';
          state += "break_point=" + std::to_string(result.break_point) + '\n';
          state += "flip_grid_binding=" +
              std::to_string(static_cast<int>(result.flip_grid_binding)) + '\n';
          state += "max_threads_x=" +
              std::to_string(device->maxThreadsDim[0]) + '\n';
          state += "max_threads_y=" +
              std::to_string(device->maxThreadsDim[1]) + '\n';
          state += "max_threads_per_block=" +
              std::to_string(device->maxThreadsPerBlock) + '\n';
          state += "max_threads_per_sm=" +
              std::to_string(device->maxThreadsPerMultiProcessor) + '\n';
          state += "max_grid_x=" + std::to_string(device->maxGridSize[0]) + '\n';
          state += "max_grid_y=" + std::to_string(device->maxGridSize[1]) + '\n';
          state += "grid_y_split_threshold=65535\n";
          state += exact_add2x16_continuation
              ? "unroll_status=bounded_by_candidate\n"
              : "unroll_status=pending\n";
          state += exact_add2x16_continuation
              ? "grid_split_status=none_for_carrier\n"
              : "grid_split_status=pending\n";
          if (exact_add2x16_continuation) {
            state += "carrier=sm89_fp32_add_2x16\n";
            state += "expr_count=1\n";
            state += "unroll_max_cap=64\n";
            state += "unroll_expr_width_budget=8192\n";
            state += "left_elems=2\n";
          }
        } else {
          state = "pointwise.launch.v1\n";
          state += "n_elems=" + std::to_string(n_elems) + '\n';
          state +=
              "right_elems=" + std::to_string(result.right_elem_count) + '\n';
          state += "vector_width=" + std::to_string(max_vect_factor) + '\n';
          state += "warp_size=" + std::to_string(threads_per_warp) + '\n';
          state += "cta_threads=" + std::to_string(kThreadX) + '\n';
        }
        const std::string native_choice =
            jev_mode == scheduler::jev::Mode::Record
            ? "x" + std::to_string(result.bdimx)
            : std::string();
        const auto selected = semantic
            ? scheduler::jev::chooseSemanticDecision(
                  "PW3.launch",
                  state,
                  "Choose the legal 2D CTA thread split for this fusion. "
                  "The breakpoint, grid binding, and vector width are already "
                  "selected. Compare every candidate's CTA threads, "
                  "preliminary grid coverage, grid X/Y binding and the 65535 "
                  "grid Y split threshold, vector work, and idle or tail "
                  "lanes. Later unroll and grid splitting can change the "
                  "launch; return one listed split for expected runtime.",
                  options,
                  native_choice)
            : scheduler::jev::chooseDecision(
                  "PW3.launch", state, options, native_choice);
        bool matched = false;
        for (const auto& option : options) {
          if (selected == option.id) {
            result.bdimx = std::stoll(option.id.substr(1));
            matched = true;
            break;
          }
        }
        NVF_ERROR(matched, "PW3.launch response is outside hard CTA bounds");
      }
    } else if (jev_mode == scheduler::jev::Mode::Record && block_quantization) {
      scheduler::jev::recordCoverageSkip("PW3.launch", "block_quantization");
    }
    result.bdimy = kThreadX / result.bdimx;
    result.gdim_left = ceilDiv(cur_left_elem_count, result.bdimy);
    result.gdim_right =
        ceilDiv(result.right_elem_count, result.bdimx * max_vect_factor);
  }

  return result;
}

std::optional<CommonScheduleInfo> commonPointwiseSchedule(
    Fusion* fusion,
    int64_t break_point) {
  CommonScheduleInfo info;

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // Cache inputs
  info.cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  info.cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  refineCachePolicy(fusion);

  // Filter and collect input tensors (excluding hanging tensors)
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      info.input_tvs.push_back(tv);
    }
  }
  auto output_tvs_view = ir_utils::filterByType<TensorView>(fusion->outputs());
  info.output_tvs =
      std::vector<TensorView*>(output_tvs_view.begin(), output_tvs_view.end());

  // Find maximum dimensions across all inputs and outputs
  int64_t max_dims = 0;
  for (auto inp : info.input_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(inp), max_dims);
  }

  for (auto out : info.output_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(out), max_dims);
  }

  // If everything is zero dim tensors, return nullopt
  if (max_dims == 0) {
    return std::nullopt;
  }

  // Get reference tensor
  info.reference_tv = getReferenceTensor(fusion);
  NVF_ERROR(
      info.reference_tv != nullptr,
      "Could not find a fully broadcasted output to reference schedule on.");
  auto ref_orig_loop = info.reference_tv->getLoopDomain();

  // Move non-concretized broadcasts innermost
  scheduler_utils::moveNonConcretizedBroadcastInnermost(
      fusion, {info.reference_tv});
  int64_t num_parallel_dims =
      scheduler_utils::countLeadingParallelDimensions(info.reference_tv);
  int64_t device_aware_break_point = break_point + num_parallel_dims;

  // Positions of rhs and lhs after merging all dimensions.
  int64_t rhs_i = -1;
  int64_t lhs_i = -1;

  if (!ir_utils::getReshapeOps(fusion).empty()) {
    // Propagate reshape transforms through the graph, expecially the reference.
    scheduler_utils::propagateReshapeTransforms(fusion);

    // Reorder reference_tv after propagating the view operation. This will
    // reorder for better merging.
    info.reference_tv->reorder(
        scheduler_utils::domainReorderAsLogicalMap(info.reference_tv));
    // Reorder so that DeviceDims are in front
    reorderParallelizedToFront(info.reference_tv);

    // Break point is relative to logical domain, find the loop domain ID's in
    // the left/right side, we really need the values in domain, but easiest way
    // to do this is with Dependency check which will grab all intermediate
    // values too.
    auto lhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + num_parallel_dims,
         ref_orig_loop.begin() + device_aware_break_point},
        {info.reference_tv->getLoopDomain().begin() + num_parallel_dims,
         info.reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> lhs_all_vals_set(
        lhs_all_vals.begin(), lhs_all_vals.end());

    auto rhs_all_vals = DependencyCheck::getAllValsBetween(
        {ref_orig_loop.begin() + device_aware_break_point, ref_orig_loop.end()},
        {info.reference_tv->getLoopDomain().begin() + num_parallel_dims,
         info.reference_tv->getLoopDomain().end()});

    std::unordered_set<Val*> rhs_all_vals_set(
        rhs_all_vals.begin(), rhs_all_vals.end());

    // Make sure lhs and rhs groups are disjoint.
    for (auto lhs_val : lhs_all_vals) {
      if (rhs_all_vals_set.count(lhs_val) != 0) {
        std::ostringstream os;
        IrTransformPrinter printer(os);
        printer.printTransforms(info.reference_tv);
        NVF_THROW(
            "Error in pointwise scheduler. LHS and RHS of the 2D scheduler are "
            "not disjoint. ",
            lhs_val->toString(),
            " belongs to both. device_aware_break_point = ",
            device_aware_break_point,
            ". reference_tv = ",
            info.reference_tv->toString(),
            " and its transforms are:\n",
            os.str());
      }
    }
    NVF_ERROR(
        !rhs_all_vals.empty(),
        "Expecting at least one dimension in the RHS of the pointwise "
        "scheduler.");

    // Merge rhs, then lhs.
    IterDomain* rhs_id = nullptr;
    IterDomain* lhs_id = nullptr;
    for (int64_t pos = info.reference_tv->nDims() - 1; pos >= 0; pos--) {
      // Merge from right to left
      auto id = info.reference_tv->axis(pos);
      if (lhs_all_vals_set.count(id) > 0) {
        if (lhs_id == nullptr) {
          lhs_id = id;
          lhs_i = pos;
        } else {
          info.reference_tv->merge(pos, lhs_i);
          lhs_i = pos;
          if (rhs_i > lhs_i) {
            rhs_i--;
          }
        }
      } else if (rhs_all_vals_set.count(id) > 0) {
        if (rhs_id == nullptr) {
          rhs_id = id;
          rhs_i = pos;
        } else {
          info.reference_tv->merge(pos, rhs_i);
          rhs_i = pos;
          if (lhs_i > rhs_i) {
            lhs_i--;
          }
        }
      }
    }
    // Find the iter domains that should be in the lhs, and rhs.
  } else {
    // Don't need to worry about view transformations, just merge reference tv
    // as we normally would.

    std::unordered_map<int64_t, int64_t> loop_reorder_map =
        scheduler_utils::reorderLoopAsAllocationMap(info.reference_tv);
    if (!loop_reorder_map.empty()) {
      info.reference_tv->reorder(loop_reorder_map);
    }
    reorderParallelizedToFront(info.reference_tv);

    // Merge right side of break point
    for (int64_t i = info.reference_tv->nDims(); i > device_aware_break_point;
         i--) {
      auto axis_i = i - 1;
      if (rhs_i == -1) {
        rhs_i = axis_i;
      } else {
        info.reference_tv->merge(axis_i, rhs_i);
        rhs_i = axis_i;
      }
    }
    if (rhs_i >= 0) {
      // If there's an rhs
      info.reference_tv->reorder({{rhs_i, -1}});
    }

    // Merge left side of break point
    for (int64_t i = device_aware_break_point; i > num_parallel_dims; i--) {
      auto axis_i = i - 1;
      if (lhs_i == -1) {
        lhs_i = axis_i;
      } else {
        info.reference_tv->merge(axis_i, lhs_i);
        lhs_i = axis_i;
      }
    }
  }

  // Store the positions of lhs and rhs merged dimensions
  info.lhs_i = lhs_i;
  info.rhs_i = rhs_i;

  return info;
}

} // namespace pointwise_utils
} // namespace nvfuser
