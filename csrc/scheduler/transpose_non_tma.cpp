// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/transpose_non_tma.h"

#include <ATen/cuda/CUDAContext.h>

#include "debug.h"
#include "instrumentation.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/reduction_utils.h"
#include "scheduler/registry_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/transpose_utils.h"
#include "scheduler/utils.h"
#include "scheduler/vectorize_helper.h"

namespace nvfuser {
namespace transpose {
namespace non_tma {

namespace {

// If a fusion is segmented, the segmenter will create fusions whose inputs
// contain reduction IterDomains. These reduction IterDomains on input
// TensorViews does not have any meaning, and should just be left untouched. See
// https://github.com/NVIDIA/Fuser/issues/1659#issuecomment-1907053830
//
// This function checks the inner `n` iterdomain and reorder reduction
// iterdomain to the beginning.
void moveReductionsOut(TensorView* tv, int n) {
  if (!tv->isFusionInput()) {
    return;
  }

  std::unordered_map<int64_t, int64_t> old2new;

  int64_t target = 0;
  for (int64_t i = 0; i < n; i++) {
    if (tv->axis(-1 - i)->isReduction()) {
      old2new[-1 - i] = target++;
    }
  }

  tv->reorder(old2new);
}

// This is for preventing dangling broadcast IDs from interferring the
// scheduling of the fusion. Returns the number of moved IDs.
int64_t moveDanglingBroadcastInner(TensorView* tv) {
  std::unordered_map<int64_t, int64_t> old2new;

  int64_t target = -1;
  for (const auto& [i, id] :
       enumerate(tv->getLoopDomain()) | std::views::reverse) {
    if (std::ranges::find(tv->getLogicalDomain(), id) ==
        tv->getLogicalDomain().end()) {
      continue;
    }

    // We are interested in non-scheduled dangling broadcast. For the
    // use case of this function, a dangling ID should be a broadcast.
    NVF_ERROR(
        id->isBroadcast(),
        "Unexpected dangling loop ID: ",
        id->toString(),
        " of ",
        tv->toString());
    old2new[i] = target--;
  }

  tv->reorder(old2new);

  return std::ssize(old2new);
}

HeuristicDataCacheEntry<HeuristicCompileTime::TransposeDomainMap> getDomainMap(
    HeuristicDataCache* data_cache,
    Fusion* fusion) {
  auto domain_map_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::TransposeDomainMap>(
          data_cache, [fusion]() {
            return std::make_unique<scheduler_tools::TransposeDomainMap>(
                fusion);
          });
  return domain_map_entry;
}

HeuristicDataCacheEntry<HeuristicCompileTime::InputsOutputsInnerDimGroups>
getInputsOutputsGroups(
    HeuristicDataCache* data_cache,
    scheduler_tools::TransposeDomainMap& domain_map) {
  auto grouped_inputs_outputs_entry = HeuristicDataCacheEntry<
      HeuristicCompileTime::InputsOutputsInnerDimGroups>(
      data_cache, [&domain_map]() {
        return std::make_unique<std::vector<std::vector<TensorView*>>>(
            domain_map.groupInputsOutputsByInnerDim());
      });
  auto& grouped_inputs_outputs = grouped_inputs_outputs_entry.get();

  NVF_ERROR(
      grouped_inputs_outputs.size() >= 2,
      "Can not find mismatched inner most dim, should use pointwise "
      "scheduler.");

  return grouped_inputs_outputs_entry;
}

HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensorsForGroups>
getReferenceTensors(
    HeuristicDataCache* data_cache,
    scheduler_tools::TransposeDomainMap& domain_map,
    std::vector<std::vector<TensorView*>>& grouped_inputs_outputs) {
  auto reference_tensors_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::ReferenceTensorsForGroups>(
          data_cache, [&domain_map, &grouped_inputs_outputs]() {
            std::vector<TensorView*> data{
                domain_map.findReferenceFor(grouped_inputs_outputs[0]),
                domain_map.findReferenceFor(grouped_inputs_outputs[1])};
            return std::make_unique<std::vector<TensorView*>>(std::move(data));
          });
  auto& reference_tensors = reference_tensors_entry.get();
  NVF_ERROR(reference_tensors.size() == 2);
  TensorView* reference1 = reference_tensors[0];
  TensorView* reference2 = reference_tensors[1];
  NVF_ERROR(
      reference1 != nullptr, "Unable to find reference tensor for group 1");
  NVF_ERROR(
      reference2 != nullptr, "Unable to find reference tensor for group 2");
  return reference_tensors_entry;
}

std::pair<std::vector<int64_t>, int64_t> getLoopDomainSizes(
    HeuristicDataCache* data_cache,
    SchedulerRuntimeInfo& runtime_info,
    TensorView* reference,
    scheduler_tools::TransposeDomainMap& domain_map) {
  auto ref_loop = reference->getLoopDomain();
  std::vector<int64_t> shape_in_ref;
  shape_in_ref.reserve(reference->nDims());
  int64_t n_elems = 1;
  for (auto id : ref_loop) {
    auto concrete_id = domain_map.getComputeAtMap().getConcreteMappedID(
        id, IdMappingMode::EXACT);
    auto inferred_val =
        runtime_info.expressionEvaluator().evaluate(concrete_id->extent());
    NVF_ERROR(
        inferred_val.hasValue(),
        "Error inferring size for pointwise scheduler: ",
        id->extent()->toInlineString());
    int64_t size = inferred_val.as<int64_t>();
    n_elems *= size;
    shape_in_ref.push_back(size);
  }
  return {shape_in_ref, n_elems};
}

HeuristicDataCacheEntry<HeuristicCompileTime::InnerMostDimInfo>
getInnerMostDimInfoInReference(
    HeuristicDataCache* data_cache,
    const std::vector<TensorView*>& group_references,
    TensorView* global_reference,
    scheduler_tools::TransposeDomainMap& domain_map) {
  auto innermost_info_entry =
      HeuristicDataCacheEntry<HeuristicCompileTime::InnerMostDimInfo>(
          data_cache, [&]() {
            std::vector<int64_t> data;
            data.reserve(group_references.size());
            for (auto ref_tv : group_references) {
              auto inner_most_id = scheduler_utils::innerMostAllocDim(ref_tv);
              auto inner_most_pos_in_global_ref =
                  domain_map.getInnerLeafDim(global_reference, inner_most_id);
              data.push_back(inner_most_pos_in_global_ref);
            }
            return std::make_unique<std::vector<int64_t>>(std::move(data));
          });
  return innermost_info_entry;
}

} // namespace

std::unique_ptr<TransposeParams> getTransposeHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache) {
  FusionGuard fg(fusion);

  // Incase any buffer is of type DataType::Index
  const auto index_type = runtime_info.getIndexType();

  auto domain_map_entry = getDomainMap(data_cache, fusion);
  auto& domain_map = dynamic_cast<scheduler_tools::TransposeDomainMap&>(
      domain_map_entry.get());
  auto grouped_inputs_outputs_entry =
      getInputsOutputsGroups(data_cache, domain_map);
  auto grouped_inputs_outputs = grouped_inputs_outputs_entry.get();
  auto reference_tensors_entry =
      getReferenceTensors(data_cache, domain_map, grouped_inputs_outputs);
  auto reference_tensors = reference_tensors_entry.get();
  TensorView* reference1 = reference_tensors[0];
  TensorView* reference2 = reference_tensors[1];
  auto [shape_in_ref1, n_elems] =
      getLoopDomainSizes(data_cache, runtime_info, reference1, domain_map);

  const int64_t device_multiprocessor_count =
      (int64_t)at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  auto innermost_info_entry = getInnerMostDimInfoInReference(
      data_cache, reference_tensors, reference1, domain_map);
  auto innermost_info = innermost_info_entry.get();

  auto inner_most_pos1_in_ref1 = innermost_info[0];
  auto inner_most_pos2_in_ref1 = innermost_info[1];
  // No exact innermost loop dimension mapping on referenc1. cannot schedule
  if (inner_most_pos1_in_ref1 < 0 || inner_most_pos2_in_ref1 < 0) {
    NVF_THROW(
        "Transpose scheduler requires exact mapping on inner most dimension on "
        "reference tensor.");
  }

  auto tparams = std::make_unique<TransposeParams>();
  tparams->tag = "Transpose heuristics";
  tparams->cparams.index_type = index_type;

  // The one-wave test is a performance preference. Keep the exact native
  // merge/split constructor and the view guard as hard schedule constraints.
  const auto jev_mode = scheduler::jev::mode();
  const bool has_view = !scheduler_utils::getViewTVs(fusion).empty();
  if (jev_mode == scheduler::jev::Mode::Off) {
    utils::maybeBuildVirtualInnerDims(
        tparams.get(),
        device_multiprocessor_count,
        n_elems,
        shape_in_ref1,
        inner_most_pos1_in_ref1,
        inner_most_pos2_in_ref1);
  } else {
    // Preview the exact source constructor without mutating the selected
    // plan. Its one-wave return and already-satisfied axes can both make the
    // nominal virtual option identical to the original-axis plan.
    auto virtual_preview = std::make_unique<TransposeParams>();
    virtual_preview->tile_size1 = tparams->tile_size1;
    virtual_preview->tile_size2 = tparams->tile_size2;
    utils::maybeBuildVirtualInnerDims(
        virtual_preview.get(),
        device_multiprocessor_count,
        n_elems,
        shape_in_ref1,
        inner_most_pos1_in_ref1,
        inner_most_pos2_in_ref1);
    const bool virtual_changes_axes =
        utils::hasSmallTransposeDimensions(virtual_preview);
    if (has_view) {
      // Preserve the pre-existing unsupported view combination even when a
      // remote answer would otherwise request the original axes.
      NVF_ERROR(
          !virtual_changes_axes,
          "combination of view op with small transpose dimensions are not "
          "supported by transpose scheduler");
    }

    if (has_view || !virtual_changes_axes) {
      scheduler::jev::recordNoChoiceDecision(
          "TR1",
          "original",
          has_view ? "view_disallows_virtual_axes"
                   : "virtual_axis_constructor_no_effect",
          "native_axis_plan");
    } else {
      std::string state = "transpose.virtual_axes.v1\nshape=";
      for (size_t i = 0; i < shape_in_ref1.size(); ++i) {
        if (i != 0) {
          state += ',';
        }
        state += std::to_string(shape_in_ref1[i]);
      }
      state += "\ninner1=" + std::to_string(inner_most_pos1_in_ref1);
      state += "\ninner2=" + std::to_string(inner_most_pos2_in_ref1);
      state += "\ntile1=" + std::to_string(tparams->tile_size1);
      state += "\ntile2=" + std::to_string(tparams->tile_size2);
      state += "\nsm_count=" + std::to_string(device_multiprocessor_count);
      state += "\nn_elems=" + std::to_string(n_elems);
      state += "\nhas_view=" + std::to_string(has_view);

      std::vector<scheduler::jev::ChoiceOption> options = {
          {"original", "Tile the original inner axes"},
          {"virtual", "Construct the source-supported virtual inner axes"}};

      std::string native_choice;
      if (jev_mode == scheduler::jev::Mode::Record) {
        const int64_t wave_elements = device_multiprocessor_count *
            std::min<int64_t>(shape_in_ref1[inner_most_pos1_in_ref1],
                              tparams->tile_size1) *
            std::min<int64_t>(shape_in_ref1[inner_most_pos2_in_ref1],
                              tparams->tile_size2);
        native_choice = wave_elements < n_elems ? "virtual" : "original";
      }
      const auto selected = scheduler::jev::chooseDecision(
          "TR1", state, options, native_choice);
      NVF_ERROR(
          selected == "original" || selected == "virtual",
          "TR1 response is outside native hard bounds");
      if (selected == "virtual") {
        // The API chooses whether to request this complete native plan; it never
        // supplies individual merge indices or split factors.
        utils::maybeBuildVirtualInnerDims(
            tparams.get(),
            device_multiprocessor_count,
            n_elems,
            shape_in_ref1,
            inner_most_pos1_in_ref1,
            inner_most_pos2_in_ref1);
      }
    }
  }

  NVF_ERROR(
      !utils::hasSmallTransposeDimensions(tparams) || !has_view,
      "combination of view op with small transpose dimensions are not "
      "supported by transpose scheduler");

  // Choose a complete, square tile before the vector/unroll and CTA analyses.
  // The smaller candidate is confined to untransformed original axes whose
  // extents divide both tiles. All dependent parameters are recomputed below.
  if (jev_mode != scheduler::jev::Mode::Off) {
    std::string state = "transpose.tile.v1\nshape=";
    for (size_t i = 0; i < shape_in_ref1.size(); ++i) {
      if (i != 0) {
        state += ',';
      }
      state += std::to_string(shape_in_ref1[i]);
    }
    state += "\ninner1=" + std::to_string(inner_most_pos1_in_ref1);
    state += "\ninner2=" + std::to_string(inner_most_pos2_in_ref1);
    state += "\ntile1=" + std::to_string(tparams->tile_size1);
    state += "\ntile2=" + std::to_string(tparams->tile_size2);
    state += "\nsm_count=" + std::to_string(device_multiprocessor_count);
    state += "\nn_elems=" + std::to_string(n_elems);
    state += "\nhas_view=" + std::to_string(has_view);

    const bool square_16_eligible =
        !has_view && !utils::hasSmallTransposeDimensions(tparams) &&
        tparams->tile_size1 == TransposeParams::getDefaultTileSize() &&
        tparams->tile_size2 == TransposeParams::getDefaultTileSize() &&
        shape_in_ref1[inner_most_pos1_in_ref1] >= 32 &&
        shape_in_ref1[inner_most_pos2_in_ref1] >= 32 &&
        shape_in_ref1[inner_most_pos1_in_ref1] % 32 == 0 &&
        shape_in_ref1[inner_most_pos2_in_ref1] % 32 == 0 &&
        n_elems / device_multiprocessor_count >= 32 * 32;
    std::vector<scheduler::jev::ChoiceOption> options = {
        {"tile32", "Tile both transpose axes by 32"}};
    if (square_16_eligible) {
      options.push_back({"tile16", "Tile both transpose axes by 16"});
    }
    const auto selected = scheduler::jev::chooseDecision(
        "TR2.tile",
        state,
        options,
        jev_mode == scheduler::jev::Mode::Record ? "tile32" : "");
    if (selected == "tile16" && square_16_eligible) {
      tparams->tile_size1 = 16;
      tparams->tile_size2 = 16;
    } else {
      NVF_ERROR(
          selected == "tile32",
          "TR2.tile response is outside native hard bounds");
    }
  }

  // Note [vectorization and unroll of input and output]
  //
  // The choice of vectorization size, block size and tile sizes needs to be
  // consistent with each other. Consider the following:
  //
  // The number of threads in one block is
  //   num_threads = blockDim.x * blockDim.y
  // and the number of elements per each tile is
  //   num_elems_per_tile = tparams->tile_size1 * tparams->tile_size2
  // So each thread needs to process
  //   num_elems_per_thread = num_elems_per_tile / num_threads
  // elements. That is, once the tile sizes and block size are determined, the
  // `num_elems_per_thread` is determined, regardless of vectorizability of
  // input/output tensors.
  //
  // To make the selection of tile sizes othogonal to vectorizability, we
  // support having both vectorization and unrolling in the same tensor. For
  // example, if we have num_elems_per_tile == 1024 and num_threads = 256, then
  // we have num_elems_per_thread being 4. And if we have vector size 2, then we
  // will do unroll 2 * vectorize 2 at the same tensor.
  //
  // Also, since the inner most dim of different groups are not the same, it is
  // natural to consider their vectorizability separately and allow them to have
  // different vectorize/unroll sizes.

  constexpr int64_t kSixteen = 16; // clang tidy

  int64_t max_io_dtype_size = 1;
  int64_t n_io_tensors = 0;
  auto scan_max_dtype_size = [&](const auto& vals) {
    for (auto inp : ir_utils::filterByType<TensorView>(vals)) {
      max_io_dtype_size = std::max(
          max_io_dtype_size, dataTypeSizeByte(inp->getDataType(), index_type));
      n_io_tensors++;
    }
  };
  scan_max_dtype_size(fusion->inputs());
  scan_max_dtype_size(fusion->outputs());

  int64_t max_unroll_factor = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    // Keep the original numeric cap in off mode. Record uses it to identify
    // the native Choice; the selected effective cap is applied below.
    max_unroll_factor = ceilDiv(
        // Available unrolling based on size of data type
        kSixteen / max_io_dtype_size,
        // Reduce max unrolling factor if we have many inputs/outputs to unroll
        // as it could start consuming a lot of registers.
        std::max(
            (scheduler_utils::lastPow2(
                 (int64_t)grouped_inputs_outputs[0].size() +
                 (int64_t)grouped_inputs_outputs[1].size()) >>
             2),
            (int64_t)1));

    // Don't unroll at the cost of getting a full wave on the GPU
    auto max_unroll_factor_occupancy = ceilDiv(
        n_elems,
        device_multiprocessor_count * (int64_t)tparams->tile_size1 *
            (int64_t)tparams->tile_size2);
    max_unroll_factor =
        std::min(max_unroll_factor, max_unroll_factor_occupancy);

    // Don't unroll at the cost of getting a full warp, useful for the case
    // where tile sizes are small
    auto max_unroll_factor_block = ceilDiv(
        (int64_t)tparams->tile_size1 * (int64_t)tparams->tile_size2, 32l);
    max_unroll_factor = std::min(max_unroll_factor, max_unroll_factor_block);
  }

  if (jev_mode != scheduler::jev::Mode::Off) {
    // A cap above 16 bytes per I/O element can form a vector instruction that
    // the native rule never offers. The tile and byte limits are structural;
    // neither uses the native register, wave, or warp preference.
    const int64_t tile_elements =
        (int64_t)tparams->tile_size1 * (int64_t)tparams->tile_size2;
    const int64_t hard_cap =
        std::min(tile_elements, kSixteen / max_io_dtype_size);
    std::vector<scheduler::jev::ChoiceOption> options;
    if (hard_cap == 0) {
      // Preserve the legacy zero input for an I/O dtype wider than 16 bytes.
      options.push_back({"u0", "Use the legacy zero helper input"});
    } else {
      for (int64_t factor = 1; factor <= hard_cap; factor *= 2) {
        options.push_back(
            {"u" + std::to_string(factor),
             "Limit the per-thread vector/unroll cap to " +
                 std::to_string(factor)});
      }
    }

    const std::string state =
        "transpose.unroll_cap.v1\nmax_io_dtype_bytes=" +
        std::to_string(max_io_dtype_size) + "\ngroup1_io=" +
        std::to_string(grouped_inputs_outputs[0].size()) + "\ngroup2_io=" +
        std::to_string(grouped_inputs_outputs[1].size()) + "\nn_elems=" +
        std::to_string(n_elems) + "\nsm_count=" +
        std::to_string(device_multiprocessor_count) + "\ntile1=" +
        std::to_string(tparams->tile_size1) + "\ntile2=" +
        std::to_string(tparams->tile_size2);
    const std::string native_choice =
        jev_mode == scheduler::jev::Mode::Record
        ? "u" + std::to_string(
                    max_unroll_factor == 0
                        ? 0
                        : scheduler_utils::maxVectorizationWidth(
                              max_unroll_factor))
        : "";
    const auto selected = scheduler::jev::chooseDecision(
        "TR2.unroll", state, options, native_choice);
    // The helper uses maxVectorizationWidth, so Record's selected power of two
    // is the same effective cap as the native raw value. Apply the selected cap
    // in both modes to keep equivalent downstream width requests identical.
    bool found = false;
    if (hard_cap == 0 && selected == "u0") {
      max_unroll_factor = 0;
      found = true;
    } else {
      for (int64_t factor = 1; factor <= hard_cap; factor *= 2) {
        if (selected == "u" + std::to_string(factor)) {
          max_unroll_factor = factor;
          found = true;
          break;
        }
      }
    }
    NVF_ERROR(found, "TR2.unroll response is outside native hard bounds");
  }

  // Note: [Computing Vectorization Width for Transpose]
  //
  // With support of small transpose dimension (see Note [Supporting small
  // transpose dimensions]), we need to consider the transformation applied on
  // our tile to compute the safe vectorization width. e.g. For a simple
  // transpose:
  //    (i0, i1, i2, i3, i4) size (2, 3, 65536, 4, 7)
  // -> (i3, i4, i2, i0, i1)
  //
  // transpose scheduler would apply transformation and tile on virtual
  // innermost dimensions. ( (i0*i1*i2/2), (2*i3*i4)/32) So we need to use the
  // size of the virtual innermost dimensions to compute our vectorization
  // width. In the example above, we are looking at 2*3*65536/2, 2*4*7.
  //
  // Currently there's limitation on our iter domain mapping. Since we can only
  // do it on rfactor/root domain, we cannot map across `split` domains. So the
  // example above will only have vectorization size of 2 and 4 repsectively for
  // the merge virtual innermost dimensions, rather than considering the split
  // and merged i2/2 & 2.
  //
  // TODO: We use ContiguousInnerDimensionsMapper to compute the size of virtual
  // innermost dimension. The analysis right now is limited on logical domain
  // only, so we can't actually map the `split` iter domains, which limits the
  // vectorization width we can apply. We need to fix that.
  // TODO 2: Small transpose dimensions transformation should also consider the
  // vectorization impact. i.e. when split_before_tiling, we should try to split
  // on a factor that allows vectorization.
  {
    // duplicating reference1's TensorDomain, since the transformations applied
    // is not persistent and only needed for us to compute vectorization width.
    TensorDomain* cloned_1_td =
        IrBuilder::create<TensorDomain>(reference1->domain());
    // Adding a domain_guard so we can transform reference1
    ir_utils::TVDomainGuard domain_guard(reference1, cloned_1_td);

    std::vector<int64_t> to_update(
        {inner_most_pos1_in_ref1, inner_most_pos2_in_ref1});
    // we only apply split here, since we want to merge split dimensions, we can
    // simply map those merged domains via ContiguousInnerDimensionsMapper
    scheduler_utils::splitDims(
        reference1, tparams->split_before_tiling, to_update);
    inner_most_pos1_in_ref1 = to_update[0];
    inner_most_pos2_in_ref1 = to_update[1];

    auto choose_group_width = [&](int64_t group,
                                  int64_t inner_dim,
                                  const std::vector<int64_t>& merged_dims,
                                  const std::vector<TensorView*>& group_tvs)
        -> int64_t {
      if (jev_mode == scheduler::jev::Mode::Off) {
        return vectorize_helper::getVectorizationFactorTransposeGroup(
            runtime_info,
            reference1,
            inner_dim,
            merged_dims,
            group_tvs,
            max_unroll_factor);
      }

      // The mapper, not the API, determines which contiguous inner regions
      // each tensor can vectorize. All smaller powers of two remain legal.
      const auto mapped_sizes =
          vectorize_helper::getTransposeGroupContiguousInnerSizes(
              runtime_info, reference1, inner_dim, merged_dims, group_tvs);
      NVF_ERROR(!mapped_sizes.empty(), "Transpose group has no I/O tensors");
      int64_t max_legal =
          scheduler_utils::maxVectorizationWidth(max_unroll_factor);
      for (auto size : mapped_sizes) {
        NVF_ERROR(size > 0, "Transpose contiguous inner size must be positive");
        max_legal = std::min(
            max_legal, scheduler_utils::maxVectorizationWidth(size));
      }
      NVF_ERROR(max_legal > 0, "Transpose vector width must be positive");

      // With a normal cap, TR2.unroll has already limited each width to the
      // tile work and 16 bytes per widest I/O type. The legacy zero helper
      // input is different: it means an effectively unbounded cap, not width 0.
      const bool legacy_zero_cap = max_unroll_factor == 0;
      if (!legacy_zero_cap) {
        NVF_ERROR(
            max_legal <=
                    (int64_t)tparams->tile_size1 * tparams->tile_size2 &&
                max_legal <= kSixteen / max_io_dtype_size,
            "Transpose group width exceeds native tile or byte bound");
      }

      std::vector<scheduler::jev::ChoiceOption> options;
      if (legacy_zero_cap) {
        // Retain the original helper's exact sentinel behavior; no new
        // width above the original result or zero-width option is exposed.
        options.push_back(
            {"w" + std::to_string(max_legal),
             "Use the only legacy-zero-cap group width"});
      } else {
        for (int64_t factor = 1; factor <= max_legal; factor *= 2) {
          options.push_back(
              {"w" + std::to_string(factor),
               "Vectorize this transpose group by " + std::to_string(factor)});
        }
      }

      std::string state =
          "transpose.group_width.v1\ngroup=" + std::to_string(group) +
          "\nmax_unroll_input=" + std::to_string(max_unroll_factor) +
          "\nmax_io_dtype_bytes=" + std::to_string(max_io_dtype_size) +
          "\ntile1=" + std::to_string(tparams->tile_size1) +
          "\ntile2=" + std::to_string(tparams->tile_size2) +
          "\nmapped_sizes=";
      for (size_t i = 0; i < mapped_sizes.size(); ++i) {
        if (i != 0) {
          state += ',';
        }
        state += std::to_string(mapped_sizes[i]);
      }

      std::string native_choice;
      if (jev_mode == scheduler::jev::Mode::Record) {
        const int64_t native_width =
            vectorize_helper::getVectorizationFactorTransposeGroup(
                runtime_info,
                reference1,
                inner_dim,
                merged_dims,
                group_tvs,
                max_unroll_factor);
        NVF_ERROR_EQ(native_width, max_legal);
        native_choice = "w" + std::to_string(native_width);
      }
      const auto selected = scheduler::jev::chooseDecision(
          "TR2.width" + std::to_string(group),
          state,
          options,
          native_choice);
      for (const auto& option : options) {
        if (selected == option.id) {
          return std::stoll(selected.substr(1));
        }
      }
      NVF_THROW("TR2 group width response is outside native hard bounds");
    };

    tparams->vectorize_factor1 = choose_group_width(
        1,
        inner_most_pos1_in_ref1,
        tparams->dims_merged_with_1,
        grouped_inputs_outputs[0]);

    // TODO: Since group2 only has global->shared and shared->global set op, we
    // can have fine-grained control of unroll/vectorization at per tensor
    // level. We should not be using a single global vectorize factor for the
    // entire group 2
    tparams->vectorize_factor2 = choose_group_width(
        2,
        inner_most_pos2_in_ref1,
        tparams->dims_merged_with_2,
        grouped_inputs_outputs[1]);
  }

  if (jev_mode != scheduler::jev::Mode::Off) {
    // The tile and group widths have already been chosen. Both groups use the
    // same TIDx extent, while the tile, block grid, and shared allocations are
    // unchanged by this choice. Require exact tile coverage, a bounded amount
    // of per-thread tile work, and the device launch bound for alternatives.
    const int64_t tile_elements = tparams->tile_size1 * tparams->tile_size2;
    const int64_t native_threads = tparams->getThreadsPerBlock();
    const int64_t tile_vectors1 =
        ceilDiv(tile_elements, tparams->vectorize_factor1);
    const int64_t tile_vectors2 =
        ceilDiv(tile_elements, tparams->vectorize_factor2);
    const int64_t device_max_threads =
        at::cuda::getCurrentDeviceProperties()->maxThreadsPerBlock;
    const int64_t device_max_threads_x =
        at::cuda::getCurrentDeviceProperties()->maxThreadsDim[0];
    std::vector<scheduler::jev::ChoiceOption> options = {
        {"t" + std::to_string(native_threads),
         "Use " + std::to_string(native_threads) +
             " threads per CTA for both transpose groups"}};
    const bool exact_tile_vectors =
        tile_elements % tparams->vectorize_factor1 == 0 &&
        tile_elements % tparams->vectorize_factor2 == 0;
    for (const int64_t threads : {64L, 256L}) {
      if (threads == native_threads || !exact_tile_vectors ||
          threads > device_max_threads || threads > device_max_threads_x ||
          tile_elements % threads != 0 ||
          tile_elements / threads > 16 || tile_vectors1 % threads != 0 ||
          tile_vectors2 % threads != 0) {
        continue;
      }
      options.push_back(
          {"t" + std::to_string(threads),
           "Use " + std::to_string(threads) +
               " threads per CTA for both transpose groups"});
    }
    const std::string state =
        "transpose.cta.v1\ntile1=" + std::to_string(tparams->tile_size1) +
        "\ntile2=" + std::to_string(tparams->tile_size2) +
        "\nwidth1=" + std::to_string(tparams->vectorize_factor1) +
        "\nwidth2=" + std::to_string(tparams->vectorize_factor2) +
        "\nmax_threads_per_block=" + std::to_string(device_max_threads) +
        "\nmax_threads_x=" + std::to_string(device_max_threads_x);
    const auto selected = scheduler::jev::chooseDecision(
        "TR2.cta",
        state,
        options,
        jev_mode == scheduler::jev::Mode::Record
            ? "t" + std::to_string(native_threads)
            : "");
    if (jev_mode == scheduler::jev::Mode::Remote) {
      bool found = false;
      for (const auto& option : options) {
        if (selected == option.id) {
          if (selected != "t" + std::to_string(native_threads)) {
            tparams->threads_per_block_cap = std::stoll(selected.substr(1));
          }
          found = true;
          break;
        }
      }
      NVF_ERROR(found, "TR2.cta response is outside native hard bounds");
    }
  }

  tparams->lparams.bind(tparams->getThreadsPerBlock(), ParallelType::TIDx);

  if (isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n===== Transpose Stats ========\n"
            << "inputs: " << ir_utils::toString(fusion->inputs()) << "\n"
            << "outputs: " << ir_utils::toString(fusion->outputs()) << "\n"
            << "shape: " << shape_in_ref1 << "\n"
            << "num_elems: " << n_elems << "\n"
            << "n_io_tensors: " << n_io_tensors << "\n"
            << "max_io_dtype_size: " << max_io_dtype_size << "\n"
            << "group 1: " << ir_utils::toString(grouped_inputs_outputs[0])
            << "\n"
            << "reference1: " << reference1->toString() << "\n"
            << "inner_most_id1 position: " << inner_most_pos1_in_ref1
            << " (in reference 1)\n"
            << "group 2: " << ir_utils::toString(grouped_inputs_outputs[1])
            << "\n"
            << "reference2: " << reference2->toString() << "\n"
            << "inner_most_id2 position: " << inner_most_pos2_in_ref1
            << " (in reference 1)\n";
    if (utils::hasSmallTransposeDimensions(tparams)) {
      debug() << "small transposed dim, needs virtual inner-most dim\n";
    }
    debug() << '\n';
    debug() << tparams->toString() << '\n';
  }

  return tparams;
}

void scheduleTranspose(Fusion* fusion, const TransposeParams* tparams) {
  FusionGuard fg(fusion);

  // Make sure we don't have global memory set on intermediate tensors from
  // fusion segmentation
  scheduler_utils::clearMemorySpace(fusion);

  // maybe has_reduction for scheduling should be done on a per output tensor
  // basis.
  NVF_ERROR(
      !ir_utils::hasAnyReductionOps(fusion),
      "This scheduler only handles pointwise ops.");

  // Cache inputs
  auto cached_inputs = scheduler_utils::cacheInputs(fusion, true);

  // Cache and fork outputs
  auto cached_outputs = scheduler_utils::cacheAndForkOutputs(fusion, true);

  scheduler_utils::prepareForMemoryTypePromotion(fusion);

  std::vector<TensorView*> input_tvs;
  {
    auto filtered_tvs = ir_utils::filterByType<TensorView>(fusion->inputs());
    // Remove hanging tensor views
    for (auto tv : filtered_tvs) {
      if (tv->uses().empty()) {
        continue;
      }
      input_tvs.push_back(tv);
    }
  }
  auto output_tvs = ir_utils::filterByType<TensorView>(fusion->outputs());

  int64_t max_dims = 0;
  for (auto inp : input_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(inp), max_dims);
  }

  for (auto out : output_tvs) {
    max_dims = std::max(scheduler_utils::nLogicalDims(out), max_dims);
  }

  // If everything is zero dim tensors, just return.
  if (max_dims == 0) {
    return;
  }

  scheduler_tools::TransposeDomainMap domain_map(fusion);
  auto grouped_inputs_outputs = domain_map.groupInputsOutputsByInnerDim();
  NVF_ERROR(grouped_inputs_outputs.size() >= 2);

  /*
   * We need something similar to `cacheFork` for input tensors in group 2. We
   * need this because we will want to propagate to the entire DAG except group
   * 2 and its cached inputs, so we need to make sure the DAG is still connected
   * if we remove group and its cached inputs. For example
   *    t0
   *    |
   *   cache
   *   /  \
   *  t1  t2
   * if groups = {{t1, t2}, {t0}}, then removing {t0, cache} from the DAG will
   * make it disconnected.
   */
  std::unordered_set<TensorView*> group2_and_cached_inputs(
      grouped_inputs_outputs[1].begin(), grouped_inputs_outputs[1].end());
  for (auto tv : grouped_inputs_outputs[1]) {
    if (tv->isFusionInput()) {
      auto existing_cache = ir_utils::consumerTvsOf(tv)[0];
      if (ir_utils::consumerTvsOf(existing_cache).size() > 1) {
        auto new_cache = tv->cacheAfter();
        new_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(new_cache);
      } else {
        existing_cache->setMemoryType(MemoryType::Shared);
        group2_and_cached_inputs.emplace(existing_cache);
      }
    }
  }
  // set cached outputs of group 2 to shared memory
  for (const auto& [cached_output, output_idx] : cached_outputs) {
    auto output = fusion->outputs()[output_idx]->as<TensorView>();
    if (group2_and_cached_inputs.count(output) > 0) {
      cached_output->setMemoryType(MemoryType::Shared);
    }
  }

  TensorView* reference1 =
      domain_map.findReferenceFor(grouped_inputs_outputs[0]);
  TensorView* reference2 =
      domain_map.findReferenceFor(grouped_inputs_outputs[1]);

  NVF_ERROR(
      reference1 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the "
      "first group.");

  NVF_ERROR(
      reference2 != nullptr,
      "Could not find a fully broadcasted tensor to reference schedule on the "
      "second group.");

  auto inner_most_id1 = scheduler_utils::innerMostAllocDim(reference1);
  auto inner_most_id2 = scheduler_utils::innerMostAllocDim(reference2);

  //////////////////////////////////////////
  // Step 1: Make virtual inner most dims //
  //////////////////////////////////////////

  // See note [Supporting small transpose dimensions]

  // split big dims so that we have enough dimensions available to merge with
  // inner-most dims to create the virtual inner-most dim
  scheduler_utils::splitDims(reference1, tparams->split_before_tiling);

  // prepare all dimensions in merge order for group1
  std::vector<int64_t> dims_group1 = tparams->dims_merged_with_1;
  auto inner_loop_index1 =
      domain_map.getInnerLeafDim(reference1, inner_most_id1);
  NVF_ERROR(inner_loop_index1 >= 0, "getInnerLeafDim cannot be resolved");
  int64_t inner_most_pos1_in_ref1 = inner_loop_index1;
  dims_group1.insert(dims_group1.begin(), inner_most_pos1_in_ref1);

  // prepare all dimensions in merge order for group2
  std::vector<int64_t> dims_group2 = tparams->dims_merged_with_2;
  auto inner_loop_index2 =
      domain_map.getInnerLeafDim(reference1, inner_most_id2);
  int64_t inner_most_pos2_in_ref1 = inner_loop_index2;
  NVF_ERROR(inner_loop_index2 >= 0, "getInnerLeafDim cannot be resolved");
  dims_group2.insert(dims_group2.begin(), inner_most_pos2_in_ref1);

  // merge all dimensions in group1, while updating all indices for group2
  auto merged1 =
      scheduler_utils::mergeDims(reference1, dims_group1, dims_group2);
  std::vector<int64_t> merged1_vec;
  if (merged1.has_value()) {
    merged1_vec.push_back(*merged1);
  }
  // merge all dimensions in group2, while updating merged index for group1
  auto merged2 =
      scheduler_utils::mergeDims(reference1, dims_group2, merged1_vec);

  // updating merged1 & merged2 indices if applicable
  if (merged1.has_value()) {
    inner_most_pos1_in_ref1 = merged1_vec[0];
  }
  if (merged2.has_value()) {
    inner_most_pos2_in_ref1 = *merged2;
  }

  /////////////////////////////
  // Step 2: global schedule //
  /////////////////////////////

  // make tile
  // [..., I1, .., I2, ...]
  reference1->split(inner_most_pos1_in_ref1, tparams->tile_size1);
  reference1->reorder({{inner_most_pos1_in_ref1 + 1, -1}});
  reference1->split(inner_most_pos2_in_ref1, tparams->tile_size2);
  reference1->reorder({{inner_most_pos2_in_ref1 + 1, -1}});
  // [..., I1/tile1, .., I2/tile2, ..., tile1, tile2]

  // Merge remaining dimensions ignoring reduction axes (See Issue #2317)
  // The reduction axes cannot be at any position.
  // For example: [i0, r1, i1, r2, i2] after tiling is [i0, r1, i1/tile1, r2,
  // i2/tile2, tile1, tile2] The following code merges all the outer iterdomains
  // as: [i0 * i1/tile1 * i2/tile2, r1, r2, tile1, tile2]
  int64_t rhs_i = reference1->nDims() - 3;
  for (int64_t lhs_i = reference1->nDims() - 4; lhs_i >= 0; lhs_i--) {
    if (reference1->axis(lhs_i)->isReduction() ||
        reference1->axis(lhs_i)->isDeviceDim()) {
      continue;
    }
    if (reference1->axis(rhs_i)->isReduction() ||
        reference1->axis(rhs_i)->isDeviceDim()) {
      rhs_i = lhs_i;
      continue;
    }
    reference1->merge(lhs_i, rhs_i);
    rhs_i = lhs_i;
  }

  reference1->split(rhs_i, 1);
  // [r.., merged_dim, 1, tile1, tile2]

  // parallelize non-tile dimensions
  reference1->axis(rhs_i + 1)->parallelize(ParallelType::Unswitch);
  reference1->axis(rhs_i)->parallelize(ParallelType::BIDx);
  // [r.., BIDx, Unswitch, tile1, tile2]

  // Propagate transformations so far to the entire DAG
  TransformPropagator propagator(reference1);
  MaxLogicalDomainInfoSpanningTree entire_dag(reference1);
  entire_dag.traverse(&propagator);

  // We may be propagating a reshape during the above transformation.
  //   T0[i0 * i1]     -> View ->   T1[i0  i1] (Root=[i0*i1])
  //      / \                       / \.
  // iDID{d} i0*i1/d           iDID{d} i0/d
  // When propagating from consumer to producer for the first time here, we will
  // replay reshape split on the view input ([i0*i1] -> [i0, i1]), followed by
  // DID loop split on the view input ([i0, i1] -> [d, i0/d, i1]) and any other
  // transforms on i0/d or i1 scheduled above. This will lose the
  // parallelization on d. Hence, we also parallelize_inputs_on_did here in case
  // T0 is a fusion input. This is not needed elsewhere. Once the reshape has
  // been replayed on the producer, the DID loop split does not need to be
  // replayed. It will be consistently present in all tvs of the fusion.
  // TODO: An alternative would be to explictly propagateReshapeTransform as in
  // other schedulers.

  scheduler_utils::parallelizeAllLike(
      reference1,
      /*selected_tvs=*/{},
      /*selected_parallel_types=*/{},
      /*propagate_padding=*/true,
      /*parallelize_inputs_on_did=*/true);

  // For a transpose scheduling, all we need is to bind threadIdx.x differently
  // for inputs and outputs. This swap of binding could happen at any tensor on
  // the path from input to output, especially, it does not have to be in the
  // transpose tensor. Here, we naively do the binding swap at cached
  // input/output for simplicity. We might need to find a better set of swap
  // tensors in the future to reduce shared memory usage.

  //////////////////////////////
  // Step 3: Schedule group 2 //
  //////////////////////////////

  // transform tile for vectorization/unroll
  // See note [vectorization and unroll of input and output]

  // There can be non-concretized broadcast IDs lying around in loop
  // domains of intermediate or output tensors but not in fusion
  // inputs, including reference1. Those broadcast IDs may be present
  // in the middle of loop domains, which may prevent ideal
  // inlining. More importantly, the code at this point assumes the
  // innermost two dimensions are the tiling dimensions, but their
  // positions may need to be adjusted if there are non-concretized
  // broadcast IDs. To make the overall scheduling work consistently,
  // move all dangling broadcast IDs to innermost. We can identify
  // such broadcast IDs by just finding Serial broadcast loop IDs
  // since all broadcast IDs present in reference1 should have been
  // merged and parallelized.
  const int64_t num_innermost_broadcast_ids =
      moveDanglingBroadcastInner(reference2);
  int64_t pos = reference2->nDims() - 2 - num_innermost_broadcast_ids;
  // [..., tile1, tile2, b..]
  moveReductionsOut(reference2, 2);

  // Sanity check: pos and pos+1 axes should be generated by the
  // tiling splits
  auto validate_innermost_tiling = [&](IterDomain* tile1, IterDomain* tile2) {
    auto tile1_split = dynamic_cast<Split*>(tile1->definition());
    NVF_ERROR(
        tile1_split != nullptr, "Not a split output: ", tile1->toString());
    NVF_ERROR(
        tile1_split->innerSplit(),
        "Not an inner split: ",
        tile1_split->toString());
    NVF_ERROR_EQ(
        tile1_split->factor()->evaluate().as<int64_t>(),
        tparams->tile_size1,
        "Unexpected split factor");
    auto tile2_split = dynamic_cast<Split*>(tile2->definition());
    NVF_ERROR(
        tile2_split != nullptr, "Not a split output: ", tile2->toString());
    NVF_ERROR(
        tile2_split->innerSplit(),
        "Not an inner split: ",
        tile2_split->toString());
    NVF_ERROR_EQ(
        tile2_split->factor()->evaluate().as<int64_t>(),
        tparams->tile_size2,
        "Unexpected split factor");
  };

  validate_innermost_tiling(reference2->axis(pos), reference2->axis(pos + 1));
  reference2->merge(pos);
  reference2->split(pos, tparams->vectorize_factor2);
  reference2->split(pos, tparams->getThreadsPerBlock());
  // [..., Unroll, TIDx, Vectorize, b...]

  // Propagate transformations of reference2 to the entire DAG except
  // group 1. We actually only want to propagate to the fusion outputs, but
  // inputs and outputs themselves are disconnected, so we have to borrow the
  // entire DAG and use its spanning tree.
  {
    auto all_tvs_except1 = ir_utils::allTvsExcept(
        fusion,
        {grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end()});
    SetSelector selector({all_tvs_except1.begin(), all_tvs_except1.end()});
    MaxLogicalDomainInfoSpanningTree entire_dag_except1(reference2, &selector);
    TransformPropagator propagator(reference2);
    entire_dag_except1.traverse(&propagator);
  }

  // parallelize group2 and its cached inputs
  {
    if (tparams->vectorize_factor2 > 1) {
      reference2->axis(-1 - num_innermost_broadcast_ids)
          ->parallelize(ParallelType::Vectorize);
    }
    reference2->axis(-2 - num_innermost_broadcast_ids)
        ->parallelize(ParallelType::TIDx);
    reference2->axis(-3 - num_innermost_broadcast_ids)
        ->parallelize(ParallelType::Unroll);

    ComputeAtMap ca_map(fusion);

    scheduler_utils::parallelizeAllLike(
        reference2,
        {group2_and_cached_inputs.begin(), group2_and_cached_inputs.end()},
        {ParallelType::TIDx});

    // Only vectorize the axes that exactly maps to the vectorized axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> vectorized_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-1), IdMappingMode::EXACT);
              })) {
        vectorized_group2_cached_inputs.push_back(gin);
      }
    }
    if (!vectorized_group2_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference2,
          vectorized_group2_cached_inputs,
          {ParallelType::Vectorize});
    }

    // Only unroll the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group2_cached_inputs;
    for (auto gin : group2_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference2](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference2->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group2_cached_inputs.push_back(gin);
      }
    }
    if (!unrolled_group2_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference2, unrolled_group2_cached_inputs, {ParallelType::Unroll});
    }
  }

  //////////////////////////////
  // Step 4: Schedule group 1 //
  //////////////////////////////

  // schedule group 1
  reference1->reorder({{-2, -1}});
  // [..., tile2, tile1]
  pos = reference1->nDims() - 2;
  moveReductionsOut(reference1, 2);
  validate_innermost_tiling(reference1->axis(pos + 1), reference1->axis(pos));
  reference1->merge(pos);
  reference1->split(pos, tparams->vectorize_factor1);
  reference1->split(pos, tparams->getThreadsPerBlock());
  if (tparams->vectorize_factor1 > 1) {
    reference1->axis(-1)->parallelize(ParallelType::Vectorize);
  }
  reference1->axis(-2)->parallelize(ParallelType::TIDx);
  reference1->axis(-3)->parallelize(ParallelType::Unroll);
  // [..., Unroll, TIDx, Vectorize]

  // Propagate transformations, parallelization of the reference1 to the entire
  // DAG except group 2 and its corresponding cached outputs.
  {
    auto all_tvs_except2 =
        ir_utils::allTvsExcept(fusion, group2_and_cached_inputs);
    SetSelector selector({all_tvs_except2.begin(), all_tvs_except2.end()});
    MaxLogicalDomainInfoSpanningTree entire_dag_except_outputs(
        reference1, &selector);
    TransformPropagator propagator(reference1);
    entire_dag_except_outputs.traverse(&propagator);
    scheduler_utils::parallelizeAllLike(
        reference1, all_tvs_except2, {ParallelType::TIDx});
  }

  // vectorize and unroll group 1's output and cached input
  {
    ComputeAtMap ca_map(fusion);
    std::vector<TensorView*> group1_and_cached_inputs(
        grouped_inputs_outputs[0].begin(), grouped_inputs_outputs[0].end());
    for (auto tv : grouped_inputs_outputs[0]) {
      if (tv->isFusionInput()) {
        group1_and_cached_inputs.emplace_back(ir_utils::consumerTvsOf(tv)[0]);
      }
    }

    // Only vectorize the axes that exactly maps to the vectorized axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> vectorized_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-1), IdMappingMode::EXACT);
              })) {
        vectorized_group1_cached_inputs.push_back(gin);
      }
    }
    if (!vectorized_group1_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference1,
          vectorized_group1_cached_inputs,
          {ParallelType::Vectorize});
    }

    // Only unroll the axes that exactly maps to the unrolled axes
    //  on reference as support for permissively mapped axes are not
    //  yet clearly defined.
    std::vector<TensorView*> unrolled_group1_cached_inputs;
    for (auto gin : group1_and_cached_inputs) {
      if (std::any_of(
              gin->getLoopDomain().begin(),
              gin->getLoopDomain().end(),
              [&ca_map, reference1](IterDomain* id) {
                return ca_map.areMapped(
                    id, reference1->axis(-3), IdMappingMode::EXACT);
              })) {
        unrolled_group1_cached_inputs.push_back(gin);
      }
    }

    if (!unrolled_group1_cached_inputs.empty()) {
      scheduler_utils::parallelizeAllLike(
          reference1, unrolled_group1_cached_inputs, {ParallelType::Unroll});
    }
  }

  ////////////////////////////////
  // Step 5: Cleanup and inline //
  ////////////////////////////////

  // cleanup parallelization from reference1 and reference2 if they are fusion
  // inputs
  for (auto tv : {reference1, reference2}) {
    if (tv->isFusionInput()) {
      for (auto id : tv->getLoopDomain()) {
        // DIDs are given as inputs instead of artifacts of this scheduler. So
        // do not reset them.
        if (!id->isDeviceDim()) {
          id->parallelize(ParallelType::Serial);
        }
      }
    }
  }

  // Inline
  inlineMost();

  scheduler_utils::promoteProducerMemoryTypes(fusion, cached_inputs);
}

} // namespace non_tma
} // namespace transpose
} // namespace nvfuser
