// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "scheduler/cutlass.h"
#include "scheduler/greedy.h"
#include "scheduler/matmul_heuristic.h"
#include "scheduler/native_diagnostic_jsonl.h"
#include "scheduler/normalization_inner_tma.h"
#include "scheduler/pointwise_heuristic.h"
#include "scheduler/reduction_heuristic.h"
#include "scheduler/reduction_outer_tma.h"
#include "scheduler/reduction_tma.h"
#include "scheduler/resize_heuristic.h"
#include "scheduler/transpose_heuristic.h"

// A read-only, typed description of scheduling fields. Keep this independent
// of HeuristicParams::toString(), whose display text is incomplete and differs
// between upstream and the experimental branch for identical Resize plans.
namespace nvfuser::scheduler::native_diagnostic {
namespace detail {

template <typename P>
bool symbolic1dCta(const P& p) {
  if constexpr (requires { p.symbolic_1d_cta; }) {
    return p.symbolic_1d_cta;
  } else {
    return false;
  }
}

template <typename P>
int64_t resizeCtaThreads(const P& p) {
  if constexpr (requires { p.cta_threads; }) {
    return p.cta_threads;
  } else {
    return 128;
  }
}

template <typename P>
int64_t transposeThreadsCap(const P& p) {
  if constexpr (requires { p.threads_per_block_cap; }) {
    return p.threads_per_block_cap;
  } else {
    return 128;
  }
}

template <typename T>
std::string streamText(const T& value) {
  std::ostringstream os;
  os << value;
  return os.str();
}

inline std::string numberArray(const std::vector<int64_t>& values) {
  std::string result = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) result += ',';
    result += std::to_string(values[i]);
  }
  return result + ']';
}

inline std::string pairArray(
    const std::vector<std::pair<int64_t, int64_t>>& values) {
  std::string result = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) result += ',';
    result += '[' + std::to_string(values[i].first) + ',' +
        std::to_string(values[i].second) + ']';
  }
  return result + ']';
}

inline std::string stringArray(const std::vector<std::string>& values) {
  std::string result = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) result += ',';
    result += quote(values[i]);
  }
  return result + ']';
}

inline std::string bufferNames(const std::vector<TensorView*>& buffers) {
  std::string result = "[";
  for (size_t i = 0; i < buffers.size(); ++i) {
    if (i != 0) result += ',';
    result += buffers[i] == nullptr ? "null" :
        std::to_string(buffers[i]->name());
  }
  return result + ']';
}

inline std::string sortedGreedyLines(std::string_view text) {
  std::istringstream input{std::string(text)};
  std::vector<std::string> lines;
  for (std::string line; std::getline(input, line);) {
    if (!line.empty()) lines.push_back(std::move(line));
  }
  std::sort(lines.begin(), lines.end());
  return stringArray(lines);
}

inline std::string circularBuffer(const CircularBufferOptions& options) {
  Object result;
  result.add("stage", options.stage);
  result.add("prefetch", options.prefetch);
  result.add("type", streamText(options.type));
  return result.str();
}

inline std::string gemmTile(const GemmTile& tile) {
  Object result;
  result.add("m", tile.m);
  result.add("n", tile.n);
  result.add("k", tile.k);
  return result.str();
}

inline std::string intPair(int64_t first, int64_t second) {
  return '[' + std::to_string(first) + ',' + std::to_string(second) + ']';
}

inline std::string enumText(MatmulParams::TileRasterizationOrder value) {
  switch (value) {
    case MatmulParams::TileRasterizationOrder::RowMajor: return "RowMajor";
    case MatmulParams::TileRasterizationOrder::ColumnMajor: return "ColumnMajor";
  }
  return "Unknown(" + std::to_string(static_cast<int>(value)) + ')';
}

inline std::string enumText(MatmulParams::TilingStrategy value) {
  switch (value) {
    case MatmulParams::TilingStrategy::OneTilePerCTA: return "OneTilePerCTA";
    case MatmulParams::TilingStrategy::DistributeTilesAcrossSMs: return "DistributeTilesAcrossSMs";
    case MatmulParams::TilingStrategy::DistributeStagesAcrossSMs: return "DistributeStagesAcrossSMs";
  }
  return "Unknown(" + std::to_string(static_cast<int>(value)) + ')';
}

inline std::string enumText(MatmulParams::BufferingLoopLevel value) {
  switch (value) {
    case MatmulParams::BufferingLoopLevel::CTATiles: return "CTATiles";
    case MatmulParams::BufferingLoopLevel::WarpTiles: return "WarpTiles";
  }
  return "Unknown(" + std::to_string(static_cast<int>(value)) + ')';
}

inline std::string enumText(MatmulParams::CircularBufferingStrategy value) {
  switch (value) {
    case MatmulParams::CircularBufferingStrategy::Pipelined: return "Pipelined";
    case MatmulParams::CircularBufferingStrategy::WarpSpecialized: return "WarpSpecialized";
  }
  return "Unknown(" + std::to_string(static_cast<int>(value)) + ')';
}

} // namespace detail

struct SchedulerFieldResult {
  std::string params_kind;
  std::string fields_json;
};

inline SchedulerFieldResult describeSchedulerFields(const HeuristicParams& params) {
  Object fields;
  std::string kind;
#define NVF_ADD(p, field) fields.add(#field, (p).field)
#define NVF_PAR(p, field) fields.add(#field, detail::streamText((p).field))
  if (const auto* p = dynamic_cast<const PointwiseParams*>(&params)) {
    kind = "PointwiseParams";
    NVF_ADD(*p, break_point);
    NVF_ADD(*p, split_block);
    fields.add("symbolic_1d_cta", detail::symbolic1dCta(*p));
    NVF_ADD(*p, split_grid_y_dim);
    NVF_ADD(*p, flip_grid_binding);
    NVF_ADD(*p, vectorization_factor);
    NVF_ADD(*p, vectorize_casts);
    NVF_ADD(*p, unroll_factor_outer);
    NVF_ADD(*p, unroll_factor_inner);
    NVF_ADD(*p, use_tma_load);
    NVF_ADD(*p, use_tma_store);
    NVF_ADD(*p, tma_domain_inner);
    NVF_ADD(*p, tma_tile_outer);
    NVF_ADD(*p, tma_tile_inner);
  } else if (const auto* p = dynamic_cast<const ResizeParams*>(&params)) {
    kind = "ResizeParams";
    NVF_ADD(*p, split_grid_x_dim);
    NVF_ADD(*p, largest_input);
    NVF_ADD(*p, vectorization_factor);
    fields.add("cta_threads", detail::resizeCtaThreads(*p));
  } else if (const auto* p = dynamic_cast<const TransposeParams*>(&params)) {
    kind = "TransposeParams";
    fields.addRaw("split_before_tiling", detail::pairArray(p->split_before_tiling));
    fields.addRaw("dims_merged_with_1", detail::numberArray(p->dims_merged_with_1));
    fields.addRaw("dims_merged_with_2", detail::numberArray(p->dims_merged_with_2));
    NVF_ADD(*p, use_tma_load);
    NVF_ADD(*p, vectorize_factor1);
    NVF_ADD(*p, vectorize_factor2);
    fields.add("threads_per_block_cap", detail::transposeThreadsCap(*p));
    NVF_ADD(*p, tile_size1);
    NVF_ADD(*p, tile_size2);
  } else if (const auto* p = dynamic_cast<const TmaInnerReductionParams*>(&params)) {
    kind = "TmaInnerReductionParams";
    NVF_ADD(*p, vectorization_factor);
    NVF_ADD(*p, threads_per_block);
    NVF_ADD(*p, unroll_factor);
  } else if (const auto* p = dynamic_cast<const TmaOuterReductionParams*>(&params)) {
    kind = "TmaOuterReductionParams";
    NVF_ADD(*p, bdimx);
    NVF_ADD(*p, bdimy);
    NVF_ADD(*p, tma_tile_i);
    NVF_ADD(*p, tma_tile_r);
    NVF_ADD(*p, iter_unroll_factor);
    NVF_ADD(*p, grdim);
  } else if (const auto* p = dynamic_cast<const InnerNormTmaParams*>(&params)) {
    kind = "InnerNormTmaParams";
    NVF_ADD(*p, vectorization_factor);
    NVF_ADD(*p, project_persistent_buffers);
    NVF_ADD(*p, vectorize_load_smem_to_regs);
    NVF_ADD(*p, persistent_batch_size);
    NVF_ADD(*p, pre_load_ldg_tvs);
    NVF_ADD(*p, tma_load_non_persistent_buffers);
    NVF_ADD(*p, is_circular_buffer_regs_cached);
    NVF_ADD(*p, n_grouped_rows);
    fields.addRaw("circular_buffer_options", detail::circularBuffer(p->circular_buffer_options));
  } else if (const auto* p = dynamic_cast<const ReductionParams*>(&params)) {
    kind = "ReductionParams";
    NVF_ADD(*p, fastest_dim);
    NVF_ADD(*p, persistent_kernel);
    NVF_ADD(*p, project_persistent_buffers);
    NVF_ADD(*p, schedule_3d);
    NVF_ADD(*p, flip_grid);
    NVF_ADD(*p, cross_block_inner_reduction);
    NVF_ADD(*p, cross_grid_inner_reduction);
    NVF_ADD(*p, cross_cluster_reduction);
    NVF_ADD(*p, unroll_factor_inner_reduction);
    NVF_ADD(*p, unroll_factor_top_of_vectorization);
    NVF_ADD(*p, vectorize_inner_reduction);
    NVF_ADD(*p, vectorize_casts);
    NVF_ADD(*p, split_grid_dim_inner_reduction);
    NVF_ADD(*p, pad_inner_reduction_to_warp);
    NVF_ADD(*p, batches_per_block_inner_reduction);
    NVF_PAR(*p, block_dim_inner_reduction);
    NVF_PAR(*p, grid_dim_inner_reduction);
    NVF_ADD(*p, multiple_reds_per_blk);
    NVF_ADD(*p, unroll_factor_iter_dom);
    NVF_ADD(*p, vectorize_iter_dom);
    NVF_ADD(*p, split_grid_dim_iter_dom_inner);
    NVF_ADD(*p, split_grid_dim_iter_dom_outer);
    NVF_PAR(*p, block_dim_iter_dom);
    NVF_PAR(*p, grid_dim_iter_dom);
    NVF_ADD(*p, cross_block_outer_reduction);
    NVF_ADD(*p, cross_grid_outer_reduction);
    NVF_ADD(*p, split_grid_dim_outer_reduction);
    NVF_ADD(*p, batches_per_block_outer_reduction);
    NVF_ADD(*p, unroll_factor_outer_reduction);
    NVF_PAR(*p, block_dim_outer_reduction);
    NVF_PAR(*p, grid_dim_outer_reduction);
    NVF_ADD(*p, compute_persistent_buffer_with_first_consumer);
    NVF_ADD(*p, static_bdimx);
    NVF_ADD(*p, static_bdimy);
    NVF_ADD(*p, combined_inner_outer);
    NVF_ADD(*p, tidx_for_outer_reduction);
    NVF_ADD(*p, pad_outer_reduction_to_warp);
    NVF_ADD(*p, combined_split_grid_inner_dim);
    NVF_ADD(*p, tma_warp_specialized);
    NVF_ADD(*p, is_good_ws_heuristic);
    NVF_ADD(*p, is_non_circular_buffer_gmem_to_regs);
    NVF_ADD(*p, is_circular_buffer_regs_cached);
    fields.addRaw("circular_buffer_options", detail::circularBuffer(p->circular_buffer_options));
    NVF_ADD(*p, computation_warp_groups);
    NVF_ADD(*p, vectorization_factor_outer);
    NVF_ADD(*p, vectorization_factor_tmp_gmem_write);
    NVF_PAR(*p, block_dim_inner_reduction_extra);
    fields.addRaw("smem_persistent_buffers", detail::bufferNames(p->smem_persistent_buffers));
  } else if (const auto* p = dynamic_cast<const MatmulParams*>(&params)) {
    kind = "MatmulParams";
    // MatmulParams::toString embeds tag, which is display evidence rather than
    // part of a scheduling plan. Serialize each nested option as typed data.
    Object vectorization, tiles, buffering, cluster;
    vectorization.add("a", p->supported_vec_size.a);
    vectorization.add("b", p->supported_vec_size.b);
    vectorization.add("epilogue", p->supported_vec_size.epilogue);
    tiles.addRaw("cta_tile", detail::gemmTile(p->tile_sizes.cta_tile));
    tiles.addRaw("warp_tile", detail::gemmTile(p->tile_sizes.warp_tile));
    buffering.add("circular_buffer_smem_write", p->circular_buffer_options.circular_buffer_smem_write);
    buffering.add("circular_buffer_smem_read", p->circular_buffer_options.circular_buffer_smem_read);
    buffering.add("smem_circular_buffer_stage", static_cast<int64_t>(p->circular_buffer_options.smem_circular_buffer_stage));
    buffering.add("smem_circular_buffer_prefetch_gap", static_cast<int64_t>(p->circular_buffer_options.smem_circular_buffer_prefetch_gap));
    cluster.add("m", p->cluster_dims.m);
    cluster.add("n", p->cluster_dims.n);
    fields.addRaw("supported_vec_size", vectorization.str());
    fields.add("async_gmem_load_operands", p->async_gmem_load_operands);
    fields.addRaw("tile_sizes", tiles.str());
    fields.add("mma_macro", nvfuser::toString(p->mma_macro));
    fields.add("tiling_strategy", detail::enumText(p->tiling_strategy));
    fields.add("buffering_loop_level", detail::enumText(p->buffering_loop_level));
    fields.add("circular_buffering_strategy", detail::enumText(p->circular_buffering_strategy));
    fields.add("cta_order", detail::enumText(p->cta_order));
    fields.addRaw("circular_buffer_options", buffering.str());
    fields.addRaw("grid_traversal_factor", detail::intPair(p->grid_traversal_factor.first, p->grid_traversal_factor.second));
    fields.add("use_smem_epilogue", p->use_smem_epilogue);
    fields.add("promote_prologue_smem_reuse", p->promote_prologue_smem_reuse);
    fields.add("use_ldst_matrix", p->use_ldst_matrix);
    fields.add("splitk_factor", static_cast<int64_t>(p->splitk_factor));
    fields.addRaw("cluster_dims", cluster.str());
  } else if (const auto* p = dynamic_cast<const GreedyParams*>(&params)) {
    kind = "GreedyParams";
    // GreedyParams stores two private unordered maps. Their rendered entries
    // are complete, but order is not semantically meaningful or stable.
    fields.addRaw("sorted_lines", detail::sortedGreedyLines(p->toString()));
  } else if (const auto* p = dynamic_cast<const CutlassParams*>(&params)) {
    kind = "CutlassParams";
    fields.addRaw("mma_tile", detail::gemmTile(p->mma_tile));
    fields.addRaw("per_sm_tile", detail::gemmTile(p->per_sm_tile));
    fields.addRaw("cluster_shape", detail::gemmTile(p->cluster_shape));
  } else if (params.isStrictlyA<HeuristicParams>()) {
    kind = "HeuristicParams";
  } else {
    // Preserve an explicit unknown row so strict readers fail closed.
    kind = "Unknown";
  }
#undef NVF_ADD
#undef NVF_PAR
  return {std::move(kind), fields.str()};
}

inline std::string schedulerFields(const HeuristicParams& params) {
  return describeSchedulerFields(params).fields_json;
}

} // namespace nvfuser::scheduler::native_diagnostic
