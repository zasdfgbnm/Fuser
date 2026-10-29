// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include "scheduler/pointwise_tma.h"

#include <ATen/cuda/CUDAContext.h>

#include <algorithm>

#include "exceptions.h"
#include "ir/utils.h"
#include "scheduler/debug_utils.h"
#include "scheduler/jev_client.h"
#include "scheduler/pointwise_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/inlining.h"
#include "scheduler/utils.h"
#include "scheduler/vectorize_helper.h"
#include "transform_iter.h"
#include "transform_replay.h"

namespace nvfuser {
namespace pointwise {
namespace tma {

// ============================================================================
// TMA TERMINOLOGY GUIDE:
// ============================================================================
// The TMA scheduler uses a 2-level hierarchy to organize data:
//
// 1. TMA DOMAIN: The logical split of the entire problem space into two parts:
//    - tma_domain_inner: Size of the inner (contiguous) dimension
//    - tma_domain_outer: Size of the outer dimension (n_elems /
//    tma_domain_inner)
//    - requirement: n_elems % tma_domain_inner == 0
//    Example: For 2048 elements with tma_domain_inner=512:
//      TMA Domain structure: [4, 512]
//
// 2. TMA TILE: The actual hardware tile size for each TMA load operation
//    - tma_tile_inner: Number of elements along the inner dimension per tile
//    - tma_tile_outer: Number of elements along the outer dimension per tile
//    Example: For TMA domain [4, 512] with tiles [2, 128]:
//      Each TMA load fetches a [2 x 128] tile, requiring 2 x 4 = 8 tiles total
//
//    Note: In general TMA terminology, a "box" is the dense rectangular region
//    loaded from global memory, while a "tile" is a potentially strided subset
//    selected from that box. The pointwise scheduler always uses dense tiles
//    (tile = box), so we use "TMA tile" throughout to refer to both concepts.
//
// Transformation sequence: logical domain -> TMA domain -> TMA tile
//  [I0, I1, ...] -- > [I] -> [Do, Di] -> [Do/to, to, Di/ti, ti]
//   where Do=tma_domain_outer, Di=tma_domain_inner,
//         to=tma_tile_outer, ti=tma_tile_inner
// ============================================================================

// TODO: This can be further relaxed to allow more tensor views with fewer
// dimensions, e.g., outer broadcast inputs [B, I] can also be loaded with TMA.
bool isTvSuitableForTma(const TensorView* tv, int64_t n_valid_dims) {
  // When tv has broadcast dimensions, it may cause one of the following issues:
  // 1. Merge of iteration domain with broadcast dimension.
  // This is not supported by TMA and will trigger tma lowering validation
  // error. The TMA domain must be equivalent to the allocation domain of the
  // gmem tensor.
  // 2. Right side or left side contains a single broadcast domain, which is
  // against our 2D tile assumption. This restriction can be lifted if we
  // further revise the scheduler.
  if (std::any_of(
          tv->getLogicalDomain().begin(),
          tv->getLogicalDomain().end(),
          [](const IterDomain* id) { return id->isBroadcast(); })) {
    return false;
  }
  return scheduler_utils::nLogicalDims(tv) == n_valid_dims;
};

// Returns the total bits required to load one element from each TMA-loaded
// input. This is used to determine how many elements should be loaded from each
// input to achieve the required bits in flight.
int64_t getInputBitsPerElement(
    const pointwise_utils::FusionRuntimeProperties& prop) {
  int64_t bits_per_element = 0;
  int64_t n_valid_dims = scheduler_utils::nLogicalDims(prop.largest_out);
  for (const auto& tv : prop.vectorizable_inputs_outputs) {
    if (tv->isFusionInput() && isTvSuitableForTma(tv, n_valid_dims)) {
      bits_per_element += dataTypeSizeBit(tv->getDataType(), prop.index_type);
    }
  }
  return bits_per_element;
}

std::unique_ptr<PointwiseParams> getPointwiseHeuristicsForBreakPoint(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop,
    const pointwise_utils::BreakPointInfo& bp_info,
    scheduler::jev::Mode jev_mode,
    bool* hard_valid = nullptr,
    bool emit_debug = true,
    int64_t tma_domain_inner_target = 512) {
  constexpr int64_t threads_per_warp = 32;

  auto params = std::make_unique<PointwiseParams>();
  params->tag = "Pointwise TMA heuristics";
  params->cparams.index_type = prop.index_type;
  params->use_tma_load = true;

  params->break_point = bp_info.break_point;

  // check which input is suitable for TMA load
  // summarize bits of all suitable inputs
  // if bits_per_element is 0, no suitable input is found, return nullptr
  // Note: we can move this check upfront when dispatching to TMA and non-TMA
  // versions, however, we may want to further extend to use break point for
  // more fine-grained control of TMA loads, e.g. mixing 2D tile and 1D tile.
  const int64_t bits_per_element = getInputBitsPerElement(prop);
  if (bits_per_element == 0) {
    return nullptr;
  }

  // ========== Step 1: Compute TMA Domain Dimensions ==========
  // The TMA domain splits the entire problem into
  // [tma_domain_outer, tma_domain_inner]
  // Target: Make the inner domain ~512 elements for good memory locality

  // tma_domain_inner: Size of the innermost contiguous dimension
  // This represents how many consecutive elements are treated as a single "row"

  // tma_domain_outer: Number of "rows" after splitting
  // The full TMA domain is now [tma_domain_outer, tma_domain_inner]
  // where tma_domain_outer * tma_domain_inner = n_elems
  // The one-dimensional row target is an empirical locality preference.
  // Materialize each complete alternate plan with the native scheduler before
  // offering it. Off-mode preflight uses no shared breakpoint-dependent cache
  // and emits neither nested Choices nor SchedulerDebug output.
  if (bp_info.break_point == 0 && jev_mode != scheduler::jev::Mode::Off &&
      prop.n_elems > (1LL << 40)) {
    if (jev_mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "PW4.domain_inner", "extent exceeds bounded divisor replay");
    } else {
      NVF_ERROR(false, "PW4.domain_inner extent exceeds bounded divisor replay");
    }
  }
  if (bp_info.break_point == 0 && jev_mode != scheduler::jev::Mode::Off &&
      prop.n_elems <= (1LL << 40)) {
    std::vector<scheduler::jev::ChoiceOption> options;
    for (const int64_t target : {256L, 512L, 1024L}) {
      bool candidate_valid = false;
      std::unique_ptr<PointwiseParams> candidate;
      try {
        candidate = getPointwiseHeuristicsForBreakPoint(
            fusion,
            runtime_info,
            nullptr,
            prop,
            bp_info,
            scheduler::jev::Mode::Off,
            &candidate_valid,
            /*emit_debug=*/false,
            target);
      } catch (const nvfError&) {
        continue;
      }
      if (candidate == nullptr || !candidate_valid) {
        continue;
      }
      const int64_t inner = candidate->tma_domain_inner;
      options.push_back(
          {"t" + std::to_string(target),
           "Target " + std::to_string(target) +
               " elements; TMA domain " + std::to_string(inner) + "x" +
               std::to_string(prop.n_elems / inner)});
    }

    const auto native = std::find_if(
        options.begin(), options.end(), [](const auto& option) {
          return option.id == "t512";
        });
    if (jev_mode == scheduler::jev::Mode::Record && native == options.end()) {
      scheduler::jev::recordCoverageSkip(
          "PW4.domain_inner",
          "native 512-element target is outside the complete-plan menu");
    } else {
      NVF_ERROR(
          native != options.end(),
          "PW4.domain_inner has no complete native TMA plan");
      std::string state = "pointwise.tma_domain_inner.v1\n";
      state += "n_elems=" + std::to_string(prop.n_elems) + '\n';
      state += "min_dtype_bits=" +
          std::to_string(prop.min_dtype_size_bit_for_vectorization) + '\n';
      const auto selected = scheduler::jev::chooseDecision(
          "PW4.domain_inner",
          state,
          options,
          jev_mode == scheduler::jev::Mode::Record ? "t512" : std::string());
      bool matched = false;
      for (const auto& option : options) {
        if (selected == option.id) {
          tma_domain_inner_target = std::stoll(selected.substr(1));
          matched = true;
          break;
        }
      }
      NVF_ERROR(matched, "PW4.domain_inner response is outside hard bounds");
    }
  }

  const int64_t tma_domain_inner = bp_info.break_point == 0
      ? scheduler_utils::getTmaDomainInner(
            prop.n_elems,
            tma_domain_inner_target,
            prop.min_dtype_size_bit_for_vectorization)
      : bp_info.right_elem_count;
  NVF_ERROR(
      tma_domain_inner > 1 && prop.n_elems % tma_domain_inner == 0,
      "Illegal tma_domain_inner size: ",
      tma_domain_inner,
      ", n_elems: ",
      prop.n_elems);

  // TMA requires 128-bit alignment. When break_point > 0, tma_domain_inner is
  // determined by right_elem_count, which may not satisfy this requirement.
  // Reject such cases early to avoid generating invalid TMA schedules.
  //
  // TODO: The break_point selection logic should account for alignment
  // requirements. For instance, a 512x127 tensor could use a 1D scheduler
  // (break_point=0) but fails with a 2D scheduler if 127 elements don't meet
  // 128-bit alignment. This limitation affects both TMA and non-TMA pointwise
  // schedulers. See TmaDomainBroadcastIllegal test for an example.
  if (bp_info.break_point > 0) {
    const int64_t ref_elem_size_bits =
        dataTypeSizeBit(prop.largest_out->getDataType(), prop.index_type);
    const int64_t tma_domain_inner_bits = tma_domain_inner * ref_elem_size_bits;
    if (tma_domain_inner_bits % 128 != 0) {
      return nullptr;
    }
  }

  const int64_t tma_domain_outer = prop.n_elems / tma_domain_inner;
  params->tma_domain_inner = tma_domain_inner;

  // ========== Step 2: Determine Target Elements Per CTA ==========
  // We calculate how many elements each CTA should load based on memory
  // bandwidth requirements.
  // bits_per_element: Total bits to load for one position across all TMA inputs
  // (e.g., if we have 2 float32 inputs, this would be 64 bits)
  // elements_per_cta: Target number of elements for each CTA to process
  // Each CTA has 128 threads, round up to 1024 for divisible by 128 and leave 8
  // for vectorization and unroll.
  constexpr int64_t cta_per_sm = 8;
  const int64_t bits_per_sm = scheduler_utils::getRequiredBitsInFlight();
  const int64_t bits_per_cta = bits_per_sm / cta_per_sm;
  // This occupancy target is a performance preference. Remote mode sends the
  // raw measurements and lets the service recompute it.
  const int64_t elements_per_cta = jev_mode == scheduler::jev::Mode::Remote
      ? 0
      : scheduler_utils::roundUpToN(
            ceilDiv(bits_per_cta, bits_per_element), 1024);

  // ========== Step 3: Compute TMA Tile Dimensions ==========
  // TMA tiles define the tile size loaded by each TMA operation:
  // [tma_tile_outer, tma_tile_inner]
  // Constraints:
  // 1. Hardware limit: Each tile dimension must be ≤ 256 elements
  // 2. 2D TMA requirement: Need at least 2 tiles along inner dimension
  //    (i.e., tma_tile_inner ≤ tma_domain_inner / 2) to maintain 2D structure
  // 3. Don't exceed domain boundaries

  // tma_tile_inner_max: Maximum size for tma tile inner dimension
  // Division by 2 ensures at least 2 tiles fit within tma_domain_inner
  // Don't exceed the hardware limit
  const int64_t tma_tile_inner_max =
      std::min(tma_domain_inner / 2, kMaxElementsPerTmaTileDim);

  // tma_tile_outer_max: Maximum size for tma tile outer dimension
  // Don't exceed the total number of "rows" in tma_domain_outer
  // Don't exceed the hardware limit
  const int64_t tma_tile_outer_max =
      std::min(tma_domain_outer, kMaxElementsPerTmaTileDim);

  // tma_tile_inner: Actual tma tile in inner dimension
  // Start with warp size, then grow by powers of 2 until reaching
  // tma_tile_inner_max e.g. 16, 32, 64, 128, 256
  int64_t tma_tile_inner = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    tma_tile_inner = std::min(tma_tile_inner_max, threads_per_warp);
    while (tma_tile_inner * 2 <= tma_tile_inner_max) {
      tma_tile_inner *= 2;
    }
  }

  const auto* device = at::cuda::getCurrentDeviceProperties();
  const int64_t min_dtype_bits = prop.min_dtype_size_bit_for_vectorization;
  const int64_t max_dtype_bits = prop.max_dtype_size_bit_for_vectorization;
  const int64_t max_grid_x_dim = device->maxGridSize[0];
  const int64_t max_grid_y_dim = device->maxGridSize[1];
  const int64_t smem_per_block_bytes = device->sharedMemPerBlockOptin;
  const int64_t max_threads_per_block = device->maxThreadsPerBlock;
  NVF_ERROR(
      min_dtype_bits > 0 && max_dtype_bits > 0,
      "PW4.tile requires positive dtype widths");
  int64_t tma_input_count = 0;
  int64_t smem_bits_per_element = 0;
  const int64_t n_valid_dims =
      scheduler_utils::nLogicalDims(prop.largest_out);
  for (const auto* input : fusion->inputs()) {
    const auto* tv = dynamic_cast<const TensorView*>(input);
    if (tv != nullptr && isTvSuitableForTma(tv, n_valid_dims)) {
      ++tma_input_count;
      smem_bits_per_element +=
          dataTypeSizeBit(tv->getDataType(), prop.index_type);
    }
  }
  bool has_block_quantization = false;
  bool requires_quant_width_two = false;
  for (auto* op : ir_utils::getOpsOfType<BlockQuantizationOp>(fusion)) {
    has_block_quantization = true;
    requires_quant_width_two |= dataTypeSizeBit(
        op->quantizedOutput()->getDataType(), prop.index_type) % 8 != 0;
  }
  for (auto* op : ir_utils::getOpsOfType<GroupedBlockQuantizationOp>(fusion)) {
    has_block_quantization = true;
    requires_quant_width_two |= dataTypeSizeBit(
        op->quantizedOutput()->getDataType(), prop.index_type) % 8 != 0;
  }

  // The box's contiguous dimension must satisfy every TMA input's 16-byte
  // alignment. The width bound also keeps the later vectorized split inside
  // one tile. Small native boxes can be non-powers of two; larger native boxes
  // grow from one warp by doubling. This complete bounded menu includes both.
  auto legal_inner = [&](int64_t candidate) {
    return candidate > 0 && candidate <= tma_tile_inner_max &&
        candidate * min_dtype_bits % 128 == 0 &&
        candidate >= 128 / max_dtype_bits &&
        ceilDiv(tma_domain_inner, candidate) <= max_grid_x_dim;
  };

  // Compute the source outer preference only in off/record modes. In remote
  // mode the service must derive it from the raw bandwidth measurements.
  int64_t tma_tile_outer = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    tma_tile_outer = std::max(
        1L, std::min(elements_per_cta / tma_tile_inner, tma_tile_outer_max));
  }

  // Count all TMA-loaded inputs for the shared-memory bound, including inputs
  // excluded from the narrower bandwidth preference calculation above.
  auto legal_outer = [&](int64_t inner, int64_t outer) {
    if (inner <= 0 || outer <= 0 || outer > tma_tile_outer_max) {
      return false;
    }
    const int64_t area = inner * outer;
    const int64_t smem_upper_bytes =
        ceilDiv(area * smem_bits_per_element, 8) +
        128 * tma_input_count + 4096;
    const int64_t bdimx = std::min(threads_per_warp, inner);
    const int64_t bdimy = std::min(128L / bdimx, outer);
    const int64_t grid_y = ceilDiv(tma_domain_outer, outer);
    return smem_upper_bytes <= smem_per_block_bytes &&
        bdimx * bdimy <= max_threads_per_block &&
        (!has_block_quantization || grid_y <= max_grid_y_dim);
  };
  const int64_t first_outer = tma_tile_outer_max == 256 ? 2 : 1;
  const int64_t minimum_outer = std::max(
      first_outer,
      has_block_quantization
          ? ceilDiv(tma_domain_outer, max_grid_y_dim)
          : 1L);
  auto legal_inner_plan = [&](int64_t candidate) {
    return legal_inner(candidate) &&
        legal_outer(candidate, minimum_outer);
  };

  if (hard_valid != nullptr) {
    *hard_valid = tma_domain_outer > 1 && bits_per_sm >= 8 &&
        legal_inner_plan(tma_tile_inner) &&
        tma_tile_outer >= first_outer &&
        legal_outer(tma_tile_inner, tma_tile_outer);
  }

  bool route_tile_choices = jev_mode != scheduler::jev::Mode::Off;
  if (jev_mode == scheduler::jev::Mode::Record) {
    std::string skip_reason;
    if (bits_per_sm < 8) {
      skip_reason = "bandwidth estimate below one CTA target bit";
    } else if (!legal_inner_plan(tma_tile_inner)) {
      skip_reason = "native inner tile has no aligned, resource-valid TMA plan";
    } else if ((tma_tile_outer < first_outer) ||
               !legal_outer(tma_tile_inner, tma_tile_outer)) {
      skip_reason = "native outer tile exceeds CTA, grid, or shared-memory bound";
    }
    if (!skip_reason.empty()) {
      scheduler::jev::recordCoverageSkip("PW4.tile", skip_reason);
      route_tile_choices = false;
    }
  }
  if (jev_mode == scheduler::jev::Mode::Remote) {
    NVF_ERROR(bits_per_sm >= 8, "PW4.tile bandwidth state has no legal CTA target");
  }

  if (route_tile_choices) {
    std::vector<scheduler::jev::ChoiceOption> options;
    for (int64_t candidate = 1;
         candidate <= std::min<int64_t>(tma_tile_inner_max, 32);
         ++candidate) {
      if (legal_inner_plan(candidate)) {
        options.push_back(
            {"i" + std::to_string(candidate),
             "TMA inner box " + std::to_string(candidate) + " elements"});
      }
    }
    for (int64_t candidate = 64; candidate <= tma_tile_inner_max;
         candidate *= 2) {
      if (legal_inner_plan(candidate)) {
        options.push_back(
            {"i" + std::to_string(candidate),
             "TMA inner box " + std::to_string(candidate) + " elements"});
      }
    }
    NVF_ERROR(!options.empty(), "PW4.tile_inner has no hard-valid candidate");
    std::string state = "pointwise.tma_tile_inner.v1\n";
    state += "domain_inner=" + std::to_string(tma_domain_inner) + '\n';
    state += "min_dtype_bits=" + std::to_string(min_dtype_bits) + '\n';
    state += "max_dtype_bits=" + std::to_string(max_dtype_bits) + '\n';
    state += "max_grid_x=" + std::to_string(max_grid_x_dim) + '\n';
    state += "domain_outer=" + std::to_string(tma_domain_outer) + '\n';
    state += "max_grid_y=" + std::to_string(max_grid_y_dim) + '\n';
    state += "max_threads_per_block=" +
        std::to_string(max_threads_per_block) + '\n';
    state += "smem_per_block_bytes=" +
        std::to_string(smem_per_block_bytes) + '\n';
    state += "tma_input_count=" + std::to_string(tma_input_count) + '\n';
    state += "has_block_quantization=" +
        std::to_string(has_block_quantization) + '\n';
    state += "bits_per_sm=" + std::to_string(bits_per_sm) + '\n';
    state += "bits_per_element=" + std::to_string(bits_per_element) + '\n';
    state += "smem_bits_per_element=" +
        std::to_string(smem_bits_per_element) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "i" + std::to_string(tma_tile_inner)
        : std::string();
    const std::string selected = scheduler::jev::chooseDecision(
        "PW4.tile_inner", state, options, native_choice);
    bool matched = false;
    for (const auto& option : options) {
      if (selected == option.id) {
        tma_tile_inner = std::stoll(selected.substr(1));
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "PW4.tile_inner response is outside hard bounds");
  }

  // Each candidate is a complete tile, CTA, and grid plan. Reserve space for
  // per-input shared-memory alignment and scheduler bookkeeping. The bound is
  // deliberately conservative because a larger tile can exceed the device's
  // per-block shared-memory allocation even when each box dimension is legal.
  if (route_tile_choices) {
    std::vector<scheduler::jev::ChoiceOption> options;
    // At outer_max=256, omit only one-row tiles: the native target is at least
    // 1024 elements and inner <=256, so its preference is always >=4 here.
    for (int64_t candidate = first_outer; candidate <= tma_tile_outer_max;
         ++candidate) {
      if (!legal_outer(tma_tile_inner, candidate)) {
        continue;
      }
      const int64_t bdimx = std::min(threads_per_warp, tma_tile_inner);
      const int64_t bdimy = std::min(128L / bdimx, candidate);
      const int64_t grid_y = ceilDiv(tma_domain_outer, candidate);
      options.push_back(
          {"o" + std::to_string(candidate),
           "TMA outer box " + std::to_string(candidate) +
               "; CTA " + std::to_string(bdimx) + "x" +
               std::to_string(bdimy) + "; grid " +
               std::to_string(ceilDiv(tma_domain_inner, tma_tile_inner)) +
               "x" + std::to_string(grid_y) +
               "; split_y " + std::to_string(grid_y > max_grid_y_dim)});
    }
    NVF_ERROR(!options.empty(), "PW4.tile_outer has no hard-valid candidate");
    std::string state = "pointwise.tma_tile_outer.v1\n";
    state += "domain_inner=" + std::to_string(tma_domain_inner) + '\n';
    state += "domain_outer=" + std::to_string(tma_domain_outer) + '\n';
    state += "tile_inner=" + std::to_string(tma_tile_inner) + '\n';
    state += "min_dtype_bits=" + std::to_string(min_dtype_bits) + '\n';
    state += "max_dtype_bits=" + std::to_string(max_dtype_bits) + '\n';
    state += "max_grid_x=" + std::to_string(max_grid_x_dim) + '\n';
    state += "max_grid_y=" + std::to_string(max_grid_y_dim) + '\n';
    state += "max_threads_per_block=" +
        std::to_string(max_threads_per_block) + '\n';
    state += "smem_per_block_bytes=" +
        std::to_string(smem_per_block_bytes) + '\n';
    state += "tma_input_count=" + std::to_string(tma_input_count) + '\n';
    state += "has_block_quantization=" +
        std::to_string(has_block_quantization) + '\n';
    state += "bits_per_sm=" + std::to_string(bits_per_sm) + '\n';
    state += "bits_per_element=" + std::to_string(bits_per_element) + '\n';
    state += "smem_bits_per_element=" +
        std::to_string(smem_bits_per_element) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "o" + std::to_string(tma_tile_outer)
        : std::string();
    const std::string selected = scheduler::jev::chooseDecision(
        "PW4.tile_outer", state, options, native_choice);
    bool matched = false;
    for (const auto& option : options) {
      if (selected == option.id) {
        tma_tile_outer = std::stoll(selected.substr(1));
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "PW4.tile_outer response is outside hard bounds");
  }

  params->tma_tile_inner = tma_tile_inner;
  params->tma_tile_outer = tma_tile_outer;

  // typical max gdimy is 65535, while max gdimx is 2^31 − 1 for most devices no
  // need to check for gdimx
  int64_t gdimy = ceilDiv(tma_domain_outer, tma_tile_outer);
  params->split_grid_y_dim = gdimy > max_grid_y_dim;

  // ========== Step 4: Configure Thread Block Dimensions ==========
  // bdimx strategy:
  // - Use min(32, tma_tile_inner) to avoid using more threads than elements
  // - For small tiles (e.g., tma_tile_inner=8 when input is 17x16), bdimx=8
  //   since it doesn't make sense to use more than 8 threads for 8 elements
  // - For most cases, tma_tile_inner is a multiple of 32, so bdimx=32
  //
  // Benefits of bdimx=32 (when possible):
  // (a) Avoids bank conflicts: 32 threads accessing 32 elements = no conflict
  //     Using fewer threads (e.g., 16) would cause multi-way bank conflicts if
  //     vectorized load from shared memory to registers is not used.
  // (b) Enables vectorization: With max tma_tile_inner=256, using 32 threads
  //     means each thread processes 8 elements, enabling vectorized writes to
  //     global memory (e.g., 8-element vectors for optimal performance)
  constexpr int64_t threads_per_cta = 128;
  const int64_t bdimx = std::min(threads_per_warp, tma_tile_inner);
  const int64_t bdimy = std::min(threads_per_cta / bdimx, tma_tile_outer);
  params->lparams.bindUnsafe(bdimx, ParallelType::TIDx);
  params->lparams.bindUnsafe(bdimy, ParallelType::TIDy);

  // ========== Step 5: Determine Vectorization Factor ==========
  // Don't limit vectorization factor by number of IO tensors or wave count
  // since some of inputs are TMA-loaded which uses shared memory instead of
  // registers.
  const auto logical_reorder_map = pointwise_utils::getLogicalReorderMap(
      prop.largest_out, prop.has_reshapes, data_cache);
  int64_t vectorization_factor = 0;
  if (jev_mode != scheduler::jev::Mode::Remote) {
    // Keep the original analysis in off and record modes. TMA disables both
    // optional soft caps, so its native preference is the layout limit.
    vectorization_factor = vectorize_helper::getVectorizationFactor(
        runtime_info,
        prop.largest_out,
        data_cache,
        bp_info.break_point,
        /*max_vectorization_size_in_bit=*/128,
        prop.min_dtype_size_bit_for_vectorization,
        prop.max_dtype_size_bit_for_vectorization,
        /*n_vectorizable_tensors=*/-1,
        /*n_waves=*/-1,
        logical_reorder_map);
  }
  if (jev_mode != scheduler::jev::Mode::Off) {
    // Layout and alignment analysis is a correctness bound. It does not
    // calculate the native preference in remote mode.
    const int64_t layout_max = vectorize_helper::getVectorizationFactor(
        runtime_info,
        prop.largest_out,
        data_cache,
        bp_info.break_point,
        /*max_vectorization_size_in_bit=*/128,
        /*min_dtype_size_bit=*/-1,
        /*max_dtype_size_bit=*/-1,
        /*n_vectorizable_tensors=*/-1,
        /*n_waves=*/-1,
        logical_reorder_map);
    const int64_t min_dtype_bits = prop.min_dtype_size_bit_for_vectorization;
    const int64_t byte_min = min_dtype_bits < 8
        ? scheduler_utils::safeDiv(8L, min_dtype_bits)
        : 1;

    // The PointWise wrapper caps block quantization at four and rejects
    // sub-byte quantization below width two. Apply those same hard bounds to
    // this menu so a remote choice cannot newly reject a schedulable fusion.
    const int64_t hard_max =
        has_block_quantization ? std::min<int64_t>(layout_max, 4) : layout_max;
    const int64_t hard_min =
        std::max<int64_t>(byte_min, requires_quant_width_two ? 2 : 1);
    NVF_ERROR(
        layout_max >= 1 && layout_max <= 128 &&
            scheduler_utils::lastPow2(layout_max) == layout_max &&
            scheduler_utils::lastPow2(byte_min) == byte_min,
        "PW4.width layout and byte bounds must be powers of two in 1..128");

    std::vector<scheduler::jev::ChoiceOption> options;
    const int64_t first_width = std::min(hard_min, hard_max);
    for (int64_t width = first_width; width <= hard_max; width *= 2) {
      options.push_back(
          {"w" + std::to_string(width),
           "Vectorize " + std::to_string(width) + " element(s) per thread"});
    }
    // If the layout limit is below a native minimum, the existing source
    // returns that limit and canScheduleRunTime decides whether to reject it.
    std::string state = "pointwise.tma_width.v1\n";
    state += "layout_max=" + std::to_string(layout_max) + '\n';
    state += "min_dtype_bits=" + std::to_string(min_dtype_bits) + '\n';
    state += "max_dtype_bits=" +
        std::to_string(prop.max_dtype_size_bit_for_vectorization) + '\n';
    state += "n_vectorizable_tensors=" +
        std::to_string(prop.vectorizable_inputs_outputs.size()) + '\n';
    state += "has_block_quantization=" +
        std::to_string(has_block_quantization) + '\n';
    state += "requires_quant_width_two=" +
        std::to_string(requires_quant_width_two) + '\n';
    const std::string native_choice = jev_mode == scheduler::jev::Mode::Record
        ? "w" + std::to_string(std::min(vectorization_factor, hard_max))
        : std::string();
    const std::string selected = scheduler::jev::chooseDecision(
        "PW4.width", state, options, native_choice);
    bool matched = false;
    for (int64_t width = first_width; width <= hard_max; width *= 2) {
      if (selected == "w" + std::to_string(width)) {
        vectorization_factor = width;
        matched = true;
        break;
      }
    }
    NVF_ERROR(matched, "PW4.width response is outside native hard bounds");
  }
  params->vectorization_factor = vectorization_factor;

  // TMA store
  params->use_tma_store = false;

  if (emit_debug && isDebugDumpEnabled(DebugDumpOption::SchedulerDebug)) {
    debug() << "\n==== Pointwise TMA Scheduler Heuristics ====\n";
    debug() << "Domain sizes:\n";
    debug() << "  n_elems: " << prop.n_elems << "\n";
    debug() << "  break_point: " << bp_info.break_point << "\n";
    debug() << "  tma_domain_inner: " << tma_domain_inner << "\n";
    debug() << "  tma_domain_outer: " << tma_domain_outer << "\n";
    debug() << "\nMemory and CTA configuration:\n";
    debug() << "  cta_per_sm: " << cta_per_sm << "\n";
    debug() << "  bits_per_sm: " << bits_per_sm << "\n";
    debug() << "  bits_per_cta: " << bits_per_cta << "\n";
    debug() << "  bits_per_element: " << bits_per_element << "\n";
    debug() << "  elements_per_cta: " << elements_per_cta << "\n";
    debug() << "\nTMA tile configuration:\n";
    debug() << "  kMaxElementsPerTmaTileDim: " << kMaxElementsPerTmaTileDim
            << "\n";
    debug() << "  tma_tile_inner_max: " << tma_tile_inner_max << "\n";
    debug() << "  tma_tile_inner: " << tma_tile_inner << "\n";
    debug() << "  tma_tile_outer: " << tma_tile_outer << "\n";
    debug() << "  tma_tile_size: " << (tma_tile_inner * tma_tile_outer) << "\n";
    debug() << "  use_tma_load: " << params->use_tma_load << "\n";
    debug() << "  use_tma_store: " << params->use_tma_store << "\n";
    debug() << "\nThread block configuration:\n";
    debug() << "  blockDim.x (TIDx): " << bdimx << "\n";
    debug() << "  blockDim.y (TIDy): " << bdimy << "\n";
    debug() << "  threads_per_cta: " << (bdimx * bdimy) << "\n";
    debug() << "\nVectorization:\n";
    debug() << "  max_dtype_size_bit: "
            << prop.max_dtype_size_bit_for_vectorization << "\n";
    debug() << "  min_dtype_size_bit: "
            << prop.min_dtype_size_bit_for_vectorization << "\n";
    debug() << "  vectorization_factor: " << params->vectorization_factor
            << "\n";
    debug() << "============================================\n";
  }
  return params;
}

std::string_view tmaBreakPointRemoteRouteabilityReason(
    std::size_t loop_axes,
    int64_t n_elems,
    int64_t break_point,
    int64_t layout_width,
    int64_t min_dtype_bits) {
  if (loop_axes > 32) {
    return "more than 32 loop dimensions";
  }
  if (break_point == 0 && n_elems > (1LL << 40)) {
    return "extent exceeds bounded divisor replay";
  }
  // The Off materialization uses the uncapped layout factor: its optional
  // register and wave limits are disabled. This is the later PW4.width menu's
  // layout_max, so checking it here does not compute another preference.
  if (layout_width < 1 || layout_width > 128 ||
      scheduler_utils::lastPow2(layout_width) != layout_width ||
      min_dtype_bits <= 0 ||
      (min_dtype_bits < 8 && 8 % min_dtype_bits != 0)) {
    return "PW4.width layout and byte bounds must be powers of two in 1..128";
  }
  const int64_t byte_min = min_dtype_bits < 8 ? 8 / min_dtype_bits : 1;
  if (scheduler_utils::lastPow2(byte_min) != byte_min) {
    return "PW4.width layout and byte bounds must be powers of two in 1..128";
  }
  if (layout_width < byte_min) {
    return "PW4.width layout is below the required byte floor";
  }
  return {};
}

namespace {

TmaBreakPointAvailability materializeTmaBreakPointAvailability(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const pointwise_utils::FusionRuntimeProperties& prop,
    const pointwise_utils::TmaBreakPointSearch& search) {
  TmaBreakPointAvailability result;
  result.state = search.state;
  const int64_t min_dtype_bits = prop.min_dtype_size_bit_for_vectorization;
  const int64_t reference_dtype_bits =
      dataTypeSizeBit(prop.largest_out->getDataType(), prop.index_type);
  for (const auto& candidate : search.candidates) {
    // The structural guards precede full native parameter construction.
    if (candidate.break_point == 0) {
      if (min_dtype_bits <= 0 || min_dtype_bits > 128 ||
          prop.n_elems % (256 / min_dtype_bits) != 0) {
        continue;
      }
      const auto inner = scheduler_utils::getTmaDomainInner(
          prop.n_elems, 512, min_dtype_bits);
      if (inner <= 1 || inner >= prop.n_elems) {
        continue;
      }
    } else if (min_dtype_bits <= 0 || min_dtype_bits > 128 ||
               candidate.right_elem_count < 256 / min_dtype_bits ||
               prop.n_elems % candidate.right_elem_count != 0 ||
               ((candidate.right_elem_count % 128) *
                (reference_dtype_bits % 128)) % 128 != 0) {
      continue;
    }
    bool hard_valid = false;
    std::unique_ptr<PointwiseParams> materialized;
    try {
      // ResizeVectorizationFactors depends on break_point. Never seed a
      // shared cache while inspecting alternatives, and emit no child Choice
      // or SchedulerDebug output for a plan that may not be selected.
      materialized = getPointwiseHeuristicsForBreakPoint(
          fusion,
          runtime_info,
          nullptr,
          prop,
          candidate,
          scheduler::jev::Mode::Off,
          &hard_valid,
          /*emit_debug=*/false);
    } catch (const nvfError&) {
      continue;
    }
    if (materialized == nullptr || !hard_valid) {
      continue;
    }
    const auto id = "bp" + std::to_string(candidate.break_point);
    result.options.push_back(
        {id,
         candidate.break_point == 0
             ? "Flatten to the one-dimensional TMA domain"
             : "Split before loop axis " +
                   std::to_string(candidate.break_point)});
    result.plans.push_back(candidate);
    result.materialized_layout_widths.push_back(
        materialized->vectorization_factor);
    // Native Off materialization with hard_valid=true already proves the
    // preferred domain/tile fits resource bounds. PW4.domain_inner retains
    // that t512 option; PW4.tile_inner and PW4.tile_outer each retain at
    // least one compatible candidate. PW4.width then has a nonempty menu if
    // its source layout and byte constraints have the required form.
    result.remote_unavailability_reasons.emplace_back(
        tmaBreakPointRemoteRouteabilityReason(
            prop.elem_counts.size(),
            prop.n_elems,
            candidate.break_point,
            materialized->vectorization_factor,
            min_dtype_bits));
  }
  return result;
}

} // namespace

TmaBreakPointAvailability getTmaBreakPointAvailability(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  // This public preview is for a future parent-first PW1 comparison. Both
  // structural search and candidate materialization are cache independent.
  const auto search = pointwise_utils::getTmaBreakPointSearch(prop, nullptr);
  return materializeTmaBreakPointAvailability(
      fusion, runtime_info, prop, search);
}

std::unique_ptr<PointwiseParams> getNativePointwiseHeuristicsPreview(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  const auto native = pointwise_utils::getBreakPoint(
      fusion, prop, nullptr, /*is_tma=*/true);
  return getPointwiseHeuristicsForBreakPoint(
      fusion,
      runtime_info,
      nullptr,
      prop,
      native,
      scheduler::jev::Mode::Off,
      nullptr,
      /*emit_debug=*/false);
}

std::unique_ptr<PointwiseParams> getPointwiseHeuristicsPreviewForBreakPoint(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    const pointwise_utils::FusionRuntimeProperties& prop,
    const pointwise_utils::BreakPointInfo& break_point) {
  return getPointwiseHeuristicsForBreakPoint(
      fusion,
      runtime_info,
      nullptr,
      prop,
      break_point,
      scheduler::jev::Mode::Off,
      nullptr,
      /*emit_debug=*/false);
}

TmaBreakPointAvailability remoteRouteableTmaBreakPoints(
    const TmaBreakPointAvailability& materialized,
    int64_t n_elems,
    std::size_t loop_axes,
    int64_t min_dtype_bits) {
  NVF_ERROR(
      materialized.options.size() == materialized.plans.size() &&
          materialized.options.size() ==
              materialized.materialized_layout_widths.size() &&
          materialized.options.size() ==
              materialized.remote_unavailability_reasons.size(),
      "PW4.breakpoint availability vectors must remain aligned");
  TmaBreakPointAvailability result;
  std::string widths;
  for (size_t i = 0; i < materialized.options.size(); ++i) {
    if (i != 0) {
      widths += ',';
    }
    widths += materialized.options[i].id + ':' +
        std::to_string(materialized.materialized_layout_widths[i]);
    if (!materialized.remote_unavailability_reasons[i].empty()) {
      result.remote_filter_applied = true;
      continue;
    }
    result.options.push_back(materialized.options[i]);
    result.plans.push_back(materialized.plans[i]);
    result.materialized_layout_widths.push_back(
        materialized.materialized_layout_widths[i]);
    result.remote_unavailability_reasons.emplace_back();
  }
  if (!result.remote_filter_applied) {
    return materialized;
  }
  constexpr std::string_view v1_tag = "pointwise.tma_breakpoint.v1\n";
  NVF_ERROR(
      materialized.state.rfind(v1_tag, 0) == 0,
      "PW4.breakpoint filtered menu requires the v1 base state");
  result.state = "pointwise.tma_breakpoint.v2\n" +
      materialized.state.substr(v1_tag.size());
  result.state += "n_elems=" + std::to_string(n_elems) + '\n';
  result.state += "loop_axes=" + std::to_string(loop_axes) + '\n';
  result.state += "min_dtype_bits=" + std::to_string(min_dtype_bits) + '\n';
  result.state += "materialized_layout_widths=" + widths + '\n';
  result.state += "remote_max_loop_axes=32\n";
  result.state += "bp0_replay_max_elems=" +
      std::to_string(1LL << 40) + '\n';
  result.state += "width_max=128\n";
  return result;
}

std::string_view tmaRecordBreakPointCoverageReason(
    const TmaBreakPointAvailability& materialized,
    const TmaBreakPointAvailability& remote_menu,
    std::string_view native_id) {
  const auto contains = [native_id](const auto& options) {
    return std::any_of(
        options.begin(), options.end(), [&](const auto& option) {
          return option.id == native_id;
        });
  };
  if (contains(remote_menu.options)) {
    return {};
  }
  return contains(materialized.options)
      ? "native TMA breakpoint has no Remote-reachable PW4 continuation"
      : "native TMA plan is outside the conservative complete-plan menu";
}

std::unique_ptr<PointwiseParams> getPointwiseHeuristics(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    HeuristicDataCache* data_cache,
    const pointwise_utils::FusionRuntimeProperties& prop) {
  const auto mode = scheduler::jev::mode();
  if (mode == scheduler::jev::Mode::Off) {
    const auto native = pointwise_utils::getBreakPoint(
        fusion, prop, data_cache, /*is_tma=*/true);
    return getPointwiseHeuristicsForBreakPoint(
        fusion, runtime_info, data_cache, prop, native, mode);
  }

  // Build the legacy plan first in record mode. It preserves the existing
  // fallback to ordinary pointwise when TMA cannot materialize and records
  // the downstream tile/width decisions exactly once.
  pointwise_utils::BreakPointInfo native;
  std::unique_ptr<PointwiseParams> native_params;
  if (mode == scheduler::jev::Mode::Record) {
    native = pointwise_utils::getBreakPoint(
        fusion, prop, data_cache, /*is_tma=*/true);
    native_params = getPointwiseHeuristicsForBreakPoint(
        fusion, runtime_info, data_cache, prop, native, mode);
    if (native_params == nullptr) {
      return nullptr;
    }
  }

  const auto search = pointwise_utils::getTmaBreakPointSearch(prop, data_cache);
  if (prop.elem_counts.size() > 32) {
    if (mode == scheduler::jev::Mode::Record) {
      scheduler::jev::recordCoverageSkip(
          "PW4.breakpoint", "more than 32 loop dimensions");
      return native_params;
    }
    NVF_ERROR(false, "PW4.breakpoint supports at most 32 loop dimensions");
  }

  const auto availability =
      materializeTmaBreakPointAvailability(fusion, runtime_info, prop, search);
  const auto menu = remoteRouteableTmaBreakPoints(
      availability,
      prop.n_elems,
      prop.elem_counts.size(),
      prop.min_dtype_size_bit_for_vectorization);
  const auto& options = menu.options;
  const auto& plans = menu.plans;

  if (mode == scheduler::jev::Mode::Record) {
    const auto native_id = "bp" + std::to_string(native.break_point);
    const auto skip_reason =
        tmaRecordBreakPointCoverageReason(availability, menu, native_id);
    if (!skip_reason.empty()) {
      scheduler::jev::recordCoverageSkip(
          "PW4.breakpoint", skip_reason);
      return native_params;
    }
    (void)scheduler::jev::chooseDecision(
        "PW4.breakpoint", menu.state, options, native_id);
    return native_params;
  }

  NVF_ERROR(
      !options.empty(),
      menu.remote_filter_applied
          ? "PW4.breakpoint has no Remote-routeable TMA plan"
          : "PW4.breakpoint has no native-valid TMA plan");
  const auto selected = scheduler::jev::chooseDecision(
      "PW4.breakpoint", menu.state, options);
  for (size_t i = 0; i < options.size(); ++i) {
    if (options[i].id != selected) {
      continue;
    }
    // The API selects only from plans that passed native materialization.
    // Rebuild the selected plan in remote mode for the downstream Choices.
    return getPointwiseHeuristicsForBreakPoint(
        fusion, runtime_info, data_cache, prop, plans[i], mode);
  }
  NVF_ERROR(false, "PW4.breakpoint response is outside the materialized menu");
}

// TODO: Inline intermediate operations (avoid inlining unrolled/vectorized
// input/output caches)
void schedulePointwise(Fusion* fusion, const PointwiseParams* pparams) {
  FusionGuard fg(fusion);

  // ========== Phase 1: Common Setup and Caching ==========
  // Merge all dimensions into a single iteration domain. The TMA domain split
  // will be handled later via domain size parameters.
  auto schedule_info_opt =
      pointwise_utils::commonPointwiseSchedule(fusion, pparams->break_point);
  if (!schedule_info_opt.has_value()) {
    // Zero-dimensional tensors, nothing to schedule
    return;
  }
  auto& schedule_info = schedule_info_opt.value();

  auto& cached_inputs = schedule_info.cached_inputs;
  auto& cached_outputs = schedule_info.cached_outputs;

  // reference_tv: The tensor view used as a template for transformations
  TensorView* reference_tv = schedule_info.reference_tv;

  // Get inputs/outputs that have the innermost dimension and can be vectorized
  auto inputs_outputs =
      scheduler_utils::getInputsOutputsWithInnerDim(reference_tv, true, true);
  std::unordered_set<TensorView*> vectorizable_io_tvs(
      inputs_outputs.begin(), inputs_outputs.end());

  // ========== Phase 2: Classify Inputs for TMA vs LDG Loading ==========
  // TMA (Tensor Memory Accelerator): Hardware-accelerated global->shared loads
  // LDG (LoaD Global): Standard global memory loads for tensors not suitable
  // for TMA

  // n_valid_dims: Number of logical dimensions in the reference tensor
  int64_t n_valid_dims = scheduler_utils::nLogicalDims(reference_tv);

  // tma_tvs: Inputs that will use TMA load (same dimensionality as reference)
  std::vector<TensorView*> tma_tvs;

  // ldg_tvs: Inputs that will use standard global loads (different
  // dimensionality)
  std::vector<TensorView*> ldg_tvs;

  for (const auto& [tv, _] : cached_inputs) {
    if (!isTvSuitableForTma(tv, n_valid_dims)) {
      ldg_tvs.push_back(tv);
      continue;
    }
    // Configure the load operation to use TMA
    auto load_op = dynamic_cast<LoadStoreOp*>(tv->definition());
    if (load_op) {
      load_op->setOpType(LoadStoreOpType::CpAsyncBulkTensorTile);
    }
    // TMA loads data into shared memory
    tv->setMemoryType(MemoryType::Shared);
    // Create a register cache after the shared memory tensor
    tv->cacheAfter();
    tma_tvs.push_back(tv);
  }

  // ========== Phase 3: Split Into TMA Domain and Tiles ==========
  // Transform the flattened domain through a two-level hierarchy:
  // Step 1: [I0] -> [tma_domain_outer, tma_domain_inner]
  //   Split into outer and inner domains based on tma_domain_inner
  // Note: Skip this split if break_point != 0, as the reference tensor has
  // already been transformed into [lhs, rhs] domains at the break point by
  // commonPointwiseSchedule, which provides the required 2D TMA structure.
  // TODO: consider device domain and non-concretized domain.
  if (pparams->break_point == 0) {
    NVF_ERROR_EQ(reference_tv->nDims(), 1);
    reference_tv->split(0, pparams->tma_domain_inner);
  } else {
    NVF_ERROR_EQ(reference_tv->nDims(), 2);
  }

  // Step 2: [tma_domain_outer, tma_domain_inner] ->
  //         [tma_domain_outer/tma_tile_outer, tma_tile_outer,
  //          tma_domain_inner/tma_tile_inner, tma_tile_inner]
  //   Split each domain into tiles based on tma_tile sizes
  //   This creates: [outer_grid, outer_tile, inner_grid, inner_tile]
  int64_t ogpos = 0; // outer grid position
  reference_tv->split(ogpos + 1, pparams->tma_tile_inner);
  reference_tv->split(ogpos, pparams->tma_tile_outer);
  if (pparams->split_grid_y_dim) {
    const int64_t max_grid_y_dim =
        at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    reference_tv->split(0, max_grid_y_dim);
    ogpos++; // 1
  }
  // [outer_serial[optional], outer_grid, outer_tile, inner_grid, inner_tile]
  // reorder to:
  // [outer_serial[optional], outer_grid, inner_grid, outer_tile, inner_tile]
  reference_tv->reorder({{ogpos + 1, ogpos + 2}});

  // Propagate these transformations to all tensors in the fusion
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);

  // ========== Phase 4: Parallelize TMA Tensors ==========
  // Parallelization strategy for TMA:
  //   axis(ogpos) [tma_domain_outer/tma_tile_outer]: Grid (BIDy or BIDx)
  //   axis(ogpos+1) [tma_domain_inner/tma_tile_inner]: Grid (BIDx or BIDy)
  //   axis(ogpos+2) [tma_tile_outer]:                  Bulk (TMA tile)
  //   axis(ogpos+3) [tma_tile_inner]:                  Bulk (TMA tile)

  // outer_cord_pt/inner_cord_pt: Grid parallelization types (BIDx/BIDy)
  auto outer_cord_pt = ParallelType::BIDy;
  auto inner_cord_pt = ParallelType::BIDx;
  if (pparams->flip_grid_binding) {
    std::swap(outer_cord_pt, inner_cord_pt);
  }

  // Apply TMA parallelization to reference
  reference_tv->axis(ogpos)->parallelize(outer_cord_pt); // Outer grid
  reference_tv->axis(ogpos + 1)->parallelize(inner_cord_pt); // Inner grid
  reference_tv->axis(ogpos + 2)->parallelize(ParallelType::Bulk); // Outer tile
  reference_tv->axis(ogpos + 3)->parallelize(ParallelType::Bulk); // Inner tile

  // Apply same parallelization to all TMA input tensors
  scheduler_utils::parallelizeAllLike(reference_tv, tma_tvs);

  // Reset reference tensor's tile axes to Serial for subsequent scheduling
  // (TMA tensors keep Bulk parallelization; reference is for non-TMA
  // scheduling)
  reference_tv->axis(ogpos + 2)->parallelize(ParallelType::Serial);
  reference_tv->axis(ogpos + 3)->parallelize(ParallelType::Serial);

  // ========== Phase 5: Schedule Non-TMA Tensors ==========
  // Starting structure: [outer_grid, inner_grid, outer_tile, inner_tile]
  // Target structure:   [outer_grid, inner_grid, outer_tile/y, y,
  // inner_tile/v/x, x, v]
  //   where y = TIDy (threads), x = TIDx (threads), v = vectorization

  // Split inner tile: inner_tile -> [inner_tile/v/x, x, v]
  reference_tv->split(ogpos + 3, pparams->vectorization_factor);
  reference_tv->split(ogpos + 3, pparams->lparams.bdimx());

  // Split outer tile: tma_tile_outer -> [tma_tile_outer/y, y]
  reference_tv->split(ogpos + 2, pparams->lparams.bdimy());

  // from: [..., inner_grid, outer_tile/y, y, inner_tile/v/x, x, v]
  //   to: [..., inner_grid, outer_tile/y, inner_tile/v/x, y, x, v]
  // basically swap [y] with [inner_tile/v/x]
  reference_tv->reorder({{ogpos + 3, ogpos + 4}});
  // Propagate these transformations to all non-TMA tensors
  // (TMA tensors already have their final schedule from Phase 4)
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&non_tma_propagator);

  // ========== Phase 6: Apply Thread Parallelization ==========
  // Final axis structure: [outer_grid, inner_grid, outer_tile/y,
  // inner_tile/v/x, y, x, v]
  //   axis(ogpos): Outer grid dimension
  //   axis(ogpos + 1): Inner grid dimension
  //   axis(ogpos + 2): Outer tile / TIDy serial
  //   axis(ogpos + 3): Inner tile / vect / TIDx serial
  //   axis(ogpos + 4): Thread block Y dimension (TIDy)
  //   axis(ogpos + 5): Thread block X dimension (TIDx)
  //   axis(ogpos + 6): Vectorization dimension
  reference_tv->axis(ogpos + 4)->parallelize(ParallelType::TIDy); // Thread Y
  reference_tv->axis(ogpos + 5)->parallelize(ParallelType::TIDx); // Thread X

  int64_t vect_pos = ogpos + 6; // Position of vectorization axis
  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // ========== Phase 7: Apply Vectorization ==========
  // Vectorize register <-> global memory transfers for non-TMA tensors

  // Vectorize output stores (register -> global)
  if (!pparams->use_tma_store && pparams->vectorization_factor > 1) {
    for (const auto& [_, original_idx] : cached_outputs) {
      auto output_tv =
          dynamic_cast<TensorView*>(fusion->outputs().at(original_idx));
      if (output_tv && vectorizable_io_tvs.contains(output_tv)) {
        output_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  // Vectorize LDG input loads (global -> register)
  if (pparams->vectorization_factor > 1) {
    for (auto ldg_tv : ldg_tvs) {
      if (vectorizable_io_tvs.contains(ldg_tv)) {
        ldg_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
      }
    }
  }

  // ========== Phase 8: Inline ==========
  // Inline TMA and LDG loads right after the last block parallelization axis.
  // This ensures global memory loads are issued early, before computations.
  // LDG tensors are typically broadcasts with minimal register usage.
  auto getLastBlockParallelizationAxisPosition = [](TensorView* tv) -> int64_t {
    for (auto i : arange(tv->nDims()) | std::views::reverse) {
      if (tv->axis(i)->isBlockDim()) {
        return i + 1;
      }
    }
    return 0;
  };
  std::vector<TensorView*> tma_or_ldg_tvs(tma_tvs.begin(), tma_tvs.end());
  tma_or_ldg_tvs.insert(tma_or_ldg_tvs.end(), ldg_tvs.begin(), ldg_tvs.end());
  for (auto tv : tma_or_ldg_tvs) {
    int64_t inline_pos = getLastBlockParallelizationAxisPosition(tv);
    NVF_ERROR_LT(inline_pos, tv->nDims());
    // All domains after inline position must have const extent
    NVF_ERROR(
        std::ranges::all_of(
            tv->getLoopDomain() | std::views::drop(inline_pos),
            [](const IterDomain* id) { return id->extent()->isConst(); }),
        "All loop domains after inline position ",
        inline_pos,
        " must have constant extent for TensorView ",
        tv->toString());
    tv->inlineAt(inline_pos);
  }
  // inline other tensors to minimize register pressure
  std::vector<TensorView*> compute_tvs = ir_utils::allTvsExcept(
      fusion, {tma_or_ldg_tvs.begin(), tma_or_ldg_tvs.end()});
  inlineMost(compute_tvs);
}

} // namespace tma
} // namespace pointwise
} // namespace nvfuser
