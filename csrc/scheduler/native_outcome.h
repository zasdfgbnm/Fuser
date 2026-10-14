// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "scheduler/native_diagnostic_jsonl.h"

// The JSON writer is independent of nvFuser and CUDA headers, so its wire
// format and failure behavior can be checked with a small host-only test.
namespace nvfuser::scheduler::native_diagnostic {

struct Outcome {
  std::string scheduler_type;
  std::string tag;
  std::string heuristic_text;
  std::string params_kind = "Unknown";
  std::string scheduler_fields_json = "{}";
  struct Launch {
    int64_t grid_x_raw = -1;
    int64_t grid_y_raw = -1;
    int64_t grid_z_raw = -1;
    int64_t block_x_raw = -1;
    int64_t block_y_raw = -1;
    int64_t block_z_raw = -1;
    int64_t dynamic_smem_bytes = 0;
  } launch;
  struct Compile {
    std::optional<int64_t> index_type;
    int64_t maxrregcount = 255;
    bool enable_magic_zero = true;
    bool enable_ptxas_verbose = false;
    std::optional<std::string> device;
    std::vector<std::string> include_paths;
    std::optional<int64_t> block_x;
    std::optional<int64_t> block_y;
    std::optional<int64_t> block_z;
    std::optional<int64_t> compute_block_x;
    std::optional<int64_t> compute_block_y;
    std::optional<int64_t> compute_block_z;
  } compile;
};

inline std::atomic<uint64_t> next_outcome_invocation{0};

inline void recordOutcome(const Outcome& outcome) noexcept {
  if (!enabled()) {
    return;
  }
  try {
    Object launch;
    launch.add("grid_x_raw", outcome.launch.grid_x_raw);
    launch.add("grid_y_raw", outcome.launch.grid_y_raw);
    launch.add("grid_z_raw", outcome.launch.grid_z_raw);
    launch.add("block_x_raw", outcome.launch.block_x_raw);
    launch.add("block_y_raw", outcome.launch.block_y_raw);
    launch.add("block_z_raw", outcome.launch.block_z_raw);
    launch.add("dynamic_smem_bytes", outcome.launch.dynamic_smem_bytes);

    Object compile;
    const auto add_optional_number = [&](std::string_view key,
                                         const std::optional<int64_t>& value) {
      if (value.has_value()) {
        compile.add(key, *value);
      } else {
        compile.addRaw(key, "null");
      }
    };
    add_optional_number("index_type", outcome.compile.index_type);
    compile.add("maxrregcount", outcome.compile.maxrregcount);
    compile.add("enable_magic_zero", outcome.compile.enable_magic_zero);
    compile.add("enable_ptxas_verbose", outcome.compile.enable_ptxas_verbose);
    if (outcome.compile.device.has_value()) {
      compile.add("device", *outcome.compile.device);
    } else {
      compile.addRaw("device", "null");
    }
    std::string include_paths = "[";
    for (size_t i = 0; i < outcome.compile.include_paths.size(); ++i) {
      if (i != 0) {
        include_paths += ',';
      }
      include_paths += quote(outcome.compile.include_paths[i]);
    }
    include_paths += ']';
    compile.addRaw("include_paths", include_paths);
    add_optional_number("block_x", outcome.compile.block_x);
    add_optional_number("block_y", outcome.compile.block_y);
    add_optional_number("block_z", outcome.compile.block_z);
    add_optional_number("compute_block_x", outcome.compile.compute_block_x);
    add_optional_number("compute_block_y", outcome.compile.compute_block_y);
    add_optional_number("compute_block_z", outcome.compile.compute_block_z);

    const char* case_key_env = std::getenv("NVFUSER_JEV_CASE_KEY");
    const auto sequence =
        next_outcome_invocation.fetch_add(1, std::memory_order_relaxed) + 1;
    Object row;
    row.add("schema", "nvfuser.native_outcome.v2");
    row.add("kind", "final");
    row.add("mode", logMode());
    row.add("case_key", case_key_env == nullptr ? "" : case_key_env);
    row.add(
        "invocation_id",
        std::to_string(::getpid()) + "-outcome-" +
            std::to_string(sequence));
    row.add("scheduler_type", outcome.scheduler_type);
    row.add("tag", outcome.tag);
    row.add("heuristic_text", outcome.heuristic_text);
    row.add("params_kind", outcome.params_kind);
    row.addRaw("scheduler_fields", outcome.scheduler_fields_json);
    row.addRaw("launch", launch.str());
    row.addRaw("compile", compile.str());
    appendLine(row.str());
  } catch (...) {
    // A failed observation must not change native scheduling or exceptions.
  }
}

} // namespace nvfuser::scheduler::native_diagnostic
