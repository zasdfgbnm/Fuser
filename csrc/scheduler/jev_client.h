// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace nvfuser::scheduler::jev {

enum class Mode { Off, Record, Remote };
Mode mode();

// The identifier is the wire-format Choice key. Descriptions provide the
// model with context; neither field contains the native decision.
struct ChoiceOption {
  std::string id;
  std::string description;
};

// NVFUSER_JEV_MODE: off (default), record, or remote.
// NVFUSER_JEV_LOG: required JSONL path in record/remote mode.
// NVFUSER_JEV_PORT: loopback server port (default 8765) in remote mode.
// NVFUSER_JEV_CASE_KEY: optional benchmark case identifier for log matching.
// native_choice is required in off/record modes. Call mode() before computing
// the native preference: remote mode accepts an empty native_choice so a
// scheduler can skip that computation entirely.
// It is never serialized into the HTTP request or remote-mode log.
std::string chooseDecision(
    std::string_view site,
    std::string_view state,
    const std::vector<ChoiceOption>& options,
    std::string_view native_choice = {},
    std::string_view case_key = {});

// Record why a native plan could not be represented by a conservative Choice
// menu. This is a coverage event, not a decision; it preserves native params.
void recordCoverageSkip(
    std::string_view site,
    std::string_view reason,
    std::string_view case_key = {});

} // namespace nvfuser::scheduler::jev
