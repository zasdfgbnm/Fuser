// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace nvfuser::scheduler::jev {

enum class Mode { Off, Record, Remote };
Mode mode();

// Force native choices within one scheduler's correctness-sensitive plan.
// The override is thread-local and nestable; the configured mode is restored
// when the scope ends, so surrounding Record diagnostics remain active.
class ScopedNativeDecisions final {
 public:
  ScopedNativeDecisions() noexcept;
  ~ScopedNativeDecisions() noexcept;
  ScopedNativeDecisions(const ScopedNativeDecisions&) = delete;
  ScopedNativeDecisions& operator=(const ScopedNativeDecisions&) = delete;

 private:
  bool previous_ = false;
};

// Reuse the client's JSON string escaping for local diagnostic snapshots.
std::string jsonQuote(std::string_view value);

// Opt in with NVFUSER_JEV_PROTOCOL=semantic-v2. The default preserves the
// historical text-state experiment. A semantic request is one complete,
// model-visible JSON state object; callers must not send a truncated snapshot.
bool semanticProtocolEnabled();
std::string_view computeInvocationInputSnapshot() noexcept;
std::string_view computeInvocationInputUnavailableReason() noexcept;

// The identifier is the wire-format Choice key. Descriptions provide the
// model with context; neither field contains the native decision.
struct ChoiceOption {
  std::string id;
  std::string description;
};

// NVFUSER_JEV_MODE: off (default), record, or remote.
// NVFUSER_JEV_LOG: required JSONL path in record/remote mode.
// NVFUSER_JEV_PORT: loopback server port (default 8765) in remote mode.
// NVFUSER_JEV_TIMEOUT_MS: total loopback POST deadline in milliseconds
// (default 2000, allowed 1..600000) in remote mode.
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

std::string chooseSemanticDecision(
    std::string_view site,
    std::string_view state_json,
    std::string_view instructions,
    const std::vector<ChoiceOption>& options,
    std::string_view native_choice = {});

// Record why a native plan could not be represented by a conservative Choice
// menu. This is a coverage event, not a decision; it preserves native params.
void recordCoverageSkip(
    std::string_view site,
    std::string_view reason,
    std::string_view case_key = {});

// A passive span around one computeHeuristics call in Record/Remote mode.
// Begin/final/no_choice rows go to the separate native outcome JSONL; Choice
// and coverage rows carry the same compute_invocation_id and an increasing
// compute_event_index. A no_choice row records the actual selected value when
// a structural gate leaves no model menu; it is never a Choice or HTTP event.
// The entry snapshot is owned by the caller and must outlive this scope. An
// exception leaves the invocation without a final row.
bool computeTraceEnabled() noexcept;
class ComputeHeuristicsScope final {
 public:
  ComputeHeuristicsScope(
      std::string_view input_snapshot,
      std::string_view input_snapshot_unavailable_reason) noexcept;
  ~ComputeHeuristicsScope() noexcept;
  ComputeHeuristicsScope(const ComputeHeuristicsScope&) = delete;
  ComputeHeuristicsScope& operator=(const ComputeHeuristicsScope&) = delete;

  bool observing() const noexcept;
  std::string_view invocationId() const noexcept;
  std::string_view inputSnapshot() const noexcept;
  std::string_view inputSnapshotUnavailableReason() const noexcept;
  uint64_t nextEventIndex() noexcept;
  void recordNoChoice(
      std::string_view site,
      std::string_view selected,
      std::string_view reason,
      std::string_view selection_source) noexcept;
  void complete(std::string_view scheduler_type) noexcept;

 private:
  void write(
      std::string_view kind,
      std::string_view scheduler_type = {},
      std::string_view site = {},
      std::string_view selected = {},
      std::string_view reason = {},
      std::string_view selection_source = {}) noexcept;

  ComputeHeuristicsScope* previous_ = nullptr;
  std::string_view input_snapshot_;
  std::string_view input_snapshot_unavailable_reason_;
  std::string invocation_id_;
  std::string case_key_;
  std::string mode_;
  uint64_t event_index_ = 0;
  bool active_ = false;
  bool completed_ = false;
};

// Passive observation inside the active computeHeuristics scope. Diagnostics
// may be absent on write failure but can never affect a schedule or API call.
void recordNoChoiceDecision(
    std::string_view site,
    std::string_view selected,
    std::string_view reason,
    std::string_view selection_source) noexcept;

// Pointwise Record uses a separate outcome JSONL so the existing Choice-only
// decision log and its record/remote comparator keep their wire contract.
// NVFUSER_JEV_POINTWISE_OUTCOME_LOG overrides the default
// NVFUSER_JEV_LOG + ".pointwise.jsonl". Each invocation has a begin row and
// exactly one final row on successful computeHeuristics return. Choice rows
// carry the same invocation_id and a shared event_index (begin is zero);
// an incomplete invocation has no final row.
class PointwiseRecordScope {
 public:
  PointwiseRecordScope();
  ~PointwiseRecordScope() noexcept;
  PointwiseRecordScope(const PointwiseRecordScope&) = delete;
  PointwiseRecordScope& operator=(const PointwiseRecordScope&) = delete;

  bool recording() const {
    return recording_;
  }

  // JSON objects are the complete returned parameters and raw launch/compile
  // settings after all native correctness caps. "empty" denotes no runtime
  // properties; other successful backends are "tma" or "ordinary".
  void complete(
      std::string_view backend,
      std::string_view final_params_json,
      std::string_view launch_json,
      std::string_view compile_json);

 private:
  std::string invocation_id_;
  std::string case_key_;
  std::string previous_invocation_id_;
  std::string previous_case_key_;
  uint64_t previous_event_index_ = 0;
  bool previous_completed_ = false;
  bool recording_ = false;
  bool completed_ = false;
};

} // namespace nvfuser::scheduler::jev
