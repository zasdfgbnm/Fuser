// SPDX-License-Identifier: BSD-3-Clause
// Host only: c++ -std=c++20 -Icsrc tests/cpp/test_jev_outcome_standalone.cpp -o /tmp/test_jev_outcome
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

#include "scheduler/native_outcome.h"

int main() {
  namespace diag = nvfuser::scheduler::native_diagnostic;
  char decision_path[] = "/tmp/nvfuser-jev-decision-XXXXXX";
  const int fd = ::mkstemp(decision_path);
  assert(fd >= 0);
  assert(::close(fd) == 0);
  assert(::unlink(decision_path) == 0);
  const std::string outcome_path =
      std::string(decision_path) + ".outcome.jsonl";
  char explicit_path[] = "/tmp/nvfuser-jev-outcome-XXXXXX";
  const int explicit_fd = ::mkstemp(explicit_path);
  assert(explicit_fd >= 0);
  assert(::close(explicit_fd) == 0);
  assert(::unlink(explicit_path) == 0);

  diag::Outcome outcome;
  outcome.scheduler_type = "Reduction";
  outcome.tag = "modified\nplan";
  outcome.heuristic_text = "complete \"reduction\" plan";
  outcome.params_kind = "HeuristicParams";
  outcome.scheduler_fields_json = "{}";
  outcome.launch.block_x_raw = 128;
  outcome.launch.dynamic_smem_bytes = 4096;
  outcome.compile.index_type = 6;
  outcome.compile.maxrregcount = 96;
  outcome.compile.enable_magic_zero = false;
  outcome.compile.device = "cuda:0";
  outcome.compile.include_paths = {"/first", "/quoted\"path"};
  outcome.compile.block_x = 128;
  outcome.compile.compute_block_x = 64;

  assert(::unsetenv(diag::kLogEnvironment) == 0);
  assert(::unsetenv(diag::kDecisionLogEnvironment) == 0);
  assert(::unsetenv(diag::kModeEnvironment) == 0);
  diag::recordOutcome(outcome);
  assert(::access(outcome_path.c_str(), F_OK) != 0);

  assert(::setenv(diag::kDecisionLogEnvironment, decision_path, 1) == 0);
  assert(::setenv(diag::kModeEnvironment, "record", 1) == 0);
  assert(::setenv("NVFUSER_JEV_CASE_KEY", "case-1", 1) == 0);
  diag::recordOutcome(outcome);
  assert(::setenv(diag::kModeEnvironment, "remote", 1) == 0);
  diag::recordOutcome(outcome);
  std::ifstream stream(outcome_path);
  const std::string data(
      (std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  assert(!data.empty() && data.back() == '\n');
  assert(std::count(data.begin(), data.end(), '\n') == 2);
  assert(data.find("\"schema\":\"nvfuser.native_outcome.v2\"") != std::string::npos);
  assert(data.find("\"kind\":\"final\"") != std::string::npos);
  assert(data.find("\"mode\":\"record\"") != std::string::npos);
  assert(data.find("\"mode\":\"remote\"") != std::string::npos);
  assert(data.find("\"case_key\":\"case-1\"") != std::string::npos);
  assert(data.find("\"scheduler_type\":\"Reduction\"") != std::string::npos);
  assert(data.find("\"tag\":\"modified\\nplan\"") != std::string::npos);
  assert(data.find("\"heuristic_text\":\"complete \\\"reduction\\\" plan\"") != std::string::npos);
  assert(data.find("\"params_kind\":\"HeuristicParams\"") != std::string::npos);
  assert(data.find("\"scheduler_fields\":{}") != std::string::npos);
  assert(data.find("\"block_x_raw\":128") != std::string::npos);
  assert(data.find("\"dynamic_smem_bytes\":4096") != std::string::npos);
  assert(data.find("\"maxrregcount\":96") != std::string::npos);
  assert(data.find("\"enable_magic_zero\":false") != std::string::npos);
  assert(data.find("\"compute_block_x\":64") != std::string::npos);
  assert(data.find("\"include_paths\":[\"/first\",\"/quoted\\\"path\"]") != std::string::npos);
  const std::string key = "\"invocation_id\":\"";
  const auto first_id = data.find(key);
  assert(first_id != std::string::npos);
  const auto first_end = data.find('"', first_id + key.size());
  const auto second_id = data.find(key, first_end);
  assert(second_id != std::string::npos);
  const auto second_end = data.find('"', second_id + key.size());
  const auto first_value = data.substr(first_id + key.size(), first_end - first_id - key.size());
  const auto second_value = data.substr(second_id + key.size(), second_end - second_id - key.size());
  assert(first_value != second_value);
  assert(first_value.starts_with(std::to_string(::getpid()) + "-outcome-"));
  assert(second_value.starts_with(std::to_string(::getpid()) + "-outcome-"));

  // An explicit outcome path also works without enabling Jev decisions.
  assert(::setenv(diag::kModeEnvironment, "off", 1) == 0);
  assert(::setenv(diag::kLogEnvironment, explicit_path, 1) == 0);
  diag::recordOutcome(outcome);
  std::ifstream explicit_stream(explicit_path);
  const std::string explicit_data(
      (std::istreambuf_iterator<char>(explicit_stream)),
      std::istreambuf_iterator<char>());
  assert(std::count(explicit_data.begin(), explicit_data.end(), '\n') == 1);
  assert(explicit_data.find("\"mode\":\"off\"") != std::string::npos);

  // The diagnostic must ignore a path it cannot use, including a directory.
  assert(::setenv(diag::kLogEnvironment, "/tmp", 1) == 0);
  diag::recordOutcome(outcome);
  assert(::unlink(outcome_path.c_str()) == 0);
  assert(::unlink(explicit_path) == 0);
}
