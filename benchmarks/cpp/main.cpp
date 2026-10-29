// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include "utils.h"

#include <options.h>
#include <runtime/fusion_kernel_runtime.h>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

namespace {

std::string getHostName() {
#if defined(__linux__)
  constexpr int len = 128;
  char name[len];
  gethostname(name, len);
  return std::string(name);
#else
  return "UNKNOWN";
#endif
}

void addGPUBenchmarkContext() {
  int dev_idx = 0;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDevice(&dev_idx));

  cudaDeviceProp prop;
  NVFUSER_CUDA_RT_SAFE_CALL(cudaGetDeviceProperties(&prop, dev_idx));

  int gpu_clock_khz, gpu_mem_clock_khz;
  cudaDeviceGetAttribute(&gpu_clock_khz, cudaDevAttrClockRate, dev_idx);
  cudaDeviceGetAttribute(
      &gpu_mem_clock_khz, cudaDevAttrMemoryClockRate, dev_idx);

  ::benchmark::AddCustomContext("gpu_name", prop.name);
  ::benchmark::AddCustomContext(
      "gpu_gmem_bytes", std::to_string(prop.totalGlobalMem));
  ::benchmark::AddCustomContext(
      "gpu_smem_bytes_per_block", std::to_string(prop.sharedMemPerBlock));
  ::benchmark::AddCustomContext(
      "gpu_regs_per_block", std::to_string(prop.regsPerBlock));
  ::benchmark::AddCustomContext("gpu_clock_khz", std::to_string(gpu_clock_khz));
  ::benchmark::AddCustomContext(
      "gpu_mem_clock_khz", std::to_string(gpu_mem_clock_khz));
  ::benchmark::AddCustomContext(
      "gpu_mem_bus_width_bits", std::to_string(prop.memoryBusWidth));
  ::benchmark::AddCustomContext(
      "gpu_compute_capability_major", std::to_string(prop.major));
  ::benchmark::AddCustomContext(
      "gpu_compute_capability_minor", std::to_string(prop.minor));
  ::benchmark::AddCustomContext(
      "gpu_sm_count", std::to_string(prop.multiProcessorCount));
  ::benchmark::AddCustomContext(
      "gpu_l2_bytes", std::to_string(prop.l2CacheSize));
  ::benchmark::AddCustomContext(
      "gpu_max_threads_per_sm",
      std::to_string(prop.maxThreadsPerMultiProcessor));
  ::benchmark::AddCustomContext(
      "gpu_max_threads_per_block", std::to_string(prop.maxThreadsPerBlock));
}

// NVFUSER_JEV_BATCH_MANIFEST has one ordinal<TAB>exact-name row per selected
// benchmark registration, in unfiltered registry order. The ordinal
// disambiguates registrations with the same printed name.
struct BatchCase {
  size_t ordinal;
  std::string name;

  std::string key() const {
    return std::to_string(ordinal) + ":" + name;
  }
};

std::vector<BatchCase> readBatchManifest(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("cannot open batch manifest: " + path);
  }

  std::vector<BatchCase> cases;
  std::string line;
  while (std::getline(input, line)) {
    const auto tab = line.find('\t');
    if (tab == std::string::npos || tab == 0 || tab + 1 == line.size() ||
        line.find('\t', tab + 1) != std::string::npos ||
        line.find('\r') != std::string::npos ||
        line.find('\0') != std::string::npos) {
      throw std::runtime_error("invalid batch manifest row");
    }

    size_t ordinal = 0;
    const char* begin = line.data();
    const char* end = begin + tab;
    const auto parsed = std::from_chars(begin, end, ordinal);
    if (parsed.ec != std::errc() || parsed.ptr != end || ordinal == 0 ||
        std::to_string(ordinal) != line.substr(0, tab) ||
        (!cases.empty() && ordinal <= cases.back().ordinal)) {
      throw std::runtime_error("batch manifest ordinals must increase");
    }
    cases.push_back({ordinal, line.substr(tab + 1)});
  }
  if (!input.eof() || cases.empty()) {
    throw std::runtime_error("batch manifest is empty or unreadable");
  }
  return cases;
}

void validateBatchArguments(int argc, char** argv) {
  bool filter = false;
  bool repetitions = false;
  bool output = false;
  bool output_format = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg.starts_with("--benchmark_filter=")) {
      if (filter || arg == "--benchmark_filter=") {
        throw std::runtime_error("batch mode needs one nonempty filter");
      }
      filter = true;
    } else if (arg.starts_with("--benchmark_repetitions")) {
      if (repetitions || arg != "--benchmark_repetitions=1") {
        throw std::runtime_error("batch mode requires one repetition");
      }
      repetitions = true;
    } else if (arg.starts_with("--benchmark_out_format")) {
      if (output_format || arg != "--benchmark_out_format=json") {
        throw std::runtime_error("batch mode requires JSON file output");
      }
      output_format = true;
    } else if (arg.starts_with("--benchmark_out")) {
      if (output || arg == "--benchmark_out=") {
        throw std::runtime_error("batch mode needs one output path");
      }
      output = true;
    } else if (arg.starts_with("--benchmark_format") &&
               arg != "--benchmark_format=console") {
      throw std::runtime_error("batch mode supports console display format");
    } else if (arg.starts_with("--benchmark_enable_random_interleaving") &&
               arg != "--benchmark_enable_random_interleaving=false") {
      throw std::runtime_error("batch mode forbids random interleaving");
    } else if ((arg.starts_with("--benchmark_report_aggregates_only") &&
                arg != "--benchmark_report_aggregates_only=false") ||
               (arg.starts_with("--benchmark_display_aggregates_only") &&
                arg != "--benchmark_display_aggregates_only=false") ||
               arg.starts_with("--benchmark_list_tests")) {
      throw std::runtime_error("batch mode requires per-registration reports");
    }
  }
  if (!filter || !repetitions || !output || !output_format) {
    throw std::runtime_error(
        "batch mode requires filter, repetitions=1, and JSON output");
  }
}

class BatchCaseReporter : public ::benchmark::ConsoleReporter {
 public:
  explicit BatchCaseReporter(std::vector<BatchCase> cases)
      : ConsoleReporter(OO_None), cases_(std::move(cases)) {}

  bool ReportContext(const Context& context) override {
    activateCase();
    return ConsoleReporter::ReportContext(context);
  }

  void ReportRuns(const std::vector<Run>& reports) override {
    if (reports.empty()) {
      throw std::runtime_error("batch reporter received an empty run group");
    }
    const bool aggregates = std::all_of(
        reports.begin(), reports.end(),
        [](const Run& run) { return run.run_type == Run::RT_Aggregate; });
    if (aggregates) {
      ConsoleReporter::ReportRuns(reports);
      return;
    }
    if (reports.size() != 1 ||
        reports.front().run_type != Run::RT_Iteration ||
        reports.front().repetitions != 1 ||
        reports.front().repetition_index != 0 ||
        next_ >= cases_.size() ||
        reports.front().run_name.str() != cases_[next_].name) {
      throw std::runtime_error("batch reporter registration order mismatch");
    }
    if (!reports.front().error_occurred &&
        cases_[next_].name.starts_with("NvFuserScheduler_")) {
      const uint64_t recorded = numericGateEvidenceCount() - evidence_before_;
      if (recorded == 0) {
        // Setup-only benchmarks may compute heuristics without going through
        // either iteration helper. Their lack of output parity is explicit.
        recordUncoveredNumericGate();
      } else if (recorded != 1) {
        throw std::runtime_error(
            "batch registration emitted multiple numeric evidence rows");
      }
    }
    ConsoleReporter::ReportRuns(reports);
    ++next_;
    if (next_ < cases_.size()) {
      activateCase();
    } else if (::unsetenv("NVFUSER_JEV_CASE_KEY") != 0) {
      throw std::runtime_error("cannot clear final batch case key");
    }
  }

  void Finalize() override {
    ConsoleReporter::Finalize();
    if (next_ != cases_.size()) {
      throw std::runtime_error("batch reporter missed registrations");
    }
  }

  size_t expected() const {
    return cases_.size();
  }

 private:
  void activateCase() {
    const std::string key = cases_.at(next_).key();
    if (::setenv("NVFUSER_JEV_CASE_KEY", key.c_str(), 1) != 0) {
      throw std::runtime_error("cannot set batch case key");
    }
    evidence_before_ = numericGateEvidenceCount();
  }

  std::vector<BatchCase> cases_;
  size_t next_ = 0;
  uint64_t evidence_before_ = 0;
};

} // namespace

// Copied from BENCHMARK_MAIN with extra custom settings
int main(int argc, char** argv) {
  const char* manifest_env = std::getenv("NVFUSER_JEV_BATCH_MANIFEST");
  const bool batch_mode = manifest_env != nullptr;
  std::vector<BatchCase> batch_cases;
  if (batch_mode) {
    try {
      validateBatchArguments(argc, argv);
      if (std::getenv("NVFUSER_JEV_CASE_KEY") != nullptr) {
        throw std::runtime_error("batch case key is already set");
      }
      batch_cases = readBatchManifest(manifest_env);
    } catch (const std::exception& error) {
      std::cerr << "nvFuser batch mode: " << error.what() << '\n';
      return 2;
    }
  }

  ::benchmark::Initialize(&argc, argv);
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }

  ::benchmark::AddCustomContext("host", getHostName());

  addGPUBenchmarkContext();

  // Disable kernel reuse during all benchmarks.
  // This is important since some benchmarks use FusionExecutorCache in order to
  // benchmark scheduling and segmentation. However, when benchmarking multiple
  // input shapes, this can lead to re-using suboptimal FusionKernelRuntimes, in
  // some extreme cases resulting in the use of a cached segmented runtime when
  // it would be possible to schedule the Fusion without segmentation.
  // See https://github.com/NVIDIA/Fuser/pull/563 for more info
  DisableOptionsGuard og;
  DisableOptionsGuard::getCurOptions().set(DisableOption::KernelReuse);

  if (batch_mode) {
    try {
      BatchCaseReporter reporter(std::move(batch_cases));
      const size_t matched = ::benchmark::RunSpecifiedBenchmarks(&reporter);
      if (matched != reporter.expected()) {
        throw std::runtime_error("batch filter matched the wrong registration count");
      }
    } catch (const std::exception& error) {
      std::cerr << "nvFuser batch mode: " << error.what() << '\n';
      ::benchmark::Shutdown();
      return 2;
    }
  } else {
    ::benchmark::RunSpecifiedBenchmarks();
  }

  ::benchmark::Shutdown();
  return 0;
}
