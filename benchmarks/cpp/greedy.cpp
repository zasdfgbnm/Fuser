// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <benchmarks/cpp/utils.h>

#include <fusion.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/fusion_kernel_runtime.h>

#include <ATen/ATen.h>
#include <benchmark/benchmark.h>

using namespace nvfuser;

namespace {

constexpr int64_t kRows = 8;
constexpr int64_t kColumns = 2048;
constexpr int64_t kScanCount = 4;

void setupGreedyFourScan(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard fg(fusion);
  auto* tv = makeContigConcreteTensor({kRows, kColumns}, dtype);
  fusion->addInput(tv);
  for (int64_t i = 0; i < kScanCount; ++i) {
    tv = scan(tv, -1, BinaryOpType::Add);
    fusion->addOutput(tv);
  }
}

void runGreedyFourScan(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  NVF_ERROR(state.range(0) == kRows && state.range(1) == kColumns);
  auto input = at::zeros(
      {kRows, kColumns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  // The fourth prefix stays below 2^24, so Float integer sums are exact.
  input.select(-1, 1600).fill_(1.0);
  KernelArgumentHolder args({input});

  // Compile and validate every output before entering the timed loop.
  auto outputs = executor_cache->runFusionWithInputs(args);
  NVF_ERROR(outputs.size() == kScanCount, "Expected four Scan outputs");
  auto reference = input;
  for (int64_t i = 0; i < kScanCount; ++i) {
    reference = reference.cumsum(-1);
    NVF_ERROR(
        at::equal(outputs[i].as<at::Tensor>(), reference),
        "Greedy Scan output ",
        i,
        " differs from ATen");
  }
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_ERROR(runtime != nullptr);
  NVF_ERROR(!runtime->isSegmented(), "Expected an unsegmented Greedy schedule");
  const auto& groups = runtime->fusionSegments()->groups();
  NVF_ERROR(
      groups.size() == 1 &&
          groups.front()->schedulerType() == SchedulerType::Greedy,
      "Expected one Greedy schedule for four Scans");

  runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      (kScanCount + 1) * kRows * kColumns * sizeof(float));
}

} // namespace

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Greedy_FourScan_fp32,
    setupGreedyFourScan,
    runGreedyFourScan,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Greedy_FourScan_fp32)
    ->Args({kRows, kColumns})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
