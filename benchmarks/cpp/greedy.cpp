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
#include <tests/cpp/utils.h>

#include <ATen/ATen.h>
#include <benchmark/benchmark.h>

using namespace nvfuser;

namespace {

constexpr int64_t kRows = 8;
constexpr int64_t kColumns = 2048;
constexpr int64_t kScanCount = 4;
constexpr int64_t kSingleScanRows = 10;

void assertSingleGreedySchedule(
    FusionExecutorCache* executor_cache,
    const char* benchmark_name) {
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_ERROR(runtime != nullptr, benchmark_name, ": missing kernel runtime");
  NVF_ERROR(
      !runtime->isSegmented(), benchmark_name, ": expected one kernel group");
  const auto& groups = runtime->fusionSegments()->groups();
  NVF_ERROR(
      groups.size() == 1 &&
          groups.front()->schedulerType() == SchedulerType::Greedy,
      benchmark_name,
      ": expected one Greedy schedule");
}

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
  assertSingleGreedySchedule(executor_cache, "Greedy FourScan");

  runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) *
      (kScanCount + 1) * kRows * kColumns * sizeof(float));
}

void setupGreedyScanAdd(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard fg(fusion);
  auto* input = makeContigConcreteTensor({kSingleScanRows, kColumns}, dtype);
  fusion->addInput(input);
  fusion->addOutput(add(scan(input, -1, BinaryOpType::Add), fusion->oneVal()));
}

void runGreedyScanAdd(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  NVF_ERROR(
      state.range(0) == kSingleScanRows && state.range(1) == kColumns);
  auto input = at::zeros(
      {kSingleScanRows, kColumns},
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  // Integer-valued prefixes remain exactly representable in fp32.
  input.select(-1, 1600).fill_(1.0);
  KernelArgumentHolder args({input});

  auto outputs = executor_cache->runFusionWithInputs(args);
  NVF_ERROR(outputs.size() == 1, "Expected one Scan+Add output");
  auto reference = input.cumsum(-1) + 1.0;
  NVF_ERROR(
      at::equal(outputs[0].as<at::Tensor>(), reference),
      "Greedy Scan+Add differs from ATen");
  assertSingleGreedySchedule(executor_cache, "Greedy ScanAdd");

  runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * 2 * kSingleScanRows *
      kColumns * sizeof(float));
}

void setupGreedyArgsortAdd(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard fg(fusion);
  auto* input = makeContigConcreteTensor({kRows, kColumns}, dtype);
  fusion->addInput(input);
  auto* indices = argsort(input, -1);
  fusion->addOutput(add(indices, fusion->oneVal(DataType::Int)));
}

void runGreedyArgsortAdd(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  NVF_ERROR(state.range(0) == kRows && state.range(1) == kColumns);
  auto columns = at::arange(
      kColumns,
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
  // 37 is coprime to 2048: each row has distinct, deterministic keys.
  auto input = at::remainder(columns * 37 + 19, kColumns)
                   .to(at::kFloat)
                   .repeat({kRows, 1});
  KernelArgumentHolder args({input});

  auto outputs = executor_cache->runFusionWithInputs(args);
  NVF_ERROR(outputs.size() == 1, "Expected one Argsort+Add output");
  auto reference = input.argsort(-1, false) + 1;
  NVF_ERROR(
      at::equal(outputs[0].as<at::Tensor>(), reference),
      "Greedy Argsort+Add differs from ATen");
  assertSingleGreedySchedule(executor_cache, "Greedy ArgsortAdd");

  runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * kRows * kColumns *
      (sizeof(float) + sizeof(int64_t)));
}

void setupGreedyScatterPermutation(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Int);
  FusionGuard fg(fusion);
  auto* indices = makeContigConcreteTensor({kColumns}, dtype);
  fusion->addInput(indices);
  auto* base = zeros({IrBuilder::create<Val>(kColumns)}, dtype);
  auto* source = arange(
      fusion->zeroVal(dtype),
      IrBuilder::create<Val>(kColumns, dtype),
      dtype);
  fusion->addOutput(scatter(base, 0, indices, source));
}

void runGreedyScatterPermutation(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Int);
  NVF_ERROR(state.range(0) == kColumns);
  auto positions = at::arange(
      kColumns,
      at::TensorOptions().dtype(at::kLong).device(at::kCUDA, 0));
  // index[j] = (37*j + 19) mod 2048 is a permutation of [0, 2048).
  auto indices = at::remainder(positions * 37 + 19, kColumns);
  KernelArgumentHolder args({indices});

  auto outputs = executor_cache->runFusionWithInputs(args);
  NVF_ERROR(outputs.size() == 1, "Expected one Scatter output");
  // 941 is the inverse of 37 modulo 2048. This checks each destination
  // independently of an ATen Scatter implementation.
  auto reference = at::remainder((positions - 19) * 941, kColumns);
  NVF_ERROR(
      at::equal(outputs[0].as<at::Tensor>(), reference),
      "Greedy Scatter permutation differs from the inverse permutation");
  assertSingleGreedySchedule(executor_cache, "Greedy ScatterPermutation");

  runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * 2 * kColumns *
      sizeof(int64_t));
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

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Greedy_ScanAdd_fp32,
    setupGreedyScanAdd,
    runGreedyScanAdd,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Greedy_ScanAdd_fp32)
    ->Args({kSingleScanRows, kColumns})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Greedy_ArgsortAdd_fp32,
    setupGreedyArgsortAdd,
    runGreedyArgsortAdd,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Greedy_ArgsortAdd_fp32)
    ->Args({kRows, kColumns})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Greedy_ScatterPermutation_i64,
    setupGreedyScatterPermutation,
    runGreedyScatterPermutation,
    DataType::Int);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Greedy_ScatterPermutation_i64)
    ->Arg(kColumns)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
