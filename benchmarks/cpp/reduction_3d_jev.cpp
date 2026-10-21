// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <benchmarks/cpp/utils.h>

#include <runtime/fusion_kernel_runtime.h>
#include <tests/cpp/utils.h>

#include <ATen/ATen.h>
#include <benchmark/benchmark.h>

using namespace nvfuser;

namespace {

void setupReduction3D(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard guard(fusion);
  auto* input = makeContigTensor(3, dtype);
  fusion->addInput(input);
  fusion->addOutput(reductionOp(
      BinaryOpType::Add, {0, 2}, IrBuilder::create<Val>(0.0), input));
}

void runReduction3D(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  const int64_t outer = state.range(0);
  const int64_t iteration = state.range(1);
  const int64_t inner = state.range(2);
  NVF_ERROR(outer > 1 && iteration > 1 && inner > 1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // All partial sums stay below 2^24, making the integer-valued ATen
  // reference exact despite different legal reduction orders.
  auto input = at::arange(outer * iteration * inner, options)
                   .remainder(7)
                   .reshape({outer, iteration, inner});
  KernelArgumentHolder args({input});

  // Verify the compiled path and its result before entering the timed loop.
  auto outputs = executor_cache->runFusionWithInputs(args);
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_ERROR(runtime != nullptr && !runtime->isSegmented());
  const auto& heuristics = runtime->schedulerHeuristics()->heuristicsList();
  NVF_ERROR(heuristics.size() == 1 && heuristics.front() != nullptr);
  auto* params = heuristics.front().get();
  NVF_ERROR(
      params != nullptr && params->scheduler_type == SchedulerType::Reduction,
      "Expected one Reduction scheduler");
  NVF_ERROR(
      params->as<ReductionParams>()->schedule_3d,
      "Expected a genuine 3D Reduction plan");
  NVF_ERROR(outputs.size() == 1);
  auto reference = input.sum(std::vector<int64_t>{0, 2});
  NVF_ERROR(
      at::equal(outputs[0].as<at::Tensor>(), reference),
      "3D Reduction output differs from ATen");

  const int64_t io_bytes = runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * io_bytes);
}

} // namespace

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Reduction3D_fp32,
    setupReduction3D,
    runReduction3D,
    DataType::Float);

// Reuse shapes from the SM89 3D reduction numeric tests. They exercise
// distinct native warp/CTA, work-split, grid-wave, and outer-unroll choices.
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction3D_fp32)
    ->Args({4, 32, 128})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction3D_fp32)
    ->Args({8, 8, 128})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction3D_fp32)
    ->Args({256, 65, 4})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction3D_fp32)
    ->Args({2, 65, 8192})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction3D_fp32)
    ->Args({8, 32, 3})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
