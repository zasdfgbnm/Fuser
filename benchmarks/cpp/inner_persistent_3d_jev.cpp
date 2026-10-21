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

#include <vector>

using namespace nvfuser;

namespace {

void setupInnerPersistent3D(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard guard(fusion);
  auto* input = makeContigTensor(3, dtype);
  fusion->addInput(input);
  auto* exponentials = exp(input);
  auto* sums = sum(exponentials, {0, 2});
  auto* sums_bcast = broadcast(sums, {true, false, true});
  fusion->addOutput(div(exponentials, sums_bcast));
}

void runInnerPersistent3D(
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
  // Bounded, reproducible values avoid exponential overflow. Every shape here
  // has fewer than 2^24 elements, so arange represents its indices exactly.
  auto input = at::arange(outer * iteration * inner, options)
                   .remainder(31)
                   .sub(15)
                   .div(10)
                   .reshape({outer, iteration, inner});
  KernelArgumentHolder args({input});

  // Check the actual FEC-selected scheduler and numerical result before time
  // is collected. A different scheduler is a fixture failure, not IP6 data.
  auto outputs = executor_cache->runFusionWithInputs(args);
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_ERROR(runtime != nullptr && !runtime->isSegmented());
  const auto& heuristics = runtime->schedulerHeuristics()->heuristicsList();
  NVF_ERROR(heuristics.size() == 1 && heuristics.front() != nullptr);
  auto* params = heuristics.front().get();
  NVF_ERROR(
      params->scheduler_type == SchedulerType::InnerPersistent,
      "Expected one InnerPersistent scheduler");
  const auto* reduction_params = params->as<ReductionParams>();
  NVF_ERROR(
      reduction_params->schedule_3d && reduction_params->persistent_kernel,
      "Expected a 3D persistent reduction plan");
  NVF_ERROR(outputs.size() == 1);
  auto exp_reference = at::exp(input.to(at::kDouble));
  auto reference =
      (exp_reference /
       exp_reference.sum(std::vector<int64_t>{0, 2}, true))
          .to(at::kFloat);
  NVF_ERROR(
      at::allclose(
          outputs[0].as<at::Tensor>(), reference, 1e-4, 1e-6, false),
      "3D InnerPersistent output differs from the double-precision ATen "
      "reference");

  const int64_t io_bytes = runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * io_bytes);
}

} // namespace

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_InnerPersistent3D_fp32,
    setupInnerPersistent3D,
    runInnerPersistent3D,
    DataType::Float);

// The SM89 3D persistent GTests force this scheduler at these shapes. The
// checks above require the benchmark's automatic scheduler to do the same.
NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InnerPersistent3D_fp32)
    ->Args({4, 256, 16})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InnerPersistent3D_fp32)
    ->Args({8, 8, 128})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_InnerPersistent3D_fp32)
    ->Args({4, 256, 5120})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
