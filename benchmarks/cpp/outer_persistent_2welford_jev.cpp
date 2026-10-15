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

void setupOuterPersistentTwoWelford(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard guard(fusion);
  auto* input = makeContigTensor(2, dtype);
  fusion->addInput(input);

  // Match Gpu3Test.TemplateFunctionTypeMismatch: two Welford reductions
  // connected by a pointwise exponential and normalization.
  auto* copied = set(input);
  auto* doubled = add(copied, copied);
  auto first_stats = Welford(doubled, {0});
  auto* first_mean = broadcast(first_stats.avg, {true, false});
  auto* first_output = div(doubled, first_mean);
  auto* exponentials = exp(first_output);
  auto second_stats = Welford(exponentials, {0});
  auto* second_mean = broadcast(second_stats.avg, {true, false});
  auto* second_output = div(exponentials, second_mean);
  fusion->addOutput(first_output);
  fusion->addOutput(second_output);
}

void runOuterPersistentTwoWelford(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  const int64_t batch = state.range(0);
  const int64_t hidden = state.range(1);
  NVF_ERROR(batch > 1 && hidden > 1);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // Positive, bounded values keep both normalization means away from zero.
  auto input = at::arange(batch * hidden, options)
                   .remainder(31)
                   .add(1)
                   .reshape({batch, hidden});
  KernelArgumentHolder args({input});

  // An automatic, single OuterPersistent plan and correct outputs are
  // prerequisites for collecting OP2 measurements.
  auto outputs = executor_cache->runFusionWithInputs(args);
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_ERROR(runtime != nullptr && !runtime->isSegmented());
  const auto& heuristics = runtime->schedulerHeuristics()->heuristicsList();
  NVF_ERROR(heuristics.size() == 1 && heuristics.front() != nullptr);
  auto* params = heuristics.front().get();
  NVF_ERROR(
      params->scheduler_type == SchedulerType::OuterPersistent,
      "Expected one OuterPersistent scheduler");
  NVF_ERROR(outputs.size() == 2);

  auto doubled = input.to(at::kDouble) * 2.0;
  auto first_reference = doubled / doubled.mean(std::vector<int64_t>{0}, true);
  auto exponentials = first_reference.exp();
  auto second_reference =
      exponentials / exponentials.mean(std::vector<int64_t>{0}, true);
  // The reductions and exponential are computed in fp32 by the fusion;
  // double-precision references allow fp32 parallel-reduction roundoff.
  NVF_ERROR(
      at::allclose(
          outputs[0].as<at::Tensor>(),
          first_reference.to(at::kFloat),
          1e-4,
          1e-5,
          false),
      "First OuterPersistent output differs from ATen");
  NVF_ERROR(
      at::allclose(
          outputs[1].as<at::Tensor>(),
          second_reference.to(at::kFloat),
          1e-4,
          1e-5,
          false),
      "Second OuterPersistent output differs from ATen");

  const int64_t io_bytes = runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * io_bytes);
}

} // namespace

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_OuterPersistentTwoWelford_fp32,
    setupOuterPersistentTwoWelford,
    runOuterPersistentTwoWelford,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_OuterPersistentTwoWelford_fp32)
    ->Args({8192, 1024})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
