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

void setupOddInnerReduction(Fusion* fusion, DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  FusionGuard guard(fusion);
  auto* input = makeContigTensor(2, dtype);
  fusion->addInput(input);
  fusion->addOutput(reductionOp(
      BinaryOpType::Add, {1}, IrBuilder::create<Val>(0.0), input));
}

void runOddInnerReduction(
    benchmark::State& state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(dtype == DataType::Float);
  const int64_t outputs = state.range(0);
  const int64_t reduction = state.range(1);
  NVF_ERROR(
      outputs > 1 && reduction > 1 && reduction % 2 == 1 &&
      reduction <= (1LL << 24) / 6);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  // Each exact integer sum is at most 6 * reduction, below 2^24 here.
  auto input = at::arange(outputs * reduction, options)
                   .remainder(7)
                   .reshape({outputs, reduction});
  KernelArgumentHolder args({input});

  // Verify the auto-scheduled path and result before the timed loop.
  auto outputs_tensor = executor_cache->runFusionWithInputs(args);
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_ERROR(runtime != nullptr && !runtime->isSegmented());
  const auto& heuristics = runtime->schedulerHeuristics()->heuristicsList();
  NVF_ERROR(heuristics.size() == 1 && heuristics.front() != nullptr);
  auto* params = heuristics.front().get();
  NVF_ERROR(
      params->scheduler_type == SchedulerType::Reduction,
      "Expected one Reduction scheduler");
  const auto* reduction_params = params->as<ReductionParams>();
  NVF_ERROR(
      reduction_params->fastest_dim && !reduction_params->schedule_3d,
      "Expected a 2D inner Reduction plan");
  NVF_ERROR(outputs_tensor.size() == 1);
  auto reference = input.sum(std::vector<int64_t>{1});
  NVF_ERROR(
      at::equal(outputs_tensor[0].as<at::Tensor>(), reference),
      "Odd inner Reduction output differs from ATen");

  const int64_t io_bytes = runBenchmarkIterations(state, executor_cache, args);
  state.SetBytesProcessed(
      static_cast<int64_t>(state.iterations()) * io_bytes);
}

} // namespace

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Reduction2D_OddInner_fp32,
    setupOddInnerReduction,
    runOddInnerReduction,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Reduction2D_OddInner_fp32)
    ->Args({1024, 1025})
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
