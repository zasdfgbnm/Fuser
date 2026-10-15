// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <csrc/exceptions.h>
#include <device_lower/lower2device.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <ir/utils.h>
#include <ops/all_ops.h>
#include <runtime/executor.h>
#include <runtime/fusion_kernel_runtime.h>
#include <scheduler/all_schedulers.h>

#include <benchmark/benchmark.h>

#include <cuda_runtime.h>

#include <benchmarks/cpp/utils.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

//------------------------------------------------------------------------------

static void setupRMSNorm(Fusion* fusion, DataType dtype) {
  NVF_ERROR(
      dtype == DataType::Float || dtype == DataType::Half ||
      dtype == DataType::BFloat16);

  FusionGuard fg(fusion);

  const float kEps = 1e-6;

  Val* eps_ptr = IrBuilder::create<Val>(kEps);

  // setup fusion
  auto input = makeContigTensor(2, dtype);
  auto weight = makeContigTensor(1, dtype);

  fusion->addInput(input);
  fusion->addInput(weight);

  if (dtype == DataType::Half) {
    input = castOp(DataType::Float, input);
    weight = castOp(DataType::Float, weight);
  }

  auto rms_norm_results = rms_norm(input, 1, weight, eps_ptr);

  auto output = rms_norm_results.output;

  if (dtype != DataType::Float) {
    output = castOp(dtype, output);
  }

  fusion->addOutput(output);
}

static void NvFuserScheduler_RMSNorm(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    DataType dtype) {
  NVF_ERROR(
      dtype == DataType::Float || dtype == DataType::Half ||
      dtype == DataType::BFloat16);

  std::vector<int64_t> input_shape{
      benchmark_state.range(0), benchmark_state.range(1)};

  // inputs
  at::manual_seed(0);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  at::Tensor input = at::randn(input_shape, options);
  at::Tensor weight = at::randn({input_shape[1]}, options);

  KernelArgumentHolder args({input, weight});

  // c24 must exercise one automatic InnerPersistent plan and produce the
  // weighted RMSNorm result before any timed iterations are collected.
  if (dtype == DataType::Float && input_shape[0] == 128 &&
      input_shape[1] == 8192) {
    auto outputs = executor_cache->runFusionWithInputs(args);
    auto* runtime = executor_cache->getMostRecentKernelRuntime();
    NVF_ERROR(runtime != nullptr && !runtime->isSegmented());
    const auto& heuristics = runtime->schedulerHeuristics()->heuristicsList();
    NVF_ERROR(heuristics.size() == 1 && heuristics.front() != nullptr);
    NVF_ERROR(
        heuristics.front()->scheduler_type == SchedulerType::InnerPersistent,
        "Expected one InnerPersistent scheduler for c24 RMSNorm");
    NVF_ERROR(outputs.size() == 1);

    auto x = input.to(at::kDouble);
    auto w = weight.to(at::kDouble);
    auto mean_square = at::pow(x, 2).mean(std::vector<int64_t>{1}, true);
    auto reference =
        (x * at::pow(mean_square + 1e-6, -0.5) * w).to(at::kFloat);
    NVF_ERROR(
        at::allclose(
            outputs[0].as<at::Tensor>(), reference, 1e-4, 1e-5, false),
        "c24 RMSNorm output differs from the double-precision ATen "
        "reference");
  }

  runBenchmarkIterations(benchmark_state, executor_cache, args);

  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) *
      (2 * input.numel() + weight.numel()) * dataTypeSizeByte(dtype));
}

//------------------------------------------------------------------------------
NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_RMSNorm_fp16,
    setupRMSNorm,
    NvFuserScheduler_RMSNorm,
    DataType::Half);

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_RMSNorm_fp32,
    setupRMSNorm,
    NvFuserScheduler_RMSNorm,
    DataType::Float);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp16)
    ->Apply(addCasesOneWave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp16)
    ->Apply(addCases16Wave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp32)
    ->Apply(addCasesOneWave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_fp32)
    ->Apply(addCases16Wave128To32K)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();

// TODO: Automatically disable/enable if bf16 is supported
// NVFUSER_BENCHMARK_DEFINE(
//     NvFuserScheduler_RMSNorm_bf16,
//     setupRMSNorm,
//     NvFuserScheduler_RMSNorm,
//     DataType::BFloat16);

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{16, 64}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{18, 56}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{22, 44}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();

// NVFUSER_BENCHMARK_RUN(NvFuserScheduler_RMSNorm_bf16)
//     ->RangeMultiplier(2)
//     ->Ranges({{24, 48}})
//     ->Unit(benchmark::kMicrosecond)
//     ->UseManualTime();
