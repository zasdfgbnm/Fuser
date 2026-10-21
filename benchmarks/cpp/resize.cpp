// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <benchmarks/cpp/utils.h>
#include <scheduler/resize_heuristic.h>
#include <tests/cpp/utils.h>

using namespace nvfuser;

namespace {

void setupResizePaddedTranspose(Fusion* fusion, int64_t pad_right) {
  FusionGuard guard(fusion);
  auto* input = makeContigConcreteTensor({5, 7, 64}, DataType::Float);
  input->setAllocationDomain(
      {input->axis(1), input->axis(0), input->axis(2)}, true);
  fusion->addInput(input);
  fusion->addOutput(
      pad(input, {fusion->zeroVal(), IrBuilder::create<Val>(pad_right)}));
}

void runResizePaddedTranspose(
    benchmark::State& benchmark_state,
    FusionExecutorCache* executor_cache,
    int64_t pad_right) {
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto input = at::arange(5 * 7 * 64, options)
                   .reshape({7, 5, 64})
                   .permute({1, 0, 2});
  NVF_CHECK(
      input.size(0) == 5 && input.size(1) == 7 && input.size(2) == 64 &&
          input.stride(0) == 64 && input.stride(1) == 320 &&
          input.stride(2) == 1,
      "Unexpected resize benchmark input layout");
  KernelArgumentHolder args = {input};

  // Compile and check the same Resize path used by the timed iterations.
  auto outputs = executor_cache->runFusionWithInputs(args);
  auto* runtime = executor_cache->getMostRecentKernelRuntime();
  NVF_CHECK(runtime != nullptr && !runtime->isSegmented(),
            "Resize benchmark was segmented or has no runtime");
  const auto& heuristics = runtime->schedulerHeuristics()->heuristicsList();
  NVF_CHECK(heuristics.size() == 1 && heuristics.front() != nullptr,
            "Resize benchmark expected one scheduler heuristic");
  auto* params = heuristics.front().get();
  NVF_CHECK(
      params != nullptr && params->scheduler_type == SchedulerType::Resize,
      "Resize benchmark used an unexpected scheduler");
  NVF_CHECK(outputs.size() == 1, "Resize benchmark expected one output");
  NVF_CHECK(
      at::pad(input, {0, pad_right}).equal(outputs[0].as<at::Tensor>()),
      "Resize benchmark output differs from ATen");

  const int64_t io_bytes =
      runBenchmarkIterations(benchmark_state, executor_cache, args);
  benchmark_state.SetBytesProcessed(
      int64_t(benchmark_state.iterations()) * io_bytes);
}

} // namespace

NVFUSER_BENCHMARK_DEFINE(
    NvFuserScheduler_Resize_PaddedTranspose_fp32,
    setupResizePaddedTranspose,
    runResizePaddedTranspose,
    8);

NVFUSER_BENCHMARK_RUN(NvFuserScheduler_Resize_PaddedTranspose_fp32)
    ->Unit(benchmark::kMicrosecond)
    ->UseManualTime();
