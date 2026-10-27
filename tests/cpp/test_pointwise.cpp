// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <unistd.h>

#include <ATen/cuda/CUDAContext.h>

#include "debug.h"
#include "fusion.h"
#include "ir/interface_nodes.h"
#include "options.h"
#include "ops/all_ops.h"
#include "optimization_pass.h"
#include "preseg_passes/mark_aliases_prepare.h"
#include "runtime/fusion_executor_cache.h"
#include "scheduler/compile_time_info.h"
#include "scheduler/compute_invocation_input.h"
#include "scheduler/jev_client.h"
#include "scheduler/native_diagnostic_jsonl.h"
#include "scheduler/pointwise_add_cta_continuation.h"
#include "scheduler/pointwise_backend_route.h"
#include "scheduler/pointwise_non_tma.h"
#include "scheduler/pointwise_tma.h"
#include "scheduler/pointwise_utils.h"
#include "scheduler/runtime_info.h"
#include "scheduler/tools/domain_map.h"
#include "scheduler/tools/inlining.h"
#include "tests/cpp/utils.h"
#include "type.h"
#include "validator_utils.h"

namespace nvfuser {

using PointwiseTest = NVFuserTest;

// Base class for parameterized pointwise tests using TEST_P
// Sets up IdModel configuration for parameterized tests
template <typename ParamType>
class PointwiseTestP : public NVFuserFixtureParamTest<ParamType> {
 protected:
  void SetUp() override {
    NVFuserFixtureParamTest<ParamType>::SetUp();
  }
};

namespace {

class ScopedTestEnv {
 public:
  ScopedTestEnv(std::string name, const char* value) : name_(std::move(name)) {
    if (const char* previous = std::getenv(name_.c_str())) {
      previous_ = previous;
    }
    NVF_CHECK(::setenv(name_.c_str(), value, 1) == 0);
  }

  ~ScopedTestEnv() {
    if (previous_.has_value()) {
      ::setenv(name_.c_str(), previous_->c_str(), 1);
    } else {
      ::unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::optional<std::string> previous_;
};

int64_t getVecSizeForPointwise(const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented());
  const PointwiseParams* params = runtime->schedulerHeuristics()
                                      ->heuristicsList()
                                      .at(0)
                                      ->as<PointwiseParams>();
  return params->vectorization_factor;
}

bool hasVectorizationCache(TensorView* tv) {
  NVF_CHECK(tv->isFusionInput());
  NVF_CHECK(tv->uses().size() == 1);
  auto set_expr = dynamic_cast<LoadStoreOp*>(tv->uses().at(0));
  NVF_CHECK(set_expr != nullptr && set_expr->opType() == LoadStoreOpType::Set);
  auto cached_input = set_expr->out()->as<TensorView>();
  NVF_CHECK(cached_input, "expects input to be cached");

  for (const auto* id : cached_input->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      return true;
    }
  }
  return false;
}

class DomainMapUnitTest : public scheduler_tools::DomainMap {
 public:
  DomainMapUnitTest(Fusion* fusion) : scheduler_tools::DomainMap(fusion) {}
  bool testTargetCoverage(TensorView* target_tv, TensorView* reference_tv)
      const {
    return areAllTargetIdsCoveredBy(target_tv, reference_tv);
  }
};

} // namespace

TEST_F(PointwiseTest, JevSemanticGraphEligibility) {
  constexpr std::string_view snapshot = "complete invocation snapshot";
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(1);
  auto* other = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addInput(other);
  auto* broadcasted = broadcast(input, {true, false});
  fusion.addOutput(add(broadcasted, other));

  // A normal broadcast may add a logical dimension without a hidden domain
  // transform. It must remain available to the strict semantic graph path.
  EXPECT_EQ(
      pointwise_utils::semanticPointwiseEligibilityReason(&fusion, snapshot),
      nullptr);
  EXPECT_STREQ(
      pointwise_utils::semanticPointwiseEligibilityReason(&fusion, {}),
      "semantic_graph_snapshot_unavailable");
  EXPECT_STREQ(
      pointwise_utils::semanticPointwiseEligibilityReason(
          &fusion, std::string(8193, 'x')),
      "semantic_graph_snapshot_too_large");
  EXPECT_EQ(
      pointwise_utils::semanticPointwiseEligibilityReason(
          &fusion, std::string(8193, 'x'), true),
      nullptr);
  EXPECT_EQ(
      pointwise_utils::semanticPointwiseEligibilityReason(
          &fusion, std::string(1024 * 1024, 'x'), true),
      nullptr);
  EXPECT_STREQ(
      pointwise_utils::semanticPointwiseEligibilityReason(
          &fusion, std::string(1024 * 1024 + 1, 'x'), true),
      "semantic_graph_snapshot_too_large");
}

TEST_F(PointwiseTest, JevIrMetadataV2CapturesPointwiseEligibilityFacts) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(1);
  auto* other = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addInput(other);
  auto* broadcasted = broadcast(input, {true, false});
  fusion.addOutput(add(broadcasted, other));

  const auto metadata =
      scheduler::compute_invocation_input::captureIrMetadata(&fusion);
  ASSERT_TRUE(metadata.available());
  EXPECT_THAT(
      metadata.text,
      testing::HasSubstr("\"schema\":\"nvfuser.compute_ir_meta.v2\""));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"memory_type\":\"global\""));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"dtype\":\"float\""));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"in_fusion_vals\":true"));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"has_device_mesh\":false"));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"is_cpu_scalar\":false"));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"domain_expr_count\":0"));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"is_iteration\":true"));
  EXPECT_THAT(metadata.text, testing::HasSubstr("\"is_broadcast\":true"));
  EXPECT_THAT(
      metadata.text,
      testing::HasSubstr(
          "\"kind\":\"broadcast\",\"op\":null,\"cache\":null"));
  EXPECT_THAT(
      metadata.text,
      testing::HasSubstr(
          "\"kind\":\"binary\",\"op\":\"add\",\"cache\":null"));
  Fusion clone(fusion);
  EXPECT_EQ(
      metadata.text,
      scheduler::compute_invocation_input::captureIrMetadata(&clone).text);
}

TEST_F(PointwiseTest, JevIrMetadataV2ExposesUnsupportedPointwiseFeatures) {
  using scheduler::compute_invocation_input::captureIrMetadata;
  {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto* input = makeContigTensor(1);
    fusion.addInput(input);
    auto* intermediate = add(input, input);
    intermediate->setMemoryType(MemoryType::Shared);
    fusion.addOutput(add(intermediate, input));
    const auto metadata = captureIrMetadata(&fusion);
    ASSERT_TRUE(metadata.available());
    EXPECT_THAT(
        metadata.text, testing::HasSubstr("\"memory_type\":\"shared\""));
    EXPECT_STREQ(
        pointwise_utils::semanticPointwiseEligibilityReason(&fusion, "snapshot"),
        "semantic_graph_memory_type");
  }
  {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto* input = makeContigTensor(1);
    fusion.addInput(input);
    fusion.addOutput(add(input, input));
    input->split(0, 4);
    const auto metadata = captureIrMetadata(&fusion);
    ASSERT_TRUE(metadata.available());
    EXPECT_THAT(metadata.text, testing::HasSubstr("\"domain_expr_count\":1"));
    EXPECT_STREQ(
        pointwise_utils::semanticPointwiseEligibilityReason(&fusion, "snapshot"),
        "semantic_graph_hidden_domain_transform");
  }
  {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto* input = makeContigTensor(1);
    fusion.addInput(input);
    fusion.addOutput(add(input, input));
    input->axis(0)->parallelize(ParallelType::TIDx);
    const auto metadata = captureIrMetadata(&fusion);
    ASSERT_TRUE(metadata.available());
    EXPECT_THAT(metadata.text, testing::HasSubstr("\"is_parallelized\":true"));
    EXPECT_STREQ(
        pointwise_utils::semanticPointwiseEligibilityReason(&fusion, "snapshot"),
        "semantic_graph_unsupported_axis");
  }
  {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto* input = makeContigTensor(1);
    fusion.addInput(input);
    fusion.addOutput(exp(input));
    const auto metadata = captureIrMetadata(&fusion);
    ASSERT_TRUE(metadata.available());
    EXPECT_THAT(
        metadata.text,
        testing::HasSubstr(
            "\"kind\":\"unary\",\"op\":\"exp\",\"cache\":null"));
    EXPECT_STREQ(
        pointwise_utils::semanticPointwiseEligibilityReason(&fusion, "snapshot"),
        "semantic_graph_unsupported_operator");
  }
}

TEST_F(PointwiseTest, JevSemanticGraphRejectsHiddenTransformsAndOps) {
  constexpr std::string_view snapshot = "complete invocation snapshot";
  {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto* input = makeContigTensor(1);
    fusion.addInput(input);
    fusion.addOutput(add(input, input));
    input->split(0, 4);
    EXPECT_NE(
        pointwise_utils::semanticPointwiseEligibilityReason(&fusion, snapshot),
        nullptr);
  }
  {
    Fusion fusion;
    FusionGuard fg(&fusion);
    auto* input = makeContigTensor(1);
    fusion.addInput(input);
    fusion.addOutput(exp(input));
    EXPECT_STREQ(
        pointwise_utils::semanticPointwiseEligibilityReason(&fusion, snapshot),
        "semantic_graph_unsupported_operator");
  }
}

TEST_F(PointwiseTest, JevSemanticGraphAcceptsNoncontiguousInput) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input =
      TensorViewBuilder().ndims(2).contiguity({false, true}).build();
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  EXPECT_EQ(
      pointwise_utils::semanticPointwiseEligibilityReason(
          &fusion, "complete invocation snapshot"),
      nullptr);
}

TEST_F(PointwiseTest, JevSemanticV3UnsupportedGraphFallsBackBeforeChoice_CUDA) {
  char log_path[] = "/tmp/nvfuser-pw3-unsupported-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
      ::unlink((std::string(path) + ".outcome.jsonl").c_str());
    }
  } file_cleanup{log_path};
  ScopedTestEnv protocol("NVFUSER_JEV_PROTOCOL", "semantic-v3");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);
  ScopedTestEnv invalid_port("NVFUSER_JEV_PORT", "0");

  Fusion unsupported;
  FusionGuard fg(&unsupported);
  auto* input = makeContigTensor(1);
  unsupported.addInput(input);
  unsupported.addOutput(exp(input));
  const auto tensor = at::empty(
      {16388}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&unsupported, {tensor});
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);
  const auto compute = [&](const char* configured_mode) {
    ScopedTestEnv mode("NVFUSER_JEV_MODE", configured_mode);
    return scheduler->computeHeuristics(&unsupported, runtime_info);
  };
  const auto native = compute("off");
  const auto recorded = compute("record");
  const auto remote = compute("remote");
  ASSERT_NE(native, nullptr);
  ASSERT_NE(recorded, nullptr);
  ASSERT_NE(remote, nullptr);
  EXPECT_EQ(native->toString(), recorded->toString());
  EXPECT_EQ(native->toString(), remote->toString());

  std::ifstream log_file(log_path);
  std::stringstream records;
  records << log_file.rdbuf();
  EXPECT_THAT(
      records.str(), ::testing::HasSubstr("\"kind\":\"context_fallback\""));
  EXPECT_THAT(
      records.str(),
      ::testing::HasSubstr(
          "\"reason\":\"semantic_graph_unsupported_operator\""));
  EXPECT_THAT(
      records.str(), ::testing::Not(::testing::HasSubstr("\"request\":")));

  Fusion supported;
  FusionGuard supported_guard(&supported);
  auto* supported_input = makeContigTensor(1);
  supported.addInput(supported_input);
  supported.addOutput(add(supported_input, supported_input));
  SchedulerRuntimeInfo supported_info(&supported, {tensor});
  {
    ScopedTestEnv mode("NVFUSER_JEV_MODE", "remote");
    try {
      (void)scheduler->computeHeuristics(&supported, supported_info);
      FAIL() << "Supported semantic-v3 graph bypassed the remote Choice";
    } catch (const std::runtime_error& error) {
      EXPECT_THAT(
          error.what(),
          ::testing::HasSubstr("NVFUSER_JEV_PORT must be an integer"));
    }
  }
}

// A non-CTA-multiple extent exercises the 1D tail for every offered warp
// count. The alternate numeric harness runs this with native x128 and forced
// x32/x64 choices through the same scheduler and checks the ATen result.
TEST_F(PointwiseTest, JevAlternateOneDimensionalCta_CUDA) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  TensorView* input = makeContigTensor(1);
  fusion.addInput(input);
  TensorView* output = add(input, input);
  fusion.addOutput(output);

  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor tensor = at::randn({16388}, options);
  auto result = scheduleAndRun(&fusion, SchedulerType::PointWise, {tensor});
  const auto* params = result.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(params->break_point, 0);
  ASSERT_EQ(params->vectorization_factor, 4);
  testValidate(
      &fusion,
      result.outputs,
      {tensor},
      {tensor + tensor},
      __LINE__,
      __FILE__);
}

TEST_F(PointwiseTest, VectorizeStrideContiguity2D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(2).contiguity({false, true}).build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {18, 2}, {32, 4}};

  for (auto [size, vec] : size_and_vec) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size}, options).narrow(1, 0, 16);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeStrideContiguity3D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 =
      TensorViewBuilder().ndims(3).contiguity({false, true, true}).build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  std::vector<std::pair<int, int>> size_and_vec{{17, 1}, {10, 2}, {16, 4}};

  for (auto [size, vec] : size_and_vec) {
    auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
    at::Tensor t0 = at::randn({1000000, size, 3}, options).narrow(1, 0, 8);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeStrideContiguity5D) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int>> sizes_and_vec{
      {9, 17, 1}, {9, 10, 2}, {9, 16, 4}};

  for (auto tup : sizes_and_vec) {
    auto size1 = std::get<0>(tup);
    auto size2 = std::get<1>(tup);
    auto vec = std::get<2>(tup);
    at::Tensor t0 = at::randn({4, size1, 12345, size2, 3}, options)
                        .narrow(1, 0, 8)
                        .narrow(3, 0, 4);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});

    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);

    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

// Test that vectorization is properly computed when base pointer is not aligned
// at 16 bytes. This can happen if a tensor is sliced then passed as input.
// See https://github.com/NVIDIA/Fuser/pull/2118
TEST_F(PointwiseTest, VectorizeStrideMisalignedBase) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int, int, int>> sizes_strides_align_and_vec{
      {4, 4, 4, 4, 4},
      {4, 4, 4, 2, 2},
      {4, 4, 4, 1, 1},
      {4, 4, 2, 4, 2},
      {4, 4, 2, 1, 1},
      {4, 2, 4, 4, 2},
      {4, 2, 4, 1, 1},
      {2, 4, 4, 4, 2},
      {2, 4, 4, 1, 1},
      {2, 2, 2, 4, 2},
      {2, 2, 2, 1, 1}};

  for (auto tup : sizes_strides_align_and_vec) {
    auto size = std::get<0>(tup);
    auto stride1 = std::get<1>(tup);
    auto stride2 = std::get<2>(tup);
    auto align = std::get<3>(tup);
    auto vec = std::get<4>(tup);
    std::vector<int64_t> shape = {4, 4, 12345, size, 3};
    std::vector<int64_t> stride = {
        stride1, (int64_t)stride2 * 12345, (int64_t)stride2, 3, 1};
    // Create a strided input that is misaligned by "align" elements
    //  First, find required size of align=0 tensor. Allocate this much plus
    //  align elements. Then slice and view as aligned tensor.
    int64_t alloc_size = 1l;
    for (auto i : arange(shape.size())) {
      alloc_size += (shape.at(i) - 1) * stride.at(i);
    }
    alloc_size += align;
    at::Tensor flat = at::randn({alloc_size}, options);
    at::Tensor t0 = flat.as_strided(shape, stride, /*storage_offset=*/align);
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});
    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);
    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeStrideContiguitySelfOverlapping) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(5)
                        .contiguity({false, true, false, true, true})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  std::vector<std::tuple<int, int, int, int>> sizes_strides_and_vec{
      {4, 4, 4, 4},
      {4, 4, 2, 2},
      {4, 2, 4, 2},
      {2, 4, 4, 2},
      {4, 4, 1, 1},
      {4, 1, 4, 1},
      {1, 4, 4, 1},
      {2, 2, 2, 2},
      {2, 2, 1, 1},
      {2, 1, 2, 1},
      {1, 2, 2, 1}};

  for (auto tup : sizes_strides_and_vec) {
    auto size = std::get<0>(tup);
    auto stride1 = std::get<1>(tup);
    auto stride2 = std::get<2>(tup);
    auto vec = std::get<3>(tup);
    std::vector<int64_t> shape = {4, 4, 12345, size, 3};
    std::vector<int64_t> stride = {
        stride1, (int64_t)stride2 * 12345, (int64_t)stride2, 3, 1};
    at::Tensor t0 = at::empty_strided(shape, stride, options);
    t0.random_();
    auto cg_outputs = executor_cache.runFusionWithInputs({t0});
    EXPECT_EQ(getVecSizeForPointwise(executor_cache), vec);
    testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
  }
}

TEST_F(PointwiseTest, VectorizeAllocationDomain) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .strideOrder({2, 0, 1})
                        .build();
  fusion->addInput(tv0);
  auto tv1 = add(tv0, IrBuilder::create<Val>(1.0, DataType::Float));
  tv1->setAllocationDomain({tv1->axis(0), tv1->axis(2), tv1->axis(1)}, true);
  fusion->addOutput(tv1);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 =
      at::empty_strided({1024, 128, 25}, {128 * 25, 1, 128}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0});
  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
  testValidate(fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

// All inputs & outputs share the same allocation domain permutation from root
// domain, but intermediate tv2 isn't specified a stride order. There's also a
// broadcast IterDomain on tv1, which is tricky for vectorization analysis to
// figure out which axes should be excluded from the computation of
// vectorization factor.
TEST_F(PointwiseTest, Issue1567VectorizeAllocationDomain) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .strideOrder({2, 0, 1})
                        .build();
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .shape({1, -1, 1})
                        .contiguity({std::nullopt, std::nullopt, true})
                        .strideOrder({2, 0, 1})
                        .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = add(tv2, IrBuilder::create<Val>(1.0, DataType::Float));
  tv3->setAllocationDomain({tv3->axis(0), tv3->axis(2), tv3->axis(1)}, true);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 =
      at::empty_strided({1024, 128, 25}, {128 * 25, 1, 128}, options);
  at::Tensor t1 = at::empty_strided({1, 128, 1}, {128, 1, 128}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 4);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Issue1567VectorizationFactorAnalysisCase0) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, std::nullopt})
                        .shape({-1, -1, 1})
                        .build();
  TensorView* tv1 =
      TensorViewBuilder().ndims(3).contiguity({true, true, true}).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 2, 1}, options);
  at::Tensor t1 = at::randn({1024, 2, 512}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results =
      scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1}, false);
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 4);
  EXPECT_FALSE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Issue1567VectorizationFactorAnalysisCase1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, std::nullopt, true})
                        .shape({-1, 1, -1})
                        .build();
  TensorView* tv1 =
      TensorViewBuilder().ndims(3).contiguity({true, true, true}).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 1, 2}, options);
  at::Tensor t1 = at::randn({1024, 512, 2}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Issue1567VectorizationFactorAnalysisCase2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, std::nullopt, true})
                        .shape({-1, 1, -1})
                        .build();
  TensorView* tv1 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({true, true, true})
                        .strideOrder({1, 2, 0})
                        .build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = transpose(tv2, 0, 1);
  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1024, 1, 2}, options);
  at::Tensor t1 = at::empty_strided({1024, 512, 2}, {2, 2048, 1}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 4);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, VectorizeIssue1567VectorizationFactorAnalysisCase3) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  TensorView* tv0 = TensorViewBuilder()
                        .ndims(3)
                        .contiguity({std::nullopt, true, true})
                        .shape({1, -1, -1})
                        .build();
  TensorView* tv1 =
      TensorViewBuilder().ndims(3).contiguity({true, true, true}).build();
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = transpose(tv2, 0, 1);
  fusion->addOutput(tv3);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({1, 1024, 2}, options);
  at::Tensor t1 = at::randn({512, 1024, 2}, options);

  // NOTE force pointwise scheduler here just for testing purpose
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();

  EXPECT_EQ(pparams->vectorization_factor, 2);
  EXPECT_TRUE(hasVectorizationCache(tv0));
  EXPECT_TRUE(hasVectorizationCache(tv1));

  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

namespace {
Fusion createPointwiseFusion(bool shard, int sharded_dim = -1) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  // Sharded fusion needs to add an additional sharded axis.
  TensorView* tv0 = makeContigTensor(shard ? 4 : 3);
  TensorView* tv1 = makeContigTensor(2);
  auto tv2 = add(tv0, tv0);
  std::vector<bool> bcast_mask;
  if (shard) {
    bcast_mask = {false, true, false, false};
    bcast_mask[sharded_dim] = true;
  } else {
    bcast_mask = {true, false, false};
  }
  TensorView* tv3 = broadcast(tv1, bcast_mask);
  TensorView* tv4 = add(tv2, tv3);
  fusion.addInput(tv0);
  fusion.addInput(tv1);
  fusion.addOutput(tv4);

  if (shard) {
    DeviceMesh mesh = DeviceMesh::createForNumDevices(4);
    for (TensorView* tv : {tv0, tv2, tv3, tv4}) {
      tv->setDeviceMesh(mesh);
      tv->axis(sharded_dim)->parallelize(ParallelType::DIDx);
    }
    tv1->setDeviceMesh(mesh);
  }
  return fusion;
}
} // namespace

// Check that (1) a sharded pointwise fusion returns the same
// pointwise scheduling parameters as its equivalent
// unsharded fusion and (2) the output is correct.
TEST_F(PointwiseTest, ShardedPointwise) {
  int64_t sharded_dim = 0;
  std::vector<std::vector<int64_t>> input_sizes = {
      {16, 8, 48},
      {2, 512, 4096},
      {2048, 512, 16},
      {65536, 512, 16},
      {512, 3, 65536},
  };
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);

  for (auto input_size : input_sizes) {
    at::Tensor t0 = at::randn(input_size, options);
    at::Tensor t1 = at::randn({input_size[1], input_size[2]}, options);

    KernelArgumentHolder sharded_inputs = {t0.unsqueeze(sharded_dim), t1};

    auto pwise_scheduler =
        SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);

    Fusion sharded_fusion = createPointwiseFusion(true, sharded_dim);
    SchedulerRuntimeInfo sharded_runtime_info(&sharded_fusion, sharded_inputs);
    auto sharded_params = pwise_scheduler->computeHeuristics(
        &sharded_fusion, sharded_runtime_info);
    auto sharded_pparams = sharded_params->as<PointwiseParams>();

    Fusion unsharded_fusion = createPointwiseFusion(false);
    SchedulerRuntimeInfo unsharded_runtime_info(&unsharded_fusion, {t0, t1});
    auto unsharded_params = pwise_scheduler->computeHeuristics(
        &unsharded_fusion, unsharded_runtime_info);
    auto unsharded_pparams = unsharded_params->as<PointwiseParams>();

    // Note: occasionally one of the compile parameter index types is int64_t
    // instead of int which causes PointwiseParams::sameAs to return false,
    // despite the pointwise specific parameters being identical, so we just
    // explicitly check pointwise schedule params.
    EXPECT_EQ(sharded_pparams->break_point, unsharded_pparams->break_point);
    EXPECT_EQ(sharded_pparams->split_block, unsharded_pparams->split_block);
    EXPECT_EQ(
        sharded_pparams->split_grid_y_dim, unsharded_pparams->split_grid_y_dim);
    EXPECT_EQ(
        sharded_pparams->vectorization_factor,
        unsharded_pparams->vectorization_factor);
    EXPECT_EQ(
        sharded_pparams->flip_grid_binding,
        unsharded_pparams->flip_grid_binding);

    pwise_scheduler->schedule(&sharded_fusion, sharded_params.get());
    KernelExecutor ke;
    ke.compile(&sharded_fusion, sharded_inputs, sharded_params->lparams);
    auto cg_outputs = ke.run(sharded_inputs, {}, sharded_params->lparams);
    testValidate(
        &sharded_fusion, cg_outputs, sharded_inputs, __LINE__, __FILE__);
  }
}

// Repro of issue #657
TEST_F(PointwiseTest, VectorizeWithBroadcastAndReshape1) {
  OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Sizes don't matter as long as they are large enough to trigger
  // vectorization
  std::vector<int64_t> shape1{1024, 1024};
  std::vector<int64_t> shape2{1024, 1024, 4};
  std::vector<int64_t> shape3{1024 * 1024 * 4};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion->addInput(tv0);

  auto tv1 = makeContigConcreteTensor(shape2);
  fusion->addInput(tv1);

  auto tv2 = broadcast(tv0, {false, false, true});
  fusion->addOutput(tv2);

  auto tv3 = add(tv1, tv2);
  auto tv4 = reshape(tv3, shape2, shape3);
  fusion->addOutput(tv4);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape2, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
}

// Repro of issue #657
TEST_F(PointwiseTest, VectorizeWithBroadcastAndReshape2) {
  OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // Sizes don't matter as long as they are large enough to trigger
  // vectorization
  std::vector<int64_t> shape1{1024, 1024};
  std::vector<int64_t> shape2{1024, 1024, 4};
  std::vector<int64_t> shape3{1024 * 1024 * 4};

  auto tv0 = makeContigConcreteTensor(shape1);
  fusion->addInput(tv0);

  auto tv1 = makeContigConcreteTensor(shape1);
  fusion->addInput(tv1);

  auto tv2 = makeContigConcreteTensor(shape2);
  fusion->addInput(tv2);

  auto tv3 = broadcast(tv0, {false, false, true});
  fusion->addOutput(tv3);

  auto tv4 = add(tv3, tv2);

  auto tv5 = broadcast(tv1, {false, false, true});

  auto tv6 = add(tv4, tv5);

  auto tv7 = reshape(tv6, shape2, shape3);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn(shape1, options);
  auto t1 = at::randn(shape1, options);
  auto t2 = at::randn(shape2, options);

  FusionExecutorCache executor_cache(std::move(fusion));
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1, t2});

  EXPECT_EQ(getVecSizeForPointwise(executor_cache), 4);
}

TEST_F(PointwiseTest, VectorizeWithExpandedBroadcast) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  constexpr int64_t kTensorSize = 65536;
  TensorView* in = TensorViewBuilder()
                       .dtype(DataType::Half)
                       .shape({2, kTensorSize})
                       .expanded({true, false})
                       .build();
  in->setAllocationDomain({in->axis(1), in->axis(0)}, true);
  TensorView* out = add(in, in);
  fusion->addInput(in);
  fusion->addOutput(out);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto in_tensor =
      at::randn({kTensorSize}, options).as_strided({2, kTensorSize}, {0, 1});

  FusionExecutorCache executor_cache(std::move(fusion));
  auto out_tensors = executor_cache.runFusionWithInputs({in_tensor});
  testValidate(
      executor_cache.fusion(), out_tensors, {in_tensor}, __LINE__, __FILE__);

  EXPECT_GT(getVecSizeForPointwise(executor_cache), 1);
}

using VectUnrollFactors = std::tuple<int64_t, int64_t, int64_t>;
using PointwiseParamsTest = NVFuserFixtureParamTest<VectUnrollFactors>;
TEST_P(PointwiseParamsTest, UnrollOnTopOfVectorize) {
  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv0, tv2);
  fusion->addOutput(tv3);

  int dim0 = 1024;
  int dim1 = 2048;
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim1}, options);

  // Generate heuristics
  SchedulerRuntimeInfo runtime_info(fusion.get(), {t0, t1});
  auto scheduler_instance =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);
  auto heuristic_params =
      scheduler_instance->computeHeuristics(fusion.get(), runtime_info);
  auto pparams = heuristic_params->as<PointwiseParams>();

  // Modify heuristics to enforce unroll on top of vectorization

  // Set unroll factors from test parameters
  auto [vect_factor, unroll_inner, unroll_outer] = GetParam();
  pparams->unroll_factor_inner = unroll_inner;
  pparams->unroll_factor_outer = unroll_outer;
  pparams->vectorization_factor = vect_factor;

  // Schedule, compile, run, validate
  scheduler_instance->schedule(fusion.get(), pparams);
  KernelExecutor ke;
  ke.compile(fusion.get(), {t0, t1}, pparams->lparams);
  auto cg_outputs = ke.run({t0, t1}, {}, pparams->lparams);
  const auto& lparams = ke.lastLaunchParams();
  ASSERT_EQ(lparams.gdimy(), dim0 / unroll_outer);
  ASSERT_EQ(
      lparams.gdimx(), dim1 / vect_factor / lparams.bdimx() / unroll_inner);
  testValidate(fusion.get(), cg_outputs, {t0, t1}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    PointwiseParamsTest,
    ::testing::Combine(
        testing::Values(1, 4), // vectorization factors
        testing::Values(1, 2), // inner unroll factors
        testing::Values(1, 2) // outer unroll factors
        ),
    [](const testing::TestParamInfo<VectUnrollFactors>& info) -> std::string {
      std::stringstream ss;
      ss << "vect_" << std::get<0>(info.param);
      ss << "_inner_unroll_" << std::get<1>(info.param);
      ss << "_outer_unroll_" << std::get<2>(info.param);
      return sanitizeTestName(ss.str());
    });

namespace {
int64_t getUnrollFactor(int64_t n_inputs_factor, int64_t computation_factor) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t required_bits_per_thread =
      scheduler_utils::getRequiredBitsInFlight() /
      (int64_t)dev_prop->maxThreadsPerMultiProcessor;
  constexpr int64_t vect_bits = 128L;
  int64_t unroll_factor = std::max(1L, required_bits_per_thread / vect_bits);
  if (unroll_factor > 1) {
    unroll_factor *= computation_factor;
    unroll_factor /= n_inputs_factor;
  }
  return unroll_factor;
}

} // namespace

// Test pointwise heuristics.
// current heuristics does fully unroll when have more than 8 waves
// of blocks. For device with high bandwidh, unroll factor is 2 when
// there is only one input tensor, scaled up by computation factor and
// scaled down by number of input tensors.
TEST_F(PointwiseTest, Heuristicst1Compute1Unroll2) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t vect = 4;
  int64_t threads = 128;
  int64_t unroll = getUnrollFactor(1, 1);
  int64_t dim0 = dev_prop->multiProcessorCount * 8 *
      dev_prop->maxThreadsPerMultiProcessor / threads;
  int64_t dim1 = vect * threads * unroll;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);

  auto cg_results =
      scheduleAndRun(fusion.get(), SchedulerType::PointWise, {t0});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(pparams->vectorization_factor, vect);
  ASSERT_EQ(pparams->unroll_factor_inner, unroll);
  testValidate(fusion.get(), cg_results.outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, Heuristicst1Compute2Unroll4) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t vect = 4;
  int64_t threads = 128;
  int64_t unroll = getUnrollFactor(1, 2);
  int64_t dim0 = dev_prop->multiProcessorCount * 8 *
      dev_prop->maxThreadsPerMultiProcessor / threads;
  int64_t dim1 = vect * threads * unroll;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  // outer bcast tv is not counted
  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(1);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv1, {true, false});
  auto tv3 = add(tv0, tv2);
  auto tv4 = exp(tv3);
  auto tv5 = reciprocal(tv4);
  fusion->addOutput(tv5);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim1}, options);

  auto cg_results =
      scheduleAndRun(fusion.get(), SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(pparams->vectorization_factor, vect);
  ASSERT_EQ(pparams->unroll_factor_outer, unroll);
  testValidate(fusion.get(), cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, HeuristicsInput2Compute4Unroll4) {
  auto dev_prop = at::cuda::getCurrentDeviceProperties();
  int64_t vect = 4;
  int64_t threads = 128;
  int64_t unroll = getUnrollFactor(2, 4);
  int64_t dim0 = dev_prop->multiProcessorCount * 8 *
      dev_prop->maxThreadsPerMultiProcessor / threads;
  int64_t dim1 = vect * threads * unroll;

  auto fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());

  auto tv0 = makeContigTensor(2);
  auto tv1 = makeContigTensor(2);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  auto tv3 = tanh(tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0, dim1}, options);
  auto cg_results =
      scheduleAndRun(fusion.get(), SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  ASSERT_EQ(pparams->vectorization_factor, vect);
  ASSERT_EQ(pparams->unroll_factor_inner, unroll);
  testValidate(fusion.get(), cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, VectorizePadLoweringPermuted) {
  // Pointwise scheduler applies permutation to restore contiguous memory access
  // on reference TV. Vectorization validation requires vectorized operations to
  // preserve the allocation domain of their inputs. This test checks that PadOp
  // propagates the allocation domain properly.
  auto fusion_ptr = std::make_unique<Fusion>();
  auto& fusion = *fusion_ptr;
  FusionGuard fg(fusion_ptr.get());

  // input is permuted
  auto tv0 = TensorViewBuilder()
                 .shape({1024, 1024})
                 .dtype(DataType::Float)
                 .contiguity(true)
                 .strideOrder({0, 1})
                 .build();
  fusion.addInput(tv0);
  auto tv1 = pad(tv0, {IrBuilder::create<Val>(4L), IrBuilder::create<Val>(4L)});
  auto tv2 = relu(tv1);
  fusion.addOutput(tv2);
  // output is permuted
  tv2->setAllocationDomain({tv2->axis(1), tv2->axis(0)}, true);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 =
      at::randn({1024 * 1024}, options).as_strided({1024, 1024}, {1, 1024});

  auto cg_outputs =
      scheduleAndRun(&fusion, SchedulerType::PointWise, {t0}).outputs;
  // check that we vectorize 4
  bool found_vectorize = false;
  for (auto id : fusion.outputs().at(0)->as<TensorView>()->getLoopDomain()) {
    if (id->getParallelType() == ParallelType::Vectorize) {
      EXPECT_EQ(id->extent()->evaluate(), 4);
      found_vectorize = true;
      break;
    }
  }
  EXPECT_TRUE(found_vectorize);
  testValidate(&fusion, cg_outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapTestEg0) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i0, i1}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i0, i1}
  auto tv1 = relu(tv0);
  fusion->addOutput(tv1);
  // tv2 {i0, b2, i1}
  auto tv2 = broadcast(tv1, {false, true, false});
  // tv3 {i0, b3{1 ex 4}, i1}
  auto tv3 = expand(
      tv2,
      {tv2->axis(0)->extent(),
       IrBuilder::create<Val>(4),
       tv2->axis(2)->extent()});
  // NOTE hat currently expand doesn't introduce an iter domain operation, so
  // we don't see that i4 is produced by realizing the expanded extent of b3{1
  // ex 4} tv4 {i0, i4*i1}
  auto tv4 = reshape(tv3, {2, 4, 3}, {2, 12});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // tv4 is not covered by tv1, because the expanded ID i4 participates in
  // transformation
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv1));

  // tv3 is not covered by tv1, because the missing ID b3{1 ex 4} is concretized
  // as i4, which is not mapped on tv1
  EXPECT_FALSE(domain_map.testTargetCoverage(tv3, tv1));

  // tv1 is covered by tv4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv1, tv4));

  // tv1 is not a valid reference
  EXPECT_FALSE(domain_map.isValidReference(tv1));

  // tv4 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 7}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0});
  testValidate(fusion, cg_results.outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapTestEg1) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i0, i1}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i2, i0, i1}
  TensorView* tv1 = makeContigTensor(3);
  fusion->addInput(tv1);
  // tv2 {i0*i1}
  auto tv2 = reshape(tv0, {2, 4}, {8});
  fusion->addOutput(tv2);

  // tv3 {b3, i0, i1}
  auto tv3 = broadcast(tv0, {true, false, false});
  // tv4 {i2, i0, i1}
  auto tv4 = add(tv1, tv3);
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // tv4 is not covered by tv2, because it misses i2
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv2));

  // tv2 is covered by tv4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv2, tv4));

  // tv2 is not a valid reference
  EXPECT_FALSE(domain_map.isValidReference(tv2));

  // tv4 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 4}, options);
  at::Tensor t1 = at::randn({3, 2, 4}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapTestEg2) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i0, i1}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i0, i1}
  auto tv1 = relu(tv0);
  fusion->addOutput(tv1);
  // tv2 {i0, b2, i1}
  auto tv2 = broadcast(tv1, {false, true, false});
  // tv3 {i0, b3{1 ex 4}, i1}
  auto tv3 = expand(
      tv2,
      {tv2->axis(0)->extent(),
       IrBuilder::create<Val>(4),
       tv2->axis(2)->extent()});
  fusion->addOutput(tv3);

  DomainMapUnitTest domain_map(fusion);
  // tv3 is covered by tv1, because the missing ID b3{1 ex 4} is broadcast and
  // doesn't get resolved to a concrete broadcast ID.
  EXPECT_TRUE(domain_map.testTargetCoverage(tv3, tv1));

  // tv1 is covered by tv4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv1, tv3));

  // tv1 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv1));

  // tv3 is a valid reference
  EXPECT_TRUE(domain_map.isValidReference(tv3));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({4, 7}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0});
  testValidate(fusion, cg_results.outputs, {t0}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapFactory) {
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv1 {i1}
  TensorView* tv0 = makeContigTensor(1);
  fusion->addInput(tv0);
  // tv1 {i0, i1}
  TensorView* tv1 = makeContigTensor(2);
  fusion->addInput(tv1);

  // tv2 {b2, b3, i1}
  auto tv2 = broadcast(tv0, {true, true, false});
  // NOTE tv1 will be broadcasted to {b2, i0, i1} before the add.
  // tv3 {b2, i0, i1}
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto size_val = IrBuilder::create<Val>(4.0, DataType::Int);
  auto one_val = IrBuilder::create<Val>(1, DataType::Int);
  // factory method creates an iter domain out of thin air
  // tv4 {i4{4}, b4, i1}
  auto tv4 = ones({size_val, one_val, tv0->axis(0)->extent()}, DataType::Float);
  // tv5 {i4{4}, i0, i1}
  auto tv5 = mul(tv2, tv4);
  fusion->addOutput(tv5);

  DomainMapUnitTest domain_map(fusion);

  // tv4 is not covered by tv3, because it's missing i4{4}
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv3));
  // tv1 is not covered by tv4, since it's missing i0
  EXPECT_FALSE(domain_map.testTargetCoverage(tv1, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv3));
  // tv5 has the same IDs as tv4, and is not a valid reference.
  EXPECT_FALSE(domain_map.isValidReference(tv5));

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({25}, {1}, options);
  at::Tensor t1 = at::empty_strided({7, 25}, {25, 1}, options);
  auto cg_outputs = executor_cache.runFusionWithInputs({t0, t1});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  SegmentedFusion* segmented_fusion = runtime->fusionSegments();
  // This fusion currently cannot be scheduled as a single kernel. It is
  // expected to be segmented as: g{(pointwise)
  //   inputs: tv0, tv1
  //   outputs: tv2, tv3
  //   tv2 = broadcast(tv0)
  //   tv3 = add (tv2, broadcast(tv1))
  // }
  //
  // g{(pointwise)
  //   inputs: tv2
  //   outputs: tv5
  //   tv4 = full({4, 1, i0})
  //   tv5 = mul(tv2, tv4)
  // }
  EXPECT_EQ(segmented_fusion->groups().size(), 2);

  for (SegmentedGroup* group : segmented_fusion->groups()) {
    const std::vector<Expr*>& exprs = group->exprs();

    size_t num_full = std::count_if(exprs.begin(), exprs.end(), [](Expr* expr) {
      return expr->isA<FullOp>();
    });
    if (num_full != 0) {
      // this is the segment contains the factory op.
      EXPECT_EQ(exprs.size(), 2);
      EXPECT_EQ(num_full, 1);
      auto binary_op_iter =
          std::find_if(exprs.begin(), exprs.end(), [](Expr* expr) {
            return expr->isA<BinaryOp>();
          });
      EXPECT_EQ(
          (*binary_op_iter)->as<BinaryOp>()->getBinaryOpType(),
          BinaryOpType::Mul);
      Fusion* group_fusion = group->getFusion();
      // validate that we have a valid reference in the segmented fusion
      DomainMapUnitTest group_dm(group_fusion);
      EXPECT_EQ(group_fusion->outputs().size(), 1);
      EXPECT_TRUE(group_dm.isValidReference(
          group_fusion->outputs()[0]->as<TensorView>()));
    } else {
      // validate segmentation has the correct ops
      EXPECT_EQ(exprs.size(), 3);
      EXPECT_EQ(
          std::count_if(
              exprs.begin(),
              exprs.end(),
              [](Expr* expr) { return expr->isA<BroadcastOp>(); }),
          2);
      EXPECT_EQ(
          std::count_if(
              exprs.begin(),
              exprs.end(),
              [](Expr* expr) { return expr->isA<BinaryOp>(); }),
          1);
      Fusion* group_fusion = group->getFusion();
      auto output_add = std::find_if(
          group_fusion->outputs().begin(),
          group_fusion->outputs().end(),
          [](Val* val) { return val->definition()->isA<BinaryOp>(); });
      EXPECT_TRUE(output_add != group_fusion->outputs().end());
      DomainMapUnitTest group_dm(group_fusion);
      // validate that the segmented fusion choose the add output as the
      // reference
      EXPECT_TRUE(group_dm.isValidReference((*output_add)->as<TensorView>()));
    }
  }

  testValidate(fusion, cg_outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapPad0) {
  OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {b1, i0}
  TensorView* tv0 = TensorViewBuilder().shape({1, -1}).build();
  fusion->addInput(tv0);
  // tv1 {i2, b1, i0}
  TensorView* tv1 = TensorViewBuilder().shape({-1, 1, -1}).build();
  fusion->addInput(tv1);
  // tv2 {i2, b1, i0}
  auto tv2 = add(tv1, tv0);
  fusion->addOutput(tv2);
  // i3 = resize(b1 + 4 + 4)
  // tv3 {i3, i0}
  auto tv3 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(4L)});
  // tv4 {i3*i0}
  auto tv4 = reshape(tv3, {9, 5}, {45});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);

  // tv4 is covered by tv2, because i3 is produced by b1
  EXPECT_TRUE(domain_map.testTargetCoverage(tv4, tv2));
  // tv2 is not covered by tv4, it's missing i2
  EXPECT_FALSE(domain_map.testTargetCoverage(tv2, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv4));
  EXPECT_TRUE(domain_map.isValidReference(tv2));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({1, 5}, {5, 1}, options);
  at::Tensor t1 = at::empty_strided({7, 1, 5}, {5, 5, 1}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapPad1) {
  OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {b1, i0}
  TensorView* tv0 = TensorViewBuilder().shape({1, -1}).build();
  fusion->addInput(tv0);
  // tv1 {i2, i3, i4, b5}
  TensorView* tv1 = TensorViewBuilder().shape({-1, -1, -1, 1}).build();
  fusion->addInput(tv1);

  // tv2 {b6, b7, b1, i0}
  auto tv2 = broadcast(tv0, {true, true, false, false});
  // tv3 {i2, i3, i4, i0}
  auto tv3 = add(tv1, tv2);
  fusion->addOutput(tv3);
  // i8 = resize(b1 + 4 + 4)
  // tv4 {i8, i0}
  auto tv4 =
      pad(tv0,
          {IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(0L),
           IrBuilder::create<Val>(4L),
           IrBuilder::create<Val>(4L)});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);

  // tv4 is covered by tv3, because i8 is produced by b1, a broadcast dimension
  // concretized as i4
  EXPECT_TRUE(domain_map.testTargetCoverage(tv4, tv3));
  // tv3 is not covered by tv4, it's missing i2 and i3
  EXPECT_FALSE(domain_map.testTargetCoverage(tv3, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv4));
  EXPECT_TRUE(domain_map.isValidReference(tv3));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::empty_strided({1, 5}, {5, 1}, options);
  at::Tensor t1 = at::empty_strided({2, 3, 4, 1}, {12, 4, 1, 1}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapSlice0) {
  OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i1, i0}
  TensorView* tv0 = makeContigTensor(2);
  fusion->addInput(tv0);
  // tv1 {i1, i0}
  // use concrete tensor to avoid need of concretization
  TensorView* tv1 = makeContigConcreteTensor({2, 4});
  fusion->addInput(tv1);

  // b3 = resize(i0 + 0 - 3)
  // tv2 {i1, b2}
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(1L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv2);
  // tv3 {i1, i0}
  auto tv3 = add(tv0, tv1);
  // tv4 {i1*i0}
  auto tv4 = reshape(tv3, {2, 4}, {8});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // tv2 and tv4 has the same source IDs, since b3 = resize(i0 + 0 - 3)
  EXPECT_TRUE(domain_map.testTargetCoverage(tv4, tv2));
  EXPECT_TRUE(domain_map.testTargetCoverage(tv2, tv4));

  EXPECT_TRUE(domain_map.isValidReference(tv2));
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 4}, options);
  at::Tensor t1 = at::randn({2, 4}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, DomainMapSlice1) {
  OptimizationPassGuard<preseg_passes::MarkAliasesPreparePass>
      optimization_guard(false);
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);

  // tv0 {i2, i1, i0}
  TensorView* tv0 = makeContigTensor(3);
  fusion->addInput(tv0);
  // tv1 {i1, i0}
  // use concrete tensor to avoid need of concretization
  TensorView* tv1 = makeContigConcreteTensor({2, 4});
  fusion->addInput(tv1);

  // b3 = resize(i0 + 0 - 3)
  // tv2 {i1, b3}
  auto tv2 = slice(
      tv1,
      {Slice(),
       {IrBuilder::create<Val>(0L),
        IrBuilder::create<Val>(1L),
        IrBuilder::create<Val>(1L)}});
  fusion->addOutput(tv2);
  // tv3 {i2, i1, i0}
  auto tv3 = add(tv0, tv1);
  // tv4 {i2, i1*i0}
  auto tv4 = reshape(tv3, {2, 2, 4}, {2, 8});
  fusion->addOutput(tv4);

  DomainMapUnitTest domain_map(fusion);
  // i2 is missing in tv2
  EXPECT_FALSE(domain_map.testTargetCoverage(tv4, tv2));
  EXPECT_TRUE(domain_map.testTargetCoverage(tv2, tv4));

  EXPECT_FALSE(domain_map.isValidReference(tv2));
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  // validate generated kernel
  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  at::Tensor t0 = at::randn({2, 2, 4}, options);
  at::Tensor t1 = at::randn({2, 4}, options);
  // NOTE force pointwise scheduler here for unit test
  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(NVFuserTest, DomainMapBroadcastIssue3653) {
  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;

  auto tv0 = makeConcreteTensor({2, 4, 8});
  fusion.addInput(tv0);
  auto tv1 = makeConcreteTensor({2});
  fusion.addInput(tv1);

  auto tv2 = reshape(tv0, {2, 4, 8}, {2, 32});
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = add(tv2, tv3);

  // tv4 covers source IDs {2, 4, 8}.
  fusion.addOutput(tv4);
  // meanwhile, tv3's broadcast ID map through permissive to `32`, which is not
  // directly contained by tv4's source IDs. This test ensures that we project
  // the mapped ID back to its source IDs and correctly schedule this fusion as
  // a single kernel.
  fusion.addOutput(tv3);

  DomainMapUnitTest domain_map(fusion_ptr.get());
  EXPECT_TRUE(domain_map.isValidReference(tv4));

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({2, 4, 8}, options);
  auto t1 = at::randn({2}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1});

  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  NVF_CHECK(!runtime->isSegmented());

  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(PointwiseTest, InnerDimAllocationTransformationOnProducer) {
  for (bool inner_split : {true, false}) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* in = makeContigTensor(2);
    in->split(1, 4, inner_split); // outer split
    in->setAllocationDomain(in->getLoopDomain(), /*new_contiguity=*/true);
    TensorView* out = cos(in);
    fusion->addInput(in);
    fusion->addOutput(out);
    EXPECT_EQ(
        scheduler_utils::getInputsOutputsWithInnerDim(
            out, /*inner_only=*/true, /*vectorize_pass=*/false)
            .size(),
        2);
  }
}

TEST_F(PointwiseTest, InnerDimAllocationTransformationOnConsumer) {
  for (bool inner_split : {true, false}) {
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());
    TensorView* in = makeContigTensor(2);
    TensorView* out = cos(in);
    out->split(1, 4, inner_split);
    out->setAllocationDomain(out->getLoopDomain(), /*new_contiguity=*/true);
    fusion->addInput(in);
    fusion->addOutput(out);

    // input and output have mapping inner dim.
    EXPECT_EQ(
        scheduler_utils::getInputsOutputsWithInnerDim(
            out, /*inner_only=*/true, /*vectorize_pass=*/false)
            .size(),
        2);
  }
}

TEST_F(
    PointwiseTest,
    InnerDimAllocationTransformationOnBothConsumerAndProducer) {
  for (bool in_inner_split : {true, false}) {
    for (bool out_inner_split : {true, false}) {
      auto fusion = std::make_unique<Fusion>();
      FusionGuard fg(fusion.get());
      TensorView* in = makeContigTensor(2);
      in->split(1, 4, in_inner_split);
      in->setAllocationDomain(in->getLoopDomain(), /*new_contiguity=*/true);
      TensorView* out = cos(in);
      out->split(1, 4, out_inner_split);
      out->setAllocationDomain(out->getLoopDomain(), /*new_contiguity=*/true);
      fusion->addInput(in);
      fusion->addOutput(out);

      // input and output have mapping inner dim.
      EXPECT_EQ(
          scheduler_utils::getInputsOutputsWithInnerDim(
              out, /*inner_only=*/true, /*vectorize_pass=*/false)
              .size(),
          2);
    }
  }
}

// TMA pointwise test utilities
namespace tma_check {

// Helper to check if TMA load is used in the compiled kernel
bool hasTmaLoad(const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  return runtime->schedulerHeuristics()
      ->heuristicsList()
      .at(0)
      ->as<PointwiseParams>()
      ->use_tma_load;
}

int64_t getTmaDomainInner(const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  return runtime->schedulerHeuristics()
      ->heuristicsList()
      .at(0)
      ->as<PointwiseParams>()
      ->tma_domain_inner;
}

int64_t getVectorizationFactor(const FusionExecutorCache& executor_cache) {
  FusionKernelRuntime* runtime = executor_cache.getMostRecentKernelRuntime();
  return runtime->schedulerHeuristics()
      ->heuristicsList()
      .at(0)
      ->as<PointwiseParams>()
      ->vectorization_factor;
}
} // namespace tma_check

// The native small-workload rule still selects bp0, but its choice is now
// recorded at PW2 with the raw threshold inputs and legal 2D alternatives.
TEST_F(PointwiseTest, SmallPointwiseDirectBreakpointChoiceRecord) {
  char log_path[] = "/tmp/nvfuser-pw2-direct-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
    }
  } file_cleanup{log_path};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv protocol("NVFUSER_JEV_PROTOCOL", "text-v1");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::empty(
      {2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  const auto prop = pointwise_utils::getFusionRuntimeProperties(
      &fusion, runtime_info, nullptr);
  ASSERT_TRUE(prop.has_value());

  const auto params = pointwise::non_tma::getPointwiseHeuristics(
      &fusion, runtime_info, nullptr, *prop);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->break_point, 0);

  std::ifstream log_file(log_path);
  std::stringstream log_output;
  log_output << log_file.rdbuf();
  const std::string records = log_output.str();
  EXPECT_THAT(records, ::testing::HasSubstr("\"site\":\"PW2\""));
  EXPECT_THAT(records, ::testing::HasSubstr("\"selected\":\"bp0\""));
  EXPECT_THAT(records, ::testing::HasSubstr("pointwise.breakpoint.v2"));
  EXPECT_THAT(records, ::testing::HasSubstr("n_elems=32"));
  EXPECT_THAT(
      records,
      ::testing::HasSubstr(
          "sm_count=" + std::to_string(prop->device_multiprocessor_count)));
  EXPECT_THAT(records, ::testing::HasSubstr("default_cta_threads=128"));
  EXPECT_THAT(records, ::testing::HasSubstr("\"bp1_normal\""));
  EXPECT_THAT(
      records, ::testing::Not(::testing::HasSubstr("\"site\":\"PW2.entry\"")));
}

// The small Add carrier offers bp0 and both legal bindings at bp1. The v3
// candidate cards expose local geometry and possible PW3 descendants, while
// the actual Remote answer may choose any of the hard-legal alternatives.
TEST_F(PointwiseTest, JevBreakpointCandidateEffects_CUDA) {
  char local_path[] = "/tmp/nvfuser-pw2-effects-XXXXXX";
  struct FileCleanup {
    const char* path = nullptr;
    ~FileCleanup() {
      if (path != nullptr) {
        ::unlink(path);
        ::unlink((std::string(path) + ".outcome.jsonl").c_str());
      }
    }
  } file_cleanup;
  const auto jev_mode = scheduler::jev::mode();
  std::optional<ScopedTestEnv> local_log;
  const char* log_path = std::getenv("NVFUSER_JEV_LOG");
  if (jev_mode != scheduler::jev::Mode::Off &&
      (log_path == nullptr || *log_path == '\0')) {
    const int fd = ::mkstemp(local_path);
    ASSERT_GE(fd, 0);
    ::close(fd);
    file_cleanup.path = local_path;
    local_log.emplace("NVFUSER_JEV_LOG", local_path);
    log_path = local_path;
  }
  std::streampos log_start = 0;
  std::streampos outcome_start = 0;
  std::string outcome_path;
  if (log_path != nullptr) {
    std::ifstream existing_log(log_path, std::ios::ate);
    if (existing_log.good()) {
      log_start = existing_log.tellg();
    }
    const char* outcome_override = std::getenv("NVFUSER_JEV_OUTCOME_LOG");
    outcome_path = outcome_override != nullptr && *outcome_override != '\0'
        ? outcome_override
        : std::string(log_path) + ".outcome.jsonl";
    std::ifstream existing_outcomes(outcome_path, std::ios::ate);
    if (existing_outcomes.good()) {
      outcome_start = existing_outcomes.tellg();
    }
  }
  ScopedTestEnv protocol("NVFUSER_JEV_PROTOCOL", "semantic-v3");

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::randn(
      {2, 16}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  auto scheduler =
      SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);
  std::unique_ptr<HeuristicParams> native;
  {
    ScopedTestEnv mode("NVFUSER_JEV_MODE", "off");
    native = scheduler->computeHeuristics(&fusion, runtime_info);
  }
  ASSERT_NE(native, nullptr);
  EXPECT_EQ(native->as<PointwiseParams>()->break_point, 0);
  auto results = scheduleAndRun(&fusion, SchedulerType::PointWise, {tensor});
  ASSERT_NE(results.heuristic_params, nullptr);
  const auto* params = results.heuristic_params->as<PointwiseParams>();
  if (jev_mode != scheduler::jev::Mode::Remote) {
    EXPECT_TRUE(native->sameAs(results.heuristic_params.get()));
    EXPECT_EQ(native->lparams, results.heuristic_params->lparams);
  }
  testValidate(
      &fusion,
      results.outputs,
      {tensor},
      {tensor + tensor},
      __LINE__,
      __FILE__);

  if (jev_mode != scheduler::jev::Mode::Off) {
    ASSERT_NE(log_path, nullptr);
    std::ifstream decision_log(log_path);
    ASSERT_TRUE(decision_log.good());
    decision_log.seekg(log_start);
    std::string row;
    std::string pw2_row;
    std::string cta_1d_row;
    std::string unroll_row;
    bool saw_unroll_choice = false;
    while (std::getline(decision_log, row)) {
      if (row.rfind("{\"site\":\"PW2\",", 0) == 0) {
        ASSERT_TRUE(pw2_row.empty());
        pw2_row = row;
      }
      if (row.rfind("{\"site\":\"PW3.cta_1d\",", 0) == 0) {
        ASSERT_TRUE(cta_1d_row.empty());
        cta_1d_row = row;
      }
      saw_unroll_choice |=
          row.rfind("{\"site\":\"PW3.unroll\",", 0) == 0;
      if (row.rfind("{\"site\":\"PW3.unroll\",", 0) == 0) {
        unroll_row = row;
      }
    }
    ASSERT_FALSE(pw2_row.empty());
    EXPECT_THAT(pw2_row, ::testing::HasSubstr("pointwise.breakpoint.v3"));
    EXPECT_THAT(
        pw2_row,
        ::testing::HasSubstr(
            "\"break_point\":0,\"left_elems\":0,\"right_elems\":32,"
            "\"grid_binding\":\"one_dimensional\""));
    EXPECT_THAT(
        pw2_row,
        ::testing::HasSubstr(
            "\"break_point\":1,\"left_elems\":2,\"right_elems\":16,"
            "\"grid_binding\":\"normal\""));
    EXPECT_THAT(
        pw2_row,
        ::testing::HasSubstr(
            "\"break_point\":1,\"left_elems\":2,\"right_elems\":16,"
            "\"grid_binding\":\"flipped\""));
    EXPECT_THAT(
        pw2_row,
        ::testing::HasSubstr(
            "\"pending_choices\":[\"PW3.width\",\"PW3.cta_1d\",\"PW3."
            "unroll\"]"));
    EXPECT_THAT(
        pw2_row,
        ::testing::HasSubstr(
            "\"pending_choices\":[\"PW3.width\",\"PW3.launch\",\"PW3."
            "unroll\"]"));
    if (jev_mode == scheduler::jev::Mode::Record) {
      EXPECT_THAT(pw2_row, ::testing::HasSubstr("\"selected\":\"bp0\""));
      EXPECT_EQ(params->break_point, 0);
      EXPECT_FALSE(saw_unroll_choice);
      ASSERT_FALSE(cta_1d_row.empty());
      std::ifstream outcome_log(outcome_path);
      ASSERT_TRUE(outcome_log.good());
      outcome_log.seekg(outcome_start);
      bool saw_singleton = false;
      while (std::getline(outcome_log, row)) {
        saw_singleton |=
            row.find("\"site\":\"PW3.unroll\"") != std::string::npos &&
            row.find("\"selected\":\"u1\"") != std::string::npos &&
            row.find("bounded model menu contains only u1") !=
                std::string::npos;
      }
      EXPECT_TRUE(saw_singleton);
    } else if (
        pw2_row.find("\"selected\":\"bp1_normal\"") != std::string::npos) {
      EXPECT_EQ(params->break_point, 1);
      EXPECT_FALSE(params->flip_grid_binding);
    } else if (pw2_row.find("\"selected\":\"bp1_flip\"") != std::string::npos) {
      EXPECT_EQ(params->break_point, 1);
      EXPECT_TRUE(params->flip_grid_binding);
    } else {
      EXPECT_THAT(pw2_row, ::testing::HasSubstr("\"selected\":\"bp0\""));
      EXPECT_EQ(params->break_point, 0);
    }

    if (params->break_point == 0) {
      ASSERT_FALSE(cta_1d_row.empty());
      ASSERT_NE(results.kernel_executor, nullptr);
      const auto launch = results.kernel_executor->lastLaunchParams();
      const int64_t threads = params->lparams.bdimx();
      const int64_t width = params->vectorization_factor;
      EXPECT_EQ(launch.gdimx(), 1);
      EXPECT_EQ(launch.gdimy(), 1);
      EXPECT_EQ(launch.bdimx(), threads);
      EXPECT_EQ(launch.bdimy(), 1);
      EXPECT_EQ(params->unroll_factor_inner, 1);
      EXPECT_THAT(
          cta_1d_row,
          ::testing::HasSubstr(
              "\"tail_right_positions\":" +
              std::to_string(threads * width - 32)));
      const auto* device = at::cuda::getCurrentDeviceProperties();
      const bool exact_carrier_device =
          std::string_view(device->name) == "NVIDIA GeForce RTX 4090" &&
          device->major == 8 && device->minor == 9 &&
          device->multiProcessorCount == 128 && device->warpSize == 32;
      if (exact_carrier_device) {
        EXPECT_THAT(
            cta_1d_row, ::testing::HasSubstr("pointwise.launch_1d.v2"));
        for (const int64_t offered_threads : {32L, 64L, 128L}) {
          const std::string option_id =
              "\"id\":\"x" + std::to_string(offered_threads) + "\"";
          const auto begin = cta_1d_row.find(option_id);
          ASSERT_NE(begin, std::string::npos);
          const auto next =
              cta_1d_row.find("\"id\":\"x", begin + option_id.size());
          const std::string candidate = cta_1d_row.substr(begin, next - begin);
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"final_pointwise_plan\":{\"tag\":\"Pointwise "
                  "heuristics\",\"break_point\":0,\"split_block\":false,"
                  "\"symbolic_1d_cta\":" +
                  std::string(offered_threads == 128 ? "false" : "true")));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"vectorization_factor\":" + std::to_string(width)));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"unroll_factor_outer\":1,\"unroll_factor_inner\":1"));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"raw_launch\":{\"grid_x\":-1,\"grid_y\":-1,"
                  "\"grid_z\":-1,\"block_x\":" +
                  std::to_string(offered_threads)));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"effective_launch\":{\"grid_x\":1,\"grid_y\":1,"
                  "\"grid_z\":1,\"block_x\":" +
                  std::to_string(offered_threads) + ",\"block_y\":1,"));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"right_work\":{\"right_elems\":32,"
                  "\"right_positions_per_cta\":" +
                  std::to_string(offered_threads * width)));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"tail_right_positions\":" +
                  std::to_string(offered_threads * width - 32)));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr(
                  "\"pending_choices\":[],\"continuation\":{"
                  "\"schema\":\"nvfuser.pw3.cta_1d.add2x16.v1\","));
          EXPECT_THAT(
              candidate, ::testing::HasSubstr("\"unroll_bound\":1"));
          EXPECT_THAT(
              candidate,
              ::testing::HasSubstr("\"unroll_kind\":\"no_choice\""));
          if (offered_threads == threads) {
            scheduler::native_diagnostic::Object actual_plan;
            actual_plan.add("tag", params->tag);
            actual_plan.add("break_point", params->break_point);
            actual_plan.add("split_block", params->split_block);
            actual_plan.add("symbolic_1d_cta", params->symbolic_1d_cta);
            actual_plan.add("split_grid_y_dim", params->split_grid_y_dim);
            actual_plan.add("flip_grid_binding", params->flip_grid_binding);
            actual_plan.add(
                "vectorization_factor", params->vectorization_factor);
            actual_plan.add("vectorize_casts", params->vectorize_casts);
            actual_plan.add(
                "unroll_factor_outer", params->unroll_factor_outer);
            actual_plan.add(
                "unroll_factor_inner", params->unroll_factor_inner);
            actual_plan.add("use_tma_load", params->use_tma_load);
            actual_plan.add("use_tma_store", params->use_tma_store);
            actual_plan.add("tma_domain_inner", params->tma_domain_inner);
            actual_plan.add("tma_tile_outer", params->tma_tile_outer);
            actual_plan.add("tma_tile_inner", params->tma_tile_inner);
            if (params->cparams.index_type.has_value()) {
              actual_plan.add(
                  "index_type",
                  static_cast<int64_t>(*params->cparams.index_type));
            } else {
              actual_plan.addRaw("index_type", "null");
            }
            EXPECT_THAT(
                candidate,
                ::testing::HasSubstr(
                    "\"final_pointwise_plan\":" + actual_plan.str()));
          }
        }
      }
    }

    // The forced bp1_normal/w1/x1 path makes PW3.unroll terminal for this
    // ordinary branch. Compare its selected effect with the physical launch,
    // since the heuristic's raw grid dimensions are intentionally unset.
    if (jev_mode == scheduler::jev::Mode::Remote &&
        params->break_point == 1 && !params->flip_grid_binding &&
        params->vectorization_factor == 1 && params->lparams.bdimx() == 1 &&
        params->lparams.bdimy() == 128) {
      ASSERT_FALSE(unroll_row.empty());
      const int64_t unroll = params->unroll_factor_inner;
      const int64_t grid_x = (16 + unroll - 1) / unroll;
      EXPECT_EQ(params->unroll_factor_outer, 1);
      EXPECT_THAT(
          unroll_row,
          ::testing::HasSubstr(
              "\"selected\":\"u" + std::to_string(unroll) + "\""));
      ASSERT_NE(results.kernel_executor, nullptr);
      const auto launch = results.kernel_executor->lastLaunchParams();
      EXPECT_EQ(launch.gdimx(), grid_x);
      EXPECT_EQ(launch.gdimy(), 1);
      EXPECT_EQ(launch.gdimz(), 1);
      EXPECT_EQ(launch.bdimx(), 1);
      EXPECT_EQ(launch.bdimy(), 128);
      EXPECT_EQ(launch.bdimz(), 1);
      const std::string candidate_id =
          "\"id\":\"u" + std::to_string(unroll) + "\"";
      const auto begin = unroll_row.find(candidate_id);
      ASSERT_NE(begin, std::string::npos);
      const auto next = unroll_row.find("\"id\":\"u", begin + candidate_id.size());
      const std::string candidate = unroll_row.substr(begin, next - begin);
      EXPECT_THAT(
          candidate,
          ::testing::HasSubstr(
              "\"plan_scope\":\"ordinary_pointwise_branch\","
              "\"final_pointwise_plan\":{\"tag\":\"Pointwise heuristics\","
              "\"break_point\":1,"));
      EXPECT_THAT(
          candidate,
          ::testing::HasSubstr(
              "\"tma_domain_inner\":-1,\"tma_tile_outer\":-1,"
              "\"tma_tile_inner\":-1,\"index_type\":11}"));
      EXPECT_THAT(
          candidate,
          ::testing::HasSubstr(
              "\"index_type_provenance\":\"source_hard_analysis\","
              "\"raw_launch\":{\"grid_x\":-1,\"grid_y\":-1,"
              "\"grid_z\":-1,\"block_x\":1,\"block_y\":128,"
              "\"block_z\":-1,\"dynamic_smem_bytes\":0}"));
      EXPECT_THAT(
          candidate,
          ::testing::HasSubstr(
              "\"effective_launch\":{\"grid_x\":" +
              std::to_string(grid_x) +
              ",\"grid_y\":1,\"grid_z\":1,\"block_x\":1,\"block_y\":128,"
              "\"block_z\":1}"));
      EXPECT_THAT(
          candidate,
          ::testing::HasSubstr(
              "\"unroll_factor_outer\":1,\"unroll_factor_inner\":" +
              std::to_string(unroll)));
      EXPECT_THAT(
          candidate,
          ::testing::HasSubstr(
              "\"right_work\":{\"right_elems\":16,"
              "\"right_positions_per_cta\":" +
              std::to_string(unroll) + ",\"right_ctas\":" +
              std::to_string(grid_x) + ",\"covered_right_positions\":" +
              std::to_string(grid_x * unroll) +
              ",\"tail_right_positions\":" +
              std::to_string(grid_x * unroll - 16) + "}"));
      EXPECT_THAT(candidate, ::testing::HasSubstr("\"pending_choices\":[]"));
    }
  }
}

TEST_F(PointwiseTest, AddCtaBoundedContinuation) {
  using pointwise::non_tma::addCtaContinuation;
  for (const int64_t width : {1L, 2L, 4L}) {
    for (const int64_t threads : {32L, 64L, 128L}) {
      const auto c = addCtaContinuation(32, width, threads, 1);
      ASSERT_TRUE(c.has_value()) << width << ',' << threads;
      EXPECT_EQ(c->source_total_blocks, 1);
      EXPECT_EQ(c->unroll_bound, 1);
      EXPECT_EQ(c->effective_grid_x, 1);
      EXPECT_EQ(c->covered_right_positions, threads * width);
      EXPECT_EQ(c->tail_right_positions, threads * width - 32);
    }
  }
  EXPECT_FALSE(addCtaContinuation(34, 1, 32, 1).has_value());
  EXPECT_FALSE(addCtaContinuation(32, 8, 32, 1).has_value());
  EXPECT_FALSE(addCtaContinuation(32, 1, 16, 1).has_value());
  EXPECT_FALSE(addCtaContinuation(32, 1, 32, 2).has_value());
}

TEST_F(PointwiseTest, AddCtaNonzeroStorageOffsetKeepsConditionalPreview_CUDA) {
  const auto* device = at::cuda::getCurrentDeviceProperties();
  if (std::string_view(device->name) != "NVIDIA GeForce RTX 4090" ||
      device->major != 8 || device->minor != 9 ||
      device->multiProcessorCount != 128 || device->warpSize != 32) {
    GTEST_SKIP() << "The bounded Add continuation is scoped to RTX 4090";
  }
  char log_path[] = "/tmp/nvfuser-pw3-cta-offset-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
      ::unlink((std::string(path) + ".outcome.jsonl").c_str());
    }
  } cleanup{log_path};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv protocol("NVFUSER_JEV_PROTOCOL", "semantic-v3");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto backing = at::randn(
      {36}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  const auto tensor = backing.narrow(0, 4, 32).view({2, 16});
  ASSERT_EQ(tensor.storage_offset(), 4);
  ASSERT_EQ(tensor.stride(0), 16);
  ASSERT_EQ(tensor.stride(1), 1);
  auto results = scheduleAndRun(&fusion, SchedulerType::PointWise, {tensor});
  testValidate(
      &fusion,
      results.outputs,
      {tensor},
      {tensor + tensor},
      __LINE__,
      __FILE__);

  std::ifstream decisions(log_path);
  ASSERT_TRUE(decisions.good());
  std::string row;
  bool saw_cta = false;
  while (std::getline(decisions, row)) {
    if (row.rfind("{\"site\":\"PW3.cta_1d\",", 0) != 0) {
      continue;
    }
    saw_cta = true;
    EXPECT_THAT(row, ::testing::HasSubstr("pointwise.launch_1d.v1"));
    EXPECT_THAT(row, ::testing::Not(::testing::HasSubstr(
                         "nvfuser.pw3.cta_1d.add2x16.v1")));
  }
  EXPECT_TRUE(saw_cta);
}

// The two structured protocols must describe the same conditional work for
// each offered width and 1D CTA. The v3 wrapper retains the text state used
// by the deterministic API parser and carries these previews per candidate.
TEST_F(PointwiseTest, PointwiseCandidatePreviewsMatchSemanticV2_CUDA) {
  char log_path[] = "/tmp/nvfuser-pw3-preview-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
      ::unlink((std::string(path) + ".outcome.jsonl").c_str());
    }
  } file_cleanup{log_path};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(1);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::empty(
      {16388}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  const auto prop = pointwise_utils::getFusionRuntimeProperties(
      &fusion, runtime_info, nullptr);
  ASSERT_TRUE(prop.has_value());
  const std::string hardware =
      "{\"sm_count\":" + std::to_string(prop->device_multiprocessor_count) +
      "}";

  auto read_log = [&]() {
    std::ifstream stream(log_path);
    std::stringstream contents;
    contents << stream.rdbuf();
    return contents.str();
  };
  auto run_record = [&](const char* protocol) {
    ScopedTestEnv protocol_env("NVFUSER_JEV_PROTOCOL", protocol);
    const std::string before = read_log();
    scheduler::jev::ComputeHeuristicsScope scope(
        "one-dimensional add with 16388 float elements", {}, hardware);
    auto params = pointwise::non_tma::getPointwiseHeuristics(
        &fusion, runtime_info, nullptr, *prop);
    NVF_CHECK(params != nullptr);
    NVF_CHECK(params->break_point == 0);
    const std::pair<int64_t, int64_t> plan{
        params->vectorization_factor, params->lparams.bdimx()};
    scope.complete("PointWise");
    return std::pair{read_log().substr(before.size()), plan};
  };
  const auto [v2_records, v2_plan] = run_record("semantic-v2");
  const auto [v3_records, v3_plan] = run_record("semantic-v3");
  EXPECT_EQ(v2_plan, v3_plan);
  EXPECT_EQ(v2_plan, (std::pair<int64_t, int64_t>{4, 128}));
  EXPECT_THAT(v3_records, ::testing::HasSubstr("pointwise.width.v1"));
  EXPECT_THAT(v3_records, ::testing::HasSubstr("pointwise.launch_1d.v1"));

  const auto site_row = [](const std::string& records, const std::string& site) {
    std::istringstream stream(records);
    std::string row;
    while (std::getline(stream, row)) {
      if (row.find("\"site\":\"" + site + "\"") != std::string::npos) {
        return row;
      }
    }
    return std::string();
  };
  for (const auto& [site, ids] : {
           std::pair<const char*, std::vector<const char*>>{
               "PW3.width", {"w1", "w2", "w4"}},
           {"PW3.cta_1d", {"x32", "x64", "x128"}}}) {
    const std::string v2_row = site_row(v2_records, site);
    const std::string v3_row = site_row(v3_records, site);
    ASSERT_FALSE(v2_row.empty()) << site;
    ASSERT_FALSE(v3_row.empty()) << site;
    for (const char* id : ids) {
      const std::string marker = "{\"id\":\"" + std::string(id) + "\"";
      const size_t v2_start = v2_row.find(marker);
      const size_t v3_start = v3_row.find(marker);
      ASSERT_NE(v2_start, std::string::npos) << site << ' ' << id;
      ASSERT_NE(v3_start, std::string::npos) << site << ' ' << id;
      const size_t v2_fields = v2_start + marker.size() + 1;
      const size_t v2_end = v2_row.find('}', v2_fields);
      const std::string consequence = "\"consequence\":{";
      const size_t v3_fields = v3_row.find(consequence, v3_start);
      ASSERT_NE(v3_fields, std::string::npos) << site << ' ' << id;
      const size_t v3_values = v3_fields + consequence.size();
      const size_t v3_end = v3_row.find('}', v3_values);
      ASSERT_NE(v2_end, std::string::npos);
      ASSERT_NE(v3_end, std::string::npos);
      const std::string pending = ",\"unroll_status\":\"pending\"";
      const std::string v3_fields_without_status =
          v3_row.substr(v3_values, v3_end - v3_values);
      ASSERT_GE(v3_fields_without_status.size(), pending.size());
      EXPECT_EQ(
          v3_fields_without_status.substr(
              v3_fields_without_status.size() - pending.size()),
          pending);
      EXPECT_EQ(
          v2_row.substr(v2_fields, v2_end - v2_fields),
          v3_fields_without_status.substr(
              0, v3_fields_without_status.size() - pending.size()))
          << site << ' ' << id;
    }
  }
}

// A logical right side just beyond the device's grid-X limit filters x1 but
// retains the native CTA. No tensor allocation is needed to exercise the
// exact hard menu and its conditional 2D preview.
TEST_F(PointwiseTest, PointwiseLaunchPartialMenuSemanticV3_CUDA) {
  const auto* device = at::cuda::getCurrentDeviceProperties();
  ASSERT_GT(device->maxGridSize[0], 128);
  char log_path[] = "/tmp/nvfuser-pw3-launch-preview-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
      ::unlink((std::string(path) + ".outcome.jsonl").c_str());
    }
  } file_cleanup{log_path};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  const int64_t right_elems =
      (static_cast<int64_t>(device->maxGridSize[0]) + 1) * 4 + 2;
  pointwise_utils::FusionRuntimeProperties prop;
  prop.n_elems = right_elems * 2;
  prop.device_multiprocessor_count = device->multiProcessorCount;
  pointwise_utils::BreakPointInfo bp;
  bp.break_point = 1;
  bp.right_elem_count = right_elems;
  const std::string hardware =
      "{\"sm_count\":" + std::to_string(prop.device_multiprocessor_count) +
      "}";
  auto read_log = [&]() {
    std::ifstream stream(log_path);
    std::stringstream contents;
    contents << stream.rdbuf();
    return contents.str();
  };
  auto run_record = [&](const char* protocol) {
    ScopedTestEnv protocol_env("NVFUSER_JEV_PROTOCOL", protocol);
    const std::string before = read_log();
    scheduler::jev::ComputeHeuristicsScope scope(
        "synthetic large two-dimensional pointwise", {}, hardware);
    const auto config =
        pointwise_utils::getBlockGridConfig(prop, bp, 4, 128, false);
    scope.complete("PointWise");
    return std::pair{read_log().substr(before.size()), config};
  };
  const auto [text_records, text_config] = run_record("text-v1");
  const auto [v2_records, v2_config] = run_record("semantic-v2");
  const auto [v3_records, v3_config] = run_record("semantic-v3");
  const std::string before_off = read_log();
  pointwise_utils::BlockGridConfig off_config;
  {
    ScopedTestEnv off_mode("NVFUSER_JEV_MODE", "off");
    off_config = pointwise_utils::getBlockGridConfig(prop, bp, 4, 128, false);
  }
  EXPECT_EQ(off_config.bdimx, v3_config.bdimx);
  EXPECT_EQ(read_log(), before_off);
  EXPECT_EQ(text_config.bdimx, v2_config.bdimx);
  EXPECT_EQ(text_config.bdimx, v3_config.bdimx);
  EXPECT_EQ(text_config.bdimy, v3_config.bdimy);
  EXPECT_THAT(
      text_records,
      ::testing::HasSubstr("text_v1_requires_full_hard_cta_menu"));
  EXPECT_THAT(v3_records, ::testing::HasSubstr("pointwise.launch.v2"));
  EXPECT_THAT(
      v3_records,
      ::testing::HasSubstr(
          "max_grid_x=" + std::to_string(device->maxGridSize[0])));
  EXPECT_THAT(v3_records, ::testing::HasSubstr("unroll_status=pending"));
  EXPECT_THAT(v3_records, ::testing::HasSubstr("grid_split_status=pending"));

  const auto site_row = [](const std::string& records) {
    std::istringstream stream(records);
    std::string row;
    while (std::getline(stream, row)) {
      if (row.find("\"site\":\"PW3.launch\"") != std::string::npos &&
          row.find("\"kind\":\"coverage_skip\"") == std::string::npos) {
        return row;
      }
    }
    return std::string();
  };
  const std::string v2_row = site_row(v2_records);
  const std::string v3_row = site_row(v3_records);
  ASSERT_FALSE(v2_row.empty());
  ASSERT_FALSE(v3_row.empty());
  EXPECT_EQ(v2_row.find("{\"id\":\"x1\""), std::string::npos);
  EXPECT_EQ(v3_row.find("{\"id\":\"x1\""), std::string::npos);
  const auto preview_fields = [](const std::string& row, const char* id,
                                 bool semantic_v3) {
    const std::string marker = "{\"id\":\"" + std::string(id) + "\"";
    const size_t candidate = row.find(marker);
    NVF_CHECK(candidate != std::string::npos, "missing 2D CTA candidate ", id);
    if (semantic_v3) {
      const std::string key = "\"consequence\":{";
      const size_t start = row.find(key, candidate);
      NVF_CHECK(start != std::string::npos, "missing 2D CTA consequence ", id);
      const size_t fields = start + key.size();
      const size_t end = row.find(",\"unroll_status\":\"pending\"", fields);
      NVF_CHECK(end != std::string::npos, "missing pending 2D CTA unroll ", id);
      return row.substr(fields, end - fields);
    }
    const size_t fields = candidate + marker.size() + 1;
    const size_t end = row.find('}', fields);
    NVF_CHECK(end != std::string::npos, "incomplete v2 2D CTA preview ", id);
    return row.substr(fields, end - fields);
  };
  for (const char* id : {"x2", "x32", "x128"}) {
    EXPECT_EQ(
        preview_fields(v2_row, id, false),
        preview_fields(v3_row, id, true)) << id;
  }
  EXPECT_THAT(
      v3_row,
      ::testing::HasSubstr(
          "\"unroll_status\":\"pending\",\"grid_split_status\":\"pending\""));
}

// Call the TMA heuristic directly so this checks candidate analysis on any
// CUDA device without compiling or executing a Hopper kernel.
TEST_F(PointwiseTest, TmaBreakpointPreflightDebugOnlyFinalPlan) {
  char log_path[] = "/tmp/nvfuser-pw4-preflight-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
      ::unlink((std::string(path) + ".outcome.jsonl").c_str());
    }
  } file_cleanup{log_path};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::empty(
      {32, 4096}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  HeuristicDataCache parent_cache;
  const auto prop = pointwise_utils::getFusionRuntimeProperties(
      &fusion, runtime_info, &parent_cache);
  ASSERT_TRUE(prop.has_value());
  const auto structural = pointwise_utils::getTmaBreakPointSearch(*prop, nullptr);
  EXPECT_EQ(structural.candidates.size(), 2);

  DebugDumpOptionsGuard debug_options;
  DebugDumpOptionsGuard::getCurOptions().set(DebugDumpOption::SchedulerDebug);
  std::stringstream preview_debug;
  pointwise::tma::TmaBreakPointAvailability preview;
  {
    DebugStreamGuard debug_guard(preview_debug);
    preview = pointwise::tma::getTmaBreakPointAvailability(
        &fusion, runtime_info, *prop);
  }
  EXPECT_EQ(preview.state, structural.state);
  ASSERT_EQ(preview.options.size(), preview.plans.size());
  ASSERT_EQ(
      preview.options.size(), preview.materialized_layout_widths.size());
  ASSERT_EQ(
      preview.options.size(), preview.remote_unavailability_reasons.size());
  ASSERT_EQ(preview.options.size(), 2);
  EXPECT_EQ(preview.options[0].id, "bp0");
  EXPECT_EQ(preview.options[1].id, "bp1");
  EXPECT_EQ(
      preview.options[0].description,
      "Flatten to the one-dimensional TMA domain");
  EXPECT_EQ(preview.options[1].description, "Split before loop axis 1");
  const auto positive_menu = pointwise::tma::remoteRouteableTmaBreakPoints(
      preview,
      prop->n_elems,
      prop->elem_counts.size(),
      prop->min_dtype_size_bit_for_vectorization);
  EXPECT_FALSE(positive_menu.remote_filter_applied);
  EXPECT_EQ(positive_menu.state, preview.state);
  ASSERT_EQ(positive_menu.options.size(), preview.options.size());
  for (size_t i = 0; i < preview.options.size(); ++i) {
    EXPECT_EQ(positive_menu.options[i].id, preview.options[i].id);
    EXPECT_EQ(
        positive_menu.options[i].description,
        preview.options[i].description);
  }
  for (size_t i = 0; i < preview.options.size(); ++i) {
    EXPECT_EQ(
        preview.options[i].id,
        "bp" + std::to_string(preview.plans[i].break_point));
    EXPECT_TRUE(preview.remote_unavailability_reasons[i].empty());
  }
  // The same source-used routeability gate separates a materializable bp0
  // from a Remote-reachable continuation at the divisor replay boundary.
  EXPECT_TRUE(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          2, 1LL << 40, 0, 4, 32)
          .empty());
  EXPECT_EQ(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          2, (1LL << 40) + 1, 0, 4, 32),
      "extent exceeds bounded divisor replay");
  EXPECT_TRUE(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          2, (1LL << 40) + 1, 1, 4, 32)
          .empty());
  EXPECT_EQ(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          33, 131072, 1, 4, 32),
      "more than 32 loop dimensions");
  EXPECT_EQ(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          2, 131072, 1, 3, 32),
      "PW4.width layout and byte bounds must be powers of two in 1..128");
  EXPECT_EQ(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          2, 131072, 1, 4, 3),
      "PW4.width layout and byte bounds must be powers of two in 1..128");
  EXPECT_EQ(
      pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
          2, 131072, 1, 1, 4),
      "PW4.width layout is below the required byte floor");
  EXPECT_TRUE(preview_debug.str().empty());
  std::ifstream preview_log(log_path);
  std::stringstream preview_rows;
  preview_rows << preview_log.rdbuf();
  EXPECT_TRUE(preview_rows.str().empty());
  std::ifstream preview_outcome(std::string(log_path) + ".outcome.jsonl");
  std::stringstream preview_events;
  preview_events << preview_outcome.rdbuf();
  EXPECT_TRUE(preview_events.str().empty());

  // PW1 uses these exact cache-free native projections before asking its
  // backend Choice. Neither preview may write to the caller's cache or log.
  std::stringstream parent_preview_debug;
  std::unique_ptr<PointwiseParams> native_tma_baseline;
  std::unique_ptr<PointwiseParams> ordinary_baseline;
  {
    DebugStreamGuard debug_guard(parent_preview_debug);
    scheduler::jev::ScopedNativeDecisions native_scope;
    native_tma_baseline =
        pointwise::tma::getNativePointwiseHeuristicsPreview(
            &fusion, runtime_info, *prop);
    ordinary_baseline = pointwise::non_tma::getPointwiseHeuristics(
        &fusion, runtime_info, nullptr, *prop, /*emit_debug=*/false);
  }
  ASSERT_NE(native_tma_baseline, nullptr);
  ASSERT_NE(ordinary_baseline, nullptr);
  EXPECT_TRUE(parent_preview_debug.str().empty());
  EXPECT_FALSE(parent_cache.hasEntry(
      HeuristicCompileTime::ResizeVectorizationFactors::EntryType));
  std::ifstream parent_log(log_path);
  std::stringstream parent_rows;
  parent_rows << parent_log.rdbuf();
  EXPECT_TRUE(parent_rows.str().empty());

  std::stringstream debug_output;
  std::unique_ptr<PointwiseParams> params;
  {
    DebugStreamGuard debug_guard(debug_output);
    params = pointwise::tma::getPointwiseHeuristics(
        &fusion, runtime_info, nullptr, *prop);
  }
  ASSERT_NE(params, nullptr);
  EXPECT_TRUE(params->sameAs(native_tma_baseline.get()));
  for (const auto parallel_type : {
           ParallelType::BIDx,
           ParallelType::BIDy,
           ParallelType::BIDz,
           ParallelType::TIDx,
           ParallelType::TIDy,
           ParallelType::TIDz}) {
    EXPECT_EQ(
        params->lparams.getRawVal(parallel_type),
        native_tma_baseline->lparams.getRawVal(parallel_type));
  }
  const std::string marker = "==== Pointwise TMA Scheduler Heuristics ====";
  const std::string output = debug_output.str();
  const size_t first = output.find(marker);
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(output.find(marker, first + marker.size()), std::string::npos);
  EXPECT_THAT(
      output,
      ::testing::HasSubstr(
          "break_point: " + std::to_string(params->break_point) + "\n"));

  std::ifstream log_file(log_path);
  std::stringstream log_output;
  log_output << log_file.rdbuf();
  EXPECT_THAT(log_output.str(), ::testing::HasSubstr("\"site\":\"PW4.breakpoint\""));
  EXPECT_THAT(
      log_output.str(),
      ::testing::HasSubstr("pointwise.tma_breakpoint.v1"));
  EXPECT_THAT(
      log_output.str(),
      ::testing::Not(
          ::testing::HasSubstr("pointwise.tma_breakpoint.v2")));
  EXPECT_THAT(log_output.str(), ::testing::HasSubstr("\"site\":\"PW4.tile_inner\""));
  EXPECT_THAT(log_output.str(), ::testing::HasSubstr("\"site\":\"PW4.tile_outer\""));
  EXPECT_THAT(log_output.str(), ::testing::HasSubstr("\"site\":\"PW4.width\""));
  EXPECT_THAT(log_output.str(), ::testing::HasSubstr("\"bp0\""));
  EXPECT_THAT(log_output.str(), ::testing::HasSubstr("\"bp1\""));
  EXPECT_THAT(
      log_output.str(),
      ::testing::HasSubstr("\"candidates\":[\"bp0\",\"bp1\"]"));

  // In remote mode a deliberately invalid port stops after candidate
  // preflight. This checks that the preflight did not seed the shared
  // breakpoint-dependent vectorization cache even on a non-resize fusion.
  HeuristicDataCache cache;
  ScopedTestEnv remote_mode("NVFUSER_JEV_MODE", "remote");
  ScopedTestEnv invalid_port("NVFUSER_JEV_PORT", "0");
  std::stringstream remote_preview_debug;
  pointwise::tma::TmaBreakPointAvailability remote_preview;
  {
    DebugStreamGuard debug_guard(remote_preview_debug);
    remote_preview = pointwise::tma::getTmaBreakPointAvailability(
        &fusion, runtime_info, *prop);
  }
  ASSERT_EQ(remote_preview.options.size(), preview.options.size());
  ASSERT_EQ(remote_preview.plans.size(), preview.plans.size());
  ASSERT_EQ(
      remote_preview.remote_unavailability_reasons.size(),
      preview.remote_unavailability_reasons.size());
  for (size_t i = 0; i < preview.options.size(); ++i) {
    EXPECT_EQ(remote_preview.options[i].id, preview.options[i].id);
    EXPECT_EQ(
        remote_preview.options[i].description,
        preview.options[i].description);
    EXPECT_EQ(
        remote_preview.plans[i].break_point, preview.plans[i].break_point);
    EXPECT_EQ(
        remote_preview.remote_unavailability_reasons[i],
        preview.remote_unavailability_reasons[i]);
  }
  EXPECT_TRUE(remote_preview_debug.str().empty());
  std::ifstream remote_preview_log(log_path);
  std::stringstream remote_preview_rows;
  remote_preview_rows << remote_preview_log.rdbuf();
  EXPECT_EQ(remote_preview_rows.str(), log_output.str());
  try {
    (void)pointwise::tma::getPointwiseHeuristics(
        &fusion, runtime_info, &cache, *prop);
    FAIL() << "Remote breakpoint Choice unexpectedly accepted port zero";
  } catch (const std::runtime_error& error) {
    EXPECT_THAT(
        error.what(),
        ::testing::HasSubstr("NVFUSER_JEV_PORT must be an integer"));
  }
  EXPECT_FALSE(cache.hasEntry(
      HeuristicCompileTime::ResizeVectorizationFactors::EntryType));
}

TEST_F(PointwiseTest, BackendParentFirstRouting) {
  using pointwise::BackendRouteAction;
  using scheduler::jev::Mode;
  const auto route = pointwise::pointwiseBackendRoute;
  EXPECT_EQ(
      route(Mode::Record, true, true, true, true),
      BackendRouteAction::OfferChoice);
  EXPECT_EQ(
      route(Mode::Record, true, false, true, true),
      BackendRouteAction::PreserveNativeTma);
  EXPECT_EQ(
      route(Mode::Record, true, false, false, true),
      BackendRouteAction::PreserveNativeTma);
  EXPECT_EQ(
      route(Mode::Remote, true, false, true, true),
      BackendRouteAction::OfferChoice);
  EXPECT_EQ(
      route(Mode::Remote, true, false, false, true),
      BackendRouteAction::OrdinaryOnly);
  EXPECT_EQ(
      route(Mode::Record, false, false, true, true),
      BackendRouteAction::PreserveNativeOrdinary);
  EXPECT_EQ(
      route(Mode::Record, false, false, false, true),
      BackendRouteAction::OrdinaryOnly);
  EXPECT_EQ(
      route(Mode::Record, false, false, true, false),
      BackendRouteAction::NoPlan);
  EXPECT_EQ(
      route(Mode::Remote, false, false, true, false),
      BackendRouteAction::TmaOnly);
}

// The SM89 host can construct both source baselines and the exact PW1 wire
// without claiming that a Hopper-only TMA kernel executed there.
TEST_F(PointwiseTest, BackendParentFirstV2Request) {
  char local_path[] = "/tmp/nvfuser-pw1-v2-XXXXXX";
  const char* log_path = std::getenv("NVFUSER_JEV_PW1_FIXTURE_LOG");
  bool cleanup = false;
  if (log_path == nullptr || *log_path == '\0') {
    const int fd = ::mkstemp(local_path);
    ASSERT_GE(fd, 0);
    ::close(fd);
    log_path = local_path;
    cleanup = true;
  }
  struct FileCleanup {
    const char* path;
    bool cleanup;
    ~FileCleanup() {
      if (cleanup) {
        ::unlink(path);
      }
    }
  } file_cleanup{log_path, cleanup};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv protocol("NVFUSER_JEV_PROTOCOL", "text-v1");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::empty(
      {32, 4096}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  const auto prop = pointwise_utils::getFusionRuntimeProperties(
      &fusion, runtime_info, nullptr);
  ASSERT_TRUE(prop.has_value());

  std::unique_ptr<PointwiseParams> tma_baseline;
  std::unique_ptr<PointwiseParams> ordinary_baseline;
  pointwise::tma::TmaBreakPointAvailability remote_tma;
  {
    scheduler::jev::ScopedNativeDecisions native_scope;
    tma_baseline = pointwise::tma::getNativePointwiseHeuristicsPreview(
        &fusion, runtime_info, *prop);
    auto materialized = pointwise::tma::getTmaBreakPointAvailability(
        &fusion, runtime_info, *prop);
    remote_tma = pointwise::tma::remoteRouteableTmaBreakPoints(
        materialized,
        prop->n_elems,
        prop->elem_counts.size(),
        prop->min_dtype_size_bit_for_vectorization);
    ordinary_baseline = pointwise::non_tma::getPointwiseHeuristics(
        &fusion, runtime_info, nullptr, *prop, /*emit_debug=*/false);
  }
  ASSERT_NE(tma_baseline, nullptr);
  ASSERT_NE(ordinary_baseline, nullptr);
  ASSERT_EQ(remote_tma.options.size(), 2);
  EXPECT_EQ(
      pointwise::choosePointwiseBackend(
          *prop,
          remote_tma,
          *tma_baseline,
          "source_native_projection",
          *ordinary_baseline,
          "tma"),
      "tma");
  std::ifstream request_log(log_path);
  std::stringstream rows;
  rows << request_log.rdbuf();
  EXPECT_THAT(rows.str(), ::testing::HasSubstr("\"site\":\"PW1\""));
  EXPECT_THAT(rows.str(), ::testing::HasSubstr("pointwise.tma_preference.v2"));
  EXPECT_THAT(
      rows.str(), ::testing::HasSubstr("tma_remote_breakpoints=bp0,bp1"));
  EXPECT_THAT(
      rows.str(), ::testing::HasSubstr("ordinary_baseline_available=1"));
  EXPECT_THAT(
      rows.str(), ::testing::HasSubstr("\"candidates\":[\"tma\",\"ordinary\"]"));
  EXPECT_THAT(
      rows.str(),
      ::testing::Not(::testing::HasSubstr("\"site\":\"PW4.")));
  EXPECT_THAT(
      rows.str(),
      ::testing::Not(::testing::HasSubstr("\"site\":\"PW3.")));

  char v3_path[] = "/tmp/nvfuser-pw1-v3-XXXXXX";
  const char* v3_log_path = std::getenv("NVFUSER_JEV_PW1_V3_FIXTURE_LOG");
  bool cleanup_v3 = false;
  if (v3_log_path == nullptr || *v3_log_path == '\0') {
    const int v3_fd = ::mkstemp(v3_path);
    ASSERT_GE(v3_fd, 0);
    ::close(v3_fd);
    v3_log_path = v3_path;
    cleanup_v3 = true;
  }
  struct V3Cleanup {
    const char* path;
    bool cleanup;
    ~V3Cleanup() {
      if (cleanup) {
        ::unlink(path);
        ::unlink((std::string(path) + ".outcome.jsonl").c_str());
      }
    }
  } v3_cleanup{v3_log_path, cleanup_v3};
  {
    ScopedTestEnv v3_protocol("NVFUSER_JEV_PROTOCOL", "semantic-v3");
    ScopedTestEnv v3_log("NVFUSER_JEV_LOG", v3_log_path);
    const std::string hardware = "{\"sm_count\":" +
        std::to_string(prop->device_multiprocessor_count) + "}";
    scheduler::jev::ComputeHeuristicsScope scope(
        "source-only TMA and ordinary baseline projections", {}, hardware);
    EXPECT_EQ(
        pointwise::choosePointwiseBackend(
            *prop,
            remote_tma,
            *tma_baseline,
            "source_native_projection",
            *ordinary_baseline,
            "tma"),
        "tma");
    scope.complete("PointWise");
  }
  std::ifstream v3_request(v3_log_path);
  std::stringstream v3_rows;
  v3_rows << v3_request.rdbuf();
  EXPECT_THAT(v3_rows.str(), ::testing::HasSubstr("\"consequence\""));
  EXPECT_THAT(
      v3_rows.str(), ::testing::HasSubstr("\"baseline_projection\""));
  EXPECT_THAT(
      v3_rows.str(),
      ::testing::HasSubstr(
          "\"baseline_kind\":\"source_native_projection\""));
  EXPECT_THAT(v3_rows.str(), ::testing::HasSubstr("\"pending_choices\""));
  EXPECT_THAT(
      v3_rows.str(),
      ::testing::HasSubstr("\"pending_choices_kind\":\"possible_descendants\""));
  EXPECT_THAT(v3_rows.str(), ::testing::HasSubstr("PW4.breakpoint"));
  EXPECT_THAT(v3_rows.str(), ::testing::HasSubstr("PW3.unroll"));
}

TEST_F(PointwiseTest, BackendParentFirstOrdinaryNumeric_CUDA) {
  if (at::cuda::getCurrentDeviceProperties()->major >= 9) {
    GTEST_SKIP() << "The ordinary-only control requires a pre-Hopper GPU";
  }
  char log_path[] = "/tmp/nvfuser-pw1-ordinary-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
      ::unlink((std::string(path) + ".outcome.jsonl").c_str());
    }
  } file_cleanup{log_path};
  EnableOptionsGuard::getCurOptions().set(EnableOption::TmaPointwise);
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(2);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::randn(
      {32, 4096}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  std::unique_ptr<HeuristicParams> off_params;
  {
    ScopedTestEnv off_mode("NVFUSER_JEV_MODE", "off");
    auto scheduler =
        SchedulerEntry::makeSchedulerInstance(SchedulerType::PointWise);
    off_params = scheduler->computeHeuristics(&fusion, runtime_info);
  }
  auto result = scheduleAndRun(&fusion, SchedulerType::PointWise, {tensor});
  ASSERT_NE(off_params, nullptr);
  ASSERT_NE(result.heuristic_params, nullptr);
  const auto* recorded = result.heuristic_params->as<PointwiseParams>();
  const auto* off = off_params->as<PointwiseParams>();
  EXPECT_FALSE(recorded->use_tma_load);
  EXPECT_TRUE(recorded->sameAs(off));
  for (const auto parallel_type : {
           ParallelType::BIDx,
           ParallelType::BIDy,
           ParallelType::BIDz,
           ParallelType::TIDx,
           ParallelType::TIDy,
           ParallelType::TIDz}) {
    EXPECT_EQ(
        recorded->lparams.getRawVal(parallel_type),
        off->lparams.getRawVal(parallel_type));
  }
  testValidate(
      &fusion,
      result.outputs,
      {tensor},
      {tensor + tensor},
      __LINE__,
      __FILE__);

  std::ifstream request_log(log_path);
  std::stringstream requests;
  requests << request_log.rdbuf();
  EXPECT_THAT(requests.str(), ::testing::HasSubstr("\"site\":\"PW2\""));
  EXPECT_THAT(
      requests.str(),
      ::testing::Not(::testing::HasSubstr("\"site\":\"PW1\"")));
  std::ifstream outcome_log(std::string(log_path) + ".outcome.jsonl");
  std::stringstream outcomes;
  outcomes << outcome_log.rdbuf();
  EXPECT_THAT(
      outcomes.str(), ::testing::HasSubstr("\"site\":\"PW1\""));
  EXPECT_THAT(outcomes.str(), ::testing::HasSubstr("\"kind\":\"no_choice\""));
}

// Inject route operands above the bp0 divisor-replay cap without allocating a
// terabyte tensor. This exercises the same filter used by live PW4.breakpoint
// and retains an exact v2 request for the matching server parser.
TEST_F(PointwiseTest, TmaBreakpointRemoteMenuFiltering) {
  constexpr int64_t right_elems = 1LL << 38;
  constexpr int64_t n_elems = 5 * right_elems;
  pointwise::tma::TmaBreakPointAvailability materialized;
  materialized.state =
      "pointwise.tma_breakpoint.v1\n"
      "elements=5," +
      std::to_string(right_elems) +
      "\ndisjoint=1,1\nlhs_bit_multiples=0,32\nrhs_bit_multiples=32,32\n";
  materialized.options = {
      {"bp0", "Flatten to the one-dimensional TMA domain"},
      {"bp1", "Split before loop axis 1"}};
  materialized.plans = {{}, {1, false, right_elems, false}};
  materialized.materialized_layout_widths = {4, 4};
  for (const auto& plan : materialized.plans) {
    materialized.remote_unavailability_reasons.emplace_back(
        pointwise::tma::tmaBreakPointRemoteRouteabilityReason(
            2, n_elems, plan.break_point, 4, 32));
  }
  const auto filtered = pointwise::tma::remoteRouteableTmaBreakPoints(
      materialized, n_elems, 2, 32);
  EXPECT_TRUE(filtered.remote_filter_applied);
  ASSERT_EQ(filtered.options.size(), 1);
  EXPECT_EQ(filtered.options[0].id, "bp1");
  EXPECT_EQ(filtered.options[0].description, "Split before loop axis 1");
  ASSERT_EQ(filtered.plans.size(), 1);
  EXPECT_EQ(filtered.plans[0].break_point, 1);
  EXPECT_EQ(
      pointwise::tma::tmaRecordBreakPointCoverageReason(
          materialized, filtered, "bp0"),
      "native TMA breakpoint has no Remote-reachable PW4 continuation");
  EXPECT_TRUE(
      pointwise::tma::tmaRecordBreakPointCoverageReason(
          materialized, filtered, "bp1")
          .empty());
  EXPECT_THAT(
      filtered.state,
      ::testing::HasSubstr(
          "pointwise.tma_breakpoint.v2\nelements=5," +
          std::to_string(right_elems)));
  EXPECT_THAT(
      filtered.state,
      ::testing::HasSubstr(
          "n_elems=" + std::to_string(n_elems) +
          "\nloop_axes=2\nmin_dtype_bits=32\n"
          "materialized_layout_widths=bp0:4,bp1:4\n"
          "remote_max_loop_axes=32\n"
          "bp0_replay_max_elems=1099511627776\nwidth_max=128\n"));

  auto all_excluded = materialized;
  for (auto& reason : all_excluded.remote_unavailability_reasons) {
    reason = "more than 32 loop dimensions";
  }
  const auto empty_menu = pointwise::tma::remoteRouteableTmaBreakPoints(
      all_excluded, n_elems, 33, 32);
  EXPECT_TRUE(empty_menu.remote_filter_applied);
  EXPECT_TRUE(empty_menu.options.empty());
  EXPECT_TRUE(empty_menu.plans.empty());
  EXPECT_EQ(
      pointwise::tma::tmaRecordBreakPointCoverageReason(
          materialized, empty_menu, "bp1"),
      "native TMA breakpoint has no Remote-reachable PW4 continuation");

  char local_path[] = "/tmp/nvfuser-pw4-route-XXXXXX";
  const char* log_path = std::getenv("NVFUSER_JEV_ROUTE_FIXTURE_LOG");
  bool cleanup = false;
  if (log_path == nullptr || *log_path == '\0') {
    const int fd = ::mkstemp(local_path);
    ASSERT_GE(fd, 0);
    ::close(fd);
    log_path = local_path;
    cleanup = true;
  }
  struct FileCleanup {
    const char* path;
    bool cleanup;
    ~FileCleanup() {
      if (cleanup) {
        ::unlink(path);
      }
    }
  } file_cleanup{log_path, cleanup};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv protocol("NVFUSER_JEV_PROTOCOL", "text-v1");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);
  EXPECT_EQ(
      scheduler::jev::chooseDecision(
          "PW4.breakpoint", filtered.state, filtered.options, "bp1"),
      "bp1");
  std::ifstream request_log(log_path);
  std::stringstream records;
  records << request_log.rdbuf();
  EXPECT_THAT(records.str(), ::testing::HasSubstr("pointwise.tma_breakpoint.v2"));
  EXPECT_THAT(
      records.str(),
      ::testing::HasSubstr("\"candidates\":[\"bp1\"]"));
}

// Exercise the 1D target menu without compiling a Hopper-only TMA kernel.
TEST_F(PointwiseTest, TmaDomainTargetCompletePlanAnalysis) {
  char log_path[] = "/tmp/nvfuser-pw4-domain-XXXXXX";
  const int fd = ::mkstemp(log_path);
  ASSERT_GE(fd, 0);
  ::close(fd);
  struct FileCleanup {
    const char* path;
    ~FileCleanup() {
      ::unlink(path);
    }
  } file_cleanup{log_path};
  ScopedTestEnv mode("NVFUSER_JEV_MODE", "record");
  ScopedTestEnv log("NVFUSER_JEV_LOG", log_path);

  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(1);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::empty(
      {131072}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  const auto prop = pointwise_utils::getFusionRuntimeProperties(
      &fusion, runtime_info, nullptr);
  ASSERT_TRUE(prop.has_value());

  const auto params = pointwise::tma::getPointwiseHeuristics(
      &fusion, runtime_info, nullptr, *prop);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->break_point, 0);
  EXPECT_EQ(params->tma_domain_inner, 512);

  std::ifstream log_file(log_path);
  std::stringstream log_output;
  log_output << log_file.rdbuf();
  const std::string records = log_output.str();
  const std::string site = "\"site\":\"PW4.domain_inner\"";
  const size_t first = records.find(site);
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(records.find(site, first + site.size()), std::string::npos);
  EXPECT_THAT(records, ::testing::HasSubstr("\"t256\""));
  EXPECT_THAT(records, ::testing::HasSubstr("\"t512\""));
  EXPECT_THAT(records, ::testing::HasSubstr("\"t1024\""));
}

// The external harness runs this same analysis once in record and once with
// the loopback API. No TMA kernel is compiled or launched on pre-Hopper GPUs.
TEST_F(PointwiseTest, TmaDomainTargetExternalModeAnalysis) {
  Fusion fusion;
  FusionGuard fg(&fusion);
  auto* input = makeContigTensor(1);
  fusion.addInput(input);
  fusion.addOutput(add(input, input));
  const auto tensor = at::empty(
      {131072}, at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0));
  SchedulerRuntimeInfo runtime_info(&fusion, {tensor});
  const auto prop = pointwise_utils::getFusionRuntimeProperties(
      &fusion, runtime_info, nullptr);
  ASSERT_TRUE(prop.has_value());

  const auto params = pointwise::tma::getPointwiseHeuristics(
      &fusion, runtime_info, nullptr, *prop);
  ASSERT_NE(params, nullptr);
  EXPECT_EQ(params->break_point, 0);
  EXPECT_EQ(params->tma_domain_inner, 512);
}

// Non-parameterized TMA pointwise test fixture (for TEST_F)
class TmaPointwiseTestF : public PointwiseTest {
 protected:
  void SetUp() override {
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
    PointwiseTest::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::TmaPointwise);
  }
};

// Parameterized TMA pointwise test fixture (for TEST_P)
template <typename ParamType>
class TmaPointwiseTestP : public PointwiseTestP<ParamType> {
 protected:
  void SetUp() override {
    NVFUSER_TEST_CUDA_ARCH_GUARD(9, 0);
    PointwiseTestP<ParamType>::SetUp();
    EnableOptionsGuard::getCurOptions().set(EnableOption::TmaPointwise);
  }
};

// Test scheduling pointwise kernel with 2D TMA tiles.
// dim0: Varied input size in the outermost dimension. Ensure we reject cases
//       that can't use TMA load, e.g., load size is not divisible by 16 bytes.
// ndims: Test 1, 2, and 3 dimensions. Since we always use 2D tiles, we want to
//        make sure we can handle cases with fewer than 2 dimensions and cases
//        with more than 2 dimensions.
// use_tma_store: Test with and without TMA store.
using TmaPointwiseTestParams =
    std::tuple<int64_t, int64_t, bool, bool>; // <dim0, ndims, use_tma_store,
                                              // auto_schedule>

class TmaPointwiseTest : public TmaPointwiseTestP<TmaPointwiseTestParams> {};
TEST_P(TmaPointwiseTest, NoBroadcast) {
  // This is a simple test with contiguous inputs, without broadcast, reshapes,
  // allocation domains, etc. Test that 2D TMA tiles can be used to schedule
  // inputs with different sizes and dimensions. Demonstrates how to use 2D TMA
  // tiles and how to handle cases with and without TMA store.
  auto dtype = DataType::Float;
  int64_t dtype_bytes = dataTypeSizeByte(dtype);
  auto [dim0, ndims, use_tma_store, auto_schedule] = GetParam();
  // test [dim0], [dim0, 2], [dim0, 2, 4]
  std::vector<int64_t> element_at_each_dim(ndims);
  element_at_each_dim[0] = dim0;
  int64_t total_elem_count = dim0;
  for (int64_t i = 1; i < ndims; i++) {
    int64_t dim_i = 1 << i;
    element_at_each_dim[i] = dim_i;
    total_elem_count *= dim_i;
  }
  if (total_elem_count * dtype_bytes % 16 != 0) {
    GTEST_SKIP() << "Total bytes is not divisible by 16, can't use TMA, "
                    "total_elem_count: "
                 << total_elem_count << ", dtype_bytes: " << dtype_bytes;
    return;
  }

  const int64_t min_inner_dim = 2 * 16 / dtype_bytes;
  if (total_elem_count % min_inner_dim != 0 ||
      total_elem_count == min_inner_dim) {
    GTEST_SKIP() << "Total elements is not divisible by min_inner_dim or equal "
                    "to min_inner_dim, can't use TMA, "
                    "total_elem_count: "
                 << total_elem_count << ", min_inner_dim: " << min_inner_dim;
    return;
  }

  auto fusion_ptr = std::make_unique<Fusion>();
  FusionGuard fg(fusion_ptr.get());
  Fusion& fusion = *fusion_ptr;
  auto tv0 = makeContigTensor(ndims);
  fusion.addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion.addOutput(tv1);

  auto options = at::TensorOptions().device(at::kCUDA, 0);
  auto t0 = at::randn(element_at_each_dim, options);

  if (auto_schedule) {
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto out_tensors = executor_cache.runFusionWithInputs({t0});
    // ensure TMA is used
    EXPECT_TRUE(tma_check::hasTmaLoad(executor_cache));
    testValidate(
        executor_cache.fusion(), out_tensors, {t0}, __LINE__, __FILE__);
    return;
  }

  // Create TMA loads from inputs to shared memory
  auto tv0_smem = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_smem->setMemoryType(MemoryType::Shared);

  // Cache loads from shared memory to registers
  auto tv0_regs = tv0_smem->cacheAfter();

  // Output caching: regs -> [smem ->] global memory
  TensorView* tv1_smem = nullptr;
  TensorView* tv1_regs = nullptr;
  if (use_tma_store) {
    // TMA store path: regs -> smem -> global memory (via TMA)
    tv1_smem = tv1->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
    tv1_smem->setMemoryType(MemoryType::Shared);
    tv1_regs = tv1_smem->cacheBefore();
  } else {
    // Regular store path: regs -> global memory (no TMA)
    tv1_regs = tv1->cacheBefore();
  }

  TensorView* reference = tv1;

  // ===== Step 1: Create 2D TMA Domain =====
  // Merge all logical dimensions into one flat domain, then split into 2D
  // structure [tma_domain_outer, tma_domain_inner] for TMA operations.
  // Requirements: Domains must be contiguous and split must be evenly
  // divisible. Transformation: [I0, I1, ...] -> [ALL_DIMS] -> [D0, D1]
  //   where D0 = tma_domain_outer, D1 = tma_domain_inner
  reference->flatten();

  // D1 (tma_domain_inner): Inner dimension size, computed to satisfy TMA
  // constraints
  int64_t D1 = scheduler_utils::getTmaDomainInner(
      total_elem_count, 512, dataTypeSizeBit(dtype));

  // D0 (tma_domain_outer): Outer dimension size (number of "rows")
  int64_t D0 = total_elem_count / D1;

  NVF_ERROR(
      total_elem_count % D1 == 0,
      "TMA domain can only be created with divisible split, D1: ",
      D1,
      " total_elem_count: ",
      total_elem_count);
  reference->split(0, D1);

  // ===== Step 2: Create TMA Tiles Within Domain =====
  // Split the 2D TMA domain into tiles that define the box size loaded by each
  // TMA operation. Using dense tiles (box ≡ tile).
  // Transformation: [D0, D1] -> [D0/to, to, D1/ti, ti]
  //   where to = outer tile size, ti = inner tile size
  //
  // Constraint: D1/ti > 1 (need at least 2 tiles along inner dimension)
  // If D1/ti = 1, then [to, ti] would collapse into a single TMA dimension,
  // breaking the 2D structure.

  // tma_tile_size: Target total elements per tile (to × ti)
  int64_t tma_tile_size = 4096;

  // ti: Inner tile dimension (max 256 by hardware, must be ≤ D1/2 for 2D
  // structure) to: Outer tile dimension (max 256 by hardware, capped by D0)
  int64_t ti = std::min(256L, std::max(1L, D1 / 2)),
          to = std::min(256L, std::min(tma_tile_size / ti, D0));

  reference->split(0, to); // Split D0 -> [D0/to, to]
  reference->split(2, ti); // Split D1 -> [D1/ti, ti]

  // Step 3: Propagate TMA transformation to all tensors.
  TransformPropagatorWithCheck propagator(reference);
  MaxLogicalDomainInfoSpanningTree(reference).traverse(&propagator);

  // Step 4: Parallelize TMA tensors with block and bulk parallel types.
  // Check grid dimensions and swap parallel types if needed to avoid exceeding
  // the maximum grid dimension limit (65535).
  auto pto = ParallelType::BIDy;
  auto pti = ParallelType::BIDx;
  int64_t gdim_y = ceilDiv(D0, to);
  if (gdim_y > 65535) {
    std::swap(pto, pti);
  }
  std::vector<TensorView*> tma_tvs = {tv0_smem};
  if (use_tma_store) {
    tma_tvs.push_back(tv1);
  }
  for (auto tv : tma_tvs) {
    tv->axis(0)->parallelize(pto);
    tv->axis(1)->parallelize(ParallelType::Bulk);
    tv->axis(2)->parallelize(pti);
    tv->axis(3)->parallelize(ParallelType::Bulk);
  }

  // Step 5: Schedule non-TMA tensors.
  // Calculate vectorization factor based on the inner tile dimension.
  int64_t vect_factor = 1;
  while (ti % (vect_factor * 2) == 0) {
    vect_factor *= 2;
    if (vect_factor == 4) {
      break;
    }
  }
  // Calculate thread block dimensions for non-TMA tensors.
  int64_t tidx = std::min(32L, ti / vect_factor),
          tidy = std::min(128L / tidx, to);
  std::vector<TensorView*> compute_tvs = {tv0_regs, tv1_regs};
  if (use_tma_store) {
    compute_tvs.push_back(tv1_smem);
  } else {
    compute_tvs.push_back(tv1);
  }
  for (auto tv : compute_tvs) {
    // [D0/to, to, D1/ti, ti] -> [D0/to, to/y, y, D1/ti, ti/v/x, x, v]
    tv->split(3, vect_factor);
    tv->split(3, tidx);
    tv->split(1, tidy);
    // Apply block and thread parallelization.
    tv->axis(0)->parallelize(pto);
    tv->axis(2)->parallelize(ParallelType::TIDy);
    tv->axis(3)->parallelize(pti);
    tv->axis(5)->parallelize(ParallelType::TIDx);
    // Vectorize write to shared memory or global memory.
    if (tv == tv0_regs || (!use_tma_store && tv == tv1)) {
      tv->axis(6)->parallelize(ParallelType::Vectorize);
    }
  }

  // Step 6: Inline
  inlineMost();

  KernelExecutor ke;
  ke.compile(&fusion, {t0});
  auto out_tensors = ke.run({t0});
  testValidate(&fusion, out_tensors, {t0}, __LINE__, __FILE__);
}
INSTANTIATE_TEST_SUITE_P(
    ,
    TmaPointwiseTest,
    ::testing::ValuesIn([] {
      // Generate dim0 values
      std::vector<int64_t> dim0_vals(
          Pow2Vals1to1Million.begin(), Pow2Vals1to1Million.end());
      // Add some irregular numbers
      dim0_vals.insert(
          dim0_vals.end(), {1024 * 1024 + 8, 1024 * 1024 + 7, 1023});

      std::vector<TmaPointwiseTestParams> params;
      for (auto dim0 : dim0_vals) {
        for (auto ndims : {1, 2, 3}) {
          // When auto_schedule=true, use_tma_store is ignored, so only test one
          // value
          params.emplace_back(dim0, ndims, false, /*auto_schedule=*/true);
          // When auto_schedule=false, test both use_tma_store values
          params.emplace_back(dim0, ndims, true, /*auto_schedule=*/false);
          params.emplace_back(dim0, ndims, false, /*auto_schedule=*/false);
        }
      }
      return params;
    }()),
    [](const testing::TestParamInfo<TmaPointwiseTestParams>& info) {
      int64_t dim0 = std::get<0>(info.param);
      int64_t ndims = std::get<1>(info.param);
      bool use_tma_store = std::get<2>(info.param);
      bool auto_schedule = std::get<3>(info.param);
      return "dim0_" + std::to_string(dim0) + "_ndim_" + std::to_string(ndims) +
          "_use_tma_store_" + std::to_string(use_tma_store) +
          "_auto_schedule_" + std::to_string(auto_schedule);
    });

// Parameters for TmaPointwiseBcastTest test
// <use_auto_scheduler, tma_inner_bcast, tma_outer_bcast>
using InnerOuterBcastParams = std::tuple<bool, bool, bool>;
class TmaPointwiseBcastTest : public TmaPointwiseTestP<InnerOuterBcastParams> {
};

TEST_P(TmaPointwiseBcastTest, InnerOuterBcast) {
  // Test TMA scheduling with broadcast tensors
  // Fusion: out = tv0 + broadcast(tv1, {false, true}) + broadcast(tv2, {true,
  // false}) tv1 has inner broadcast dimension (broadcasts along inner dim) tv2
  // has outer broadcast dimension (broadcasts along outer dim) Parameters
  // control whether to use auto scheduler and whether to load broadcast inputs
  // via TMA or regular global load (ldg)
  int64_t dim0 = 1024;
  int64_t dim1 = 2048;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, dtype);
  auto tv1 = makeContigTensor(1, dtype);
  auto tv2 = makeContigTensor(1, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = broadcast(tv2, {true, false});
  auto tv5 = add(tv0, tv3);
  auto tv6 = add(tv5, tv4);
  fusion->addOutput(tv6);
  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0}, options);
  auto t2 = at::randn({dim1}, options);

  // Auto scheduler won't use tma load for broadcast tensors.
  // Here we demo how to safely load broadcast tensors with TMA.
  auto [use_auto_scheduler, tma_inner_bcast, tma_outer_bcast] = GetParam();

  if (use_auto_scheduler) {
    FusionExecutorCache executor_cache(std::move(fusion_ptr));
    auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2});
    // ensure TMA is used
    EXPECT_TRUE(tma_check::hasTmaLoad(executor_cache));
    testValidate(
        executor_cache.fusion(), out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
    return;
  }

  // Manual scheduling: Test TMA load/store with broadcast tensors
  // Cache the main input (tv0) and output (tv6) using TMA
  auto tv0_smem = tv0->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv0_smem->setMemoryType(MemoryType::Shared);
  auto tv6_smem = tv6->cacheBefore(LoadStoreOpType::CpAsyncBulkTensorTile);
  tv6_smem->setMemoryType(MemoryType::Shared);
  std::vector<TensorView*> tma_tvs = {tv0_smem, tv6};
  std::vector<TensorView*> ldg_tvs = {};
  std::unordered_set<TensorView*> vectorizable_ldg_tvs = {};

  // Handle broadcast input tv1 (inner broadcast: {false, true})
  // Either use TMA or regular load to global (ldg)
  if (tma_inner_bcast) {
    auto tv1_smem = tv1->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
    tv1_smem->setMemoryType(MemoryType::Shared);
    tma_tvs.push_back(tv1_smem);
  } else {
    auto tv1_regs = tv1->cacheAfter();
    ldg_tvs.push_back(tv1_regs);
  }

  // Handle broadcast input tv2 (outer broadcast: {true, false})
  // Either use TMA or regular load to global (ldg) with vectorization
  if (tma_outer_bcast) {
    auto tv2_smem = tv2->cacheAfter(LoadStoreOpType::CpAsyncBulkTensorTile);
    tv2_smem->setMemoryType(MemoryType::Shared);
    tma_tvs.push_back(tv2_smem);
  } else {
    auto tv2_regs = tv2->cacheAfter();
    ldg_tvs.push_back(tv2_regs);
    vectorizable_ldg_tvs.insert(tv2_regs);
  }

  // Use tv6_smem as reference for transformation propagation
  // Note: We can't use tv6 directly because TMA store operations only have
  // TMA tile domains, not computation domains. tv6_smem has both.
  auto reference_tv = tv6_smem;

  // For 2D scheduling, keep the two dimensions separate (don't flatten)
  // Flattening would would merge broadcast and non-broadcast ids which may
  // lead to incorrect TMA indexing and should be avoided.
  // With the 2D approach, the original two domains are used as TMA domains.
  // The code below is disabled (false condition) to maintain 2D scheduling.
  // Schedule the TMA domain [I0, I1] (not flattened)
  if (/*is_1d_scheduler=*/false) {
    // This branch would flatten to 1D: [I0, I1] -> [I0*I1] -> [Do, Di]
    reference_tv->flatten();
    int64_t tma_domain_size_inner = dim1;
    reference_tv->split(0, tma_domain_size_inner);
  }

  // Schedule the TMA box/tile dimensions
  // Split into outer/inner blocks: [I0, I1] -> [I0/to, to, I1/ti, ti]
  int64_t tma_tile_inner = 256;
  int64_t tma_tile_outer = 8;
  reference_tv->split(1, tma_tile_inner);
  reference_tv->split(0, tma_tile_outer);

  // Propagate TMA tile transformations to all tensors in the fusion
  TransformPropagator propagator(reference_tv);
  MaxLogicalDomainInfoSpanningTree(reference_tv).traverse(&propagator);

  // Parallelize axes for TMA operations
  // Axes 0,2 are block indices (BIDy, BIDx), axes 1,3 are TMA bulk transfer
  // axes
  reference_tv->axis(0)->parallelize(ParallelType::BIDy);
  reference_tv->axis(1)->parallelize(ParallelType::Bulk);
  reference_tv->axis(2)->parallelize(ParallelType::BIDx);
  reference_tv->axis(3)->parallelize(ParallelType::Bulk);
  scheduler_utils::parallelizeAllLike(reference_tv, tma_tvs);
  // Reset bulk axes to serial for non-TMA computation
  reference_tv->axis(1)->parallelize(ParallelType::Serial);
  reference_tv->axis(3)->parallelize(ParallelType::Serial);

  // Schedule computation (non-TMA) dimensions
  // Starting from [Do/to, to, Di/ti, ti], split further for thread parallelism
  // Result: [Do/to, to/y, y, Di/ti, ti/v/x, x, v]
  int64_t opos = 1, ipos = 3, vectorization_factor = 4;
  reference_tv->split(ipos, vectorization_factor);
  reference_tv->split(ipos, /*bdimx=*/32);
  reference_tv->split(opos, /*bdimy=*/4);

  // Propagate computation transformations to non-TMA tensors only
  std::vector<TensorView*> non_tma_tvs =
      ir_utils::allTvsExcept(fusion, {tma_tvs.begin(), tma_tvs.end()});
  TransformPropagator non_tma_propagator(reference_tv);
  SetSelector selector({non_tma_tvs.begin(), non_tma_tvs.end()});
  MaxLogicalDomainInfoSpanningTree(reference_tv, &selector)
      .traverse(&non_tma_propagator);

  // Parallelize computation axes for non-TMA tensors
  // Axis 0: BIDy (block Y), Axis 2: TIDy (thread Y)
  // Axis 3: BIDx (block X), Axis 5: TIDx (thread X)
  reference_tv->axis(0)->parallelize(ParallelType::BIDy);
  reference_tv->axis(2)->parallelize(ParallelType::TIDy);
  reference_tv->axis(3)->parallelize(ParallelType::BIDx);
  reference_tv->axis(5)->parallelize(ParallelType::TIDx);

  scheduler_utils::parallelizeAllLike(reference_tv, non_tma_tvs);

  // Inline all tensors except those using load from global memory (ldg)
  // pre-loading without inlineMost shows higher performance.
  std::vector<TensorView*> non_ldg_tvs =
      ir_utils::allTvsExcept(fusion, {ldg_tvs.begin(), ldg_tvs.end()});
  inlineMost(non_ldg_tvs);
  for (auto ldg_tv : ldg_tvs) {
    if (vectorizable_ldg_tvs.contains(ldg_tv)) {
      int64_t vect_pos = 3;
      ldg_tv->axis(vect_pos)->parallelize(ParallelType::Vectorize);
    }
  }

  KernelExecutor ke;
  ke.compile(fusion, {t0, t1, t2});
  auto out_tensors = ke.run({t0, t1, t2});
  testValidate(fusion, out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    ,
    TmaPointwiseBcastTest,
    ::testing::ValuesIn([] {
      std::vector<InnerOuterBcastParams> params;
      // When use_auto_scheduler=true, other params don't matter
      params.push_back({true, false, false});
      // When use_auto_scheduler=false, test all combinations
      for (bool tma_inner : {false, true}) {
        for (bool tma_outer : {false, true}) {
          params.push_back({false, tma_inner, tma_outer});
        }
      }
      return params;
    }()),
    [](const testing::TestParamInfo<InnerOuterBcastParams>& info) {
      bool use_auto_scheduler = std::get<0>(info.param);
      bool tma_inner_bcast = std::get<1>(info.param);
      bool tma_outer_bcast = std::get<2>(info.param);
      return "use_auto_scheduler_" + std::to_string(use_auto_scheduler) +
          "_tma_inner_bcast_" + std::to_string(tma_inner_bcast) +
          "_tma_outer_bcast_" + std::to_string(tma_outer_bcast);
    });

TEST_F(TmaPointwiseTestF, TmaDomainBroadcast) {
  int64_t dim0 = 1024;
  int64_t dim1 = 64;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, dtype);
  auto tv1 = makeContigTensor(1, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1});
  // ensure TMA is used
  EXPECT_TRUE(tma_check::hasTmaLoad(executor_cache));
  // This fusion with broadcast will use 2D scheduler, the tma domain size is
  // naturally [dim0, dim1]
  EXPECT_EQ(tma_check::getTmaDomainInner(executor_cache), dim1);
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1}, __LINE__, __FILE__);
}

// load of tv0 is not vectorized in non-TMA version, not TMA loaded in TMA
// version.
TEST_F(TmaPointwiseTestF, TmaDomainBroadcastIllegal) {
  int64_t dim0 = 8192;
  int64_t dim1 = 8191;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, dtype);
  auto tv1 = makeContigTensor(1, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = broadcast(tv1, {false, true});
  auto tv3 = add(tv0, tv2);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0}, options);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1});
  EXPECT_FALSE(tma_check::hasTmaLoad(executor_cache));
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(TmaPointwiseTestF, MixedPrecisionBroadcast) {
  int64_t dim0 = 16384;
  int64_t dim1 = 16384;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  // tv0 is loaded with tma
  // tv1 is loaded with vectorization with a factor of 4, 128 bytes
  // tv2 is loaded with vectorization with a factor of 4, 64 bytes
  auto tv0 = makeContigTensor(2, DataType::Float);
  auto tv1 = makeContigTensor(1, DataType::Float);
  auto tv2 = makeContigTensor(1, DataType::BFloat16);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = broadcast(tv1, {false, true});
  auto tv4 = broadcast(tv2, {false, true});
  auto tv5 = castOp(DataType::Float, tv4);
  auto tv6 = add(tv0, tv3);
  auto tv7 = add(tv6, tv5);
  fusion->addOutput(tv7);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto options_bf16 =
      at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0}, options);
  auto t2 = at::randn({dim0}, options_bf16);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2});
  EXPECT_TRUE(tma_check::hasTmaLoad(executor_cache));
  EXPECT_EQ(tma_check::getVectorizationFactor(executor_cache), 4);
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(TmaPointwiseTestF, SplitGridDim1D) {
  maybeClearAllocator(/*max_bytes=*/0);
  int64_t dim0 = 32768;
  int64_t dim1 = 32768;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, DataType::Float);
  fusion->addInput(tv0);
  auto tv1 = add(tv0, tv0);
  fusion->addOutput(tv1);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0});
  EXPECT_TRUE(tma_check::hasTmaLoad(executor_cache));
  testValidate(executor_cache.fusion(), out_tensors, {t0}, __LINE__, __FILE__);
}

TEST_F(TmaPointwiseTestF, SplitGridDim2D) {
  maybeClearAllocator(/*max_bytes=*/0);
  // use a large dim0, ensure it is larger than the max grid y dimension
  // after split by outer tma domain
  const int64_t max_grid_y_dim =
      at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
  int64_t dim0 = max_grid_y_dim * 20;
  int64_t dim1 = 16;
  DataType dtype = DataType::BFloat16;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigTensor(2, dtype);
  auto tv1 = makeContigTensor(1, dtype);
  auto tv2 = makeContigTensor(1, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  auto tv3 = broadcast(tv1, {true, false});
  auto tv4 = broadcast(tv2, {false, true});
  tv0 = maybeCastOp(DataType::Float, tv0);
  tv3 = maybeCastOp(DataType::Float, tv3);
  tv4 = maybeCastOp(DataType::Float, tv4);
  auto tv5 = add(tv0, tv3);
  auto tv6 = add(tv5, tv4);
  tv6 = maybeCastOp(dtype, tv6);
  fusion->addOutput(tv6);

  auto options =
      at::TensorOptions().dtype(data_type_to_aten(dtype)).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim1}, options);
  auto t2 = at::randn({dim0}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2});
  EXPECT_TRUE(tma_check::hasTmaLoad(executor_cache));
  // further clear before testValidate to reduce memory usage
  maybeClearAllocator(/*max_bytes=*/0);
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1, t2}, __LINE__, __FILE__);
}

TEST_F(TmaPointwiseTestF, MixedPrecisionIllegalTma) {
  int64_t dim0 = 16384 + 8;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  // tv1 is suitable for TMA, tv0 is not. Then non-TMA version is used.
  auto tv0 = makeContigTensor(1, DataType::BFloat16);
  auto tv1 = makeContigTensor(1, DataType::Float);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = castOp(DataType::Float, tv0);
  auto tv3 = add(tv2, tv1);
  fusion->addOutput(tv3);

  auto options = at::TensorOptions().dtype(at::kBFloat16).device(at::kCUDA, 0);
  auto options_float =
      at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0}, options);
  auto t1 = at::randn({dim0}, options_float);

  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1});
  EXPECT_FALSE(tma_check::hasTmaLoad(executor_cache));
  testValidate(
      executor_cache.fusion(), out_tensors, {t0, t1}, __LINE__, __FILE__);
}

// input tvs have broadcast dimension, they are not suitable for TMA load.
// outer dimension is the broadcast dimension.
TEST_F(TmaPointwiseTestF, OuterDimOne) {
  int64_t dim1 = 8192;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigConcreteTensor({1, dim1}, dtype);
  auto tv1 = makeContigConcreteTensor({1, dim1}, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({1, dim1}, options);
  auto t1 = at::randn({1, dim1}, options);

  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  EXPECT_FALSE(pparams->use_tma_load);
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

// input tvs have broadcast dimension, they are not suitable for TMA load.
// inner dimension is the broadcast dimension.
TEST_F(TmaPointwiseTestF, InnerDimOne) {
  int64_t dim0 = 8192;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigConcreteTensor({dim0, 1}, dtype);
  auto tv1 = makeContigConcreteTensor({dim0, 1}, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, 1}, options);
  auto t1 = at::randn({dim0, 1}, options);

  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  EXPECT_FALSE(pparams->use_tma_load);
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

// input tvs have broadcast dimension, they are not suitable for TMA load.
// midddle dimension is the broadcast dimension.
TEST_F(TmaPointwiseTestF, MiddleDimOne) {
  int64_t dim0 = 8192;
  int64_t dim2 = 1024;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigConcreteTensor({dim0, 1, dim2}, dtype);
  auto tv1 = makeContigConcreteTensor({dim0, 1, dim2}, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, 1, dim2}, options);
  auto t1 = at::randn({dim0, 1, dim2}, options);

  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  EXPECT_FALSE(pparams->use_tma_load);
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

// tv0 has broadcast dimension, not suitable for TMA load
// tv1 doesn't have broadcast dimension, it is suitable for TMA load.
TEST_F(TmaPointwiseTestF, OneBcastOneNonBcast) {
  int64_t dim0 = 8192;
  int64_t dim2 = 1024;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigConcreteTensor({dim0, 1, dim2}, dtype);
  auto tv1 = makeContigConcreteTensor({dim0, 2, dim2}, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  auto tv2 = add(tv0, tv1);
  fusion->addOutput(tv2);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, 1, dim2}, options);
  auto t1 = at::randn({dim0, 2, dim2}, options);

  auto cg_results = scheduleAndRun(fusion, SchedulerType::PointWise, {t0, t1});
  auto pparams = cg_results.heuristic_params->as<PointwiseParams>();
  EXPECT_TRUE(pparams->use_tma_load);
  testValidate(fusion, cg_results.outputs, {t0, t1}, __LINE__, __FILE__);
}

TEST_F(TmaPointwiseTestF, MultipleInputs) {
  int64_t dim0 = 8192;
  int64_t dim1 = 8192;
  DataType dtype = DataType::Float;
  auto fusion_ptr = std::make_unique<Fusion>();
  auto fusion = fusion_ptr.get();
  FusionGuard fg(fusion);
  auto tv0 = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto tv1 = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto tv2 = makeContigConcreteTensor({dim0, dim1}, dtype);
  auto tv3 = makeContigConcreteTensor({dim0, dim1}, dtype);
  fusion->addInput(tv0);
  fusion->addInput(tv1);
  fusion->addInput(tv2);
  fusion->addInput(tv3);
  auto tv4 = add(tv0, tv1);
  auto tv5 = add(tv2, tv3);
  auto tv6 = add(tv4, tv5);
  fusion->addOutput(tv6);

  auto options = at::TensorOptions().dtype(at::kFloat).device(at::kCUDA, 0);
  auto t0 = at::randn({dim0, dim1}, options);
  auto t1 = at::randn({dim0, dim1}, options);
  auto t2 = at::randn({dim0, dim1}, options);
  auto t3 = at::randn({dim0, dim1}, options);
  FusionExecutorCache executor_cache(std::move(fusion_ptr));
  auto out_tensors = executor_cache.runFusionWithInputs({t0, t1, t2, t3});
  testValidate(
      executor_cache.fusion(),
      out_tensors,
      {t0, t1, t2, t3},
      __LINE__,
      __FILE__);
}

} // namespace nvfuser
