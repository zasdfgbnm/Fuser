// SPDX-License-Identifier: BSD-3-Clause
#include <gtest/gtest.h>

#include "scheduler/transpose_utils.h"

namespace nvfuser {

TEST(TransposeVirtualAxisPreviewTest, OneWaveAndSatisfiedAxesAreNoOps) {
  TransposeParams one_wave;
  transpose::utils::maybeBuildVirtualInnerDims(
      &one_wave, /*sm_count=*/128, /*n_elems=*/512,
      /*shape=*/{2, 16, 16}, /*inner1=*/1, /*inner2=*/2);
  EXPECT_TRUE(one_wave.split_before_tiling.empty());
  EXPECT_TRUE(one_wave.dims_merged_with_1.empty());
  EXPECT_TRUE(one_wave.dims_merged_with_2.empty());

  TransposeParams satisfied_axes;
  transpose::utils::maybeBuildVirtualInnerDims(
      &satisfied_axes, /*sm_count=*/128, /*n_elems=*/64 * 64 * 64,
      /*shape=*/{64, 64, 64}, /*inner1=*/1, /*inner2=*/2);
  EXPECT_TRUE(satisfied_axes.split_before_tiling.empty());
  EXPECT_TRUE(satisfied_axes.dims_merged_with_1.empty());
  EXPECT_TRUE(satisfied_axes.dims_merged_with_2.empty());

  TransposeParams changed_axes;
  transpose::utils::maybeBuildVirtualInnerDims(
      &changed_axes, /*sm_count=*/128, /*n_elems=*/1024 * 2 * 2,
      /*shape=*/{1024, 2, 2}, /*inner1=*/1, /*inner2=*/2);
  EXPECT_EQ(
      changed_axes.split_before_tiling,
      (std::vector<std::pair<int64_t, int64_t>>{{0, 16}}));
  EXPECT_EQ(changed_axes.dims_merged_with_1, (std::vector<int64_t>{1}));
  EXPECT_EQ(changed_axes.dims_merged_with_2, (std::vector<int64_t>{0}));

  TransposeParams reversed_axes;
  transpose::utils::maybeBuildVirtualInnerDims(
      &reversed_axes, /*sm_count=*/128, /*n_elems=*/1024 * 2 * 2,
      /*shape=*/{1024, 2, 2}, /*inner1=*/2, /*inner2=*/1);
  EXPECT_EQ(
      reversed_axes.split_before_tiling,
      (std::vector<std::pair<int64_t, int64_t>>{{0, 16}}));
  EXPECT_EQ(reversed_axes.dims_merged_with_1, (std::vector<int64_t>{0}));
  EXPECT_EQ(reversed_axes.dims_merged_with_2, (std::vector<int64_t>{1}));
}

} // namespace nvfuser
