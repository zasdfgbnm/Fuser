// SPDX-License-Identifier: BSD-3-Clause
// CPU-only test: no nvFuser libraries or CUDA context are needed.
#include <scheduler/bounded_snapshot_stream.h>

#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

} // namespace

int main() {
  using nvfuser::scheduler::compute_invocation_input::BoundedSnapshotBuffer;

  BoundedSnapshotBuffer exact(8);
  std::ostream exact_stream(&exact);
  exact_stream << "abc" << "defgh";
  require(exact_stream.good(), "an exact-limit write must succeed");
  require(!exact.truncated(), "an exact-limit write must not truncate");
  require(exact.view() == "abcdefgh", "an exact-limit write changed bytes");
  exact_stream << 'i';
  require(!exact_stream.good(), "an over-limit write must fail the stream");
  require(exact.truncated(), "an over-limit write must record truncation");
  require(exact.view() == "abcdefgh", "an over-limit write retained a suffix");

  constexpr size_t kLimit = 1024 * 1024;
  BoundedSnapshotBuffer bulk(kLimit);
  std::ostream bulk_stream(&bulk);
  const std::string oversized(kLimit * 4, 'x');
  bulk_stream.write(oversized.data(), oversized.size());
  require(!bulk_stream.good(), "a large bulk write must fail the stream");
  require(bulk.truncated(), "a large bulk write must record truncation");
  require(bulk.view().size() == kLimit, "a large write exceeded the memory cap");
  require(std::move(bulk).take() == oversized.substr(0, kLimit),
          "the bounded buffer changed the accepted prefix");

  BoundedSnapshotBuffer zero(0);
  std::ostream zero_stream(&zero);
  zero_stream << 'x';
  require(!zero_stream.good() && zero.truncated() && zero.view().empty(),
          "a zero-byte limit must reject the first character");
}
