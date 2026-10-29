// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include <streambuf>
#include <string>
#include <string_view>
#include <utility>

namespace nvfuser::scheduler::compute_invocation_input {

// Keep diagnostic text bounded while a producer writes to an ostream. A write
// past the limit fails the stream and records truncation; callers must discard
// the partial text rather than use it as an invocation identity.
class BoundedSnapshotBuffer final : public std::streambuf {
 public:
  explicit BoundedSnapshotBuffer(size_t max_bytes) : max_bytes_(max_bytes) {}

  bool truncated() const noexcept {
    return truncated_;
  }

  std::string_view view() const noexcept {
    return text_;
  }

  std::string take() && noexcept {
    return std::move(text_);
  }

 protected:
  std::streamsize xsputn(const char* bytes, std::streamsize count) override {
    if (count <= 0) {
      return 0;
    }
    const size_t requested = static_cast<size_t>(count);
    const size_t remaining = max_bytes_ - text_.size();
    const size_t written = requested < remaining ? requested : remaining;
    text_.append(bytes, written);
    truncated_ |= written != requested;
    return static_cast<std::streamsize>(written);
  }

  int_type overflow(int_type ch) override {
    if (traits_type::eq_int_type(ch, traits_type::eof())) {
      return traits_type::not_eof(ch);
    }
    if (text_.size() == max_bytes_) {
      truncated_ = true;
      return traits_type::eof();
    }
    text_.push_back(traits_type::to_char_type(ch));
    return ch;
  }

  int sync() override {
    return 0;
  }

 private:
  const size_t max_bytes_;
  std::string text_;
  bool truncated_ = false;
};

} // namespace nvfuser::scheduler::compute_invocation_input
