// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

// Passive JSONL output for the modified scheduler's completed parameters.
// Keep the writer independent of CUDA and the Jev client: observation failure
// must not turn a successful computeHeuristics call into a failure.
namespace nvfuser::scheduler::native_diagnostic {

inline constexpr char kLogEnvironment[] = "NVFUSER_JEV_OUTCOME_LOG";
inline constexpr char kDecisionLogEnvironment[] = "NVFUSER_JEV_LOG";
inline constexpr char kModeEnvironment[] = "NVFUSER_JEV_MODE";

inline const char* logMode() noexcept {
  const char* mode = std::getenv(kModeEnvironment);
  return mode != nullptr && mode[0] != '\0' ? mode : "off";
}

inline bool enabled() noexcept {
  const char* path = std::getenv(kLogEnvironment);
  if (path != nullptr && path[0] != '\0') {
    return true;
  }
  const std::string_view mode(logMode());
  const char* decision_path = std::getenv(kDecisionLogEnvironment);
  return (mode == "record" || mode == "remote") &&
      decision_path != nullptr && decision_path[0] != '\0';
}

inline std::string logPath() {
  const char* path = std::getenv(kLogEnvironment);
  if (path != nullptr && path[0] != '\0') {
    return path;
  }
  const char* decision_path = std::getenv(kDecisionLogEnvironment);
  return decision_path == nullptr || decision_path[0] == '\0'
      ? std::string()
      : std::string(decision_path) + ".outcome.jsonl";
}

inline std::string quote(std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string result = "\"";
  for (unsigned char ch : value) {
    switch (ch) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (ch < 0x20) {
          result += "\\u00";
          result += kHex[ch >> 4];
          result += kHex[ch & 0xf];
        } else {
          result += static_cast<char>(ch);
        }
    }
  }
  result += '"';
  return result;
}

class Object {
 public:
  void add(std::string_view key, const char* value) {
    add(key, std::string_view(value));
  }

  void add(std::string_view key, std::string_view value) {
    field(key);
    contents_ += quote(value);
  }

  void add(std::string_view key, int64_t value) {
    field(key);
    contents_ += std::to_string(value);
  }

  void add(std::string_view key, bool value) {
    field(key);
    contents_ += value ? "true" : "false";
  }

  void addRaw(std::string_view key, std::string_view json) {
    field(key);
    contents_ += json;
  }

  std::string str() const {
    return '{' + contents_ + '}';
  }

 private:
  void field(std::string_view key) {
    if (!contents_.empty()) {
      contents_ += ',';
    }
    contents_ += quote(key);
    contents_ += ':';
  }

  std::string contents_;
};

namespace detail {

// Call only while holding an exclusive advisory lock on a regular file.
// The saved EOF lets a failed partial body or failed newline be rolled back.
// Exposed to the small host-only test so both short-write paths are exercised.
template <typename Writer>
bool appendLocked(int fd, std::string_view line, Writer&& writer) noexcept {
  if (line.find('\n') != std::string_view::npos) {
    return false;
  }
  const off_t start = ::lseek(fd, 0, SEEK_END);
  if (start < 0) {
    return false;
  }
  // Do not extend an already incomplete JSONL file. The strict comparison
  // reader rejects it and the caller can retain the original evidence.
  if (start > 0) {
    char last_byte = '\0';
    if (::pread(fd, &last_byte, 1, start - 1) != 1 || last_byte != '\n') {
      return false;
    }
  }
  const auto write_all = [&](const char* data, size_t length) {
    size_t written = 0;
    while (written < length) {
      const auto count = writer(fd, data + written, length - written);
      if (count > 0) {
        written += static_cast<size_t>(count);
      } else if (count < 0 && errno == EINTR) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  };
  if (write_all(line.data(), line.size()) && write_all("\n", 1)) {
    return true;
  }
  (void)::ftruncate(fd, start);
  return false;
}

} // namespace detail

// Diagnostics must never change a schedule or throw from computeHeuristics.
// An advisory lock and rollback preserve line boundaries across cooperating
// benchmark processes. A nonregular file or failed write is ignored.
inline void appendLine(std::string_view line) noexcept {
  if (!enabled()) {
    return;
  }
  try {
    const std::string path = logPath();
    if (path.empty()) {
      return;
    }
    int fd = ::open(
        path.c_str(),
        O_RDWR | O_CREAT | O_APPEND | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK,
        0600);
    if (fd < 0) {
      return;
    }
    struct stat info {};
    if (::fstat(fd, &info) != 0 || !S_ISREG(info.st_mode)) {
      ::close(fd);
      return;
    }
    if (::flock(fd, LOCK_EX) != 0) {
      ::close(fd);
      return;
    }
    (void)detail::appendLocked(fd, line, ::write);
    (void)::flock(fd, LOCK_UN);
    ::close(fd);
  } catch (...) {
    // Even a failed path allocation cannot affect scheduler results.
  }
}

} // namespace nvfuser::scheduler::native_diagnostic
