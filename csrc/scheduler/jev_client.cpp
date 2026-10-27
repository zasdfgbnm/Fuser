// SPDX-License-Identifier: BSD-3-Clause
#include <scheduler/jev_client.h>

#include <scheduler/native_diagnostic_jsonl.h>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace nvfuser::scheduler::jev {
namespace {

constexpr size_t kMaxRequest = 64 * 1024;
constexpr size_t kMaxSemanticV3Request = 8 * 1024 * 1024;
constexpr size_t kMaxResponse = 64 * 1024;
constexpr size_t kMaxHeaders = 8 * 1024;
constexpr auto kTimeout = std::chrono::milliseconds(2000);
constexpr unsigned kMaxTimeoutMs = 600000;
constexpr std::string_view kPointwiseOutcomeSchema =
    "nvfuser.jev.pointwise.outcome.v1";
constexpr std::string_view kComputeInvocationSchema =
    "nvfuser.jev.compute_invocation.v1";

thread_local ComputeHeuristicsScope* active_compute_scope = nullptr;
thread_local bool native_decisions_forced = false;
std::atomic<uint64_t> next_compute_invocation{0};

struct ActiveInvocation {
  std::string id;
  std::string case_key;
  uint64_t event_index = 0;
  bool completed = false;
};
thread_local ActiveInvocation active_invocation;
std::atomic<uint64_t> next_invocation{0};

[[noreturn]] void fail(std::string_view detail) {
  throw std::runtime_error("nvFuser Jev client: " + std::string(detail));
}

std::string env(std::string_view name) {
  // All callers use static, NUL-terminated names.
  const char* value = std::getenv(name.data());
  return value == nullptr ? std::string() : std::string(value);
}

std::string quote(std::string_view value) {
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 2);
  result += '"';
  for (unsigned char c : value) {
    switch (c) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (c < 0x20) {
          result += "\\u00";
          result += hex[c >> 4];
          result += hex[c & 15];
        } else {
          result += static_cast<char>(c);
        }
    }
  }
  result += '"';
  return result;
}

// A small complete JSON reader keeps the experimental client independent of
// Python/Torch serialization and rejects malformed or ambiguous API replies.
struct Json {
  enum class Kind { Null, Boolean, Number, String, Object, Array } kind;
  std::string text;
  std::map<std::string, Json> object;
  std::vector<Json> array;

  const Json& at(std::string_view key) const {
    if (kind != Kind::Object) fail("expected JSON object");
    auto it = object.find(std::string(key));
    if (it == object.end()) fail("missing JSON field " + std::string(key));
    return it->second;
  }
  std::string_view asString() const {
    if (kind != Kind::String) fail("expected JSON string");
    return text;
  }
};

class JsonReader {
 public:
  explicit JsonReader(std::string_view input) : input_(input) {}

  Json parse() {
    Json result = value(0);
    space();
    if (pos_ != input_.size()) fail("trailing JSON data");
    return result;
  }

 private:
  std::string_view input_;
  size_t pos_ = 0;

  void space() {
    while (pos_ < input_.size() &&
           (input_[pos_] == ' ' || input_[pos_] == '\n' ||
            input_[pos_] == '\r' || input_[pos_] == '\t')) ++pos_;
  }
  bool take(char c) {
    space();
    if (pos_ < input_.size() && input_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!take(c)) fail("invalid JSON syntax");
  }
  bool literal(std::string_view token) {
    if (input_.substr(pos_, token.size()) != token) return false;
    pos_ += token.size();
    return true;
  }
  static int hexDigit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    fail("invalid JSON Unicode escape");
  }
  unsigned codeUnit() {
    if (input_.size() - pos_ < 4) fail("short JSON Unicode escape");
    unsigned unit = 0;
    for (int i = 0; i < 4; ++i) unit = unit * 16 + hexDigit(input_[pos_++]);
    return unit;
  }
  static void appendUtf8(std::string& out, unsigned codepoint) {
    if (codepoint <= 0x7f) out += static_cast<char>(codepoint);
    else if (codepoint <= 0x7ff) {
      out += static_cast<char>(0xc0 | (codepoint >> 6));
      out += static_cast<char>(0x80 | (codepoint & 63));
    } else if (codepoint <= 0xffff) {
      out += static_cast<char>(0xe0 | (codepoint >> 12));
      out += static_cast<char>(0x80 | ((codepoint >> 6) & 63));
      out += static_cast<char>(0x80 | (codepoint & 63));
    } else {
      out += static_cast<char>(0xf0 | (codepoint >> 18));
      out += static_cast<char>(0x80 | ((codepoint >> 12) & 63));
      out += static_cast<char>(0x80 | ((codepoint >> 6) & 63));
      out += static_cast<char>(0x80 | (codepoint & 63));
    }
  }
  std::string string() {
    expect('"');
    std::string out;
    while (pos_ < input_.size()) {
      unsigned char c = input_[pos_++];
      if (c == '"') return out;
      if (c < 0x20) fail("unescaped JSON control byte");
      if (c != '\\') { out += static_cast<char>(c); continue; }
      if (pos_ == input_.size()) fail("short JSON escape");
      char escaped = input_[pos_++];
      switch (escaped) {
        case '"': case '\\': case '/': out += escaped; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
          unsigned point = codeUnit();
          if (point >= 0xd800 && point <= 0xdbff) {
            if (input_.substr(pos_, 2) != "\\u") fail("missing JSON low surrogate");
            pos_ += 2;
            unsigned low = codeUnit();
            if (low < 0xdc00 || low > 0xdfff) fail("invalid JSON low surrogate");
            point = 0x10000 + ((point - 0xd800) << 10) + low - 0xdc00;
          } else if (point >= 0xdc00 && point <= 0xdfff) {
            fail("unpaired JSON low surrogate");
          }
          appendUtf8(out, point);
          break;
        }
        default: fail("invalid JSON escape");
      }
    }
    fail("unterminated JSON string");
  }
  Json number() {
    const size_t start = pos_;
    if (input_[pos_] == '-') ++pos_;
    if (pos_ == input_.size()) fail("short JSON number");
    if (input_[pos_] == '0') ++pos_;
    else {
      if (input_[pos_] < '1' || input_[pos_] > '9') fail("invalid JSON number");
      while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
    }
    if (pos_ < input_.size() && input_[pos_] == '.') {
      ++pos_;
      const size_t begin = pos_;
      while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
      if (pos_ == begin) fail("invalid JSON fraction");
    }
    if (pos_ < input_.size() && (input_[pos_] == 'e' || input_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < input_.size() && (input_[pos_] == '+' || input_[pos_] == '-')) ++pos_;
      const size_t begin = pos_;
      while (pos_ < input_.size() && input_[pos_] >= '0' && input_[pos_] <= '9') ++pos_;
      if (pos_ == begin) fail("invalid JSON exponent");
    }
    return {Json::Kind::Number, std::string(input_.substr(start, pos_ - start)), {}, {}};
  }
  Json value(int depth) {
    if (depth > 32) fail("JSON nesting exceeds limit");
    space();
    if (pos_ == input_.size()) fail("missing JSON value");
    if (take('{')) {
      Json result{Json::Kind::Object, {}, {}, {}};
      if (take('}')) return result;
      do {
        space();
        if (pos_ == input_.size() || input_[pos_] != '"') fail("invalid JSON object key");
        std::string key = string();
        expect(':');
        if (!result.object.emplace(std::move(key), value(depth + 1)).second)
          fail("duplicate JSON object key");
      } while (take(','));
      expect('}');
      return result;
    }
    if (take('[')) {
      Json result{Json::Kind::Array, {}, {}, {}};
      if (take(']')) return result;
      do { result.array.push_back(value(depth + 1)); } while (take(','));
      expect(']');
      return result;
    }
    if (input_[pos_] == '"') return {Json::Kind::String, string(), {}, {}};
    if (literal("true")) return {Json::Kind::Boolean, "true", {}, {}};
    if (literal("false")) return {Json::Kind::Boolean, "false", {}, {}};
    if (literal("null")) return {Json::Kind::Null, {}, {}, {}};
    if (input_[pos_] == '-' || (input_[pos_] >= '0' && input_[pos_] <= '9')) return number();
    fail("invalid JSON value");
  }
};

class Fd {
 public:
  explicit Fd(int fd) : fd_(fd) { if (fd_ < 0) fail(std::strerror(errno)); }
  ~Fd() { if (fd_ >= 0) ::close(fd_); }
  Fd(const Fd&) = delete;
  Fd& operator=(const Fd&) = delete;
  int get() const { return fd_; }
 private:
  int fd_;
};

using Clock = std::chrono::steady_clock;
void ready(int fd, short event, Clock::time_point deadline) {
  while (true) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
    if (remaining <= 0) fail("loopback API timeout");
    pollfd pfd{fd, event, 0};
    int result = ::poll(&pfd, 1, static_cast<int>(remaining));
    if (result > 0) {
      if (pfd.revents & POLLNVAL) fail("invalid loopback socket");
      if (pfd.revents & (event | POLLERR | POLLHUP)) return;
      fail("unexpected loopback socket event");
    }
    if (result == 0) fail("loopback API timeout");
    if (errno != EINTR) fail(std::strerror(errno));
  }
}

unsigned portFromEnv() {
  const std::string text = env("NVFUSER_JEV_PORT");
  if (text.empty()) return 8765;
  unsigned port = 0;
  auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), port);
  if (error != std::errc{} || end != text.data() + text.size() || port == 0 || port > 65535)
    fail("NVFUSER_JEV_PORT must be an integer in 1..65535");
  return port;
}

std::chrono::milliseconds timeoutFromEnv() {
  const std::string text = env("NVFUSER_JEV_TIMEOUT_MS");
  if (text.empty()) return kTimeout;
  unsigned timeout_ms = 0;
  auto [end, error] = std::from_chars(
      text.data(), text.data() + text.size(), timeout_ms);
  if (error != std::errc{} || end != text.data() + text.size() ||
      timeout_ms == 0 || timeout_ms > kMaxTimeoutMs) {
    fail("NVFUSER_JEV_TIMEOUT_MS must be an integer in 1..600000");
  }
  return std::chrono::milliseconds(timeout_ms);
}

std::string post(std::string_view body, size_t max_request) {
  if (body.size() > max_request) fail("request exceeds protocol size limit");
  Fd socket(::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0));
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<uint16_t>(portFromEnv()));
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const auto deadline = Clock::now() + timeoutFromEnv();
  int rc = ::connect(socket.get(), reinterpret_cast<sockaddr*>(&address), sizeof(address));
  if (rc < 0 && errno != EINPROGRESS) fail(std::strerror(errno));
  if (rc < 0) {
    ready(socket.get(), POLLOUT, deadline);
    int error = 0;
    socklen_t length = sizeof(error);
    if (::getsockopt(socket.get(), SOL_SOCKET, SO_ERROR, &error, &length) < 0 || error != 0)
      fail(error == 0 ? std::strerror(errno) : std::strerror(error));
  }
  const std::string request =
      "POST /v1/systemone HTTP/1.0\r\nHost: 127.0.0.1\r\n"
      "Content-Type: application/json\r\nContent-Length: " + std::to_string(body.size()) +
      "\r\nConnection: close\r\n\r\n" + std::string(body);
  size_t sent = 0;
  while (sent < request.size()) {
    ready(socket.get(), POLLOUT, deadline);
    const ssize_t count = ::send(socket.get(), request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
    if (count > 0) sent += static_cast<size_t>(count);
    else if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    else fail("loopback API send failed");
  }
  std::string response;
  std::array<char, 4096> buffer{};
  size_t header_end = std::string::npos;
  size_t body_size = 0;
  while (true) {
    if (header_end != std::string::npos && response.size() >= header_end + body_size)
      break;
    ready(socket.get(), POLLIN, deadline);
    ssize_t count = ::recv(socket.get(), buffer.data(), buffer.size(), 0);
    if (count == 0) fail("truncated loopback API response");
    if (count < 0 && (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)) continue;
    if (count < 0) fail(std::strerror(errno));
    response.append(buffer.data(), static_cast<size_t>(count));
    if (response.size() > kMaxHeaders + kMaxResponse) fail("loopback API response too large");
    if (header_end == std::string::npos) {
      size_t split = response.find("\r\n\r\n");
      if (split == std::string::npos) {
        if (response.size() > kMaxHeaders) fail("loopback API headers too large");
        continue;
      }
      if (split > kMaxHeaders) fail("loopback API headers too large");
      header_end = split + 4;
      const size_t first_line_end = response.find("\r\n");
      const std::string status = response.substr(0, first_line_end);
      if (status != "HTTP/1.0 200 OK" && status != "HTTP/1.1 200 OK")
        fail("loopback API returned " + status);
      bool has_length = false;
      bool has_type = false;
      size_t offset = first_line_end + 2;
      while (offset < split) {
        size_t end = response.find("\r\n", offset);
        if (end == std::string::npos || end > split) fail("malformed HTTP header");
        std::string line = response.substr(offset, end - offset);
        std::transform(line.begin(), line.end(), line.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (line.rfind("content-length: ", 0) == 0) {
          if (has_length) fail("duplicate Content-Length");
          std::string_view value(line.data() + 16, line.size() - 16);
          auto [ptr, error] = std::from_chars(value.data(), value.data() + value.size(), body_size);
          if (error != std::errc{} || ptr != value.data() + value.size() || body_size > kMaxResponse)
            fail("invalid Content-Length");
          has_length = true;
        } else if (line == "content-type: application/json") {
          if (has_type) fail("duplicate Content-Type");
          has_type = true;
        } else if (line.rfind("transfer-encoding:", 0) == 0) {
          fail("chunked loopback API response unsupported");
        }
        offset = end + 2;
      }
      if (!has_length || !has_type) fail("missing JSON HTTP headers");
    }
  }
  if (response.size() != header_end + body_size) fail("extra loopback API response bytes");
  return response.substr(header_end, body_size);
}

void appendLineToPath(const std::string& path, std::string_view line) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> guard(mutex);
  // Resolve the final component without following a symlink. O_NONBLOCK also
  // prevents a named pipe supplied as the log path from stalling the caller.
  Fd file(::open(path.c_str(),
                 O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC | O_NOFOLLOW |
                     O_NONBLOCK,
                 0644));
  struct stat file_stat {};
  if (::fstat(file.get(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode))
    fail("NVFUSER_JEV_LOG must be a regular file");
  std::string record(line);
  record += '\n';
  size_t written = 0;
  while (written < record.size()) {
    ssize_t count = ::write(file.get(), record.data() + written, record.size() - written);
    if (count > 0) written += static_cast<size_t>(count);
    else if (count < 0 && errno == EINTR) continue;
    else fail("failed to write NVFUSER_JEV_LOG");
  }
}

void appendLog(std::string_view line) {
  const std::string path = env("NVFUSER_JEV_LOG");
  if (path.empty()) fail("NVFUSER_JEV_LOG is required in record/remote mode");
  appendLineToPath(path, line);
}

void appendPointwiseOutcome(std::string_view line) {
  std::string path = env("NVFUSER_JEV_POINTWISE_OUTCOME_LOG");
  const std::string decision_path = env("NVFUSER_JEV_LOG");
  if (path.empty()) {
    if (decision_path.empty()) fail("NVFUSER_JEV_LOG is required in record mode");
    path = decision_path + ".pointwise.jsonl";
  }
  if (path == decision_path)
    fail("Pointwise outcome log must differ from the Choice log");
  appendLineToPath(path, line);
}

std::string selectedFromResponse(
    std::string_view response,
    std::string_view site,
    const std::vector<ChoiceOption>& options,
    bool semantic) {
  Json root = JsonReader(response).parse();
  const auto response_model = root.at("model").asString();
  const bool known_codex_model = response_model == "codex-gpt-6.1-sol-max";
  const bool known_semantic_model = response_model == "nvfuser-analysis-v1" ||
      response_model == "jev-latest" || response_model.starts_with("jev-") ||
      known_codex_model;
  if (semantic ? !known_semantic_model
               : response_model != "nvfuser-analysis-v1" && !known_codex_model) {
    fail("unexpected response model");
  }
  const Json& answers = root.at("answers");
  if (answers.kind != Json::Kind::Object || answers.object.size() != 1)
    fail("expected exactly one Choice answer");
  const Json& answer = answers.at(site);
  if (answer.at("type").asString() != "choice") fail("answer is not Choice");
  const std::string choice(answer.at("choice").asString());
  const bool allowed = std::any_of(options.begin(), options.end(), [&](const ChoiceOption& option) {
    return option.id == choice;
  });
  if (!allowed) fail("API selected a choice outside the allowed candidates");
  const Json& confidence = answer.at("confidence");
  if (confidence.kind != Json::Kind::Number) fail("invalid Choice confidence");
  auto probability = [](const Json& value) {
    if (value.kind != Json::Kind::Number) fail("expected numeric probability");
    double number = 0.0;
    try { number = std::stod(value.text); }
    catch (const std::exception&) { fail("invalid numeric probability"); }
    if (!std::isfinite(number) || number < 0.0 || number > 1.0)
      fail("Choice probability outside 0..1");
  };
  probability(confidence);
  const Json& probabilities = answer.at("probabilities");
  if (probabilities.kind != Json::Kind::Object || probabilities.object.size() != options.size())
    fail("invalid Choice probabilities");
  for (const auto& option : options) {
    probability(probabilities.at(option.id));
  }
  const Json& usage = root.at("usage");
  if (usage.at("input_tokens").kind != Json::Kind::Number ||
      usage.at("output_tokens").kind != Json::Kind::Number)
    fail("invalid API usage");
  return choice;
}

} // namespace

Mode mode() {
  if (native_decisions_forced)
    return Mode::Off;
  const std::string configured = env("NVFUSER_JEV_MODE");
  if (configured.empty() || configured == "off")
    return Mode::Off;
  if (configured == "record")
    return Mode::Record;
  if (configured == "remote")
    return Mode::Remote;
  fail("NVFUSER_JEV_MODE must be off, record, or remote");
}

ScopedNativeDecisions::ScopedNativeDecisions() noexcept
    : previous_(native_decisions_forced) {
  native_decisions_forced = true;
}

ScopedNativeDecisions::~ScopedNativeDecisions() noexcept {
  native_decisions_forced = previous_;
}

std::string jsonQuote(std::string_view value) {
  return quote(value);
}

bool semanticProtocolEnabled() {
  const std::string configured = env("NVFUSER_JEV_PROTOCOL");
  if (configured.empty() || configured == "text-v1")
    return false;
  if (configured == "semantic-v2")
    return true;
  if (configured == "semantic-v3")
    return false;
  fail("NVFUSER_JEV_PROTOCOL must be text-v1, semantic-v2, or semantic-v3");
}

bool semanticV3Enabled() {
  const std::string configured = env("NVFUSER_JEV_PROTOCOL");
  if (configured.empty() || configured == "text-v1" ||
      configured == "semantic-v2") {
    return false;
  }
  if (configured == "semantic-v3") return true;
  fail("NVFUSER_JEV_PROTOCOL must be text-v1, semantic-v2, or semantic-v3");
}

bool computeTraceEnabled() noexcept {
  try {
    const std::string_view configured = native_diagnostic::logMode();
    if ((configured != "record" && configured != "remote") ||
        !native_diagnostic::enabled()) {
      return false;
    }
    const std::string decision_path = env("NVFUSER_JEV_LOG");
    return !decision_path.empty() &&
        native_diagnostic::logPath() != decision_path;
  } catch (...) {
    return false;
  }
}

ComputeHeuristicsScope::ComputeHeuristicsScope(
    std::string_view input_snapshot,
    std::string_view input_snapshot_unavailable_reason,
    std::string_view hardware_json) noexcept
    : previous_(active_compute_scope),
      input_snapshot_(input_snapshot),
      input_snapshot_unavailable_reason_(input_snapshot_unavailable_reason),
      hardware_json_(hardware_json) {
  // A disabled nested call still shadows its parent invocation.
  active_compute_scope = this;
  if (!computeTraceEnabled()) return;
  try {
    mode_ = native_diagnostic::logMode();
    case_key_ = env("NVFUSER_JEV_CASE_KEY");
    invocation_id_ = std::to_string(::getpid()) + "-compute-" +
        std::to_string(
            next_compute_invocation.fetch_add(1, std::memory_order_relaxed) +
            1);
    active_ = true;
    write("begin");
  } catch (...) {
    // Observation cannot change the native scheduler or its exceptions.
    active_ = false;
  }
}

ComputeHeuristicsScope::~ComputeHeuristicsScope() noexcept {
  if (active_compute_scope == this) active_compute_scope = previous_;
}

bool ComputeHeuristicsScope::observing() const noexcept {
  return active_compute_scope == this && active_ && !completed_;
}

std::string_view ComputeHeuristicsScope::invocationId() const noexcept {
  return observing() ? std::string_view(invocation_id_) : std::string_view{};
}

std::string_view ComputeHeuristicsScope::inputSnapshot() const noexcept {
  return active_compute_scope == this ? input_snapshot_ : std::string_view{};
}

std::string_view ComputeHeuristicsScope::inputSnapshotUnavailableReason()
    const noexcept {
  return active_compute_scope == this ? input_snapshot_unavailable_reason_
                                      : std::string_view("no_compute_scope");
}

std::string_view ComputeHeuristicsScope::hardwareJson() const noexcept {
  return active_compute_scope == this ? hardware_json_ : std::string_view{};
}

std::string ComputeHeuristicsScope::priorChoicesJson() const {
  if (active_compute_scope != this) fail("semantic-v3 compute scope is inactive");
  std::string result = "[";
  for (const auto& [site, selected] : prior_choices_) {
    if (result.size() != 1) result += ',';
    result += "{\"site\":" + quote(site) + ",\"choice\":" +
        quote(selected) + '}';
  }
  return result + ']';
}

void ComputeHeuristicsScope::recordSelectedChoice(
    std::string_view site,
    std::string_view selected) {
  if (active_compute_scope != this) fail("semantic-v3 compute scope is inactive");
  prior_choices_.emplace_back(site, selected);
}

std::string_view computeInvocationInputSnapshot() noexcept {
  return active_compute_scope == nullptr ? std::string_view{}
                                         : active_compute_scope->inputSnapshot();
}

std::string_view computeInvocationInputUnavailableReason() noexcept {
  return active_compute_scope == nullptr
      ? std::string_view("no_compute_scope")
      : active_compute_scope->inputSnapshotUnavailableReason();
}

uint64_t ComputeHeuristicsScope::nextEventIndex() noexcept {
  return observing() ? ++event_index_ : 0;
}

void ComputeHeuristicsScope::write(
    std::string_view kind,
    std::string_view scheduler_type,
    std::string_view site,
    std::string_view selected,
    std::string_view reason,
    std::string_view selection_source) noexcept {
  try {
    native_diagnostic::Object row;
    row.add("schema", kComputeInvocationSchema);
    row.add("kind", kind);
    row.add("mode", mode_);
    row.add("case_key", case_key_);
    row.add("compute_invocation_id", invocation_id_);
    row.addRaw("compute_event_index", std::to_string(event_index_));
    if (kind == "begin") {
      if (input_snapshot_unavailable_reason_.empty() &&
          !input_snapshot_.empty()) {
        row.add("input_snapshot", input_snapshot_);
      } else {
        row.add(
            "input_snapshot_unavailable_reason",
            input_snapshot_unavailable_reason_.empty()
                ? std::string_view("not_captured")
                : input_snapshot_unavailable_reason_);
      }
    } else if (kind == "final") {
      row.add("scheduler_type", scheduler_type);
    } else if (kind == "no_choice") {
      row.add("site", site);
      row.add("selected", selected);
      row.add("reason", reason);
      row.add("selection_source", selection_source);
    }
    native_diagnostic::appendLine(row.str());
  } catch (...) {
    // A missing row is detected by the strict offline trace reader.
  }
}

void ComputeHeuristicsScope::recordNoChoice(
    std::string_view site,
    std::string_view selected,
    std::string_view reason,
    std::string_view selection_source) noexcept {
  if (!observing() || site.empty() || selected.empty() || reason.empty() ||
      selection_source.empty()) {
    return;
  }
  ++event_index_;
  write("no_choice", {}, site, selected, reason, selection_source);
}

void recordNoChoiceDecision(
    std::string_view site,
    std::string_view selected,
    std::string_view reason,
    std::string_view selection_source) noexcept {
  if (active_compute_scope != nullptr) {
    active_compute_scope->recordNoChoice(
        site, selected, reason, selection_source);
  }
}

void recordContextFallback(std::string_view reason) noexcept {
  try {
    if (mode() == Mode::Off || !semanticV3Enabled() || reason.empty()) return;
    std::string row =
        "{\"schema\":\"nvfuser.jev.context_fallback.v1\","
        "\"kind\":\"context_fallback\",\"site\":\"semantic-v3\","
        "\"selected\":\"native\",\"reason\":" + quote(reason) +
        ",\"mode\":" + quote(mode() == Mode::Record ? "record" : "remote") +
        ",\"case_key\":" + quote(env("NVFUSER_JEV_CASE_KEY"));
    if (active_compute_scope != nullptr &&
        !active_compute_scope->invocationId().empty()) {
      row += ",\"compute_invocation_id\":" +
          quote(active_compute_scope->invocationId());
    }
    appendLog(row + '}');
  } catch (...) {
    // A diagnostic write failure must not change the native fallback.
  }
}

void ComputeHeuristicsScope::complete(std::string_view scheduler_type) noexcept {
  if (!observing() || std::uncaught_exceptions() != 0) return;
  ++event_index_;
  completed_ = true;
  write("final", scheduler_type);
}

namespace {
struct ComputeEvent {
  std::string_view invocation_id;
  uint64_t index = 0;
};

ComputeEvent nextComputeEvent() noexcept {
  if (active_compute_scope == nullptr || !active_compute_scope->observing())
    return {};
  return {
      active_compute_scope->invocationId(),
      active_compute_scope->nextEventIndex()};
}
} // namespace

PointwiseRecordScope::PointwiseRecordScope() {
  if (mode() != Mode::Record) return;
  if (env("NVFUSER_JEV_LOG").empty())
    fail("NVFUSER_JEV_LOG is required in record mode");
  invocation_id_ = std::to_string(::getpid()) + "-" +
      std::to_string(next_invocation.fetch_add(1, std::memory_order_relaxed) + 1);
  case_key_ = env("NVFUSER_JEV_CASE_KEY");
  std::string active_id = invocation_id_;
  std::string active_case = case_key_;
  appendPointwiseOutcome(
      "{\"schema\":" + quote(kPointwiseOutcomeSchema) +
      ",\"kind\":\"begin\",\"mode\":\"record\",\"invocation_id\":" +
      quote(invocation_id_) + ",\"event_index\":0,\"case_key\":" +
      quote(case_key_) + '}');
  previous_invocation_id_.swap(active_invocation.id);
  previous_case_key_.swap(active_invocation.case_key);
  previous_event_index_ = active_invocation.event_index;
  previous_completed_ = active_invocation.completed;
  active_invocation.id.swap(active_id);
  active_invocation.case_key.swap(active_case);
  active_invocation.event_index = 0;
  active_invocation.completed = false;
  recording_ = true;
}

PointwiseRecordScope::~PointwiseRecordScope() noexcept {
  if (recording_) {
    active_invocation.id.swap(previous_invocation_id_);
    active_invocation.case_key.swap(previous_case_key_);
    active_invocation.event_index = previous_event_index_;
    active_invocation.completed = previous_completed_;
  }
}

void PointwiseRecordScope::complete(
    std::string_view backend,
    std::string_view final_params_json,
    std::string_view launch_json,
    std::string_view compile_json) {
  if (!recording_) return;
  if (completed_ || active_invocation.id != invocation_id_)
    fail("Pointwise Record invocation completion is out of order");
  if (backend != "tma" && backend != "ordinary" && backend != "empty")
    fail("invalid Pointwise final backend");
  for (auto object : {final_params_json, launch_json, compile_json}) {
    if (JsonReader(object).parse().kind != Json::Kind::Object)
      fail("Pointwise final snapshot must contain JSON objects");
  }
  const auto event_index = ++active_invocation.event_index;
  appendPointwiseOutcome(
      "{\"schema\":" + quote(kPointwiseOutcomeSchema) +
      ",\"kind\":\"final\",\"mode\":\"record\",\"invocation_id\":" +
      quote(invocation_id_) + ",\"event_index\":" +
      std::to_string(event_index) + ",\"case_key\":" + quote(case_key_) +
      ",\"backend\":" + quote(backend) +
      ",\"final_params\":" + std::string(final_params_json) +
      ",\"launch\":" + std::string(launch_json) +
      ",\"compile\":" + std::string(compile_json) + '}');
  completed_ = true;
  active_invocation.completed = true;
}

namespace {

std::string semanticV3State(
    std::string_view site,
    std::string_view analysis_inputs,
    const std::vector<const ChoiceOption*>& ordered_options) {
  if (active_compute_scope == nullptr || analysis_inputs.empty()) {
    fail("semantic-v3 requires a computeHeuristics scope and analysis inputs");
  }
  const auto snapshot = active_compute_scope->inputSnapshot();
  const auto hardware = active_compute_scope->hardwareJson();
  if (snapshot.empty() || !active_compute_scope->inputSnapshotUnavailableReason().empty() ||
      hardware.empty()) {
    fail("semantic-v3 requires a complete fusion/runtime and hardware snapshot");
  }
  if (JsonReader(hardware).parse().kind != Json::Kind::Object) {
    fail("semantic-v3 hardware snapshot must be a JSON object");
  }

  std::string result =
      "{\"schema\":\"nvfuser.systemone.context.v3\","
      "\"verification_status\":\"analysis_summary_only\","
      "\"objective\":\"Choose the listed schedule alternative expected "
      "to minimize fusion kernel runtime on the described GPU.\","
      "\"workload\":{\"snapshot_format\":"
      "\"nvfuser.compute_invocation_input.v1\",\"fusion_and_runtime\":";
  result += quote(snapshot) + ",\"hardware\":" + std::string(hardware);
  result += "},\"current_schedule\":{\"prior_choices\":" +
      active_compute_scope->priorChoicesJson() + "},\"decision\":{\"site\":" +
      quote(site) + ",\"analysis_inputs_text\":" + quote(analysis_inputs) +
      ",\"candidates\":[";
  for (const auto* option : ordered_options) {
    if (result.back() != '[') result += ',';
    result += "{\"id\":" + quote(option->id) + ",\"description\":" +
        quote(option->description);
    if (!option->consequence_json.empty()) {
      if (JsonReader(option->consequence_json).parse().kind !=
          Json::Kind::Object) {
        fail("semantic-v3 candidate consequence must be a JSON object");
      }
      result += ",\"consequence\":" + option->consequence_json;
    }
    result += '}';
  }
  return result + "]}}";
}

} // namespace

std::string chooseDecisionImpl(
    std::string_view site,
    std::string_view state,
    const std::vector<ChoiceOption>& options,
    std::string_view native_choice,
    std::string_view case_key,
    bool semantic,
    std::string_view instructions) {
  const Mode current_mode = mode();
  if (current_mode == Mode::Off) {
    if (native_choice.empty()) fail("native choice is required in off mode");
    return std::string(native_choice);
  }
  if (env("NVFUSER_JEV_LOG").empty())
    fail("NVFUSER_JEV_LOG is required in record/remote mode");
  if (site.empty() || options.empty() || options.size() > 255) fail("invalid Choice site or candidates");
  std::map<std::string, bool> seen;
  for (const auto& option : options) {
    if (option.id.empty() || !seen.emplace(option.id, true).second)
      fail("Choice candidates must have distinct nonempty IDs");
  }
  if (current_mode == Mode::Record && !seen.count(std::string(native_choice)))
    fail("native choice is not an allowed candidate");
  if (!active_invocation.id.empty() && active_invocation.completed)
    fail("Pointwise Choice followed a final Record outcome");
  const bool semantic_v3 = !semantic && semanticV3Enabled();
  std::vector<const ChoiceOption*> ordered_options;
  ordered_options.reserve(options.size());
  for (const auto& option : options) ordered_options.push_back(&option);
  if (semantic_v3) {
    std::sort(
        ordered_options.begin(),
        ordered_options.end(),
        [](const ChoiceOption* lhs, const ChoiceOption* rhs) {
          return lhs->id < rhs->id;
        });
  }
  const bool structured = semantic || semantic_v3;
  const std::string state_json = semantic_v3
      ? semanticV3State(site, state, ordered_options)
      : semantic ? std::string(state) : quote(state);
  if (structured && JsonReader(state_json).parse().kind != Json::Kind::Object)
    fail("semantic state must be a JSON object");
  const std::string log_case_key = case_key.empty()
      ? (active_invocation.id.empty() ? env("NVFUSER_JEV_CASE_KEY")
                                      : active_invocation.case_key)
      : std::string(case_key);

  const std::string question_instructions = semantic
      ? std::string(instructions)
      : semantic_v3
          ? "Choose the alternative expected to minimize fusion kernel "
            "runtime. The state contains the full invocation snapshot, "
            "hardware, prior choices, and local analysis inputs. "
            "Candidate final plans are not yet independently verified; "
            "do not assume missing effects. Return one listed ID."
          : "Choose nvFuser scheduler decision " + std::string(site);

  // Keep the exact model-visible question in both local logs. In record mode
  // this is the request that the remote mode would have sent, without the
  // native winner; comparing it catches mismatched option descriptions too.
  std::string request = "{\"model\":\"jev-latest\",\"state\":" +
      state_json +
      ",\"questions\":{" + quote(site) + ":{\"type\":\"choice\",\"instructions\":" +
      quote(question_instructions) +
      ",\"criteria\":{";
  for (size_t i = 0; i < ordered_options.size(); ++i) {
    if (i != 0) request += ',';
    request += quote(ordered_options[i]->id) + ':' +
        quote(ordered_options[i]->description);
  }
  request += "}}}}";
  const size_t max_request =
      semantic_v3 ? kMaxSemanticV3Request : kMaxRequest;
  if (semantic_v3 && request.size() > max_request)
    fail("complete semantic-v3 request exceeds 8 MiB; no snapshot was truncated");
  if (semantic && request.size() > kMaxRequest)
    fail("complete semantic request exceeds 64 KiB; no snapshot was truncated");
  std::string selected(native_choice);
  int64_t roundtrip_us = 0;
  if (current_mode == Mode::Remote) {
    const auto begin = Clock::now();
    const std::string response = post(request, max_request);
    roundtrip_us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - begin).count();
    selected = selectedFromResponse(response, site, options, structured);
  }
  std::string candidates = "[";
  for (size_t i = 0; i < options.size(); ++i) {
    if (i != 0) candidates += ',';
    candidates += quote(options[i].id);
  }
  candidates += ']';
  const auto event_index = active_invocation.id.empty()
      ? uint64_t{0}
      : ++active_invocation.event_index;
  const auto compute_event = nextComputeEvent();
  appendLog("{\"site\":" + quote(site) + ",\"case_key\":" + quote(log_case_key) +
      (active_invocation.id.empty() ? std::string()
                                    : ",\"invocation_id\":" + quote(active_invocation.id) +
                                          ",\"event_index\":" + std::to_string(event_index)) +
      (compute_event.invocation_id.empty()
           ? std::string()
           : ",\"compute_invocation_id\":" +
                 quote(compute_event.invocation_id) +
                 ",\"compute_event_index\":" +
                 std::to_string(compute_event.index)) +
      ",\"state\":" + state_json +
      ",\"candidates\":" + candidates +
      ",\"request\":" + request +
      ",\"selected\":" + quote(selected) + ",\"mode\":" +
      quote(current_mode == Mode::Record ? "record" : "remote") +
      ",\"roundtrip_us\":" + std::to_string(roundtrip_us) + '}');
  if (semantic_v3) active_compute_scope->recordSelectedChoice(site, selected);
  return selected;
}

std::string chooseDecision(
    std::string_view site,
    std::string_view state,
    const std::vector<ChoiceOption>& options,
    std::string_view native_choice,
    std::string_view case_key) {
  return chooseDecisionImpl(
      site, state, options, native_choice, case_key, false, {});
}

std::string chooseSemanticDecision(
    std::string_view site,
    std::string_view state_json,
    std::string_view instructions,
    const std::vector<ChoiceOption>& options,
    std::string_view native_choice) {
  if (!semanticProtocolEnabled() || instructions.empty())
    fail("semantic Choice requires semantic-v2 and instructions");
  return chooseDecisionImpl(
      site, state_json, options, native_choice, {}, true, instructions);
}

void recordCoverageSkip(
    std::string_view site,
    std::string_view reason,
    std::string_view case_key) {
  if (mode() != Mode::Record) fail("coverage skip requires record mode");
  if (site.empty() || reason.empty()) fail("coverage skip needs site and reason");
  if (!active_invocation.id.empty() && active_invocation.completed)
    fail("Pointwise coverage skip followed a final Record outcome");
  const std::string log_case_key = case_key.empty()
      ? (active_invocation.id.empty() ? env("NVFUSER_JEV_CASE_KEY")
                                      : active_invocation.case_key)
      : std::string(case_key);
  const auto compute_event = nextComputeEvent();
  appendLog("{\"kind\":\"coverage_skip\",\"site\":" + quote(site) +
      ",\"case_key\":" + quote(log_case_key) +
      (compute_event.invocation_id.empty()
           ? std::string()
           : ",\"compute_invocation_id\":" +
                 quote(compute_event.invocation_id) +
                 ",\"compute_event_index\":" +
                 std::to_string(compute_event.index)) +
      ",\"mode\":\"record\",\"reason\":" + quote(reason) + '}');
  if (!active_invocation.id.empty()) {
    const auto event_index = ++active_invocation.event_index;
    appendPointwiseOutcome(
        "{\"schema\":" + quote(kPointwiseOutcomeSchema) +
        ",\"kind\":\"coverage_skip\",\"mode\":\"record\",\"invocation_id\":" +
        quote(active_invocation.id) + ",\"event_index\":" +
        std::to_string(event_index) + ",\"case_key\":" + quote(log_case_key) +
        ",\"site\":" + quote(site) + ",\"reason\":" + quote(reason) + '}');
  }
}

} // namespace nvfuser::scheduler::jev
