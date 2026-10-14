// SPDX-License-Identifier: BSD-3-Clause
// Standalone, CPU-only protocol test: build with jev_client.cpp and -pthread.
#include <scheduler/jev_client.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::string readFile(const std::string& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

class FakeServer {
 public:
  explicit FakeServer(std::string body, int status = 200) : body_(std::move(body)), status_(status) {
    fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    require(fd_ >= 0, "socket failed");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(::bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0, "bind failed");
    require(::listen(fd_, 1) == 0, "listen failed");
    socklen_t size = sizeof(address);
    require(::getsockname(fd_, reinterpret_cast<sockaddr*>(&address), &size) == 0, "getsockname failed");
    port_ = ntohs(address.sin_port);
    thread_ = std::thread([this] { serve(); });
  }
  ~FakeServer() {
    if (thread_.joinable()) thread_.join();
    ::close(fd_);
  }
  unsigned port() const { return port_; }
  const std::string& request() const { return request_; }

 private:
  void serve() {
    int client = ::accept(fd_, nullptr, nullptr);
    if (client < 0) return;
    char buffer[4096];
    size_t expected = 0;
    while (true) {
      ssize_t n = ::recv(client, buffer, sizeof(buffer), 0);
      if (n <= 0) break;
      request_.append(buffer, static_cast<size_t>(n));
      const size_t split = request_.find("\r\n\r\n");
      if (split != std::string::npos) {
        if (expected == 0) {
          const std::string header = "Content-Length: ";
          size_t index = request_.find(header);
          if (index != std::string::npos) {
            expected = static_cast<size_t>(std::stoul(request_.substr(index + header.size())));
          }
        }
        if (request_.size() >= split + 4 + expected) break;
      }
    }
    const std::string reply = "HTTP/1.0 " + std::to_string(status_) +
        (status_ == 200 ? " OK" : " Failure") + "\r\nContent-Type: application/json\r\n"
        "Content-Length: " + std::to_string(body_.size()) + "\r\n\r\n" + body_;
    size_t offset = 0;
    while (offset < reply.size()) {
      ssize_t n = ::send(client, reply.data() + offset, reply.size() - offset, MSG_NOSIGNAL);
      if (n <= 0) break;
      offset += static_cast<size_t>(n);
    }
    ::close(client);
  }
  std::string body_;
  int status_;
  int fd_ = -1;
  unsigned port_ = 0;
  std::thread thread_;
  std::string request_;
};

std::string answer(const std::string& choice) {
  return "{\"model\":\"nvfuser-analysis-v1\",\"answers\":{\"PW2\":{\"type\":\"choice\","
      "\"choice\":\"" + choice + "\",\"confidence\":1.0,\"probabilities\":{\"bp0\":0.0,"
      "\"bp1\":1.0}}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}";
}

} // namespace

int main() {
  using nvfuser::scheduler::jev::ChoiceOption;
  using nvfuser::scheduler::jev::chooseDecision;
  const std::vector<ChoiceOption> options{{"bp0", "one dimensional"}, {"bp1", "split"}};
  const std::string state = "pointwise.breakpoint.v1\nelements=2,256\n";
  char path[] = "/tmp/nvfuser-jev-client-XXXXXX";
  int log_fd = ::mkstemp(path);
  require(log_fd >= 0, "mkstemp failed");
  ::close(log_fd);
  ::setenv("NVFUSER_JEV_LOG", path, 1);

  ::setenv("NVFUSER_JEV_MODE", "off", 1);
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Off,
          "mode() did not report off");
  {
    nvfuser::scheduler::jev::PointwiseRecordScope scope;
    require(!scope.recording(), "off mode enabled Pointwise diagnostics");
    scope.complete("ordinary", "{}", "{}", "{}");
  }
  require(chooseDecision("PW2", state, options, "bp0") == "bp0", "off mode changed choice");
  require(readFile(path).empty(), "off mode wrote a record");

  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Record,
          "mode() did not report record");
  require(chooseDecision("PW2", state, options, "bp0", "case-1") == "bp0", "record changed choice");
  require(readFile(path).find("\"case_key\":\"case-1\"") != std::string::npos,
          "record lacks case key");
  require(readFile(path).find("\"request\":{\"model\":\"jev-latest\"") != std::string::npos,
          "record lacks the model-visible request");
  require(readFile(path).find("\"instructions\":\"Choose nvFuser scheduler decision PW2\"") != std::string::npos,
          "record lacks Choice instructions");
  require(readFile(path).find("\"criteria\":{\"bp0\":\"one dimensional\",\"bp1\":\"split\"}") != std::string::npos,
          "record lacks candidate descriptions");
  nvfuser::scheduler::jev::recordCoverageSkip(
      "PW4.tile", "native tile exceeds conservative shared-memory bound", "case-skip");
  require(readFile(path).find("\"kind\":\"coverage_skip\",\"site\":\"PW4.tile\"") !=
              std::string::npos &&
              readFile(path).find("\"reason\":\"native tile exceeds conservative shared-memory bound\"") !=
                  std::string::npos,
          "record coverage skip lacks a reason");
  const std::string first_record = readFile(path);
  require(chooseDecision("PW2", state, options, "bp1", "case-1b") == "bp1",
          "second record changed choice");
  const std::string appended = readFile(path);
  require(appended.starts_with(first_record) &&
              appended.find("\"case_key\":\"case-1b\"") != std::string::npos &&
              appended.size() > first_record.size(),
          "record mode did not append to a regular log file");

  // Pointwise candidate plans can be discarded. Correlate their Choice rows
  // with the returned plan in a separate, complete outcome stream.
  ::setenv("NVFUSER_JEV_CASE_KEY", "pointwise-case", 1);
  const std::string outcome_path = std::string(path) + ".pointwise.jsonl";
  {
    nvfuser::scheduler::jev::PointwiseRecordScope scope;
    require(scope.recording(), "Pointwise Record scope was not enabled");
    require(chooseDecision("PW2", state, options, "bp0") == "bp0",
            "Pointwise Record scope changed the native choice");
    nvfuser::scheduler::jev::recordCoverageSkip("PW4.tile", "no safe tile");
    scope.complete(
        "tma", "{\"use_tma_load\":true,\"vectorization_factor\":4}",
        "{\"block_x_raw\":128}", "{\"index_type\":null}");
  }
  const std::string outcome = readFile(outcome_path);
  require(outcome.find("\"kind\":\"begin\"") != std::string::npos &&
              outcome.find("\"kind\":\"final\"") != std::string::npos &&
              outcome.find("\"kind\":\"coverage_skip\"") != std::string::npos &&
              outcome.find("\"event_index\":0") != std::string::npos &&
              outcome.find("\"event_index\":2") != std::string::npos &&
              outcome.find("\"event_index\":3") != std::string::npos &&
              outcome.find("\"backend\":\"tma\"") != std::string::npos &&
              outcome.find("\"final_params\":{\"use_tma_load\":true") !=
                  std::string::npos,
          "Pointwise outcome is missing its terminal plan or coverage event");
  const std::string id_key = "\"invocation_id\":\"";
  const auto id_start = outcome.find(id_key);
  require(id_start != std::string::npos, "Pointwise outcome lacks invocation ID");
  const auto id_end = outcome.find('"', id_start + id_key.size());
  const std::string invocation_id = outcome.substr(
      id_start + id_key.size(), id_end - id_start - id_key.size());
  require(!invocation_id.empty() &&
              outcome.find(id_key + invocation_id + '"', id_end) != std::string::npos &&
              readFile(path).find(id_key + invocation_id + "\",\"event_index\":1") !=
                  std::string::npos,
          "Pointwise Choice and outcome rows do not share an invocation ID");
  require(readFile(path).find("\"kind\":\"coverage_skip\",\"site\":\"PW4.tile\"") !=
              std::string::npos,
          "Choice-only log coverage contract changed");
  ::unsetenv("NVFUSER_JEV_CASE_KEY");
  const std::string before_link = readFile(path);

  const std::string link_path = std::string(path) + ".link";
  require(::symlink(path, link_path.c_str()) == 0, "symlink creation failed");
  ::setenv("NVFUSER_JEV_LOG", link_path.c_str(), 1);
  bool link_rejected = false;
  try { (void)chooseDecision("PW2", state, options, "bp0", "symlink"); }
  catch (const std::runtime_error&) { link_rejected = true; }
  require(link_rejected && readFile(path) == before_link,
          "symlink log path was followed or changed its target");
  ::unlink(link_path.c_str());

  const std::string missing_target = std::string(path) + ".missing";
  require(::symlink(missing_target.c_str(), link_path.c_str()) == 0,
          "dangling symlink creation failed");
  bool dangling_link_rejected = false;
  try { (void)chooseDecision("PW2", state, options, "bp0", "dangling"); }
  catch (const std::runtime_error&) { dangling_link_rejected = true; }
  require(dangling_link_rejected && ::access(missing_target.c_str(), F_OK) != 0,
          "dangling symlink log path created its target");
  ::unlink(link_path.c_str());

  const std::string fifo_path = std::string(path) + ".fifo";
  require(::mkfifo(fifo_path.c_str(), 0600) == 0, "named pipe creation failed");
  ::setenv("NVFUSER_JEV_LOG", fifo_path.c_str(), 1);
  bool fifo_rejected = false;
  try { (void)chooseDecision("PW2", state, options, "bp0", "fifo"); }
  catch (const std::runtime_error&) { fifo_rejected = true; }
  require(fifo_rejected, "named pipe log path was accepted");
  ::unlink(fifo_path.c_str());

  ::setenv("NVFUSER_JEV_LOG", "/dev/null", 1);
  bool device_rejected = false;
  try { (void)chooseDecision("PW2", state, options, "bp0", "device"); }
  catch (const std::runtime_error&) { device_rejected = true; }
  require(device_rejected, "non-regular device log path was accepted");
  ::setenv("NVFUSER_JEV_LOG", path, 1);

  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Remote,
          "mode() did not report remote");
  {
    nvfuser::scheduler::jev::PointwiseRecordScope scope;
    require(!scope.recording(), "remote mode enabled Pointwise Record diagnostics");
    scope.complete("ordinary", "{}", "{}", "{}");
    require(readFile(outcome_path) == outcome,
            "remote mode changed the Pointwise Record outcome log");
  }
  {
    FakeServer server(answer("bp\\u0031"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    require(chooseDecision("PW2", state, options, {}, "case-2") == "bp1",
            "remote Unicode-escaped choice was not decoded");
    require(server.request().find("POST /v1/systemone HTTP/1.0") == 0, "wrong endpoint");
    require(server.request().find("\"questions\":{\"PW2\"") != std::string::npos,
            "missing Choice question");
    require(server.request().find("native") == std::string::npos, "native winner leaked");
    require(server.request().find("winner") == std::string::npos, "winner leaked");
  }
  const std::string log = readFile(path);
  require(log.find("\"selected\":\"bp1\",\"mode\":\"remote\"") != std::string::npos,
          "remote choice not logged");
  require(log.find("\"roundtrip_us\":") != std::string::npos, "missing latency");
  {
    FakeServer server(answer("bp2"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    bool rejected = false;
    try { (void)chooseDecision("PW2", state, options); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "out-of-set choice accepted");
  }
  {
    FakeServer server("{}", 500);
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    bool rejected = false;
    try { (void)chooseDecision("PW2", state, options); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "HTTP error accepted");
  }
  {
    bool rejected = false;
    try { (void)chooseDecision("PW2", std::string(70 * 1024, 'x'), options); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "oversized request accepted");
  }
  ::unlink(path);
  ::unlink(outcome_path.c_str());
}
