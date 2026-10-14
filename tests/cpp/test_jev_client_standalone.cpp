// SPDX-License-Identifier: BSD-3-Clause
// Standalone, CPU-only protocol test: build with jev_client.cpp and -pthread.
#include <scheduler/jev_client.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
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
  require(chooseDecision("PW2", state, options, "bp0") == "bp0", "off mode changed choice");
  require(readFile(path).empty(), "off mode wrote a record");

  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Record,
          "mode() did not report record");
  require(chooseDecision("PW2", state, options, "bp0", "case-1") == "bp0", "record changed choice");
  require(readFile(path).find("\"case_key\":\"case-1\"") != std::string::npos,
          "record lacks case key");

  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Remote,
          "mode() did not report remote");
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
}
