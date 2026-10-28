// SPDX-License-Identifier: BSD-3-Clause
// Standalone, CPU-only protocol test: build with jev_client.cpp and -pthread.
#include <scheduler/jev_client.h>
#include <scheduler/inner_outer_mode_preflight.h>
#include <scheduler/matmul_hopper_tile_menu.h>
#include <scheduler/normalization_inner_smem_cta_choice.h>
#include <scheduler/normalization_inner_outer_grid_rows_admission.h>
#include <scheduler/normalization_inner_outer_tma_tmp_width.h>
#include <scheduler/normalization_tma_prefetch_choice.h>
#include <scheduler/pointwise_unroll_envelope.h>
#include <scheduler/reduction_inner_remainder_choice.h>
#include <scheduler/reduction_inner_width_choice.h>
#include <scheduler/reduction_inner3d_iter_unroll_continuation.h>
#include <scheduler/reduction_tma_unroll_choice.h>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

void testHopperLowStageMenu() {
  using namespace nvfuser::matmul_utils::hopper_tile_menu;
  // The legacy three-stage preference has no winner with this injected SMEM
  // budget. The sole hard-admissible tile can continue with one operand stage.
  const auto menu = enumerate(8192, 4096, 4096, 132, 40000, true);
  require(!menu.has_native_config, "three-stage policy unexpectedly found a tile");
  size_t positive_stages = 0;
  for (const auto& candidate : menu.candidates) {
    const auto budget = stageBudget(
        candidate.cta_m,
        candidate.cta_n,
        candidate.cta_k,
        40000,
        2,
        1);
    if (budget.max_operand_stages < 1) {
      continue;
    }
    ++positive_stages;
    require(
        candidate.cta_m == 64 && candidate.cta_n == 64 &&
            candidate.cta_k == 64 && candidate.instr_n == 64 &&
            !candidate.native_valid &&
            budget.output_smem_reserve_bytes == 8192 &&
            budget.operand_stage_bytes == 16400 &&
            budget.max_operand_stages == 1,
        "low-stage Hopper tile cannot continue under the injected budget");
  }
  require(positive_stages == 1, "low-stage Hopper menu is not the single expected tile");
  const auto legacy = enumerate(8192, 4096, 4096, 132, 40000, false);
  require(
      !legacy.has_native_config && legacy.candidates.empty(),
      "Off/Record tile admission changed under the injected budget");
}

void testInnerSmemCtaNativeSeedGate() {
  using namespace nvfuser::normalization_inner::non_tma::smem_cta_choice;
  // A 4100-element FP32 persistent buffer fits when seed 512 selects b3,
  // although the native max-CTA seed 1024 selects b2 and exceeds this cap.
  const Input input{
      4100, 4, 1024, 32, 4100 * 32, 21400 * 8, 4096 * 8,
      128 * 8, 128 * 8, 128, {{"T0", 4100 * 32, 32}}};
  const auto native = preview(input, 1024);
  const auto alternative = preview(input, 512);
  const auto menu = enumerate(input);
  require(
      native.has_value() && alternative.has_value() &&
          native->batch == 2 && native->actual_threads == 513 &&
          native->padded_threads == 544 &&
          native->roundup_overhead_bits == 1024 * 8 &&
          native->required_smem_bits == 21648 * 8 && !native->legal &&
          alternative->batch == 3 && alternative->actual_threads == 342 &&
          alternative->padded_threads == 352 &&
          alternative->roundup_overhead_bits == 512 * 8 &&
          alternative->required_smem_bits == 21136 * 8 &&
          alternative->legal && !menu.hasNativeSeed(1024) &&
          !menu.candidates.empty() &&
          menu.geometry.front().seed == 1024 &&
          menu.candidates.front().seed == 512 &&
          menu.candidates.front().batch == 3,
      "IP8 native seed was treated as shared-memory legality");
}

void testTmaTemporaryReadWidthContinuation() {
  using namespace nvfuser::inner_outer_tma_warp_specialized::tmp_width;
  const auto half = bounds(/*inner_dim=*/4096, /*inner_vector=*/8,
                           /*tmp_dtype_bits=*/16);
  require(
      half.hard_max == 8 && half.native_read == 4,
      "IOP2 native performance cap was confused with the hard read limit");
  const auto half_eight = readPlan(
      8, 4096, 8, 16, /*bdimx=*/128, /*gdimy=*/128,
      /*fixed_ws_plan_hard_valid=*/true);
  require(
      half_eight.hard_valid && !half_eight.combined_split_grid_inner_dim &&
          half_eight.tmp_write_width_seed == 8 &&
          !half_eight.tmp_write_choice_pending,
      "IOP2 w8 cannot continue through temporary write and final grid");
  const auto large_inner = readPlan(8, 1048576, 8, 16, 128, 128, true);
  require(
      large_inner.hard_valid && large_inner.combined_split_grid_inner_dim,
      "IOP2 candidate-specific grid split was not recomputed");
  require(
      !readPlan(16, 4096, 8, 16, 128, 128, true).hard_valid &&
          !readPlan(8, 4096, 8, 16, 128, 128, false).hard_valid,
      "IOP2 accepted an excessive width or a failed fixed WS plan");

  const auto byte = bounds(/*inner_dim=*/4096, /*inner_vector=*/16,
                           /*tmp_dtype_bits=*/8);
  const auto byte_eight = readPlan(8, 4096, 16, 8, 128, 128, true);
  require(
      byte.hard_max == 16 && byte.native_read == 4 &&
          byte_eight.hard_valid && byte_eight.tmp_write_width_seed == 16 &&
          byte_eight.tmp_write_choice_pending,
      "IOP2 selected w8 cannot reach a hard-valid w8/w16 write choice");
}

void testInner3dIterUnrollContinuation() {
  using namespace nvfuser::reduction::non_tma::inner3d_iter_continuation;
  const Input equivalent{512, 1, 1, 200, 16, 1, 32, 128, 65535};
  std::vector<Candidate> same_plans;
  for (int64_t factor : {1, 2, 4}) {
    auto candidate = equivalent;
    candidate.requested_unroll = factor;
    same_plans.push_back({factor, preview(candidate)});
    require(
        same_plans.back().plan.final_unroll == 1 &&
            same_plans.back().plan.final_godim == 512 &&
            same_plans.back().plan.raw_gridim == 1 &&
            same_plans.back().plan.grodim == 16 &&
            !same_plans.back().plan.rd5_grid_eligible,
        "RD5 u>1 rollback did not preserve the complete final grid plan");
  }
  const auto single = distinctCandidates(same_plans);
  require(
      single.size() == 1 && single.front().factor == 1,
      "RD5 equivalence did not retain the canonical representative");

  // Native u2 and u1 have the same realized plan, while u3 differs. The
  // Record and Remote menus must both use the canonical IDs u1 and u3;
  // Record maps native u2 to u1 without leaking its preference in the menu.
  const StateFields mixed_classes{
      257, 1, 257, 32, 1, 1, 1, 128, 512, 256, 32, 2, 1, 32,
      8,   8, 2147483647, 65535, 65535};
  const auto record_request = buildRequest(mixed_classes, 3, 2);
  const auto remote_request = buildRequest(mixed_classes, 3, std::nullopt);
  require(
      record_request.state == remote_request.state &&
          record_request.state.find(
              "reduction.inner3d_iter_unroll.v2\niteration_numel=257\n") ==
              0 &&
          record_request.candidates.size() == 2 &&
          remote_request.candidates.size() == 2 &&
          record_request.candidates[0].id == "u1" &&
          record_request.candidates[1].id == "u3" &&
          record_request.native_choice == "u1" &&
          remote_request.native_choice.empty() &&
          record_request.candidates[1].plan.clean_gridim == 7,
      "RD5 Record/Remote state or IDs differ, or native u2 did not map to u1");
  for (size_t i = 0; i < record_request.candidates.size(); ++i) {
    require(
        record_request.candidates[i].description ==
                remote_request.candidates[i].description &&
            record_request.candidates[i].consequence_json ==
                remote_request.candidates[i].consequence_json,
        "RD5 Record/Remote candidate bytes differ");
  }

  const Input trigger{512, 1, 1, 300, 1, 2, 32, 128, 65535};
  auto trigger_u2 = trigger;
  trigger_u2.requested_unroll = 2;
  const auto before = preview(trigger);
  const auto after = preview(trigger_u2);
  require(
      before.final_unroll == 1 && after.final_unroll == 1 &&
          before.final_godim == after.final_godim &&
          before.clean_gridim == 1 && after.clean_gridim == 2 &&
          before.realizedSignature() != after.realizedSignature(),
      "RD5 deduplicated factors that change the cross-grid trigger");

  // Exact 3D Reduction CUDA witness {2,512,8193}: u2 rolls back to u1 but
  // retains a six-way inner grid, whereas requested u1 has no inner grid.
  const Input cuda_witness{512, 1, 1, 512, 6, 6, 384, 128, 65535};
  auto cuda_u2_input = cuda_witness;
  cuda_u2_input.requested_unroll = 2;
  const auto cuda_u1 = preview(cuda_witness);
  const auto cuda_u2 = preview(cuda_u2_input);
  require(
      cuda_u1.final_unroll == 1 && cuda_u2.final_unroll == 1 &&
          cuda_u1.final_godim == 512 && cuda_u2.final_godim == 512 &&
          cuda_u1.clean_gridim == 1 && cuda_u2.clean_gridim == 6 &&
          cuda_u1.realizedSignature() != cuda_u2.realizedSignature(),
      "RD5 real rollback witness lost its distinct six-way grid effect");

  const Input pending_grid{3, 1, 1, 100, 64, 64, 32, 128, 65535};
  auto pending_u2 = pending_grid;
  pending_u2.requested_unroll = 2;
  const auto grid_u1 = preview(pending_grid);
  const auto grid_u2 = preview(pending_u2);
  require(
      grid_u1.raw_gridim == 64 && grid_u1.clean_gridim == 42 &&
          grid_u1.rd5_grid_eligible && grid_u2.raw_gridim == 64 &&
          grid_u2.clean_gridim == 64 && !grid_u2.rd5_grid_eligible &&
          grid_u1.realizedSignature() != grid_u2.realizedSignature(),
      "RD5 iteration factor removed a distinct later RD5.grid choice");
}

std::string readFile(const std::string& path) {
  std::ifstream stream(path);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string loggedRequest(const std::string& decision_row) {
  constexpr std::string_view key = "\"request\":";
  const auto field = decision_row.find(key);
  const auto selected = decision_row.find(",\"selected\":", field);
  require(field != std::string::npos && selected != std::string::npos,
          "Choice log lacks a complete request");
  return decision_row.substr(field + key.size(), selected - field - key.size());
}

class FakeServer {
 public:
  explicit FakeServer(
      std::string body,
      int status = 200,
      std::chrono::milliseconds reply_delay = std::chrono::milliseconds(0))
      : FakeServer(
            std::vector<std::string>{std::move(body)}, status, reply_delay) {}
  explicit FakeServer(
      std::vector<std::string> bodies,
      int status = 200,
      std::chrono::milliseconds reply_delay = std::chrono::milliseconds(0))
      : bodies_(std::move(bodies)), status_(status), reply_delay_(reply_delay) {
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
    for (const auto& body : bodies_) {
      if (bodies_.size() > 1) {
        pollfd ready{fd_, POLLIN, 0};
        if (::poll(&ready, 1, 3000) <= 0) return;
      }
      int client = ::accept(fd_, nullptr, nullptr);
      if (client < 0) return;
      std::string current_request;
      char buffer[4096];
      size_t expected = 0;
      while (true) {
        ssize_t n = ::recv(client, buffer, sizeof(buffer), 0);
        if (n <= 0) break;
        current_request.append(buffer, static_cast<size_t>(n));
        const size_t split = current_request.find("\r\n\r\n");
        if (split != std::string::npos) {
          if (expected == 0) {
            const std::string header = "Content-Length: ";
            size_t index = current_request.find(header);
            if (index != std::string::npos) {
              expected = static_cast<size_t>(
                  std::stoul(current_request.substr(index + header.size())));
            }
          }
          if (current_request.size() >= split + 4 + expected) break;
        }
      }
      request_ += current_request;
      std::this_thread::sleep_for(reply_delay_);
      const std::string reply = "HTTP/1.0 " + std::to_string(status_) +
          (status_ == 200 ? " OK" : " Failure") + "\r\nContent-Type: application/json\r\n"
          "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
      size_t offset = 0;
      while (offset < reply.size()) {
        ssize_t n = ::send(client, reply.data() + offset, reply.size() - offset, MSG_NOSIGNAL);
        if (n <= 0) break;
        offset += static_cast<size_t>(n);
      }
      ::close(client);
    }
  }
  std::vector<std::string> bodies_;
  int status_;
  std::chrono::milliseconds reply_delay_;
  int fd_ = -1;
  unsigned port_ = 0;
  std::thread thread_;
  std::string request_;
};

std::string answer(
    const std::string& choice,
    const std::string& model = "nvfuser-analysis-v1") {
  return "{\"model\":\"" + model + "\",\"answers\":{\"PW2\":{\"type\":\"choice\","
      "\"choice\":\"" + choice + "\",\"confidence\":1.0,\"probabilities\":{\"bp0\":0.0,"
      "\"bp1\":1.0}}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}";
}

std::string prefetchAnswer(
    const std::string& choice,
    const std::vector<nvfuser::normalization_inner::tma::PrefetchDistanceRange>&
        ranges) {
  std::string probabilities = "{";
  for (const auto& range : ranges) {
    if (probabilities.size() > 1) probabilities += ',';
    probabilities += "\"" + range.id() + "\":" +
        (range.id() == choice ? "1.0" : "0.0");
  }
  probabilities += '}';
  return "{\"model\":\"nvfuser-analysis-v1\",\"answers\":{\"IP3.prefetch\":{"
      "\"type\":\"choice\",\"choice\":\"" + choice +
      "\",\"confidence\":1.0,\"probabilities\":" + probabilities +
      "}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}";
}

std::vector<std::string> innerRangeReplies(
    std::string_view site,
    int64_t high,
    int64_t target) {
  using nvfuser::reduction::non_tma::innerUnrollRanges;
  std::vector<std::string> replies;
  int64_t low = 1;
  while (true) {
    const auto ranges = innerUnrollRanges(low, high);
    std::string selected;
    std::string probabilities = "{";
    for (const auto& range : ranges) {
      if (!selected.empty() && range.low <= target && target <= range.high) {
        throw std::runtime_error("RD3 target matched two ranges");
      }
      if (range.low <= target && target <= range.high) {
        selected = range.id;
      }
      if (probabilities.size() > 1) probabilities += ',';
      probabilities += "\"" + range.id + "\":" +
          (range.low <= target && target <= range.high ? "1.0" : "0.0");
    }
    require(!selected.empty(), "RD3 target is outside range hierarchy");
    probabilities += '}';
    replies.push_back(
        "{\"model\":\"nvfuser-analysis-v1\",\"answers\":{\"" +
        std::string(site) + "\":{\"type\":\"choice\",\"choice\":\"" +
        selected + "\",\"confidence\":1.0,\"probabilities\":" +
        probabilities +
        "}},\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    const auto found = std::find_if(
        ranges.begin(), ranges.end(), [&](const auto& range) {
          return range.id == selected;
        });
    if (found->low == found->high) break;
    low = found->low;
    high = found->high;
  }
  return replies;
}

void testInnerRemainderPlateauScan() {
  using nvfuser::reduction::non_tma::InnerRemainderScanState;
  using nvfuser::reduction::non_tma::positiveCeilDiv;
  using nvfuser::reduction::non_tma::scanInnerRemainderWidthByPlateaus;
  const auto reference = [](int64_t after_block,
                            int64_t target_unroll,
                            int64_t candidate_x,
                            InnerRemainderScanState& state) {
    for (int64_t factor = target_unroll; factor > 1; --factor) {
      const int64_t remainder = after_block % factor;
      if (remainder == 0) {
        state.block_x = candidate_x;
        state.unroll = factor;
        break;
      }
      if (remainder < state.current_remainder) {
        state.current_remainder = remainder;
        state.block_x = candidate_x;
        state.unroll = factor;
      }
    }
  };
  const auto compare = [&](int64_t after_vect,
                           int64_t target_unroll,
                           int64_t initial_remainder,
                           int64_t first_x,
                           int64_t last_x) {
    InnerRemainderScanState expected{3, 1, initial_remainder};
    InnerRemainderScanState actual = expected;
    for (int64_t x = first_x; x <= last_x; ++x) {
      const int64_t after_block = positiveCeilDiv(after_vect, x);
      reference(after_block, target_unroll, x, expected);
      scanInnerRemainderWidthByPlateaus(
          after_block, target_unroll, x, actual);
      require(
          expected.block_x == actual.block_x &&
              expected.unroll == actual.unroll &&
              expected.current_remainder == actual.current_remainder,
          "RD3 Record plateau scan differs from descending native scan");
    }
  };
  for (int64_t after_vect = 1; after_vect <= 128; ++after_vect) {
    for (int64_t target_unroll = 2; target_unroll <= 64; ++target_unroll) {
      for (int64_t initial_remainder = 0; initial_remainder <= 4;
           ++initial_remainder) {
        compare(after_vect, target_unroll, initial_remainder, 1, 4);
      }
    }
  }
  // A zero at the first width leaves current_remainder unchanged, so a later
  // nonzero remainder can overwrite the pair chosen by that zero.
  compare(2, 2, 2, 1, 2);
  // Both old replay limits are exceeded: target > 65,536 and 32 widths times
  // target > 2,000,000. The exact scan still agrees with the native loop.
  compare(3'141'592, 70'000, 17, 128, 159);
  require(
      positiveCeilDiv(std::numeric_limits<int64_t>::max(), 2) ==
          4'611'686'018'427'387'904LL &&
          positiveCeilDiv(std::numeric_limits<int64_t>::max(), 3) ==
              3'074'457'345'618'258'603LL,
      "positive RD3 ceil division overflowed at int64 limit");
  InnerRemainderScanState huge{128, 1, 5};
  scanInnerRemainderWidthByPlateaus(
      1'234'567'891, 1'000'000'000, 256, huge);
  require(
      huge.block_x == 256 && huge.unroll >= 2 &&
          huge.unroll <= 1'000'000'000,
      "RD3 high-factor plateau scan did not complete");
}

void testInnerOuterGridRowsAdmission() {
  using nvfuser::inner_outer_multi_wave::GridRowsAdmission;
  using nvfuser::inner_outer_multi_wave::gridRowsAdmission;
  using nvfuser::scheduler::jev::Mode;
  // The producer has already excluded r8 after its full CTA preflight, while
  // one complete r4 search survives. Remote may use it; Record keeps Off's r8.
  require(
      gridRowsAdmission(Mode::Remote, 1, false, true) ==
              GridRowsAdmission::OfferWithoutNative &&
          gridRowsAdmission(Mode::Record, 1, false, true) ==
              GridRowsAdmission::PreserveNative,
      "IOP3 rows admission treated the soft r8 target as hard legality");
  require(
      gridRowsAdmission(Mode::Remote, 2, false, true) ==
              GridRowsAdmission::OfferWithoutNative &&
          gridRowsAdmission(Mode::Remote, 1, true, true) ==
              GridRowsAdmission::OfferWithNative &&
          gridRowsAdmission(Mode::Record, 1, true, true) ==
              GridRowsAdmission::OfferWithNative,
      "IOP3 rows admission lost a fully preflighted Choice menu");
  require(
      gridRowsAdmission(Mode::Remote, 0, false, true) ==
              GridRowsAdmission::RejectRemote &&
          gridRowsAdmission(Mode::Remote, 1, false, false) ==
              GridRowsAdmission::RejectRemote &&
          gridRowsAdmission(Mode::Record, 1, true, false) ==
              GridRowsAdmission::PreserveNative &&
          gridRowsAdmission(Mode::Off, 1, true, true) ==
              GridRowsAdmission::PreserveNative,
      "IOP3 rows admission accepted an empty/oversized request or changed Off");
}

} // namespace

int main() {
  testHopperLowStageMenu();
  testInnerSmemCtaNativeSeedGate();
  testTmaTemporaryReadWidthContinuation();
  testInner3dIterUnrollContinuation();
  testInnerRemainderPlateauScan();
  testInnerOuterGridRowsAdmission();
  using nvfuser::inner_outer_mode_preflight::Action;
  using nvfuser::inner_outer_mode_preflight::actionForValidWs;
  using nvfuser::scheduler::jev::Mode;
  require(
      actionForValidWs(Mode::Record, false, false) ==
          Action::PreserveNativeMultiwave,
      "Record rejected its native multiwave plan after preview failure");
  require(
      actionForValidWs(Mode::Record, false, true) == Action::PreserveNativeWs,
      "Record rejected its native WS plan after preview failure");
  require(
      actionForValidWs(Mode::Remote, false, false) == Action::RejectRemote &&
          actionForValidWs(Mode::Remote, false, true) == Action::RejectRemote,
      "Remote offered an unproved multiwave alternative");
  require(
      actionForValidWs(Mode::Record, true, false) == Action::OfferChoice &&
          actionForValidWs(Mode::Remote, true, false) == Action::OfferChoice,
      "hard-valid IOP1 alternatives lost their Choice");
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
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope("off input", {});
    require(!scope.observing(), "off mode enabled compute observation");
    nvfuser::scheduler::jev::recordNoChoiceDecision(
        "RD5.work_split", "unroll_first", "equal_work_factors",
        "native_unroll_first_branch");
    scope.complete("Reduction");
  }
  require(
      chooseDecision("PW2", state, options, "bp0") == "bp0",
      "off mode changed choice");
  require(
      readFile(path).empty() &&
          readFile(std::string(path) + ".outcome.jsonl").empty(),
      "off mode wrote a record");

  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  require(
      nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Record,
      "mode() did not report record");
  const std::string before_native_scope = readFile(path);
  {
    nvfuser::scheduler::jev::ScopedNativeDecisions native_scope;
    require(
        nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Off,
        "native scope did not force off mode");
    {
      nvfuser::scheduler::jev::ScopedNativeDecisions nested_scope;
      require(
          chooseDecision("PW2", state, options, "bp1") == "bp1",
          "nested native scope changed the native choice");
    }
    require(
        nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Off,
        "nested native scope restored configured mode too early");
  }
  require(
      nvfuser::scheduler::jev::mode() ==
              nvfuser::scheduler::jev::Mode::Record &&
          readFile(path) == before_native_scope,
      "native scope did not restore Record without a Choice log");
  try {
    nvfuser::scheduler::jev::ScopedNativeDecisions native_scope;
    throw std::runtime_error("test exception");
  } catch (const std::runtime_error&) {
    require(
        nvfuser::scheduler::jev::mode() ==
            nvfuser::scheduler::jev::Mode::Record,
        "exception did not restore the configured mode");
  }
  require(
      chooseDecision("PW2", state, options, "bp0", "case-1") == "bp0",
      "record changed choice");
  require(
      readFile(path).find("\"case_key\":\"case-1\"") != std::string::npos,
      "record lacks case key");
  {
    const std::string before_plain = readFile(path);
    require(chooseDecision("PW2", state, options, "bp0") == "bp0",
            "text-v1 plain Record changed native choice");
    const std::string plain_request =
        loggedRequest(readFile(path).substr(before_plain.size()));
    const std::vector<ChoiceOption> enriched{
        {"bp0", "one dimensional", "{\"tile\":1}"},
        {"bp1", "split", "{\"tile\":2}"}};
    const std::string before_enriched = readFile(path);
    require(chooseDecision("PW2", state, enriched, "bp0") == "bp0",
            "text-v1 enriched Record changed native choice");
    require(
        loggedRequest(readFile(path).substr(before_enriched.size())) ==
            plain_request,
        "text-v1 wire bytes changed when consequences were supplied");
  }
  {
    // An fp32 reduction with MUFU can prefer u4 even when 128 elements and a
    // 256-thread CTA leave only u1 in the per-thread Choice menu.
    const std::vector<ChoiceOption> short_reduction{
        {"u1", "Unroll 1 reduction element per thread"}};
    const std::string before = readFile(path);
    int64_t native_unroll = 4;
    const auto selected =
        nvfuser::reduction::tma::chooseInnerUnrollIfRepresentable(
            "reduction.tma_inner_unroll.v1\nreduction_numel=128\n"
            "threads_per_block=256\n",
            short_reduction,
            native_unroll);
    if (selected) {
      native_unroll = std::stoll(selected->substr(1));
    }
    require(
        !selected && native_unroll == 4,
        "RD2 Record changed an unrepresentable native unroll");
    const std::string delta = readFile(path).substr(before.size());
    require(
        delta.find("\"kind\":\"coverage_skip\",\"site\":\"RD2.unroll\"") !=
                std::string::npos &&
            delta.find("native unroll exceeds per-thread reduction menu") !=
                std::string::npos &&
            delta.find("\"kind\":\"choice\"") == std::string::npos,
        "RD2 Record did not log only a coverage skip");
    const auto offered =
        nvfuser::reduction::tma::chooseInnerUnrollIfRepresentable(
            "reduction.tma_inner_unroll.v1\nreduction_numel=1024\n"
            "threads_per_block=256\n",
            {{"u1", "Unroll 1 reduction element per thread"},
             {"u2", "Unroll 2 reduction elements per thread"},
             {"u4", "Unroll 4 reduction elements per thread"}},
            4);
    require(
        offered && *offered == "u4",
        "RD2 Record failed to retain an offered native unroll");
  }
  require(
      readFile(path).find("\"request\":{\"model\":\"jev-latest\"") !=
          std::string::npos,
      "record lacks the model-visible request");
  require(
      readFile(path).find(
          "\"instructions\":\"Choose nvFuser scheduler decision PW2\"") !=
          std::string::npos,
      "record lacks Choice instructions");
  require(
      readFile(path).find(
          "\"criteria\":{\"bp0\":\"one dimensional\",\"bp1\":\"split\"}") !=
          std::string::npos,
      "record lacks candidate descriptions");
  nvfuser::scheduler::jev::recordCoverageSkip(
      "PW4.tile",
      "native tile exceeds conservative shared-memory bound",
      "case-skip");
  require(
      readFile(path).find("\"kind\":\"coverage_skip\",\"site\":\"PW4.tile\"") !=
              std::string::npos &&
          readFile(path).find(
              "\"reason\":\"native tile exceeds conservative shared-memory "
              "bound\"") != std::string::npos,
      "record coverage skip lacks a reason");
  const std::string first_record = readFile(path);
  require(
      chooseDecision("PW2", state, options, "bp1", "case-1b") == "bp1",
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

  // The generic compute span uses a different ID/index from Pointwise's
  // existing Record span. A nested failed call must restore its parent.
  const std::string compute_path = std::string(path) + ".outcome.jsonl";
  std::string outer_id;
  std::string nested_id;
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope outer("outer input", {});
    require(outer.observing(), "generic compute span was not enabled");
    outer_id = outer.invocationId();
    require(chooseDecision("PW2", state, options, "bp0") == "bp0",
            "generic span changed Record selection");
    try {
      nvfuser::scheduler::jev::ComputeHeuristicsScope nested("inner input", {});
      nested_id = nested.invocationId();
      nvfuser::scheduler::jev::recordCoverageSkip("RD5", "nested coverage");
      throw std::runtime_error("aborted nested compute");
    } catch (const std::runtime_error&) {
    }
    require(chooseDecision("PW2", state, options, "bp1") == "bp1",
            "nested span changed parent Record selection");
    outer.complete("PointWise");
  }
  const std::string compute_trace = readFile(compute_path);
  const std::string compute_choices = readFile(path);
  require(!outer_id.empty() && !nested_id.empty() && outer_id != nested_id,
          "generic compute invocation IDs were not unique");
  require(compute_trace.find("\"compute_invocation_id\":\"" + outer_id +
                                 "\",\"compute_event_index\":0,\"input_snapshot\":\"outer input\"") !=
              std::string::npos &&
              compute_trace.find("\"compute_invocation_id\":\"" + outer_id +
                                     "\",\"compute_event_index\":3,\"scheduler_type\":\"PointWise\"") !=
                  std::string::npos,
          "generic compute begin/final or event indexes are incomplete");
  require(compute_trace.find("\"compute_invocation_id\":\"" + nested_id +
                                 "\",\"compute_event_index\":0,\"input_snapshot\":\"inner input\"") !=
              std::string::npos &&
              compute_trace.find("\"compute_invocation_id\":\"" + nested_id +
                                     "\",\"compute_event_index\":2,\"scheduler_type\"") ==
                  std::string::npos,
          "failed nested compute acquired a final row");
  require(compute_choices.find("\"compute_invocation_id\":\"" + outer_id +
                                  "\",\"compute_event_index\":1") !=
              std::string::npos &&
              compute_choices.find("\"compute_invocation_id\":\"" + outer_id +
                                      "\",\"compute_event_index\":2") !=
                  std::string::npos &&
              compute_choices.find("\"compute_invocation_id\":\"" + nested_id +
                                      "\",\"compute_event_index\":1") !=
                  std::string::npos &&
              compute_choices.find("outer input") == std::string::npos,
          "generic compute IDs or snapshot isolation changed the Choice log");
  // A structural no-choice event occupies the same compute sequence but only
  // the outcome stream. It must not become a Choice or HTTP request.
  const std::string decisions_before_no_choice = readFile(path);
  std::string no_choice_id;
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope("no-choice input", {});
    no_choice_id = scope.invocationId();
    nvfuser::scheduler::jev::recordNoChoiceDecision(
        "RD3.remainder_x", "x128", "single_cta_width", "native_remainder_scan");
    nvfuser::scheduler::jev::recordNoChoiceDecision(
        "RD3.remainder_u", "u1", "single_cta_width", "native_remainder_scan");
    scope.complete("Reduction");
  }
  const std::string no_choice_trace = readFile(compute_path);
  require(readFile(path) == decisions_before_no_choice &&
              no_choice_trace.find("\"kind\":\"no_choice\",\"mode\":\"record\"") !=
                  std::string::npos &&
              no_choice_trace.find(
                  "\"compute_invocation_id\":\"" + no_choice_id +
                  "\",\"compute_event_index\":1,\"site\":\"RD3.remainder_x\",\"selected\":\"x128\",\"reason\":\"single_cta_width\",\"selection_source\":\"native_remainder_scan\"") !=
                  std::string::npos &&
              no_choice_trace.find(
                  "\"compute_invocation_id\":\"" + no_choice_id +
                  "\",\"compute_event_index\":2,\"site\":\"RD3.remainder_u\",\"selected\":\"u1\"") !=
                  std::string::npos &&
              no_choice_trace.find(
                  "\"compute_invocation_id\":\"" + no_choice_id +
                  "\",\"compute_event_index\":3,\"scheduler_type\":\"Reduction\"") !=
                  std::string::npos,
          "deterministic Record events changed Choice or compute sequencing");
  const std::string decisions_before_unroll = readFile(path);
  std::string record_unroll_id;
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "bounded pointwise unroll", {});
    record_unroll_id = scope.invocationId();
    const auto bypass =
        nvfuser::pointwise::non_tma::getPointwiseUnrollEnvelope(100, 4096, 1, 3);
    require(
        bypass.model_limit == 2 && bypass.native_factor == 3 &&
            !bypass.offer_choice,
        "Record over-budget unroll did not preserve the native factor");
    const auto offered =
        nvfuser::pointwise::non_tma::getPointwiseUnrollEnvelope(100, 128, 4, 3);
    require(
        offered.model_limit == 16 && offered.native_factor == 3 &&
            offered.offer_choice,
        "Record in-budget unroll did not offer the bounded menu");
    scope.complete("PointWise");
  }
  const std::string record_unroll_trace = readFile(compute_path);
  require(
      readFile(path) == decisions_before_unroll &&
          record_unroll_trace.find(
              "\"compute_invocation_id\":\"" + record_unroll_id +
              "\",\"compute_event_index\":1,\"site\":\"PW3.unroll\",\"selected\":\"u3\",\"reason\":\"native factor exceeds bounded model resource envelope\",\"selection_source\":\"native_unroll\"") !=
              std::string::npos &&
          record_unroll_trace.find(
              "\"compute_invocation_id\":\"" + record_unroll_id +
              "\",\"compute_event_index\":2,\"scheduler_type\":\"PointWise\"") !=
              std::string::npos,
      "Record over-budget unroll emitted a Choice or lost its no_choice row");
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

  // Every positive distance remains reachable, including both endpoints of
  // an interval and a range near the int64 limit.
  for (const int64_t stage_count : {int64_t{257}, int64_t{100000},
                                    std::numeric_limits<int64_t>::max()}) {
    for (const int64_t distance :
         {int64_t{1}, (stage_count - 1) / 2, stage_count - 1}) {
      int64_t low = 1;
      int64_t high = stage_count - 1;
      for (int depth = 0; low != high; ++depth) {
        require(depth < 10, "IP3.prefetch hierarchy did not converge");
        const auto ranges =
            nvfuser::normalization_inner::tma::prefetchDistanceRanges(low, high);
        require(!ranges.empty() && ranges.size() <= 255 &&
                    ranges.front().low == low && ranges.back().high == high,
                "IP3.prefetch range partition omitted a boundary");
        int64_t covered = low;
        bool found = false;
        for (const auto& range : ranges) {
          require(range.low == covered && range.high >= range.low,
                  "IP3.prefetch ranges overlap or leave a gap");
          covered = range.high + 1;
          if (range.low <= distance && distance <= range.high) {
            low = range.low;
            high = range.high;
            found = true;
          }
        }
        require(found, "IP3.prefetch cannot reach a positive distance");
      }
      require(low == distance, "IP3.prefetch ended at the wrong distance");
    }
  }
  const std::string before_large_record = readFile(path);
  require(
      nvfuser::normalization_inner::tma::chooseLargePrefetchDistance(257) == 256,
      "IP3.prefetch Record did not follow the native distance");
  const std::string large_record = readFile(path).substr(before_large_record.size());
  require(
      large_record.find("inner_persistent.tma_prefetch.v2\\nstage_count=257\\nrange_low=1\\nrange_high=256") !=
              std::string::npos &&
          large_record.find("\"selected\":\"r255_256\"") !=
              std::string::npos &&
          large_record.find("\"selected\":\"p256\"") !=
              std::string::npos,
      "IP3.prefetch Record omitted a native hierarchy level");

  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Remote,
          "mode() did not report remote");
  {
    nvfuser::scheduler::jev::ScopedNativeDecisions native_scope;
    require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Off &&
                chooseDecision("PW2", state, options, "bp0") == "bp0",
            "Remote native scope did not bypass the server");
  }
  require(nvfuser::scheduler::jev::mode() == nvfuser::scheduler::jev::Mode::Remote,
          "Remote native scope did not restore the configured mode");
  {
    const auto first =
        nvfuser::normalization_inner::tma::prefetchDistanceRanges(1, 256);
    const auto second =
        nvfuser::normalization_inner::tma::prefetchDistanceRanges(1, 2);
    FakeServer server(std::vector<std::string>{
        prefetchAnswer("r1_2", first), prefetchAnswer("p1", second)});
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    const std::string before_large_remote = readFile(path);
    require(
        nvfuser::normalization_inner::tma::chooseLargePrefetchDistance(257) == 1,
        "IP3.prefetch Remote could not choose a non-native small distance");
    const std::string large_remote =
        readFile(path).substr(before_large_remote.size());
    require(
        large_remote.find("\"selected\":\"r1_2\"") !=
                std::string::npos &&
            large_remote.find("\"selected\":\"p1\"") !=
                std::string::npos &&
            server.request().find("Prefetch TMA in [1, 2] iteration(s) ahead") !=
                std::string::npos,
        "IP3.prefetch Remote lost a hierarchy request or continuation");
  }
  {
    // Exercise the exact RD3 helpers used by the scheduler with a target
    // above the former 65,536 cap, then choose a small downstream unroll.
    auto replies = innerRangeReplies("RD3.unroll", 70'000, 70'000);
    const size_t unroll_requests = replies.size();
    auto remainder_replies = innerRangeReplies("RD3.remainder_u", 70'000, 1);
    replies.insert(
        replies.end(), remainder_replies.begin(), remainder_replies.end());
    FakeServer server(std::move(replies));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    const std::string before = readFile(path);
    const int64_t target =
        nvfuser::reduction::non_tma::chooseInnerUnrollTarget(
            "reduction.inner_unroll.v1\nreduction_numel=70000\n",
            70'000,
            0,
            nvfuser::scheduler::jev::Mode::Remote);
    require(target == 70'000, "RD3 Remote lost high target continuation");
    const int64_t remainder =
        nvfuser::reduction::non_tma::chooseInnerRemainderRange(
            "RD3.remainder_u",
            "reduction.inner_remainder_u.v1\nafter_vect=70000\n"
            "target_inner_unroll=70000\n",
            1,
            target,
            0,
            nvfuser::scheduler::jev::Mode::Remote);
    require(remainder == 1, "RD3 Remote lost a small downstream factor");
    const std::string delta = readFile(path).substr(before.size());
    require(
        unroll_requests >= 2 && remainder_replies.size() >= 2 &&
            delta.find("\"selected\":\"u70000\"") != std::string::npos &&
            delta.find("\"selected\":\"u1\"") != std::string::npos &&
            server.request().find("RD3.unroll") != std::string::npos &&
            server.request().find("RD3.remainder_u") != std::string::npos &&
            server.request().find("range_high=70000") != std::string::npos,
        "RD3 Remote omitted a high-range or downstream request");
  }
  {
    FakeServer server(
        "{\"model\":\"nvfuser-analysis-v1\",\"answers\":{\"RD2.unroll\":{"
        "\"type\":\"choice\",\"choice\":\"u1\",\"confidence\":1.0,"
        "\"probabilities\":{\"u1\":1.0}}},"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    const auto selected =
        nvfuser::reduction::tma::chooseInnerUnrollIfRepresentable(
            "reduction.tma_inner_unroll.v2\nreduction_numel=128\n"
            "threads_per_block=256\n",
            {{"u1", "Unroll 1 reduction element per thread"}},
            0);
    require(
        selected && *selected == "u1" &&
            server.request().find("\"criteria\":{\"u1\":") != std::string::npos,
        "RD2 Remote did not retain the structural menu");
  }
  {
    nvfuser::scheduler::jev::PointwiseRecordScope scope;
    require(!scope.recording(), "remote mode enabled Pointwise Record diagnostics");
    scope.complete("ordinary", "{}", "{}", "{}");
    require(readFile(outcome_path) == outcome,
            "remote mode changed the Pointwise Record outcome log");
  }
  std::string remote_compute_id;
  {
    FakeServer server(answer("bp\\u0031"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    nvfuser::scheduler::jev::ComputeHeuristicsScope compute_scope(
        "private remote snapshot", {});
    remote_compute_id = compute_scope.invocationId();
    require(chooseDecision("PW2", state, options, {}, "case-2") == "bp1",
            "remote Unicode-escaped choice was not decoded");
    compute_scope.complete("PointWise");
    require(server.request().find("POST /v1/systemone HTTP/1.0") == 0, "wrong endpoint");
    require(server.request().find("\"questions\":{\"PW2\"") != std::string::npos,
            "missing Choice question");
    require(server.request().find("native") == std::string::npos, "native winner leaked");
    require(server.request().find("winner") == std::string::npos, "winner leaked");
    require(server.request().find("private remote snapshot") == std::string::npos &&
                server.request().find("compute_invocation_id") ==
                    std::string::npos,
            "diagnostic compute fields leaked into the HTTP request");
  }
  const std::string log = readFile(path);
  require(log.find("\"selected\":\"bp1\",\"mode\":\"remote\"") != std::string::npos,
          "remote choice not logged");
  require(log.find("\"roundtrip_us\":") != std::string::npos, "missing latency");
  require(!remote_compute_id.empty() &&
              log.find("\"compute_invocation_id\":\"" + remote_compute_id +
                           "\",\"compute_event_index\":1") !=
                  std::string::npos &&
              readFile(compute_path).find(
                  "\"compute_invocation_id\":\"" + remote_compute_id +
                  "\",\"compute_event_index\":2,\"scheduler_type\":\"PointWise\"") !=
                  std::string::npos,
          "remote Choice and compute outcome cannot join");
  // The Codex mode uses the same Choice envelope, but its model identifier
  // must remain truthful and the request may take longer than the mock server.
  {
    ::setenv("NVFUSER_JEV_TIMEOUT_MS", "250", 1);
    FakeServer server(
        answer("bp1", "codex-gpt-6.1-sol-max"), 200,
        std::chrono::milliseconds(40));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    require(chooseDecision("PW2", state, options) == "bp1",
            "Codex model response or longer loopback deadline was rejected");
    ::unsetenv("NVFUSER_JEV_TIMEOUT_MS");
  }
  {
    FakeServer server(answer("bp1", "codex-gpt-6.1-sol-max-extra"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    bool rejected = false;
    try { (void)chooseDecision("PW2", state, options); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "unrecognized Codex response model was accepted");
  }
  {
    ::setenv("NVFUSER_JEV_TIMEOUT_MS", "5", 1);
    FakeServer server(
        answer("bp1"), 200, std::chrono::milliseconds(40));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    bool timed_out = false;
    try { (void)chooseDecision("PW2", state, options); }
    catch (const std::runtime_error& error) {
      timed_out = std::string(error.what()).find("loopback API timeout") !=
          std::string::npos;
    }
    require(timed_out, "short loopback deadline did not time out");
    ::unsetenv("NVFUSER_JEV_TIMEOUT_MS");
  }
  for (const char* valid : {"600001", "3600000"}) {
    ::setenv("NVFUSER_JEV_TIMEOUT_MS", valid, 1);
    FakeServer server(answer("bp1"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    require(chooseDecision("PW2", state, options) == "bp1",
            "extended loopback timeout was rejected");
  }
  for (const char* invalid : {"0", "3600001", "invalid"}) {
    ::setenv("NVFUSER_JEV_TIMEOUT_MS", invalid, 1);
    bool rejected = false;
    try { (void)chooseDecision("PW2", state, options); }
    catch (const std::runtime_error& error) {
      rejected = std::string(error.what()).find("NVFUSER_JEV_TIMEOUT_MS") !=
          std::string::npos;
    }
    require(rejected, "invalid loopback timeout was accepted");
  }
  ::unsetenv("NVFUSER_JEV_TIMEOUT_MS");
  // Semantic-v2 sends the same model-visible object in Record and Remote.
  // A real Jev response may resolve jev-latest to a versioned model name.
  ::setenv("NVFUSER_JEV_PROTOCOL", "semantic-v2", 1);
  const std::string semantic_state =
      "{\"workload\":{\"fusion_and_runtime\":\"full snapshot\"},"
      "\"decision\":{\"kind\":\"vector_width\"}}";
  const std::vector<ChoiceOption> width_options{
      {"w1", "one element per thread"}, {"w2", "two elements per thread"}};
  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  {
    const std::string before = readFile(path);
    require(nvfuser::scheduler::jev::chooseSemanticDecision(
                "PW3.width", semantic_state, "Choose vector width for this fusion",
                {{"w1", "one element per thread"}}, "w1") == "w1" &&
                readFile(path).substr(before.size()).find(
                    "{\"site\":\"PW3.width\"") == 0,
            "semantic-v2 singleton stopped using its existing Choice protocol");
  }
  const std::string before_semantic_record = readFile(path);
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope("full snapshot", {});
    require(nvfuser::scheduler::jev::computeInvocationInputSnapshot() ==
                "full snapshot",
            "semantic scope lost entry snapshot");
    require(nvfuser::scheduler::jev::chooseSemanticDecision(
                "PW3.width", semantic_state, "Choose vector width for this fusion",
                width_options, "w2") == "w2",
            "semantic Record changed native choice");
  }
  require(nvfuser::scheduler::jev::computeInvocationInputSnapshot().empty() &&
              nvfuser::scheduler::jev::computeInvocationInputUnavailableReason() ==
                  "no_compute_scope",
          "semantic snapshot escaped its compute scope");
  const std::string semantic_record = readFile(path).substr(before_semantic_record.size());
  require(semantic_record.find("\"state\":{") != std::string::npos &&
              semantic_record.find("\"native_choice\"") == std::string::npos,
          "semantic Record did not preserve a winner-free object state");
  {
    const std::vector<ChoiceOption> enriched{
        {"w1", "one element per thread", "not JSON"},
        {"w2", "two elements per thread", "[]"}};
    const std::string before_enriched = readFile(path);
    require(nvfuser::scheduler::jev::chooseSemanticDecision(
                "PW3.width", semantic_state, "Choose vector width for this fusion",
                enriched, "w2") == "w2",
            "semantic-v2 enriched Record changed native choice");
    require(
        loggedRequest(readFile(path).substr(before_enriched.size())) ==
            loggedRequest(semantic_record),
        "semantic-v2 wire bytes changed when consequences were supplied");
  }
  {
    const std::string before_overflow = readFile(path);
    const std::string oversized = "{\"payload\":\"" +
        std::string(70 * 1024, 'x') + "\"}";
    bool rejected = false;
    try {
      (void)nvfuser::scheduler::jev::chooseSemanticDecision(
          "PW3.width", oversized, "Choose width", width_options, "w2");
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && readFile(path) == before_overflow,
            "oversized semantic state was shortened or logged");
  }
  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  {
    FakeServer server(
        "{\"model\":\"jev-1.13.0\",\"answers\":{\"PW3.width\":{"
        "\"type\":\"choice\",\"choice\":\"w2\",\"confidence\":1.0,"
        "\"probabilities\":{\"w1\":0.0,\"w2\":1.0}}},"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    require(nvfuser::scheduler::jev::chooseSemanticDecision(
                "PW3.width", semantic_state, "Choose vector width for this fusion",
                width_options) == "w2",
            "versioned real-Jev model response was rejected");
    const auto body_start = server.request().find("\r\n\r\n");
    require(body_start != std::string::npos, "semantic HTTP body missing");
    const auto body = server.request().substr(body_start + 4);
    require(semantic_record.find("\"request\":" + body) != std::string::npos &&
                body.find("full snapshot") != std::string::npos &&
                body.find("native_choice") == std::string::npos,
            "Record and Remote sent different semantic requests");
  }
  {
    FakeServer server(
        "{\"model\":\"codex-gpt-6.1-sol-max\",\"answers\":{\"PW3.width\":{"
        "\"type\":\"choice\",\"choice\":\"w1\",\"confidence\":1.0,"
        "\"probabilities\":{\"w1\":1.0,\"w2\":0.0}}},"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    require(nvfuser::scheduler::jev::chooseSemanticDecision(
                "PW3.width", semantic_state, "Choose vector width for this fusion",
                width_options) == "w1",
            "semantic Codex model response was rejected");
  }
  // semantic-v3 wraps every text Choice. It carries a full invocation
  // snapshot, hardware, prior selections, and an honest verification marker.
  ::setenv("NVFUSER_JEV_PROTOCOL", "semantic-v3", 1);
  const std::string v3_hardware = "{\"sm_count\":128}";
  const std::vector<ChoiceOption> reverse_options{
      {"bp1", "split"}, {"bp0", "one dimensional"}};
  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  const std::vector<ChoiceOption> sole_grid{
      {"g1", "Use the only source-offered grid", "{\"grid_y\":1}"}};
  const std::string singleton_reason =
      "only one source-offered candidate at this step";
  const std::string singleton_source = "source_menu_singleton";
  {
    const std::string choices_before = readFile(path);
    const std::string outcomes_before = readFile(std::string(path) + ".outcome.jsonl");
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "singleton Record fusion", {}, v3_hardware);
    const std::string invocation_id(scope.invocationId());
    require(chooseDecision("RD3.grid", "one hard-valid grid", sole_grid, "g1") ==
                "g1" &&
                readFile(path) == choices_before,
            "semantic-v3 Record singleton emitted a Choice");
    require(chooseDecision("PW2", state, reverse_options, "bp0") == "bp0",
            "semantic-v3 Record downstream Choice changed");
    const std::string downstream = readFile(path).substr(choices_before.size());
    require(downstream.find("\"prior_choices\":[{\"site\":\"RD3.grid\","
                            "\"choice\":\"g1\"}]") != std::string::npos &&
                downstream.find("{\"site\":\"PW2\"") == 0,
            "semantic-v3 Record singleton was not retained as downstream prior");
    scope.complete("Reduction");
    const std::string outcomes =
        readFile(std::string(path) + ".outcome.jsonl").substr(outcomes_before.size());
    require(outcomes.find(
                "\"compute_invocation_id\":\"" + invocation_id +
                "\",\"compute_event_index\":1,\"site\":\"RD3.grid\","
                "\"selected\":\"g1\",\"reason\":\"" + singleton_reason +
                "\",\"selection_source\":\"" + singleton_source + "\"") !=
                std::string::npos &&
                outcomes.find("\"kind\":\"no_choice\",\"mode\":\"record\"") !=
                    std::string::npos &&
                outcomes.find("\"compute_event_index\":3,\"scheduler_type\":"
                              "\"Reduction\"") != std::string::npos,
            "semantic-v3 Record singleton outcome or event sequence is incomplete");
  }
  {
    const std::string choices_before = readFile(path);
    const std::string outcomes_before = readFile(std::string(path) + ".outcome.jsonl");
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "singleton validation fusion", {}, v3_hardware);
    const auto rejected = [&](std::string_view analysis,
                              const std::vector<ChoiceOption>& candidates,
                              std::string_view native) {
      try {
        (void)chooseDecision("RD3.grid", analysis, candidates, native);
      } catch (const std::runtime_error&) {
        return true;
      }
      return false;
    };
    require(rejected("one grid", sole_grid, "g2") &&
                rejected("", sole_grid, "g1") &&
                rejected("one grid", {{"g1", "grid", "[]"}}, "g1") &&
                readFile(path) == choices_before &&
                readFile(std::string(path) + ".outcome.jsonl").substr(
                    outcomes_before.size()).find("\"kind\":\"no_choice\"") ==
                    std::string::npos,
            "semantic-v3 singleton bypassed native, analysis, or consequence validation");
  }
  {
    const std::string choices_before = readFile(path);
    bool rejected = false;
    try {
      (void)chooseDecision("RD3.grid", "one grid", sole_grid, "g1");
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected && readFile(path) == choices_before,
            "semantic-v3 singleton bypassed the required compute scope");
  }
  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  {
    const std::string choices_before = readFile(path);
    const std::string outcomes_before = readFile(std::string(path) + ".outcome.jsonl");
    ::setenv("NVFUSER_JEV_PORT", "0", 1);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "singleton Remote fusion", {}, v3_hardware);
    const std::string invocation_id(scope.invocationId());
    require(chooseDecision("RD3.grid", "one hard-valid grid", sole_grid) ==
                "g1" &&
                readFile(path) == choices_before,
            "semantic-v3 Remote singleton attempted HTTP or emitted a Choice");
    FakeServer server(answer("bp1"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    require(chooseDecision("PW2", state, reverse_options) == "bp1",
            "semantic-v3 Remote downstream Choice was not served");
    const std::string remote_choice = readFile(path).substr(choices_before.size());
    const auto body_start = server.request().find("\r\n\r\n");
    require(remote_choice.find("{\"site\":\"PW2\"") == 0 &&
                remote_choice.find("\n{\"site\":") == std::string::npos &&
                body_start != std::string::npos &&
                server.request().substr(body_start + 4).find(
                    "\"prior_choices\":[{\"site\":\"RD3.grid\","
                    "\"choice\":\"g1\"}]") != std::string::npos,
            "semantic-v3 Remote singleton was not carried into the sole HTTP request");
    scope.complete("Reduction");
    const std::string outcomes =
        readFile(std::string(path) + ".outcome.jsonl").substr(outcomes_before.size());
    require(outcomes.find("\"kind\":\"no_choice\",\"mode\":\"remote\"") !=
                std::string::npos &&
                outcomes.find("\"compute_invocation_id\":\"" + invocation_id +
                              "\",\"compute_event_index\":1,\"site\":\"RD3.grid\","
                              "\"selected\":\"g1\",\"reason\":\"" +
                              singleton_reason + "\",\"selection_source\":\"" +
                              singleton_source + "\"") != std::string::npos,
            "semantic-v3 Remote singleton outcome or prior lineage is incomplete");
  }
  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  using nvfuser::reduction::non_tma::inner_width_choice::layoutCapOption;
  const auto legacy_width_cap = layoutCapOption(3, false);
  const auto contextual_width_cap = layoutCapOption(3, true);
  require(
      legacy_width_cap.id == contextual_width_cap.id &&
          legacy_width_cap.id == "w3" &&
          legacy_width_cap.description == "Use the native layout width cap" &&
          contextual_width_cap.description ==
              "Vectorize 3 element(s) per thread at the layout width cap",
      "RD3 layout cap changed its ID or legacy wording, or leaked the native winner");
  const std::string rd3_width_state =
      "reduction.inner_width.v1\nlayout_max=3\nmax_dtype_bits=32\n"
      "n_tensor_inputs=1\nhas_mufu=0\nsm_count=128\n"
      "max_threads_per_sm=1536\nmemory_bus_bits=384\n"
      "memory_clock_khz=1000000";
  const std::vector<ChoiceOption> rd3_width_options{
      contextual_width_cap, {"w1", "Vectorize 1 element(s) per thread"}};
  std::string rd3_width_record_request;
  {
    const std::string before = readFile(path);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "RD3 synthetic input", {}, v3_hardware);
    require(
        chooseDecision("RD3.width", rd3_width_state, rd3_width_options, "w3") ==
            "w3",
        "RD3 layout cap Record changed the native selection");
    rd3_width_record_request =
        loggedRequest(readFile(path).substr(before.size()));
    require(
        rd3_width_record_request.find(
            "\"criteria\":{\"w1\":\"Vectorize 1 element(s) per thread\","
            "\"w3\":\"Vectorize 3 element(s) per thread at the layout width cap\"}") !=
                std::string::npos &&
            rd3_width_record_request.find("native layout width cap") ==
                std::string::npos,
        "RD3 semantic-v3 request exposed the source-native width hint");
  }
  // IP5.width sends a conditional, unranked batch frontier only through the
  // semantic-v3 envelope. Record and Remote must send identical candidate
  // effects; the native Record answer is not part of either request.
  const std::string ip5_width_state =
      "inner_persistent.reg_width.v2\ninner_numel=64\n"
      "plan=1:1:2\nbatch_plan=1:1:64:2:64:64:32:1:0";
  const std::vector<ChoiceOption> ip5_width_options{
      {"w2", "Vector width 2, checked persistent batches 1 through 2",
       "{\"schema\":\"nvfuser.ip5.width.batch_frontier.v1\","
       "\"vector_width\":2,\"batch_plans\":[{\"persistent_batch\":1}]}"},
      {"w1", "Vector width 1, checked persistent batches 1 through 2",
       "{\"schema\":\"nvfuser.ip5.width.batch_frontier.v1\","
       "\"vector_width\":1,\"batch_plans\":[{\"persistent_batch\":1},"
       "{\"persistent_batch\":2}]}"}};
  std::string ip5_width_record_request;
  {
    const std::string before = readFile(path);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "IP5 synthetic input", {}, v3_hardware);
    require(chooseDecision("IP5.width", ip5_width_state, ip5_width_options,
                           "w2") == "w2",
            "IP5 semantic-v3 Record changed its width");
    ip5_width_record_request =
        loggedRequest(readFile(path).substr(before.size()));
    const std::string exact_cards =
        "\"candidates\":[{\"id\":\"w1\",\"description\":\"" +
        ip5_width_options[1].description + "\",\"consequence\":" +
        ip5_width_options[1].consequence_json +
        "},{\"id\":\"w2\",\"description\":\"" +
        ip5_width_options[0].description + "\",\"consequence\":" +
        ip5_width_options[0].consequence_json + "}]";
    require(
        ip5_width_record_request.find("inner_persistent.reg_width.v2") !=
                std::string::npos &&
            ip5_width_record_request.find(exact_cards) != std::string::npos &&
            ip5_width_record_request.find(
                "\"criteria\":{\"w1\":\"Vector width 1") !=
                std::string::npos &&
            ip5_width_record_request.find("native_choice") ==
                std::string::npos,
        "IP5 semantic-v3 frontier lost candidate data or exposed the winner");
  }
  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  {
    FakeServer server(
        "{\"model\":\"jev-1.13.0\",\"answers\":{\"IP5.width\":{"
        "\"type\":\"choice\",\"choice\":\"w1\",\"confidence\":1.0,"
        "\"probabilities\":{\"w1\":1.0,\"w2\":0.0}}},"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "IP5 synthetic input", {}, v3_hardware);
    require(chooseDecision("IP5.width", ip5_width_state, ip5_width_options) ==
                "w1",
            "IP5 semantic-v3 Remote rejected its width frontier");
    const auto body_start = server.request().find("\r\n\r\n");
    require(body_start != std::string::npos &&
                server.request().substr(body_start + 4) ==
                    ip5_width_record_request,
            "IP5 Record and Remote frontier requests differ");
  }
  // The wider v3 summary keeps its four policy slots in each typed card;
  // semantic transport must preserve those bytes and never add the Record
  // answer to the request sent to the service.
  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  const std::string ip5_width_v3_state =
      "inner_persistent.reg_width.v3\ninner_numel=64\n"
      "policy=1:16:16:1:1:2:64:64:32:32:1:1";
  const std::string ip5_width_v3_effect =
      "{\"schema\":\"nvfuser.ip5.width.target_policy_summaries.v1\","
      "\"policy_summaries\":[{\"register_overhead_target\":16,"
      "\"resident_warps_per_sm_target\":16,\"preflight_available\":true},"
      "{\"register_overhead_target\":16,"
      "\"resident_warps_per_sm_target\":28,\"preflight_available\":false},"
      "{\"register_overhead_target\":32,"
      "\"resident_warps_per_sm_target\":16,\"preflight_available\":false},"
      "{\"register_overhead_target\":32,"
      "\"resident_warps_per_sm_target\":28,\"preflight_available\":false}]}";
  const std::vector<ChoiceOption> ip5_width_v3_options{
      {"w2", "Vector width 2, checked persistent batches 1 through 2",
       ip5_width_v3_effect},
      {"w1", "Vector width 1, checked persistent batches 1 through 2",
       ip5_width_v3_effect}};
  std::string ip5_width_v3_record_request;
  {
    const std::string before = readFile(path);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "IP5 v3 synthetic input", {}, v3_hardware);
    require(
        chooseDecision(
            "IP5.width", ip5_width_v3_state, ip5_width_v3_options, "w2") ==
            "w2",
        "IP5 v3 Record changed its width");
    ip5_width_v3_record_request =
        loggedRequest(readFile(path).substr(before.size()));
    const std::string expected_candidates =
        "\"candidates\":[{\"id\":\"w1\",\"description\":\"" +
        ip5_width_v3_options[1].description + "\",\"consequence\":" +
        ip5_width_v3_effect +
        "},{\"id\":\"w2\",\"description\":\"" +
        ip5_width_v3_options[0].description + "\",\"consequence\":" +
        ip5_width_v3_effect + "}]";
    require(
        ip5_width_v3_record_request.find(expected_candidates) !=
                std::string::npos &&
            ip5_width_v3_record_request.find("native_choice") ==
                std::string::npos,
        "IP5 v3 candidate array changed or exposed the native width");
  }
  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  {
    FakeServer server(
        "{\"model\":\"jev-1.13.0\",\"answers\":{\"IP5.width\":{"
        "\"type\":\"choice\",\"choice\":\"w1\",\"confidence\":1.0,"
        "\"probabilities\":{\"w1\":1.0,\"w2\":0.0}}},"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "IP5 v3 synthetic input", {}, v3_hardware);
    require(chooseDecision("IP5.width", ip5_width_v3_state,
                           ip5_width_v3_options) == "w1",
            "IP5 v3 Remote rejected its policy summaries");
    const auto body_start = server.request().find("\r\n\r\n");
    require(body_start != std::string::npos &&
                server.request().substr(body_start + 4) ==
                    ip5_width_v3_record_request,
            "IP5 v3 Record and Remote requests differ");
  }
  ::setenv("NVFUSER_JEV_MODE", "record", 1);
  ::setenv("NVFUSER_JEV_PROTOCOL", "text-v1", 1);
  {
    const std::string before_singleton = readFile(path);
    require(chooseDecision("IP5.width", "legacy singleton",
                           {{"w1", "Vector width 1"}}, "w1") == "w1" &&
                readFile(path).substr(before_singleton.size()).find(
                    "{\"site\":\"IP5.width\"") == 0,
            "text-v1 singleton stopped using its existing Choice protocol");
    const std::string legacy_state =
        "inner_persistent.reg_width.v1\ninner_numel=64\nplan=1:1:2";
    const std::string before_plain = readFile(path);
    require(chooseDecision("IP5.width", legacy_state,
                           {{"w2", ip5_width_options[0].description},
                            {"w1", ip5_width_options[1].description}},
                           "w2") == "w2",
            "IP5 text-v1 Record changed its native width");
    const std::string plain =
        loggedRequest(readFile(path).substr(before_plain.size()));
    const std::string before_enriched = readFile(path);
    require(chooseDecision("IP5.width", legacy_state, ip5_width_options,
                           "w2") == "w2",
            "IP5 text-v1 enriched options changed the width");
    require(loggedRequest(readFile(path).substr(before_enriched.size())) ==
                plain,
            "IP5 text-v1 wire bytes changed with semantic consequences");
  }
  ::setenv("NVFUSER_JEV_PROTOCOL", "semantic-v3", 1);
  const std::string before_v3_record = readFile(path);
  nvfuser::scheduler::jev::recordContextFallback("snapshot_too_large");
  require(
      readFile(path).find(
          "\"schema\":\"nvfuser.jev.context_fallback.v1\","
          "\"kind\":\"context_fallback\",\"site\":\"semantic-v3\","
          "\"selected\":\"native\",\"reason\":\"snapshot_too_large\"") !=
          std::string::npos,
      "semantic-v3 native fallback was absent from the decision log");
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "complete fusion and runtime", {}, v3_hardware);
    require(chooseDecision("PW2", state, reverse_options, "bp1") == "bp1",
            "semantic-v3 Record changed native choice");
    require(chooseDecision("PW2", state, reverse_options, "bp0") == "bp0",
            "semantic-v3 prior selection changed native choice");
  }
  const auto v3_record = readFile(path).substr(before_v3_record.size());
  require(v3_record.find("\"schema\":\"nvfuser.systemone.context.v3\"") !=
                  std::string::npos &&
              v3_record.find("\"verification_status\":\"analysis_summary_only\"") !=
                  std::string::npos &&
              v3_record.find("\"fusion_and_runtime\":\"complete fusion and runtime\"") !=
                  std::string::npos &&
              v3_record.find("\"hardware\":{\"sm_count\":128}") !=
                  std::string::npos &&
              v3_record.find("\"prior_choices\":[{\"site\":\"PW2\",\"choice\":\"bp1\"}]") !=
                  std::string::npos &&
              v3_record.find("\"criteria\":{\"bp0\":\"one dimensional\",\"bp1\":\"split\"}") !=
                  std::string::npos &&
              v3_record.find("\"native_choice\"") == std::string::npos,
          "semantic-v3 envelope, history, or canonical candidate order failed");
  require(v3_record.find("\"consequence\"") == std::string::npos,
          "semantic-v3 emitted an absent consequence");
  {
    const std::string before_v2_snapshot = readFile(path);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "nvfuser.compute_invocation_input.v2\nfusion 0:", {}, v3_hardware);
    require(chooseDecision("PW2", state, reverse_options, "bp0") == "bp0",
            "semantic-v3 changed the choice with a v2 snapshot");
    const std::string request =
        loggedRequest(readFile(path).substr(before_v2_snapshot.size()));
    require(
        request.find(
            "\"snapshot_format\":\"nvfuser.compute_invocation_input.v2\"") !=
            std::string::npos,
        "semantic-v3 did not advertise the actual v2 snapshot format");
  }
  {
    const std::vector<ChoiceOption> enriched_reverse{
        {"bp1", "split", "{\"break_point\":128,\"tiles\":[64,128],\"legal\":true}"},
        {"bp0", "one dimensional", "{\"break_point\":null,\"tiles\":[256],\"legal\":true}"}};
    const std::string before_enriched = readFile(path);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "complete fusion and runtime", {}, v3_hardware);
    require(chooseDecision("PW2", state, enriched_reverse, "bp0") == "bp0",
            "semantic-v3 consequence Record changed native choice");
    const std::string request =
        loggedRequest(readFile(path).substr(before_enriched.size()));
    require(
        request.find(
            "\"candidates\":[{\"id\":\"bp0\",\"description\":\"one dimensional\","
            "\"consequence\":{\"break_point\":null,\"tiles\":[256],\"legal\":true}},"
            "{\"id\":\"bp1\",\"description\":\"split\",\"consequence\":"
            "{\"break_point\":128,\"tiles\":[64,128],\"legal\":true}}]") !=
            std::string::npos &&
            request.find("\"criteria\":{\"bp0\":\"one dimensional\",\"bp1\":\"split\"}") !=
                std::string::npos,
        "semantic-v3 candidate consequence lost its canonical ID association");
  }
  {
    const std::string before_invalid = readFile(path);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "complete fusion and runtime", {}, v3_hardware);
    for (const std::string& invalid : {
             "[]",
             "null",
             "\"string\"",
             "{bad",
             "{\"duplicate\":1,\"duplicate\":2}"}) {
      const std::vector<ChoiceOption> malformed{
          {"bp0", "one dimensional", invalid}, {"bp1", "split"}};
      bool rejected = false;
      try { (void)chooseDecision("PW2", state, malformed, "bp0"); }
      catch (const std::runtime_error&) { rejected = true; }
      require(rejected, "semantic-v3 accepted a malformed candidate consequence");
    }
    require(readFile(path) == before_invalid,
            "semantic-v3 logged a malformed candidate consequence");
  }
  {
    bool rejected = false;
    try { (void)chooseDecision("PW2", state, options, "bp0"); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "semantic-v3 accepted a Choice without compute context");
  }
  {
    const std::string before_oversized = readFile(path);
    const std::string oversized_snapshot(8 * 1024 * 1024, 'x');
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        oversized_snapshot, {}, v3_hardware);
    bool rejected = false;
    try { (void)chooseDecision("PW2", state, options, "bp0"); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected && readFile(path) == before_oversized,
            "semantic-v3 request exceeding 8 MiB was sent or logged");
  }
  ::setenv("NVFUSER_JEV_MODE", "remote", 1);
  {
    FakeServer server(
        "{\"model\":\"nvfuser-analysis-v1\",\"answers\":{\"RD3.width\":{"
        "\"type\":\"choice\",\"choice\":\"w1\",\"confidence\":1.0,"
        "\"probabilities\":{\"w1\":1.0,\"w3\":0.0}}},"
        "\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}");
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "RD3 synthetic input", {}, v3_hardware);
    require(
        chooseDecision("RD3.width", rd3_width_state, rd3_width_options) == "w1",
        "RD3 semantic-v3 Remote rejected a valid alternate width");
    const auto body_start = server.request().find("\r\n\r\n");
    require(body_start != std::string::npos,
            "RD3 semantic-v3 Remote request body is missing");
    require(server.request().substr(body_start + 4) == rd3_width_record_request,
            "RD3 width Record and Remote requests differ");
  }
  {
    FakeServer server(answer("bp1"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    const std::string large_consequence =
        "{\"payload\":\"" + std::string(70 * 1024, 'c') + "\"}";
    const std::vector<ChoiceOption> enriched_reverse{
        {"bp1", "split", large_consequence},
        {"bp0", "one dimensional", "{\"tile\":1}"}};
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "complete fusion and runtime", {}, v3_hardware);
    require(chooseDecision("PW2", state, enriched_reverse) == "bp1",
            "semantic-v3 rejected a candidate consequence larger than 64 KiB");
    const auto body_start = server.request().find("\r\n\r\n");
    require(body_start != std::string::npos, "semantic-v3 consequence HTTP body missing");
    const auto body = server.request().substr(body_start + 4);
    require(body.size() > 64 * 1024 &&
                body.find("\"consequence\":" + large_consequence) != std::string::npos &&
                body.find("\"native_choice\"") == std::string::npos,
            "semantic-v3 candidate consequence was omitted or truncated");
  }
  {
    FakeServer server(answer("bp1"));
    ::setenv("NVFUSER_JEV_PORT", std::to_string(server.port()).c_str(), 1);
    const std::string full_snapshot =
        "begin snapshot " + std::string(70 * 1024, 'x') + " end snapshot";
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        full_snapshot, {}, v3_hardware);
    require(chooseDecision("PW2", state, reverse_options) == "bp1",
            "semantic-v3 rejected a complete request larger than 64 KiB");
    const auto body_start = server.request().find("\r\n\r\n");
    require(body_start != std::string::npos, "semantic-v3 HTTP body missing");
    const auto body = server.request().substr(body_start + 4);
    require(body.size() > 64 * 1024 &&
                body.find("begin snapshot ") != std::string::npos &&
                body.find(" end snapshot") != std::string::npos,
            "semantic-v3 request was truncated to the old protocol limit");
  }
  ::unsetenv("NVFUSER_JEV_PROTOCOL");
  const std::string remote_decisions_before_no_choice = readFile(path);
  std::string remote_no_choice_id;
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "remote no-choice input", {});
    remote_no_choice_id = scope.invocationId();
    nvfuser::scheduler::jev::recordNoChoiceDecision(
        "RD3.remainder_x", "x64", "single_cta_width", "remote_remainder_seed");
    scope.complete("Reduction");
  }
  const std::string remote_no_choice_trace = readFile(compute_path);
  require(readFile(path) == remote_decisions_before_no_choice &&
              remote_no_choice_trace.find(
                  "\"kind\":\"no_choice\",\"mode\":\"remote\"") !=
                  std::string::npos &&
              remote_no_choice_trace.find(
                  "\"compute_invocation_id\":\"" + remote_no_choice_id +
                  "\",\"compute_event_index\":1,\"site\":\"RD3.remainder_x\",\"selected\":\"x64\",\"reason\":\"single_cta_width\",\"selection_source\":\"remote_remainder_seed\"") !=
                  std::string::npos &&
              remote_no_choice_trace.find(
                  "\"compute_invocation_id\":\"" + remote_no_choice_id +
                  "\",\"compute_event_index\":2,\"scheduler_type\":\"Reduction\"") !=
                  std::string::npos,
          "deterministic Remote event changed Choice or compute sequencing");
  const std::string remote_decisions_before_unroll = readFile(path);
  std::string remote_unroll_id;
  {
    nvfuser::scheduler::jev::ComputeHeuristicsScope scope(
        "remote bounded pointwise unroll", {});
    remote_unroll_id = scope.invocationId();
    const auto bypass =
        nvfuser::pointwise::non_tma::getPointwiseUnrollEnvelope(100, 4096, 1, 3);
    require(
        bypass.model_limit == 2 && bypass.native_factor == 3 &&
            !bypass.offer_choice,
        "Remote over-budget unroll did not preserve the native factor");
    scope.complete("PointWise");
  }
  const std::string remote_unroll_trace = readFile(compute_path);
  require(
      readFile(path) == remote_decisions_before_unroll &&
          remote_unroll_trace.find(
              "\"compute_invocation_id\":\"" + remote_unroll_id +
              "\",\"compute_event_index\":1,\"site\":\"PW3.unroll\",\"selected\":\"u3\",\"reason\":\"native factor exceeds bounded model resource envelope\",\"selection_source\":\"native_unroll\"") !=
              std::string::npos &&
          remote_unroll_trace.find(
              "\"compute_invocation_id\":\"" + remote_unroll_id +
              "\",\"compute_event_index\":2,\"scheduler_type\":\"PointWise\"") !=
              std::string::npos,
      "Remote over-budget unroll emitted a Choice or lost its no_choice row");
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
  ::unlink(compute_path.c_str());
}
