// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <ATen/core/Tensor.h>

#include <bit>
#include <cstddef>
#include <complex>
#include <cstdint>
#include <locale>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "fusion.h"
#include "scheduler/runtime_info.h"

// A passive, deterministic description of the inputs visible at the entry to
// computeHeuristics. It contains no tensor data or raw pointer addresses. A
// failed or oversized description is unavailable in its entirety; callers must
// not use a partial description as an invocation identity.
namespace nvfuser::scheduler::compute_invocation_input {

enum class UnavailableReason {
  None,
  NullFusion,
  TooManyInputs,
  FusionTextTooLarge,
  FusionTextContainsPointerLikeValue,
  UndefinedTensor,
  InputKindMismatch,
  UnsupportedScalar,
  SnapshotTooLarge,
  FormattingFailed,
};

inline const char* reasonName(UnavailableReason reason) noexcept {
  switch (reason) {
    case UnavailableReason::None: return "none";
    case UnavailableReason::NullFusion: return "null_fusion";
    case UnavailableReason::TooManyInputs: return "too_many_inputs";
    case UnavailableReason::FusionTextTooLarge: return "fusion_text_too_large";
    case UnavailableReason::FusionTextContainsPointerLikeValue:
      return "fusion_text_contains_pointer_like_value";
    case UnavailableReason::UndefinedTensor: return "undefined_tensor";
    case UnavailableReason::InputKindMismatch: return "input_kind_mismatch";
    case UnavailableReason::UnsupportedScalar: return "unsupported_scalar";
    case UnavailableReason::SnapshotTooLarge: return "snapshot_too_large";
    case UnavailableReason::FormattingFailed: return "formatting_failed";
  }
  return "unknown";
}

struct Snapshot {
  std::string text;
  UnavailableReason reason = UnavailableReason::None;

  bool available() const noexcept {
    return reason == UnavailableReason::None;
  }
};

inline bool containsPointerLikeValue(std::string_view text) noexcept {
  const auto is_hex = [](char ch) {
    return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') ||
        (ch >= 'A' && ch <= 'F');
  };
  for (size_t i = 0; i + 2 < text.size(); ++i) {
    if (text[i] != '0' || (text[i + 1] != 'x' && text[i + 1] != 'X')) {
      continue;
    }
    size_t j = i + 2;
    while (j < text.size() && is_hex(text[j])) {
      ++j;
    }
    // A hexadecimal graph literal might be harmless, but the printer does
    // not distinguish it from a pointer. Reject the whole snapshot instead
    // of risking disclosure or a process-specific comparison key.
    if (j > i + 2) {
      return true;
    }
  }
  return false;
}

inline void appendSized(std::ostream& out, std::string_view value) {
  out << value.size() << ':' << value;
}

inline void appendDoubleBits(std::ostream& out, double value) {
  constexpr char hex[] = "0123456789abcdef";
  auto bits = std::bit_cast<uint64_t>(value);
  for (int shift = 60; shift >= 0; shift -= 4) {
    out << hex[(bits >> shift) & 0xf];
  }
}

inline Snapshot capture(Fusion* fusion, SchedulerRuntimeInfo& runtime_info)
    noexcept {
  constexpr size_t kMaxInputs = 128;
  constexpr size_t kMaxTextBytes = 1024 * 1024;
  if (fusion == nullptr) {
    return {{}, UnavailableReason::NullFusion};
  }
  try {
    if (fusion->inputs().size() > kMaxInputs) {
      return {{}, UnavailableReason::TooManyInputs};
    }

    std::ostringstream fusion_stream;
    fusion_stream.imbue(std::locale::classic());
    fusion->print(fusion_stream, false);
    if (!fusion_stream.good()) {
      return {{}, UnavailableReason::FormattingFailed};
    }
    const std::string fusion_text = fusion_stream.str();
    if (fusion_text.size() > kMaxTextBytes) {
      return {{}, UnavailableReason::FusionTextTooLarge};
    }
    if (containsPointerLikeValue(fusion_text)) {
      return {{}, UnavailableReason::FusionTextContainsPointerLikeValue};
    }

    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "nvfuser.compute_invocation_input.v1\nfusion ";
    appendSized(out, fusion_text);
    out << "\nindex_type "
        << static_cast<int64_t>(runtime_info.getIndexType()) << "\ninputs "
        << fusion->inputs().size();

    size_t index = 0;
    for (Val* input : fusion->inputs()) {
      out << "\ninput " << index++ << " dtype ";
      std::ostringstream dtype_stream;
      dtype_stream.imbue(std::locale::classic());
      dtype_stream << input->dtype();
      if (!dtype_stream.good()) {
        return {{}, UnavailableReason::FormattingFailed};
      }
      appendSized(out, dtype_stream.str());

      const auto& value = runtime_info.expressionEvaluator().evaluate(input);
      if (value.is<at::Tensor>()) {
        if (!input->isA<TensorView>()) {
          return {{}, UnavailableReason::InputKindMismatch};
        }
        const auto& tensor = value.as<at::Tensor>();
        if (!tensor.defined()) {
          return {{}, UnavailableReason::UndefinedTensor};
        }
        out << " tensor scalar_type "
            << static_cast<int64_t>(tensor.scalar_type()) << " device ";
        appendSized(out, tensor.device().str());
        out << " storage_offset " << tensor.storage_offset() << " sizes "
            << tensor.dim();
        for (int64_t size : tensor.sizes()) {
          out << ' ' << size;
        }
        out << " strides " << tensor.dim();
        for (int64_t stride : tensor.strides()) {
          out << ' ' << stride;
        }
        if (input->as<TensorView>()->isCpuScalar()) {
          out << " alignment_bits none";
        } else {
          out << " alignment_bits "
              << runtime_info.getAlignmentSizeBit(input->as<TensorView>());
        }
      } else {
        if (input->isA<TensorView>()) {
          return {{}, UnavailableReason::InputKindMismatch};
        }
        if (value.is<int64_t>()) {
          out << " scalar_i64 " << value.as<int64_t>();
        } else if (value.is<bool>()) {
          out << " scalar_bool " << (value.as<bool>() ? 1 : 0);
        } else if (value.is<double>()) {
          out << " scalar_f64_bits ";
          appendDoubleBits(out, value.as<double>());
        } else if (value.is<std::complex<double>>()) {
          const auto number = value.as<std::complex<double>>();
          out << " scalar_c128_bits ";
          appendDoubleBits(out, number.real());
          out << ' ';
          appendDoubleBits(out, number.imag());
        } else {
          return {{}, UnavailableReason::UnsupportedScalar};
        }
      }
      if (!out.good()) {
        return {{}, UnavailableReason::FormattingFailed};
      }
      if (out.tellp() > static_cast<std::streampos>(kMaxTextBytes)) {
        return {{}, UnavailableReason::SnapshotTooLarge};
      }
    }
    if (!out.good()) {
      return {{}, UnavailableReason::FormattingFailed};
    }
    std::string text = out.str();
    if (text.size() > kMaxTextBytes) {
      return {{}, UnavailableReason::SnapshotTooLarge};
    }
    return {std::move(text), UnavailableReason::None};
  } catch (...) {
    return {{}, UnavailableReason::FormattingFailed};
  }
}

} // namespace nvfuser::scheduler::compute_invocation_input
