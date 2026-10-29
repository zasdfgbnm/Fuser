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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "fusion.h"
#include "scheduler/bounded_snapshot_stream.h"
#include "scheduler/native_diagnostic_jsonl.h"
#include "scheduler/runtime_info.h"

// A passive, deterministic description of the inputs visible at the entry to
// computeHeuristics. It contains no tensor data or raw pointer addresses. A
// failed or oversized description is unavailable in its entirety; callers must
// not use a partial description as an invocation identity.
namespace nvfuser::scheduler::compute_invocation_input {

enum class Format { V1, V2 };

enum class UnavailableReason {
  None,
  NotCaptured,
  NullFusion,
  TooManyInputs,
  FusionTextTooLarge,
  FusionTextContainsPointerLikeValue,
  UndefinedTensor,
  InputKindMismatch,
  UnsupportedScalar,
  TooManyIrValues,
  TooManyTensorViews,
  TooManyAxes,
  TooManyExpressions,
  UnknownExpressionVal,
  IrMetadataContainsPointerLikeValue,
  SnapshotTooLarge,
  FormattingFailed,
};

inline const char* reasonName(UnavailableReason reason) noexcept {
  switch (reason) {
    case UnavailableReason::None: return "none";
    case UnavailableReason::NotCaptured: return "not_captured";
    case UnavailableReason::NullFusion: return "null_fusion";
    case UnavailableReason::TooManyInputs: return "too_many_inputs";
    case UnavailableReason::FusionTextTooLarge: return "fusion_text_too_large";
    case UnavailableReason::FusionTextContainsPointerLikeValue:
      return "fusion_text_contains_pointer_like_value";
    case UnavailableReason::UndefinedTensor: return "undefined_tensor";
    case UnavailableReason::InputKindMismatch: return "input_kind_mismatch";
    case UnavailableReason::UnsupportedScalar: return "unsupported_scalar";
    case UnavailableReason::TooManyIrValues: return "too_many_ir_values";
    case UnavailableReason::TooManyTensorViews: return "too_many_tensor_views";
    case UnavailableReason::TooManyAxes: return "too_many_axes";
    case UnavailableReason::TooManyExpressions: return "too_many_expressions";
    case UnavailableReason::UnknownExpressionVal:
      return "unknown_expression_val";
    case UnavailableReason::IrMetadataContainsPointerLikeValue:
      return "ir_metadata_contains_pointer_like_value";
    case UnavailableReason::SnapshotTooLarge: return "snapshot_too_large";
    case UnavailableReason::FormattingFailed: return "formatting_failed";
  }
  return "unknown";
}

struct Snapshot {
  std::string text;
  UnavailableReason reason = UnavailableReason::NotCaptured;

  bool available() const noexcept {
    return reason == UnavailableReason::None && !text.empty();
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

// The v2 suffix describes IR facts that Fusion::print(false) does not show.
// IDs come from the fusion's own deterministic IR numbering, never addresses.
// This is a transport snapshot, not a proof that every scheduler continuation
// can be reconstructed from the metadata alone.
inline Snapshot captureIrMetadata(Fusion* fusion) noexcept {
  constexpr size_t kMaxTextBytes = 1024 * 1024;
  constexpr size_t kMaxValues = 65536;
  constexpr size_t kMaxTensorViews = 4096;
  constexpr size_t kMaxAxesPerDomain = 128;
  constexpr size_t kMaxAxisEntries = 32768;
  constexpr size_t kMaxAxisPrintBytes = 8192;
  constexpr size_t kMaxExpressions = 65536;
  if (fusion == nullptr) {
    return {{}, UnavailableReason::NullFusion};
  }
  try {
    // A copied Fusion may add special scalar values. Per-type IR names, unlike
    // insertion ordinals, remain stable across equivalent copies.
    const auto vals = fusion->deterministic_vals();
    if (vals.size() > kMaxValues) {
      return {{}, UnavailableReason::TooManyIrValues};
    }
    std::vector<TensorView*> tvs;
    std::unordered_map<uint64_t, const Val*> val_keys;
    val_keys.reserve(vals.size());
    for (Val* val : vals) {
      const uint64_t key =
          (static_cast<uint64_t>(val->vtype()) << 32) | val->name();
      const auto [found, inserted] = val_keys.emplace(key, val);
      if (!inserted && found->second != val) {
        return {{}, UnavailableReason::FormattingFailed};
      }
      if (val->isA<TensorView>()) {
        tvs.push_back(val->as<TensorView>());
      }
    }
    if (tvs.size() > kMaxTensorViews) {
      return {{}, UnavailableReason::TooManyTensorViews};
    }
    // IrPrinter renders exactly this output-reachable, topologically sorted
    // sequence. Other deterministic expressions may be dead and are not part
    // of the printed PointWise graph.
    const auto exprs = fusion->exprs();
    if (exprs.size() > kMaxExpressions) {
      return {{}, UnavailableReason::TooManyExpressions};
    }
    // The container keeps zero/one and other special scalar Vals outside
    // deterministic_vals(). They may still be operands of a printed Expr.
    // Admit only such scalar references, with the same collision check as
    // owned Vals; an unregistered TV would leave the TV table incomplete.
    for (const Expr* expr : exprs) {
      const auto register_expr_vals = [&](const std::vector<Val*>& refs) {
        for (const Val* val : refs) {
          const uint64_t key =
              (static_cast<uint64_t>(val->vtype()) << 32) | val->name();
          const auto [found, inserted] = val_keys.emplace(key, val);
          if (!inserted && found->second != val) return false;
          if (inserted && val->isA<TensorView>()) return false;
        }
        return true;
      };
      if (!register_expr_vals(expr->inputs()) ||
          !register_expr_vals(expr->outputs())) {
        return {{}, UnavailableReason::UnknownExpressionVal};
      }
    }
    if (val_keys.size() > kMaxValues) {
      return {{}, UnavailableReason::TooManyIrValues};
    }

    BoundedSnapshotBuffer metadata_buffer(kMaxTextBytes);
    std::ostream out(&metadata_buffer);
    out.imbue(std::locale::classic());
    out << "{\"schema\":\"nvfuser.compute_ir_meta.v2\",\"tvs\":[";
    const auto append_val_key = [&](const Val* val) {
      out << '[' << static_cast<int64_t>(val->vtype()) << ',' << val->name()
          << ']';
    };
    const auto known_val = [&](const Val* val) {
      const uint64_t key =
          (static_cast<uint64_t>(val->vtype()) << 32) | val->name();
      const auto found = val_keys.find(key);
      return found != val_keys.end() && found->second == val;
    };
    const auto enum_text = [](const auto& value) {
      std::ostringstream text_stream;
      text_stream.imbue(std::locale::classic());
      text_stream << value;
      return text_stream.str();
    };
    size_t axis_entries = 0;
    bool first_tv = true;
    for (TensorView* tv : tvs) {
      const auto& logical = tv->getLogicalDomain();
      const auto& loop = tv->getLoopDomain();
      const auto& allocation = tv->getMaybeAllocationDomain();
      const auto& contiguity = tv->getContiguity();
      if (logical.size() > kMaxAxesPerDomain ||
          loop.size() > kMaxAxesPerDomain ||
          allocation.size() > kMaxAxesPerDomain) {
        return {{}, UnavailableReason::TooManyAxes};
      }
      if (contiguity.size() != allocation.size()) {
        return {{}, UnavailableReason::FormattingFailed};
      }
      if (!tv->hasAllocation() && logical != allocation) {
        return {{}, UnavailableReason::FormattingFailed};
      }
      if (!first_tv) out << ',';
      first_tv = false;
      out << "{\"id\":\"T" << tv->name() << "\",\"val_key\":";
      append_val_key(tv);
      out << ",\"memory_type\":";
      switch (tv->getMemoryType()) {
        case MemoryType::Local: out << "\"local\""; break;
        case MemoryType::Shared: out << "\"shared\""; break;
        case MemoryType::Global: out << "\"global\""; break;
        case MemoryType::Tensor: out << "\"tensor\""; break;
        case MemoryType::Symmetric: out << "\"symmetric\""; break;
        default: return {{}, UnavailableReason::FormattingFailed};
      }
      out << ",\"dtype\":"
          << native_diagnostic::quote(enum_text(tv->dtype()))
          << ",\"in_fusion_vals\":"
          << (fusion->vals().count(tv) != 0 ? "true" : "false")
          << ",\"has_device_mesh\":"
          << (tv->hasDeviceMesh() ? "true" : "false")
          << ",\"is_cpu_scalar\":"
          << (tv->isCpuScalar() ? "true" : "false")
          << ",\"domain_expr_count\":" << tv->domain()->allExprs().size();
      out << ",\"logical\":";
      const auto append_axes = [&](const std::vector<IterDomain*>& axes) {
        out << '[';
        for (size_t i = 0; i < axes.size(); ++i) {
          if (i != 0) out << ',';
          out << axes[i]->name();
        }
        out << ']';
      };
      append_axes(logical);
      out << ",\"loop\":";
      append_axes(loop);
      out << ",\"allocation\":";
      append_axes(allocation);
      out << ",\"allocation_explicit\":"
          << (tv->hasAllocation() ? "true" : "false")
          << ",\"contiguity\":[";
      for (size_t i = 0; i < contiguity.size(); ++i) {
        if (i != 0) out << ',';
        out << (contiguity[i].has_value()
                    ? (*contiguity[i] ? "true" : "false")
                    : "null");
      }
      out << "],\"axes\":[";
      std::unordered_set<int64_t> seen_axes;
      bool first_axis = true;
      for (const auto* domain : {&logical, &loop, &allocation}) {
        for (const IterDomain* axis : *domain) {
          if (!seen_axes.insert(axis->name()).second) continue;
          if (++axis_entries > kMaxAxisEntries) {
            return {{}, UnavailableReason::TooManyAxes};
          }
          const std::string axis_print = axis->toString();
          if (axis_print.size() > kMaxAxisPrintBytes) {
            return {{}, UnavailableReason::SnapshotTooLarge};
          }
          if (!first_axis) out << ',';
          first_axis = false;
          out << "{\"id\":" << axis->name() << ",\"print\":"
              << native_diagnostic::quote(axis_print)
              << ",\"is_reduction\":"
              << (axis->isReduction() ? "true" : "false")
              << ",\"is_stride\":"
              << (axis->isStride() ? "true" : "false")
              << ",\"is_symbolic\":"
              << (axis->isSymbolic() ? "true" : "false")
              << ",\"is_device_dim\":"
              << (axis->isDeviceDim() ? "true" : "false")
              << ",\"has_expanded_extent\":"
              << (axis->hasExpandedExtent() ? "true" : "false")
              << ",\"is_parallelized\":"
              << (axis->isParallelized() ? "true" : "false")
              << ",\"is_iteration\":"
              << (axis->isIteration() ? "true" : "false")
              << ",\"is_broadcast\":"
              << (axis->isBroadcast() ? "true" : "false") << '}';
        }
      }
      out << "]}";
      if (!out.good()) {
        return {{}, metadata_buffer.truncated()
                        ? UnavailableReason::SnapshotTooLarge
                        : UnavailableReason::FormattingFailed};
      }
    }
    out << "],\"input_tvs\":[";
    for (size_t i = 0; i < fusion->inputs().size(); ++i) {
      if (i != 0) out << ',';
      const Val* input = fusion->inputs()[i];
      if (input->isA<TensorView>()) {
        out << "\"T" << input->name() << '"';
      } else {
        out << "null";
      }
    }
    out << "],\"input_vals\":[";
    for (size_t i = 0; i < fusion->inputs().size(); ++i) {
      if (i != 0) out << ',';
      append_val_key(fusion->inputs()[i]);
    }
    out << "],\"outputs\":[";
    for (size_t i = 0; i < fusion->outputs().size(); ++i) {
      if (i != 0) out << ',';
      const Val* output = fusion->outputs()[i];
      const AliasInfo& alias = fusion->getOutputAlias(output);
      out << "{\"val_key\":";
      append_val_key(output);
      out << ",\"tv\":";
      if (output->isA<TensorView>()) {
        out << "\"T" << output->name() << '"';
      } else {
        out << "null";
      }
      out << ",\"alias\":{\"type\":";
      switch (alias.type) {
        case AllocationType::New: out << "\"new\""; break;
        case AllocationType::ReuseBuffer: out << "\"reuse_buffer\""; break;
        case AllocationType::Evaluate: out << "\"evaluate\""; break;
        default: return {{}, UnavailableReason::FormattingFailed};
      }
      out << ",\"target_val\":";
      if (alias.aliased_io == nullptr) {
        out << "null";
      } else {
        if (!known_val(alias.aliased_io)) {
          return {{}, UnavailableReason::FormattingFailed};
        }
        append_val_key(alias.aliased_io);
      }
      out << ",\"visibility\":";
      switch (alias.visibility) {
        case OutputVisibility::kVisible: out << "\"visible\""; break;
        case OutputVisibility::kHidden: out << "\"hidden\""; break;
        default: return {{}, UnavailableReason::FormattingFailed};
      }
      out << "}}";
    }
    out << "],\"exprs\":[";
    bool first_expr = true;
    for (const Expr* expr : exprs) {
      if (!first_expr) out << ',';
      first_expr = false;
      out << "{\"kind\":";
      if (expr->isA<BroadcastOp>()) {
        out << "\"broadcast\",\"op\":null,\"cache\":null";
      } else if (expr->isA<LoadStoreOp>()) {
        const auto* op = expr->as<LoadStoreOp>();
        out << "\"load_store\",\"op\":"
            << native_diagnostic::quote(enum_text(op->opType()))
            << ",\"cache\":"
            << native_diagnostic::quote(enum_text(op->cacheOp()));
      } else if (expr->isA<UnaryOp>()) {
        out << "\"unary\",\"op\":"
            << native_diagnostic::quote(
                   enum_text(expr->as<UnaryOp>()->getUnaryOpType()))
            << ",\"cache\":null";
      } else if (expr->isA<BinaryOp>()) {
        out << "\"binary\",\"op\":"
            << native_diagnostic::quote(
                   enum_text(expr->as<BinaryOp>()->getBinaryOpType()))
            << ",\"cache\":null";
      } else {
        // Represent unsupported operators too, so the server can reject the
        // complete graph instead of treating an omitted node as eligible.
        out << "\"other\",\"op\":null,\"cache\":null";
      }
      const auto append_expr_vals = [&](const std::vector<Val*>& expr_vals) {
        out << '[';
        for (size_t i = 0; i < expr_vals.size(); ++i) {
          const Val* val = expr_vals[i];
          if (!known_val(val)) return false;
          if (i != 0) out << ',';
          out << "{\"val_key\":";
          append_val_key(val);
          out << ",\"dtype\":"
              << native_diagnostic::quote(enum_text(val->dtype()))
              << ",\"is_const_scalar\":"
              << (val->isConstScalar() ? "true" : "false") << '}';
        }
        out << ']';
        return true;
      };
      out << ",\"inputs\":";
      if (!append_expr_vals(expr->inputs())) {
        return {{}, UnavailableReason::UnknownExpressionVal};
      }
      out << ",\"outputs\":";
      if (!append_expr_vals(expr->outputs())) {
        return {{}, UnavailableReason::UnknownExpressionVal};
      }
      out << '}';
      if (!out.good()) {
        return {{}, metadata_buffer.truncated()
                        ? UnavailableReason::SnapshotTooLarge
                        : UnavailableReason::FormattingFailed};
      }
    }
    out << "]}";
    if (metadata_buffer.truncated()) {
      return {{}, UnavailableReason::SnapshotTooLarge};
    }
    if (!out.good()) {
      return {{}, UnavailableReason::FormattingFailed};
    }
    std::string text = std::move(metadata_buffer).take();
    if (containsPointerLikeValue(text)) {
      return {{}, UnavailableReason::IrMetadataContainsPointerLikeValue};
    }
    return {std::move(text), UnavailableReason::None};
  } catch (...) {
    return {{}, UnavailableReason::FormattingFailed};
  }
}

inline Snapshot capture(
    Fusion* fusion,
    SchedulerRuntimeInfo& runtime_info,
    Format format = Format::V1)
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

    BoundedSnapshotBuffer fusion_buffer(kMaxTextBytes);
    std::ostream fusion_stream(&fusion_buffer);
    fusion_stream.imbue(std::locale::classic());
    fusion->print(fusion_stream, false);
    if (fusion_buffer.truncated()) {
      return {{}, UnavailableReason::FusionTextTooLarge};
    }
    if (!fusion_stream.good()) {
      return {{}, UnavailableReason::FormattingFailed};
    }
    const std::string_view fusion_text = fusion_buffer.view();
    if (containsPointerLikeValue(fusion_text)) {
      return {{}, UnavailableReason::FusionTextContainsPointerLikeValue};
    }

    BoundedSnapshotBuffer snapshot_buffer(kMaxTextBytes);
    std::ostream out(&snapshot_buffer);
    out.imbue(std::locale::classic());
    out << (format == Format::V2 ? "nvfuser.compute_invocation_input.v2\nfusion "
                                 : "nvfuser.compute_invocation_input.v1\nfusion ");
    appendSized(out, fusion_text);
    out << "\nindex_type "
        << static_cast<int64_t>(runtime_info.getIndexType()) << "\ninputs "
        << fusion->inputs().size();
    if (snapshot_buffer.truncated()) {
      return {{}, UnavailableReason::SnapshotTooLarge};
    }

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
        if (snapshot_buffer.truncated()) {
          return {{}, UnavailableReason::SnapshotTooLarge};
        }
        return {{}, UnavailableReason::FormattingFailed};
      }
    }
    if (format == Format::V2) {
      const Snapshot metadata = captureIrMetadata(fusion);
      if (!metadata.available()) {
        return {{}, metadata.reason};
      }
      out << "\nir_meta ";
      appendSized(out, metadata.text);
    }
    if (snapshot_buffer.truncated()) {
      return {{}, UnavailableReason::SnapshotTooLarge};
    }
    if (!out.good()) {
      return {{}, UnavailableReason::FormattingFailed};
    }
    std::string text = std::move(snapshot_buffer).take();
    return {std::move(text), UnavailableReason::None};
  } catch (...) {
    return {{}, UnavailableReason::FormattingFailed};
  }
}

} // namespace nvfuser::scheduler::compute_invocation_input
