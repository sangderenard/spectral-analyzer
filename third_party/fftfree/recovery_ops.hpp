// recovery_ops.hpp
#pragma once

#include <cstddef>
#include <cstring>
#include <vector>

// Helpers for column-slice snapshot/restore used by algorithms that mutate
// columnar regions (lanes x rows). Snapshots are contiguous copies sized
// rowsN * width * elem_bytes.

struct ColumnSlice {
  std::byte* base = nullptr;       // start of column 0 (lane 0)
  std::ptrdiff_t axis_stride = 0;  // bytes to next row in a column
  std::ptrdiff_t batch_stride = 0; // bytes to next lane/column
  int rowsN = 0;                   // number of rows (N)
  int width = 0;                   // active lanes in this chunk
  int elem_bytes = 0;              // element size in bytes
};

struct ColumnSnapshot {
  std::vector<std::byte> bytes; // contiguous snapshot: rowsN * width * elem_bytes
  int rowsN = 0;
  int width = 0;
  int elem_bytes = 0;
};

inline void capture_columns(const ColumnSlice& s, ColumnSnapshot& out) {
  out.rowsN = s.rowsN;
  out.width = s.width;
  out.elem_bytes = s.elem_bytes;
  const size_t total = static_cast<size_t>(s.rowsN) * static_cast<size_t>(s.width) * static_cast<size_t>(s.elem_bytes);
  out.bytes.resize(total);
  std::byte* dst = out.bytes.data();
  for (int i = 0; i < s.rowsN; ++i) {
    const std::ptrdiff_t row_off = static_cast<std::ptrdiff_t>(i) * s.axis_stride;
    for (int lane = 0; lane < s.width; ++lane) {
      std::byte* src = s.base + static_cast<std::ptrdiff_t>(lane) * s.batch_stride + row_off;
      std::memcpy(dst, src, static_cast<size_t>(s.elem_bytes));
      dst += s.elem_bytes;
    }
  }
}

inline void restore_columns(const ColumnSlice& s, const ColumnSnapshot& in) {
  const std::byte* src = in.bytes.data();
  for (int i = 0; i < s.rowsN; ++i) {
    const std::ptrdiff_t row_off = static_cast<std::ptrdiff_t>(i) * s.axis_stride;
    for (int lane = 0; lane < s.width; ++lane) {
      std::byte* dst = s.base + static_cast<std::ptrdiff_t>(lane) * s.batch_stride + row_off;
      std::memcpy(dst, src, static_cast<size_t>(s.elem_bytes));
      src += s.elem_bytes;
    }
  }
}

