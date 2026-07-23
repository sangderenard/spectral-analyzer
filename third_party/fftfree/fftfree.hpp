// fftfree.hpp
// Thin alias header for the public C API formerly exposed via fft_cffi.hpp
#pragma once

#include "fft_cffi.hpp"

// This header exists to allow downstreams to include "fftfree.hpp"
// while we transition build targets and packaging names from *cffi* to *fftfree*.
// The exported function names remain the same (fft_init_full, fft_execute_batched, ...).

