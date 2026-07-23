SHELL := /bin/sh

# Default build directories
BUILD_DIR ?= build
DBG_DIR := $(BUILD_DIR)/debug
REL_DIR := $(BUILD_DIR)/release

# Choose generator if desired (defaults to CMake default)
GEN ?=

.PHONY: all debug release clean trace_threads test_threads

all: debug

$(DBG_DIR):
	@mkdir -p $(DBG_DIR)

$(REL_DIR):
	@mkdir -p $(REL_DIR)

debug: | $(DBG_DIR)
	cmake -S . -B $(DBG_DIR) $(if $(GEN),-G "$(GEN)") -DCMAKE_BUILD_TYPE=Debug
	cmake --build $(DBG_DIR) --config Debug -- -j

release: | $(REL_DIR)
	cmake -S . -B $(REL_DIR) $(if $(GEN),-G "$(GEN)") -DCMAKE_BUILD_TYPE=Release
	cmake --build $(REL_DIR) --config Release -- -j

# Build the thread tracing test with tracing enabled (no optimizations via Debug)
trace_threads: | $(DBG_DIR)
	cmake -S . -B $(DBG_DIR) $(if $(GEN),-G "$(GEN)") -DCMAKE_BUILD_TYPE=Debug -DEIGFFT_TRACE_THREADS=ON
	cmake --build $(DBG_DIR) --config Debug --target fft_threads_trace -- -j

test_threads: trace_threads
	ctest --test-dir $(DBG_DIR) -C Debug -R fftfree.threads_trace -V

clean:
	rm -rf $(BUILD_DIR)

