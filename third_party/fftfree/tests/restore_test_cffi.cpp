// CFFI-driven restore test
// Exercises the C API (fft_init_full + fft_execute_batched) to ensure
// the C ABI code path (compute_effective_runtime, PoolDispatcher) is used.

#include "../fft_cffi.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <atomic>

// Globals used by the restore observer instrumentation.
static std::vector<unsigned char>* s_restore_flags_ptr = nullptr;
static std::atomic<size_t> s_restore_calls(0);
static std::atomic<size_t> s_restored_columns(0);

extern "C" void test_restore_observer_c(size_t s, size_t e) {
  if (!s_restore_flags_ptr) return;
  const size_t maxidx = s_restore_flags_ptr->size();
  for (size_t i = s; i < e; ++i) {
    if (i < maxidx) (*s_restore_flags_ptr)[i] = 1;
  }
  s_restore_calls.fetch_add(1);
  s_restored_columns.fetch_add((e > s) ? (e - s) : 0);
}

int main(int argc, char** argv) {
  int N = 1024;
  int B = 64; // frames
  int threads = 1;
  bool quick = false;
  // Default to exercising the CFFI recovery coordinator
  std::string mode = "restore"; // "roundtrip" or "restore"
  std::string transform_mode = "r2c"; // "r2c" (default) or "c2c"

  // Default: deterministic dropout so restores are guaranteed
  int debug_dropout = 0;
  // Deterministic dropout controls
  int dropout_period = 0;
  int dropout_on_len = 0;
  int dropout_phase = 0;
  long long dropout_single = -1;
  bool disable_crash_reports = false;
  bool dropout_persistent = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--N=", 0) == 0) N = std::stoi(a.substr(4));
    else if (a.rfind("--B=", 0) == 0) B = std::stoi(a.substr(4));
    else if (a.rfind("--threads=", 0) == 0) threads = std::stoi(a.substr(10));
    else if (a == "--quick") quick = true;
    else if (a.rfind("--mode=", 0) == 0) mode = a.substr(7);
    else if (a.rfind("--transform=", 0) == 0) transform_mode = a.substr(12);
    else if (a.rfind("--debug-dropout=", 0) == 0) debug_dropout = std::stoi(a.substr(16));
    else if (a.rfind("--dropout-pattern=", 0) == 0) {
      // Format: --dropout-pattern=period,on_len[,phase]
      std::string v = a.substr(18);
      int values[3] = {0, 0, 0};
      int idxv = 0;
      size_t startp = 0;
      while (startp <= v.size() && idxv < 3) {
        size_t comma = v.find(',', startp);
        std::string tok = (comma == std::string::npos) ? v.substr(startp) : v.substr(startp, comma - startp);
        if (!tok.empty()) {
          try { values[idxv++] = std::stoi(tok); } catch (...) { ++idxv; }
        } else {
          ++idxv; // empty token -> skip
        }
        if (comma == std::string::npos) break;
        startp = comma + 1;
      }
      if (idxv >= 2) {
        dropout_period = values[0];
        dropout_on_len = values[1];
        dropout_phase = (idxv >= 3) ? values[2] : 0;
      }
    }
    else if (a.rfind("--dropout-single=", 0) == 0) {
      try { dropout_single = std::stoll(a.substr(17)); } catch (...) { dropout_single = -1; }
    }
    else if (a == "--disable-crash-reports" || a == "--silent-crash") disable_crash_reports = true;
    else if (a == "--dropout-persistent") dropout_persistent = true;
  }
  if (quick) { N = 256; B = 8; }
  // Clamp to at least one thread but otherwise honor the requested value so
  // single-thread baselines can disable worker dropouts when needed.
  if (threads < 1) threads = 1;

  // Prepare input PCM: B consecutive frames of length N.
  const size_t pcm_len = static_cast<size_t>(N) * static_cast<size_t>(B);
  std::vector<float> pcm(pcm_len);
  std::mt19937_64 rng(1337);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (size_t i = 0; i < pcm_len; ++i) pcm[i] = static_cast<float>(dist(rng));

  // Configure CFFI-side dropout so the DLL recovery path is exercised.
  // Clear one-shot history at test start so flags take effect deterministically per run
  fft_clear_workerpool_dropout_history();
  // If no pattern or single specified, install a default deterministic pattern
  if (debug_dropout == 0 && dropout_period == 0 && dropout_on_len == 0 && dropout_single < 0) {
    dropout_period = 4; dropout_on_len = 1; dropout_phase = 0;
  }
  if (debug_dropout != 0) {
    fft_set_workerpool_debug_dropout(debug_dropout);
  }
  fft_set_workerpool_dropout_persistent(dropout_persistent ? 1 : 0);
  if (dropout_period > 0 && dropout_on_len > 0) {
    fft_set_workerpool_dropout_pattern(dropout_period, dropout_on_len, dropout_phase);
  }
  if (dropout_single >= 0) {
    fft_set_workerpool_dropout_single(dropout_single);
  }

  // prefer API-level control: do not use environment variables for internal tests.
  // `disable_crash_reports` will be passed into `fft_init_full` as the
  // `silent_crash_reports` argument so the CFFI initializer sets flags before
  // installing the handler.

  // Shared helper to create a context
  auto make_ctx = [&](int inverse, int transform)->void* {
    return fft_init_full(
        static_cast<size_t>(N),
        threads,
        1,               // lanes
        inverse,         // inverse
        0,               // kernel=auto
        0,               // radix=auto
        nullptr,
        0,
        1,               // pad_mode: pad last frame if needed
        N,               // window = N
        N,               // hop = N
        1,               // stft_mode = batched
        transform,       // transform
        0,               // reduce_magnitude
        0,               // store_polar
        1,               // half_spectrum
        1,               // allow_outer_parallel
        0,               // allow_inner_parallel
        0,               // inner_threads
        0,               // save_crash_logs: default OFF
        (disable_crash_reports ? 1 : 0) /* silent_crash_reports */
    );
  };

  if (mode == "roundtrip") {
    // Choose transform ids based on requested mode
    int fwd_transform = (transform_mode == "c2c") ? 0 : 1; // 0=C2C, 1=R2C
    int inv_transform = (transform_mode == "c2c") ? 0 : 2; // 0=C2C, 2=C2R

    // Forward
    void* ctx = make_ctx(0, fwd_transform);
    if (!ctx) {
      std::cerr << "fft_init_full failed\n";
      return 2;
    }

    size_t bins = (static_cast<size_t>(N) / 2) + 1;
    const size_t max_frames = static_cast<size_t>(B);
    const size_t out_size = max_frames * bins;
    std::vector<float> out_real(out_size);
    std::vector<float> out_imag(out_size);
    std::vector<float> out_mag(out_size);

    size_t produced = fft_execute_batched(ctx, pcm.data(), pcm_len, out_real.data(), out_imag.data(), out_mag.data(), 1 /*pad_mode*/, 0 /*enable_backup*/, max_frames);
    if (produced == 0) {
      std::cerr << "fft_execute_batched returned 0 frames (error)\n";
      fft_free(ctx);
      return 3;
    }

    std::cout << "ROUNDTRIP: forward produced=" << produced << " frames\n";

    // Inverse (C2R)
    void* inv = make_ctx(1, inv_transform);
    if (!inv) {
      std::cerr << "fft_init_full (inverse) failed\n";
      fft_free(ctx);
      return 4;
    }

    std::vector<float> restored(pcm_len);
    size_t recovered = fft_execute_complex_batched(inv, out_real.data(), out_imag.data(), produced, restored.data(), 1 /*pad_mode*/, 0 /*enable_backup*/, produced);
    if (recovered == 0) {
      std::cerr << "fft_execute_complex_batched failed\n";
      fft_free(ctx);
      fft_free(inv);
      return 5;
    }

    // Compare restored PCM to original (allow small float error)
    double max_abs = 0.0;
    double max_err = 0.0;
    for (size_t i = 0; i < pcm_len; ++i) {
      double a = pcm[i];
      double b = restored[i];
      max_abs = std::max(max_abs, std::abs(a));
      max_err = std::max(max_err, std::abs(a - b));
    }
    double rel = max_err / (max_abs + 1e-30);
    std::cout << "ROUND_TRIP: max_err=" << max_err << " rel=" << rel << "\n";

    fft_free(ctx);
    fft_free(inv);

    const double tol = 1e-3; // float tolerance
    if (rel <= tol) {
      std::cout << "ROUNDTRIP: PASS\n";
      return 0;
    }
    std::cout << "ROUNDTRIP: FAIL (rel_err=" << rel << ")\n";
    return 6;
  }

  // mode == "restore" (instrumented)
  {
    int fwd_transform = (transform_mode == "c2c") ? 0 : 1; // 0=C2C, 1=R2C
    int inv_transform = (transform_mode == "c2c") ? 0 : 2; // 0=C2C, 2=C2R

    void* ctx = make_ctx(0, fwd_transform);
    if (!ctx) {
      std::cerr << "fft_init_full failed\n";
      return 2;
    }
    size_t bins = (static_cast<size_t>(N) / 2) + 1;
    const size_t max_frames = static_cast<size_t>(B);
    const size_t out_size = max_frames * bins;
    std::vector<float> out_real(out_size);
    std::vector<float> out_imag(out_size);
    std::vector<float> out_mag(out_size);

    std::cout << "RESTORE TEST: running forward with enable_backup=1\n";
    // Observe forward-stage restores
    std::vector<unsigned char> fwd_restore_flags(B);
    std::fill(fwd_restore_flags.begin(), fwd_restore_flags.end(), 0);
    s_restore_flags_ptr = &fwd_restore_flags;
    s_restore_calls.store(0);
    s_restored_columns.store(0);
    fft_register_restore_observer(&test_restore_observer_c);

    size_t produced = fft_execute_batched(ctx, pcm.data(), pcm_len, out_real.data(), out_imag.data(), out_mag.data(), 1 /*pad_mode*/, 1 /*enable_backup*/, max_frames);
    // Unregister forward observer
    fft_register_restore_observer(nullptr);
    s_restore_flags_ptr = nullptr;
    std::cout << "RESTORE TEST: produced=" << produced << " frames\n";
    std::cout << "FORWARD_STATS: restore_calls=" << s_restore_calls.load() << " restored_columns_total=" << s_restored_columns.load() << "\n";
    // Forward restore grid: r=restored, .=no-restore (only for produced frames)
    size_t fcols = static_cast<size_t>(std::min<size_t>(64, produced));
    std::cout << "FORWARD_GRID: legend: r=restore, .=none\n";
    for (size_t r = 0; r * fcols < produced; ++r) {
      size_t row_start = r * fcols;
      size_t row_end = std::min(produced, row_start + fcols);
      for (size_t f = row_start; f < row_end; ++f) {
        const bool had_restore = (f < fwd_restore_flags.size() && fwd_restore_flags[f] != 0);
        std::cout << (had_restore ? 'r' : '.');
      }
      std::cout << "\n";
    }

    size_t workers = fft_ctx_worker_threads(ctx);
    size_t effective = fft_ctx_effective_threads(ctx);
    std::cout << "CFFI_CTX: workers=" << workers << " effective_plan_threads=" << effective << "\n";

    // Warm an inverse context and run restore path with enable_backup too.
    // For one‑shot dropout semantics, clear history so inverse can progress
    // independently when using very high dropout rates.
    fft_clear_workerpool_dropout_history();
    void* inv = make_ctx(1, inv_transform);
    if (!inv) {
      std::cerr << "fft_init_full (inverse) failed\n";
      fft_free(ctx);
      return 4;
    }

    std::vector<float> restored(pcm_len);
    std::cout << "RESTORE TEST: running inverse with enable_backup=1\n";
    // Prepare flags sized by produced frames and register observer.
    std::vector<unsigned char> restore_flags(produced);
    std::fill(restore_flags.begin(), restore_flags.end(), 0);
    s_restore_flags_ptr = &restore_flags;
    s_restore_calls.store(0);
    s_restored_columns.store(0);
    fft_register_restore_observer(&test_restore_observer_c);

    size_t recovered = fft_execute_complex_batched(inv, out_real.data(), out_imag.data(), produced, restored.data(), 1 /*pad_mode*/, 1 /*enable_backup*/, produced);
    // Unregister observer immediately after the call so other tests aren't affected.
    fft_register_restore_observer(nullptr);
    s_restore_flags_ptr = nullptr;

    std::cout << "RESTORE TEST: recovered=" << recovered << " frames\n";
    std::cout << "RESTORE_STATS: restore_calls=" << s_restore_calls.load() << " restored_columns_total=" << s_restored_columns.load() << "\n";

    // Provide simple instrumentation metrics AND an ASCII grid visualization
    double max_abs = 0.0;
    double max_err = 0.0;
    size_t mismatches = 0;
    const double per_frame_thresh = 1e-2; // frame-level threshold

    // Per-frame bad flag (true if that frame has any sample error > thresh)
    std::vector<unsigned char> frame_bad(produced);
    for (size_t f = 0; f < produced; ++f) {
      double frame_max_err = 0.0;
      const size_t base = f * static_cast<size_t>(N);
      for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        const double a = pcm[base + i];
        const double b = restored[base + i];
        const double err = std::abs(a - b);
        frame_max_err = std::max(frame_max_err, err);
        max_abs = std::max(max_abs, std::abs(a));
        max_err = std::max(max_err, err);
      }
      frame_bad[f] = (frame_max_err > per_frame_thresh) ? 1 : 0;
      if (frame_bad[f]) ++mismatches;
    }
    double rel = max_err / (max_abs + 1e-30);

    // Print ASCII grid: glyph per frame (O/x/X/o)
    size_t cols = static_cast<size_t>(std::min<size_t>(64, produced));
    std::cout << "RESTORE_GRID: legend: O=ok, x=restore->ok, X=restore->bad, o=bad(no restore)\n";
    size_t count_O = 0, count_x = 0, count_X = 0, count_o = 0;
    for (size_t r = 0; r * cols < produced; ++r) {
      size_t row_start = r * cols;
      size_t row_end = std::min(produced, row_start + cols);
      for (size_t f = row_start; f < row_end; ++f) {
        const bool had_restore = (restore_flags[f] != 0);
        const bool bad = (frame_bad[f] != 0);
        char glyph = '?';
        if (!had_restore && !bad) { glyph = 'O'; ++count_O; }
        else if (had_restore && !bad) { glyph = 'x'; ++count_x; }
        else if (had_restore && bad) { glyph = 'X'; ++count_X; }
        else if (!had_restore && bad) { glyph = 'o'; ++count_o; }
        std::cout << glyph;
      }
      std::cout << "\n";
    }

    std::cout << "RESTORE_COUNTS: O=" << count_O << " x=" << count_x << " X=" << count_X << " o=" << count_o << "\n";
    std::cout << "RESTORE_METRICS: max_err=" << max_err << " rel=" << rel << " mismatches=" << mismatches << "\n";

    fft_free(ctx);
    fft_free(inv);
    // Return non-zero if too many mismatches to surface failures to CI
    if (mismatches > 0) return 7;
    return 0;
  }
}
