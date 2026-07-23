// Large transform correctness test
// Performs forward then inverse FFT on a large N x B matrix using the
// selected algorithm and precision, then compares the round-trip result to
// the original and reports PASS/FAIL with relative epsilon metrics.

#include "../eigen_fft.hpp"
#include "../plan_support.hpp"
#include <Eigen/Core>
#include <chrono>
#include <complex>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <utility>

using namespace eigfft;

struct PoolDispatcherLocal : public JobDispatcher {
  WorkerPool* pool = nullptr;
  void parallel_for(size_t total, size_t chunk, const Fn& fn) override {
    if (!pool || total == 0) {
      InlineDispatcher::instance().parallel_for(total, chunk, fn);
      return;
    }
    pool->parallel_for(total, chunk, fn);
  }

  void parallel_for_with_restore(size_t total,
                                 size_t chunk,
                                 const Fn& fn,
                                 std::function<void(size_t,size_t)> restore_fn,
                                 RecoveryOps ops,
                                 void* job_ctx) override {
    if (!pool || total == 0) {
      InlineDispatcher::instance().parallel_for_with_restore(total, chunk, fn,
                                                            std::move(restore_fn),
                                                            ops, job_ctx);
      return;
    }
    pool->parallel_for_with_restore_ex(total, (chunk == 0) ? 1 : chunk, fn,
                                       std::move(restore_fn), ops, job_ctx);
  }
};

template <typename Scalar>
int run_transform_test(int N, int B, const std::string& alg, int threads) {
  using Complex = std::complex<Scalar>;
  using MatrixXc = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  PlanRuntimeConfig cfg;
  cfg.threads = threads;
  cfg.lanes = 1;
  cfg.transform = 0; // C2C

  PoolDispatcherLocal dispatcher_local;
  std::unique_ptr<WorkerPool> pool;
  // Always construct a WorkerPool when the test requests at least one
  // thread so we exercise the pool code paths even for threads==1.
  if (threads >= 1) {
    pool = std::make_unique<WorkerPool>(threads);
    dispatcher_local.pool = pool.get();
    try {
      std::cout << "POOL_CREATED: size=" << pool->size() << " (threads requested=" << threads << ")\n";
    } catch (...) {}
  }

  // Create plan for forward
  PlanEnvironment<Scalar> env_fwd;
  env_fwd.initialize(N, /*inverse=*/false, cfg);
  try { env_fwd.plan().set_dispatcher(&dispatcher_local); } catch(...) {}
  // Select algorithm
  if (alg == "stockham") env_fwd.plan().use_kernel(KernelKind::Stockham);
  else env_fwd.plan().use_kernel(KernelKind::CooleyTukey);

  // Allocate data
  MatrixXc data(N, B);
  MatrixXc original(N, B);
  std::mt19937_64 rng(1337);
  std::normal_distribution<double> dist(0.0, 1.0);
  for (int j = 0; j < B; ++j) {
    for (int i = 0; i < N; ++i) {
      const Scalar re = static_cast<Scalar>(dist(rng));
      const Scalar im = static_cast<Scalar>(dist(rng));
      data(i, j) = Complex(re, im);
      original(i, j) = data(i, j);
    }
  }

  // Forward
  const auto t0 = std::chrono::high_resolution_clock::now();
  fft_inplace_batched<Scalar>(data, env_fwd.plan());
  const auto t1 = std::chrono::high_resolution_clock::now();

  // Capture intermediate forward output for later per-element inspection
  MatrixXc after_fwd = data;
  if (const char* dump_env = std::getenv("FFTFREE_DUMP_AFTER_FWD")) {
    std::string path = dump_env;
    if (!path.empty()) {
      std::ofstream ofs(path);
      if (ofs.is_open()) {
        ofs << std::setprecision(12);
        for (int j = 0; j < B; ++j) {
          for (int i = 0; i < N; ++i) {
            const Complex v = after_fwd(i, j);
            ofs << i << ',' << j << ',' << v.real() << ',' << v.imag() << '\n';
          }
        }
      }
    }
  }

  // Inverse
  PlanEnvironment<Scalar> env_inv;
  cfg.inverse = true;
  env_inv.initialize(N, /*inverse=*/true, cfg);
  try { env_inv.plan().set_dispatcher(&dispatcher_local); } catch(...) {}
  if (alg == "stockham") env_inv.plan().use_kernel(KernelKind::Stockham);
  else env_inv.plan().use_kernel(KernelKind::CooleyTukey);

  fft_inplace_batched<Scalar>(data, env_inv.plan());
  const auto t2 = std::chrono::high_resolution_clock::now();

  if (const char* dump_env = std::getenv("FFTFREE_DUMP_AFTER_INV")) {
    std::string path = dump_env;
    if (!path.empty()) {
      std::ofstream ofs(path);
      if (ofs.is_open()) {
        ofs << std::setprecision(12);
        for (int j = 0; j < B; ++j) {
          for (int i = 0; i < N; ++i) {
            const Complex v = data(i, j);
            ofs << i << ',' << j << ',' << v.real() << ',' << v.imag() << '\n';
          }
        }
      }
    }
  }

  // Compute errors: try both unscaled and scaled-by-N inverse to be robust
  double max_abs_orig = 0.0;
  double max_abs_err_unscaled = 0.0;
  double max_abs_err_scaled = 0.0;
  for (int j = 0; j < B; ++j) {
    for (int i = 0; i < N; ++i) {
      const Complex o = original(i, j);
      const Complex r = data(i, j);
      const double ao = std::abs(o);
      const double ar = std::abs(r);
      max_abs_orig = std::max(max_abs_orig, ao);
      const double diff_un = std::abs(r - o);
      const double diff_sc = std::abs((r / static_cast<Scalar>(N)) - o);
      max_abs_err_unscaled = std::max(max_abs_err_unscaled, diff_un);
      max_abs_err_scaled = std::max(max_abs_err_scaled, diff_sc);
    }
  }

  const double min_err = std::min(max_abs_err_unscaled, max_abs_err_scaled);
  const double rel_err = min_err / (max_abs_orig + 1e-30);

  const double ms = std::chrono::duration<double>(t1 - t0).count();
  const double mr = std::chrono::duration<double>(t2 - t1).count();

  const double tol = std::is_same<Scalar, float>::value ? 1e-3 : 1e-9;

  std::cout << "ALG=" << alg << " N=" << N << " B=" << B << " threads=" << threads
            << " forward_s=" << ms << " inv_s=" << mr << " rel_err=" << rel_err << "\n";

  // Print plan configuration summary to help verify forward/inverse wiring
  try {
    const auto& pf = env_fwd.plan();
    const auto& pi = env_inv.plan();
    std::cout << "PLAN_SUMMARY: fwd.kernel=" << (pf.kernel().name ? pf.kernel().name : "(null)")
              << " inv.kernel=" << (pi.kernel().name ? pi.kernel().name : "(null)")
              << " fwd.inverse=" << (pf.inverse ? "true" : "false")
              << " inv.inverse=" << (pi.inverse ? "true" : "false")
              << " fwd.half_spectrum=" << (pf.half_spectrum ? "true" : "false")
              << " inv.half_spectrum=" << (pi.half_spectrum ? "true" : "false")
              << " fwd.store_polar=" << (pf.store_polar ? "true" : "false")
              << " inv.store_polar=" << (pi.store_polar ? "true" : "false")
              << " fwd.reduce_magnitude=" << (pf.reduce_magnitude ? "true" : "false")
              << " inv.reduce_magnitude=" << (pi.reduce_magnitude ? "true" : "false")
              << "\n";
  } catch (...) {
    // best-effort; ignore if plans unavailable
  }

  if (rel_err <= tol) {
    std::cout << "RESTORE_TEST: PASS\n";
    return 0;
  }
  std::cout << "RESTORE_TEST: FAIL (rel_err=" << rel_err << " tol=" << tol << ")\n";

  // Dump a table of mismatches (every element not within tolerance).
  // Columns: i,j,orig_re,orig_im,after_fwd_re,after_fwd_im,res_re,res_im,abs_err_unscaled,abs_err_scaled
  std::cout << "MISMATCHES: i,j,orig_re,orig_im,after_fwd_re,after_fwd_im,res_re,res_im,abs_err_unscaled,abs_err_scaled\n";
  const double thresh = tol * (max_abs_orig + 1e-30);
  size_t mismatch_count = 0;
  const size_t max_dump = 256; // only print this many detailed rows; still count all mismatches
  size_t printed = 0;
  for (int j = 0; j < B; ++j) {
    for (int i = 0; i < N; ++i) {
      const Complex o = original(i, j);
      const Complex f = after_fwd(i, j);
      const Complex r = data(i, j);
      const double diff_un = std::abs(r - o);
      const double diff_sc = std::abs((r / static_cast<Scalar>(N)) - o);
      const double eff = std::min(diff_un, diff_sc);
      if (eff > thresh) {
        ++mismatch_count;
        if (printed < max_dump) {
          ++printed;
          std::cout.setf(std::ios::scientific);
          std::cout << i << "," << j << ","
                    << std::setprecision(9) << o.real() << "," << o.imag() << ","
                    << std::setprecision(9) << f.real() << "," << f.imag() << ","
                    << std::setprecision(9) << r.real() << "," << r.imag() << ","
                    << std::setprecision(6) << diff_un << "," << diff_sc << "\n";
          std::cout.unsetf(std::ios::scientific);
        }
      }
    }
  }
  std::cout << "MISMATCH_COUNT: " << mismatch_count << " (printed=" << printed << ")\n";
  return 2;
}

int main(int argc, char** argv) {
  int N = 8192;
  int B = 8192;
  std::string alg = "ct";
  std::string prec = "f32";
  int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  bool quick = false;

  int debug_dropout = 0;
  bool disable_crash_reports = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--N=", 0) == 0) N = std::stoi(a.substr(4));
    else if (a.rfind("--B=", 0) == 0) B = std::stoi(a.substr(4));
    else if (a.rfind("--alg=", 0) == 0) alg = a.substr(6);
    else if (a.rfind("--precision=", 0) == 0) prec = a.substr(12);
    else if (a.rfind("--threads=", 0) == 0) threads = std::stoi(a.substr(10));
    else if (a == "--quick") quick = true;
    else if (a.rfind("--debug-dropout=", 0) == 0) debug_dropout = std::stoi(a.substr(16));
    else if (a == "--disable-crash-reports" || a == "--silent-crash") disable_crash_reports = true;
  }

  if (quick) { N = 512; B = 512; }
  if (debug_dropout != 0) eigfft::set_workerpool_debug_dropout(debug_dropout);
  // Honor test-level opt-out for crash diagnostics: set env var so
  // crash_handler::install_crash_handler will early-return.
  if (disable_crash_reports) {
    // Prefer API-level control: request silent crash handler via runtime flag.
    fftfree::set_crash_handler_silent(true);
  }
  // Normalize alg token
  if (alg == "ct") alg = "cooleytukey"; 
  if (alg == "cooleytukey") alg = "ct"; // keep "ct" or "stockham" comparison

  if (prec == "f64" || prec == "double") {
    return run_transform_test<double>(N, B, (alg == "stockham" ? "stockham" : "ct"), threads);
  } else {
    return run_transform_test<float>(N, B, (alg == "stockham" ? "stockham" : "ct"), threads);
  }
}
