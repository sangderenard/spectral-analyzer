// Lightweight crash/minidump helper. Install early in process to capture
// unhandled crashes and write diagnostics to disk. Minimal, header-only.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <processthreadsapi.h>
#include <timeapi.h>
#include <io.h>
#include <fcntl.h>
// For symbol resolution
#include <dbghelp.h>
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#else
#include <signal.h>
#if defined(__linux__) || defined(__APPLE__)
#include <execinfo.h>
#include <unistd.h>
#endif
#endif

namespace fftfree {

static std::atomic<bool> g_crash_handler_installed{false};

// Control whether crash diagnostics write files (minidumps / .log). Default
// is false to avoid leaving artifacts during normal test runs. Tests or CFFI
// callers may opt-in via the provided setter.
static std::atomic<bool> g_crash_handler_write_files{false};

// When true, suppress any console/minidump/log output; used for "silent"
// failure mode in tests where we want no crash artifacts or console noise.
static std::atomic<bool> g_crash_handler_silent{false};

inline void set_crash_handler_silent(bool on) { g_crash_handler_silent.store(on ? true : false); }
inline bool crash_handler_silent_enabled() { return g_crash_handler_silent.load(); }

// Setter/getter for runtime control (call from C ABI to opt-in file writes).
inline void set_crash_handler_write_files(bool on) { g_crash_handler_write_files.store(on ? true : false); }
inline bool crash_handler_write_files_enabled() { return g_crash_handler_write_files.load(); }

inline std::string timestamp_string() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t = system_clock::to_time_t(now);
    std::tm tm;
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d-%H%M%S");
    return ss.str();
}

#if defined(_WIN32)

// Attempt to write a minidump via dbghelp.dll when an unhandled exception occurs.
inline void write_minidump(EXCEPTION_POINTERS* exinfo) {
    // Dynamic load to avoid link dependency in CI/builds that don't provide dbghelp.
    HMODULE hDbg = LoadLibraryA("dbghelp.dll");
    if (!hDbg) return;
    // If silent mode requested, do nothing.
    if (g_crash_handler_silent.load()) { FreeLibrary(hDbg); return; }
    using MiniDumpWriteDump_t = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, DWORD, void*, void*, void*);
    auto fn = reinterpret_cast<MiniDumpWriteDump_t>(GetProcAddress(hDbg, "MiniDumpWriteDump"));
    if (!fn) { FreeLibrary(hDbg); return; }

    DWORD pid = GetCurrentProcessId();
    HANDLE proc = GetCurrentProcess();

    // If file writing is disabled, avoid creating a minidump file.
    if (!g_crash_handler_write_files.load()) { FreeLibrary(hDbg); return; }
    std::string fname = "fftfree-crash-" + timestamp_string() + ".dmp";
    HANDLE fh = CreateFileA(fname.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) { FreeLibrary(hDbg); return; }

    // MINIDUMP_TYPE 0 => MiniDumpNormal
    const DWORD dumpType = 0;

    struct MINIDUMP_EXCEPTION_INFORMATION_WRAPPER {
        DWORD ThreadId;
        EXCEPTION_POINTERS* ExInfo;
        BOOL ClientPointers;
    } mei;
    mei.ThreadId = GetCurrentThreadId();
    mei.ExInfo = exinfo;
    mei.ClientPointers = FALSE;

    // Call the function
    fn(proc, pid, fh, dumpType, &mei, nullptr, nullptr);

    CloseHandle(fh);
    // Notify user unless silent mode is enabled
    if (!g_crash_handler_silent.load()) {
        std::fprintf(stderr, "Wrote minidump\n");
    }
    FreeLibrary(hDbg);
}

// Write a human-readable stack trace (best-effort) to stderr and a .log file.
inline void write_text_backtrace(EXCEPTION_POINTERS* /*exinfo*/) {
    if (g_crash_handler_silent.load()) return;
    const int kMaxFrames = 62;
    void* frames[kMaxFrames];
    USHORT captured = CaptureStackBackTrace(0, kMaxFrames, frames, nullptr);

    // Try to resolve symbols via dbghelp
    HMODULE hDbg = LoadLibraryA("dbghelp.dll");
    HANDLE proc = GetCurrentProcess();
    if (hDbg) {
        using SymInitialize_t = BOOL(WINAPI*)(HANDLE, PCSTR, BOOL);
        using SymFromAddr_t = BOOL(WINAPI*)(HANDLE, DWORD64, PDWORD64, PSYMBOL_INFO);
        using SymSetOptions_t = DWORD(WINAPI*)(DWORD);

        auto pSymInitialize = reinterpret_cast<SymInitialize_t>(GetProcAddress(hDbg, "SymInitialize"));
        auto pSymFromAddr = reinterpret_cast<SymFromAddr_t>(GetProcAddress(hDbg, "SymFromAddr"));
        auto pSymSetOptions = reinterpret_cast<SymSetOptions_t>(GetProcAddress(hDbg, "SymSetOptions"));

        if (pSymSetOptions) {
            // Request undecorated names
            pSymSetOptions(SYMOPT_DEFERRED_LOADS | SYMOPT_UNDNAME);
        }
        if (pSymInitialize) {
            pSymInitialize(proc, nullptr, TRUE);
        }

    // Only open a file when file-writing is enabled.
        FILE* f = nullptr;
        if (g_crash_handler_write_files.load()) {
            std::string fname = "fftfree-crash-" + timestamp_string() + ".log";
            f = std::fopen(fname.c_str(), "w");
            if (f) {
                std::fprintf(f, "CaptureStackBackTrace frames=%hu\n", captured);
            }
        }

    for (USHORT i = 0; i < captured; ++i) {
            DWORD64 addr = reinterpret_cast<DWORD64>(frames[i]);
            char symbol_buf[sizeof(SYMBOL_INFO) + 1024];
            PSYMBOL_INFO pSym = reinterpret_cast<PSYMBOL_INFO>(symbol_buf);
            std::memset(pSym, 0, sizeof(symbol_buf));
            pSym->SizeOfStruct = sizeof(SYMBOL_INFO);
            pSym->MaxNameLen = 1024;
            DWORD64 displacement = 0;
            const char* name = "<unknown>";
            if (pSymFromAddr && pSymFromAddr(proc, addr, &displacement, pSym)) {
                name = pSym->Name;
            }
            if (!g_crash_handler_silent.load()) std::fprintf(stderr, "#%02u %p %s +0x%llx\n", (unsigned)i, (void*)addr, name, (unsigned long long)displacement);
            if (f) std::fprintf(f, "#%02u %p %s +0x%llx\n", (unsigned)i, (void*)addr, name, (unsigned long long)displacement);
        }
        if (f) {
            std::fprintf(f, "Wrote text backtrace\n");
            std::fclose(f);
        }

        if (pSymInitialize) {
            // SymCleanup is optional; try to call if available
            auto pSymCleanup = reinterpret_cast<BOOL(WINAPI*)(HANDLE)>(GetProcAddress(hDbg, "SymCleanup"));
            if (pSymCleanup) pSymCleanup(proc);
        }
        FreeLibrary(hDbg);
    } else {
        // Fallback: just print addresses (unless silent)
        if (!g_crash_handler_silent.load()) {
            std::fprintf(stderr, "Stack frames (addresses): captured=%hu\n", captured);
            for (USHORT i = 0; i < captured; ++i) {
                std::fprintf(stderr, "#%02u %p\n", (unsigned)i, frames[i]);
            }
        }
    }
}

inline LONG WINAPI vectored_exception_handler(EXCEPTION_POINTERS* exinfo) {
    DWORD code = exinfo && exinfo->ExceptionRecord ? exinfo->ExceptionRecord->ExceptionCode : 0;
    if (code == 0x40010006 /* DBG_PRINTEXCEPTION_C */ ||
        code == 0x4001000A /* DBG_PRINTEXCEPTION_WIDE_C (undoc) */ ||
        code == 0x80000003 /* EXCEPTION_BREAKPOINT */ ||
        code == 0x406D1388 /* SetThreadName */) {
        return EXCEPTION_CONTINUE_SEARCH;
    }

    static thread_local bool in_handler = false;
    if (in_handler) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    struct HandlerScope {
        bool& flag;
        explicit HandlerScope(bool& f) : flag(f) { flag = true; }
        ~HandlerScope() { flag = false; }
    } scope(in_handler);

    write_text_backtrace(exinfo);
    write_minidump(exinfo);
    return EXCEPTION_CONTINUE_SEARCH;
}

inline void install_crash_handler_impl() {
    if (g_crash_handler_installed.exchange(true)) return;
    // Install vectored handler so we get called for crashes
    AddVectoredExceptionHandler(1, reinterpret_cast<PVECTORED_EXCEPTION_HANDLER>(vectored_exception_handler));
}

// Helper invoked inside the SEH filter expression. Must be a normal function
// (not a local definition) because filter expressions are evaluated in a
// restricted context where some intrinsics are only valid.
inline int fftfree_seh_on_exception(EXCEPTION_POINTERS* ep) {
    write_text_backtrace(ep);
    write_minidump(ep);
    return EXCEPTION_EXECUTE_HANDLER;
}

// C-callable thin wrapper to execute a user-provided thunk under SEH and
// produce diagnostics (minidump + text backtrace) on exception. This is
// inline in the header so callers can use it without adding a new TU.
// Returns 0 on normal return or 1 if an SEH exception occurred and was
// handled (diagnostics written). The return value allows callers to decide
// whether to invoke recovery/restore logic.
extern "C" inline int fftfree_run_with_seh(void (*thunk)(void*), void* ctx) {
    __try {
        thunk(ctx);
        return 0;
    } __except (fftfree_seh_on_exception(GetExceptionInformation())) {
        // Diagnostics already written in the filter expression; signal caller.
        return 1;
    }
}

#else

inline void write_backtrace_to_file(int signo, void* context) {
#if defined(_WIN32)
    (void)context; (void)signo;
    // On Windows this path is not used; keep signature but no-op when file
    // writing is disabled.
#else
    // Respect silent mode first
    if (g_crash_handler_silent.load()) return;
    if (!g_crash_handler_write_files.load()) {
        std::fprintf(stderr, "Received signal %d (backtrace suppressed)\n", signo);
        return;
    }
    std::string fname = "fftfree-crash-" + timestamp_string() + ".log";
    FILE* f = std::fopen(fname.c_str(), "w");
    if (!f) return;
    std::fprintf(f, "Received signal %d\n", signo);
#if defined(__linux__) || defined(__APPLE__)
    void* bt[64];
    int n = backtrace(bt, static_cast<int>(std::size(bt)));
    backtrace_symbols_fd(bt, n, fileno(f));
#endif
    std::fprintf(f, "Wrote backtrace\n");
    std::fclose(f);
    // Also write to stderr unless silent
    if (!g_crash_handler_silent.load()) std::fprintf(stderr, "Wrote crash log\n");
#endif
}

inline void signal_handler(int signo, siginfo_t* si, void* context) {
    (void)si;
    write_backtrace_to_file(signo, context);
    // Re-raise default to allow usual crash behavior (core, abort)
    signal(signo, SIG_DFL);
    raise(signo);
}

inline void install_crash_handler_impl() {
    if (g_crash_handler_installed.exchange(true)) return;
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = signal_handler;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGABRT, &sa, nullptr);
    sigaction(SIGFPE, &sa, nullptr);
    sigaction(SIGILL, &sa, nullptr);
}

#endif

// Public installer. Safe to call multiple times.
inline void install_crash_handler() {
    // If runtime silent flag set, do not install handler (API-level opt-out).
    if (g_crash_handler_silent.load()) return;
    // Honor environment opt-out for backward compatibility
    const char* env = std::getenv("FFTFREE_DISABLE_CRASH_HANDLER");
    if (env && std::strcmp(env, "1") == 0) return;
    install_crash_handler_impl();
}

} // namespace fftfree
