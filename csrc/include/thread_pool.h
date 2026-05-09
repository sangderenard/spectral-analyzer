/**
 * thread_pool.h — Minimal std::thread-based work-stealing thread pool.
 *
 * Header-only.  C++17.  No external dependencies.
 *
 * Usage
 * -----
 *   ThreadPool pool;                         // hardware_concurrency threads
 *   ThreadPool pool(4);                      // explicit thread count
 *
 *   auto fut = pool.enqueue([](int x){ return x*x; }, 7);
 *   int result = fut.get();                  // 49
 *
 *   // Parallel-for helper (partitions [0, n) across threads):
 *   ThreadPool::parallel_for(pool, 0, n, [](size_t i){ ... });
 *
 * Safety
 * ------
 *   - Destructor drains the queue, then joins all threads.
 *   - Exceptions propagated through std::future.
 *   - Re-entrant enqueue is safe (the caller's thread does not execute tasks).
 */
#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

class ThreadPool {
public:
    explicit ThreadPool(size_t n_threads = 0)
        : _stop(false)
    {
        if (n_threads == 0)
            n_threads = std::max<size_t>(1u, std::thread::hardware_concurrency());

        _workers.reserve(n_threads);
        for (size_t i = 0; i < n_threads; ++i) {
            _workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(_mutex);
                        _cv.wait(lk, [this]{ return _stop || !_tasks.empty(); });
                        if (_stop && _tasks.empty()) return;
                        task = std::move(_tasks.front());
                        _tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lk(_mutex);
            _stop = true;
        }
        _cv.notify_all();
        for (auto& w : _workers) {
            if (w.joinable()) w.join();
        }
    }

    /* Enqueue any callable + args; returns a future for the result. */
    template<typename F, typename... Args>
    auto enqueue(F&& f, Args&&... args)
        -> std::future<typename std::invoke_result<F, Args...>::type>
    {
        using Ret = typename std::invoke_result<F, Args...>::type;

        auto task_ptr = std::make_shared<std::packaged_task<Ret()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<Ret> fut = task_ptr->get_future();
        {
            std::unique_lock<std::mutex> lk(_mutex);
            if (_stop)
                throw std::runtime_error("ThreadPool: enqueue on stopped pool");
            _tasks.emplace([task_ptr]{ (*task_ptr)(); });
        }
        _cv.notify_one();
        return fut;
    }

    size_t n_threads() const { return _workers.size(); }

    /* ── Parallel-for helper ─────────────────────────────────────────────── */
    /* Calls body(i) for each i in [begin, end) partitioned across threads.
     * Blocks until all items are processed.  Rethrows first exception. */
    template<typename IndexType, typename Body>
    static void parallel_for(ThreadPool& pool,
                             IndexType begin, IndexType end,
                             Body&& body)
    {
        const size_t n      = static_cast<size_t>(end - begin);
        const size_t nw     = pool.n_threads();
        const size_t chunk  = (n + nw - 1) / nw;

        std::vector<std::future<void>> futs;
        futs.reserve(nw);

        for (size_t t = 0; t < nw; ++t) {
            const size_t lo = begin + t * chunk;
            const size_t hi = std::min<size_t>(lo + chunk, static_cast<size_t>(end));
            if (lo >= static_cast<size_t>(end)) break;

            futs.push_back(pool.enqueue([&body, lo, hi]{
                for (size_t i = lo; i < hi; ++i)
                    body(i);
            }));
        }

        std::exception_ptr ep = nullptr;
        for (auto& f : futs) {
            try { f.get(); }
            catch (...) {
                if (!ep) ep = std::current_exception();
            }
        }
        if (ep) std::rethrow_exception(ep);
    }

    /* Non-copyable, non-movable. */
    ThreadPool(const ThreadPool&)            = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&)                 = delete;
    ThreadPool& operator=(ThreadPool&&)      = delete;

private:
    std::vector<std::thread>          _workers;
    std::queue<std::function<void()>> _tasks;
    std::mutex                        _mutex;
    std::condition_variable           _cv;
    bool                              _stop;
};
