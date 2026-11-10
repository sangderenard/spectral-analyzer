# Retained Live yield primitive

The existing Live worker accepts server-authored `scheduler_state` (`paused`
or `running`) and monotonically increasing integer `scheduler_generation`
inside its existing `live-control.json`. Scheduler generation is independent
of camera/display revision. A stale generation cannot resume newer paused work.

Pause completes the current sensor sweep and calls `glFinish` before writing
`state: yielded`. The same process retains the accumulator, linear history,
forward field, random seed and GL context. Status includes `retained_pass`,
`retained_row`, `retained_history_bytes`, `gpu_dispatch_yielded: true` and
`gpu_memory_released: false`. Yielding execution does not free GPU memory.
Admission of another GPU job therefore still requires a real combined memory
budget; this primitive alone is not a concurrency scheduler.

While scheduler-paused, an expired browser lease does not destroy retained
work. A resume command with an expired lease remains yielded until the existing
browser-control renewal arrives. Explicit job cancellation remains the host's
existing cancellation path. The host must own and expose paused-job lifetime,
not leave an invisible indefinite allocation. Authenticated Nodus-backed job
storage/load management belongs to the host, not a second worker-side queue.

No production scheduler or server restart was activated here. The unchanged
default remains running when no scheduler fields are present. Camera changes
retain the existing explicit sweep-boundary reset semantics.

Verification: `python -m pytest tests/test_marble_live_ray_batch.py -q` passed
16 tests, including unchanged row order and seed progression across yield,
no work while waiting, expired-lease retention and stale-generation rejection.
An isolated real optical worker test exists at
`tests/studio_live_yield_gpu_check.py`, but its first run lost tool transport
with a blocked telemetry request before a success receipt. It is not a
verified GPU pause/resume result. Do not infer completion from the partial log.
