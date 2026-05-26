"""Headless BDPT evaluation — mirrors the exact call chain from thick_lens_focus_lab.py.

Runs forward + backward traces, drains the pipeline between steps, shadow-
connects endpoints, and saves the accumulated linear image as a PNG.
No pygame, no display, no GL window.

Usage:
    python eval_bdpt.py
    python eval_bdpt.py --steps 20 --rpe 2 --res 32 --out bdpt_out.png
    python eval_bdpt.py --steps 5 --rpe 4 --sensor-gain 8 --emitter-gain 1
"""

import argparse
import gc
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thick_lens_focus_lab import (
    DEFAULT_FREQ_HZ,
    ForwardCppLensBench,
    FreeFrequencySidecar,
    SceneConfig,
    _SHADER_DIR,
)


def save_png(path: str, rgb_f32: np.ndarray) -> None:
    import struct, zlib
    h, w = rgb_f32.shape[:2]
    pixels = np.clip(rgb_f32 * 255.0, 0, 255).astype(np.uint8)

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(h))
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as f:
        f.write(png)
    print(f"[eval-bdpt] saved {path} ({w}x{h})", flush=True)


def drain_to_empty(bench, timeout: float = 120.0, poll: float = 0.05) -> bool:
    """Block until in_flight_count() == 0 or timeout. Returns True if clean."""
    t0 = time.perf_counter()
    while True:
        try:
            n = int(bench.tracer.in_flight_count())
        except Exception:
            n = 0
        if n <= 0:
            return True
        if time.perf_counter() - t0 > timeout:
            print(f"[eval-bdpt] drain timeout — still {n} in flight after {timeout:.0f}s", flush=True)
            return False
        time.sleep(poll)


def run(
    steps: int = 10,
    rpe: int = 2,
    res: int = 32,
    max_bounces: int = 1,
    sensor_gain: float = 1.0,
    emitter_gain: float = 1.0,
    sensor_min_amp: float = 0.0,
    compute_mode: str = "gpu",
    drain_timeout: float = 60.0,
    out: str = "bdpt_eval.png",
    band_stride: int = 4,
) -> None:
    scene = SceneConfig()
    scene.image_plate.sensor_res = int(max(4, res))

    # Sub-sample the spectral bands for faster evaluation.
    # Every 4th band = 8 bands at ~40 nm spacing; still covers 380-700 nm.
    freq_hz = DEFAULT_FREQ_HZ[::band_stride].copy()
    sidecar = FreeFrequencySidecar.lazy_prepare(int(freq_hz.size))
    bench = ForwardCppLensBench(
        scene=scene,
        freq_hz=freq_hz,
        view_h=360,
        view_w=640,
        sidecar=sidecar,
    )
    bench.compute_mode = compute_mode
    bench.sensor_amp_gain = float(sensor_gain)
    bench.emitter_amp_gain = float(emitter_gain)
    bench.sensor_min_amplitude = float(sensor_min_amp)

    bench.tracer.configure_sensor_image(
        float(scene.image_plate.x),
        float(scene.image_plate.radius),
        int(max(16, scene.image_plate.sensor_res)),
        0.008,
    )

    if compute_mode in ("gpu", "mixed"):
        print("[eval-bdpt] ensuring GPU pipeline …", flush=True)
        bench.tracer.ensure_pipeline(
            max_children=2,
            seed=13579,
            min_amplitude=float(bench._min_amplitude),
            use_gpu_compute=True,
            gpu_all_stages=(compute_mode == "gpu"),
            shader_dir=_SHADER_DIR,
        )
        print("[eval-bdpt] GPU pipeline ready", flush=True)

    n_emitters = int(bench.src_pos.shape[0]) if hasattr(bench, "src_pos") else "?"
    print(
        f"[eval-bdpt] steps={steps} rpe={rpe} res={res} bounces={max_bounces}"
        f" compute={compute_mode} n_emitters={n_emitters}"
        f" sensor_gain={sensor_gain} emitter_gain={emitter_gain}",
        flush=True,
    )

    # ── accumulation loop ─────────────────────────────────────────────────────
    # Key requirement: drain completely between steps so the shadow pass
    # starts with in_flight ≈ 0.  Without this, shadow rays queue behind
    # thousands of unfinished main-pipeline rays and the 15s shadow timeout
    # fires with first_hit=inf → everything "visible" → sparkles.
    for step in range(int(max(1, steps))):
        seed = 20260601 + step * 137
        t0 = time.perf_counter()

        bench.trace_forward(rpe, seed, max_bounces=max_bounces)
        bench.trace_sensor_cast(rpe, seed ^ 0x5A5A, max_bounces=max_bounces)

        # Wait until the GPU has finished all submitted rays.
        clean = drain_to_empty(bench, timeout=drain_timeout)

        fwd_n = bench._async_bdpt_forward_count()
        bwd_n = bench._async_bdpt_backward_count()
        print(
            f"[eval-bdpt] step {step+1}/{steps}"
            f"  fwd_ep={fwd_n:_}  bwd_ep={bwd_n:_}"
            f"  elapsed={time.perf_counter()-t0:.1f}s"
            f"  drain={'ok' if clean else 'TIMEOUT'}",
            flush=True,
        )

    # ── shadow connection pass ────────────────────────────────────────────────
    # The pipeline is now idle; shadow rays will drain promptly.
    print("[eval-bdpt] running shadow connection …", flush=True)
    plate = bench.trace_forward_backward_sensor_rgb(
        pixels=int(scene.image_plate.sensor_res),
        aperture_samples=8,
        seed=20260601,
        max_bounces=max_bounces,
        n_rays_bdpt=1_000_000,
    )

    # ── diagnostics ──────────────────────────────────────────────────────────
    ss = bench.bdpt_last_shadow_stats
    lin = bench._bdpt_plate_linear_accum
    wgt = bench._bdpt_plate_weight_accum

    print(
        f"\n[eval-bdpt] === RESULTS ===",
        f"\n  endpoints  fwd={ss.get('n_fwd',0)}  bwd={ss.get('n_bwd',0)}",
        f"\n  pairs built={ss.get('n_pairs',0)}",
        f"\n  shadow rays={ss.get('shadow_rays', ss.get('n_pairs',0))}",
        f"\n  visible={ss.get('visible',0)}  blocked={ss.get('blocked',0)}",
        flush=True,
    )
    if lin is not None and lin.size > 0:
        nonzero = int(np.count_nonzero(lin))
        total   = int(lin.size // 3)
        print(
            f"  lit pixels={nonzero//3}/{total} ({100*nonzero//3/max(1,total):.0f}%)",
            f"\n  linear max={float(np.max(lin)):.3e}  sum={float(np.sum(lin)):.3e}",
            f"\n  p_fwd={float(bench._emitter_launch_pdf):.3e}",
            f"  p_bwd=1/A_ap={float(bench._last_backward_aperture_pdf):.3e}",
            f"  cos_exit={float(bench._last_backward_cos_exit):.3f}",
            flush=True,
        )
    else:
        print("  WARNING: linear accumulator empty — no connections formed", flush=True)

    if plate is not None and plate.size > 0:
        save_png(out, np.asarray(plate, dtype=np.float32))
    else:
        print("[eval-bdpt] plate empty — not saving", flush=True)

    bench._drain_stop.set()
    if bench._drain_thread is not None:
        bench._drain_thread.join(timeout=2.0)
    del bench
    gc.collect()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Headless BDPT evaluator")
    ap.add_argument("--steps",        type=int,   default=10)
    ap.add_argument("--rpe",          type=int,   default=2,   help="Rays per emitter per step")
    ap.add_argument("--res",          type=int,   default=32,  help="Sensor grid side (pixels)")
    ap.add_argument("--bounces",      type=int,   default=1,   help="Max bounces (1=direct only)")
    ap.add_argument("--sensor-gain",  type=float, default=1.0)
    ap.add_argument("--emitter-gain", type=float, default=1.0)
    ap.add_argument("--sensor-min-amp", type=float, default=0.0)
    ap.add_argument("--compute-mode", type=str,   default="gpu", choices=["cpu","gpu","mixed"])
    ap.add_argument("--drain-timeout",type=float, default=60.0, help="Per-step drain timeout (s)")
    ap.add_argument("--band-stride",  type=int,   default=4,    help="Sub-sample bands: 1=all 32, 4=every 4th=8 bands (default), 8=4 bands")
    ap.add_argument("--out",          type=str,   default="bdpt_eval.png")
    args = ap.parse_args()

    run(
        steps=args.steps,
        rpe=args.rpe,
        res=args.res,
        max_bounces=args.bounces,
        sensor_gain=args.sensor_gain,
        emitter_gain=args.emitter_gain,
        sensor_min_amp=args.sensor_min_amp,
        compute_mode=args.compute_mode,
        drain_timeout=args.drain_timeout,
        out=args.out,
        band_stride=args.band_stride,
    )
