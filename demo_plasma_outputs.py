"""demo_plasma_outputs.py

Plasma demo — all three projection worlds saved as audio and images.

Outputs (written to ./plasma_demo_out/):
  mono.wav             — interference world: real projection, phase folding visible
  stereo_quadrature.wav — rotation world: (L,R)=(cos,sin), no phase folding
  phase_gradient.wav   — ray world: integrate f_inst(t) → cos, no interference
  tf_field.png         — complex TF field (magnitude + phase panels)
  phase_gradient.png   — instantaneous frequency vs time
  lissajous.png        — (L,R) Lissajous figure showing phase-space trajectory

Run:
    python demo_plasma_outputs.py [--duration 4] [--n-rays 256] [--device cuda]
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np


PI2 = 2.0 * math.pi
OUT = "plasma_demo_out"


# ---------------------------------------------------------------------------
# Scene construction
# ---------------------------------------------------------------------------

def build_scene(
    duration:    float = 4.0,
    n_rays:      int   = 256,
    fmin:        float = 100.0,
    fmax:        float = 2000.0,
    B_field:     float = 0.003,
    synchrotron: float = 0.0001,
    device:      str   = "cuda",
    seed:        int   = 42,
    sample_rate: float = 48_000.0,
):
    from tf_ray_engine import TFRayGPU, TFRayScene
    import numpy as np

    tracer = TFRayGPU(
        scene_end        = duration,
        f_min            = fmin,
        f_max            = fmax,
        h_surfs          = [],
        v_surfs          = [],
        freq_mirror_loss = 0.02,
        time_mirror_loss = 0.02,
        scatter_rate     = 0.0,
        amp_loss         = 0.0,
        n_scattered      = 1,
        max_chirp        = 1500.0,
        max_batch        = 4096,
        max_segments     = 500_000,
        min_amplitude    = 0.02,
        max_bounces      = 120,
        device           = device,
        seed             = seed,
        phase_mode       = "physical",
        density_step     = 0.002,
    )

    tracer.set_plasma(
        B_field          = B_field,
        synchrotron_coeff= synchrotron,
        initial_charge   = -1.0,   # electron
    )

    # Grid of initial rays (forward + backward, spread over f and chirp)
    n_side = max(1, int(math.ceil(math.sqrt(n_rays / 2))))
    freqs  = np.linspace(fmin + 80, fmax - 150, n_side)
    chirps = np.linspace(-800.0, 800.0, n_side)
    rows   = []
    for vt0 in (1.0, -1.0):
        t0 = 0.0 if vt0 > 0 else duration
        for f0 in freqs:
            for vf0 in chirps:
                rows.append([t0, float(f0), vt0, float(vf0), 0.0, 1.0])
    init_arr = np.array(rows, dtype=np.float64)

    print(f"Launching {init_arr.shape[0]} rays on {device} …")
    all_segs = tracer.propagate(init_arr)
    print(f"  → {len(all_segs)} chirp segments")

    scene = TFRayScene(all_segs, scene_end=duration,
                       output_sample_rate=sample_rate, fade_s=0.002)
    return scene, all_segs


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def save_phase_gradient_png(f_inst: np.ndarray, t_axis: np.ndarray,
                             path: str, sr: float, dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), dpi=dpi)

    # Top: f_inst time series
    axes[0].plot(t_axis, f_inst, lw=0.5, color="steelblue")
    axes[0].set_ylabel("Instantaneous frequency (Hz)")
    axes[0].set_title("Phase gradient field  f(t) = dφ/dt / 2π")
    axes[0].set_xlim(t_axis[0], t_axis[-1])

    # Bottom: spectrogram-style density — f_inst as a scatter heatmap
    # bin into (time × freq) grid and show density
    n_t, n_f = 800, 400
    t0, t1   = float(t_axis[0]), float(t_axis[-1])
    active   = f_inst > 0
    if active.any():
        f_vals   = f_inst[active]
        t_vals   = t_axis[active]
        f0_v     = float(np.percentile(f_vals, 1))
        f1_v     = float(np.percentile(f_vals, 99))
        H, xedge, yedge = np.histogram2d(
            t_vals, f_vals,
            bins=[n_t, n_f],
            range=[[t0, t1], [f0_v, f1_v]],
        )
        axes[1].imshow(np.log1p(H.T), aspect="auto", origin="lower",
                       extent=[t0, t1, f0_v, f1_v],
                       cmap="plasma", interpolation="bilinear")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Phase gradient density (ray-world  TF)")

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_lissajous_png(L: np.ndarray, R: np.ndarray,
                       path: str, dpi: int = 150,
                       max_pts: int = 200_000) -> None:
    import matplotlib.pyplot as plt

    # Subsample for plotting if very long
    step = max(1, len(L) // max_pts)
    Ls, Rs = L[::step], R[::step]

    # Colour by time to show trajectory direction
    colours = np.linspace(0, 1, len(Ls))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    sc = ax.scatter(Ls, Rs, c=colours, cmap="hsv", s=0.3, linewidths=0, alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Time (normalised)")
    ax.set_aspect("equal")
    ax.set_xlabel("L  =  A·cos(φ)")
    ax.set_ylabel("R  =  A·sin(φ)")
    ax.set_title("Quadrature Lissajous  —  phase-space trajectory\n"
                 "(each point = unique phase state, colour = time)")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Plasma demo — all projection worlds")
    ap.add_argument("--duration",    type=float, default=4.0,    help="Scene duration (s)")
    ap.add_argument("--n-rays",      type=int,   default=256,    help="Initial ray count")
    ap.add_argument("--fmin",        type=float, default=100.0,  help="Min frequency (Hz)")
    ap.add_argument("--fmax",        type=float, default=2000.0, help="Max frequency (Hz)")
    ap.add_argument("--B-field",     type=float, default=0.003,  help="Cyclotron B field")
    ap.add_argument("--synchrotron", type=float, default=0.0001, help="Synchrotron coeff")
    ap.add_argument("--sample-rate", type=float, default=48_000.0)
    ap.add_argument("--device",      default="cuda",             help="cuda or cpu")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--dpi",         type=int,   default=150)
    ap.add_argument("--out",         default=OUT,                help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- build scene -------------------------------------------------------
    scene, _ = build_scene(
        duration    = args.duration,
        n_rays      = args.n_rays,
        fmin        = args.fmin,
        fmax        = args.fmax,
        B_field     = args.B_field,
        synchrotron = args.synchrotron,
        device      = args.device,
        seed        = args.seed,
        sample_rate = args.sample_rate,
    )

    # ---- audio outputs -----------------------------------------------------
    print("\n── Audio ──")
    scene.write_wav(os.path.join(args.out, "mono.wav"))
    scene.write_wav_stereo(os.path.join(args.out, "stereo_quadrature.wav"))
    scene.write_wav_phase_gradient(os.path.join(args.out, "phase_gradient.wav"))

    # ---- image outputs -----------------------------------------------------
    print("\n── Images ──")

    # Complex TF field (magnitude + phase panels)
    scene.save_field_png(
        os.path.join(args.out, "tf_field.png"),
        n_time            = 800,
        n_freq            = 600,
        freq_min          = args.fmin,
        freq_max          = args.fmax,
        log_freq          = False,
        heisenberg_spread = 2.0,
        log_magnitude     = True,
        dpi               = args.dpi,
        summation         = "complex",
        amplitude_percentile = 99.5,
    )
    print(f"Saved: {os.path.join(args.out, 'tf_field.png')}")

    # Phase gradient image
    f_inst, t_axis = scene.render_phase_gradient()
    save_phase_gradient_png(
        f_inst, t_axis,
        os.path.join(args.out, "phase_gradient.png"),
        sr=args.sample_rate,
        dpi=args.dpi,
    )

    # Lissajous (quadrature stereo)
    L, R = scene.render_audio_quadrature()
    save_lissajous_png(
        L, R,
        os.path.join(args.out, "lissajous.png"),
        dpi=args.dpi,
    )

    print(f"\nAll outputs written to: {os.path.abspath(args.out)}/")
    print("  mono.wav              — interference world")
    print("  stereo_quadrature.wav — rotation world  (L=cos, R=sin)")
    print("  phase_gradient.wav    — ray world        (integrate f_inst → cos)")
    print("  tf_field.png          — complex TF  (magnitude + phase)")
    print("  phase_gradient.png    — f_inst(t) field + density map")
    print("  lissajous.png         — phase-space trajectory")


if __name__ == "__main__":
    main()
