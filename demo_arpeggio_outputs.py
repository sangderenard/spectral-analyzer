"""demo_arpeggio_outputs.py

Arpeggiation demo — all three projection worlds saved as audio and images,
using the discrete harmonic signal engine (signal_generator_v2 / sequence_engine).

Source:  ArpeggioRule → NoteSchedule → SequenceRenderer → AdaptiveSampleBuffer
Outputs (written to ./arpeggio_demo_out/):
  mono.wav              — interference world: real projection, phase folding
  stereo_quadrature.wav — rotation world: (L,R)=(cos,sin), no phase folding
  phase_gradient.wav    — ray world: integrate f_inst(t) → cos
  tf_analytic.png       — analytical TF (frequency ridges from phase differences)
  phase_gradient.png    — instantaneous frequency field + density map
  lissajous.png         — (L,R) phase-space trajectory coloured by time

Run:
    python demo_arpeggio_outputs.py [options]
    python demo_arpeggio_outputs.py --help
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from signal_generator_v2 import (
    AdaptiveProjector,
    AnalyticalTFRenderer,
    DensityPolicy,
    PhaseWarpedManifold,
    ProjectionPolicy,
    SmoothSinusoidalDriftModel,
    WitnessThresholds,
    PI2,
)
from sequence_engine import (
    ArpeggioRule,
    LatticeBuilder,
    MoodPreset,
    SequenceRenderer,
    TimbrePreset,
    CHORD_PROGRESSIONS,
    MODAL_SCALES,
    MOOD_PRESETS,
    SCALE_MOODS,
    adsr_envelope_factory,
    monotone_envelope_factory,
    piecewise_envelope_factory,
    spline_envelope_factory,
)

OUT = "arpeggio_demo_out"


# ---------------------------------------------------------------------------
# Build and render the arpeggiation
# ---------------------------------------------------------------------------

_DEFAULT_KNOTS = [(0.0, 0.0), (0.01, 1.0), (0.1, 0.72), (0.85, 0.72), (1.0, 0.0)]


def build_arpeggio(
    root_hz:       float = 110.0,
    scale:         str   = "pentatonic_minor",
    bpm:           float = 120.0,
    repeats:       int   = 2,
    partial_count: int   = 6,
    sample_rate:   float = 48_000.0,
    carry_phase:   bool  = True,
    harmonic_lock: bool  = False,
    # Envelope — which type to use (adsr / spline / monotone / linear)
    env_type:      str   = "adsr",
    use_envelope:  bool  = True,
    # ADSR parameters (only used when env_type="adsr")
    env_attack:    float = 0.008,
    env_decay:     float = 0.04,
    env_sustain:   float = 0.72,
    env_release:   float = 0.06,
    env_peak:      float = 1.0,
    # Knot-based parameters (used when env_type is spline / monotone / linear)
    # Each knot is (normalized_time_0_to_1, amplitude_0_to_1).
    env_knots = None,
):
    rule = ArpeggioRule(
        root_hz       = root_hz,
        scale         = scale,
        pattern       = [0, 2, 4, 6, 7, 6, 4, 2,
                         1, 3, 5, 7, 8, 7, 5, 3],
        rhythm_beats  = [0.375],
        bpm           = bpm,
        velocity_curve= [1.0, 0.75, 0.85, 0.70,
                         0.90, 0.70, 0.80, 0.65,
                         1.0, 0.75, 0.85, 0.70,
                         0.90, 0.70, 0.80, 0.65],
        legato_fraction = 0.82,
        partial_count   = partial_count,
        octave_span     = 2,
        repeats         = repeats,
    )

    schedule = rule.generate()
    print(f"Schedule: {len(schedule.events)} notes  "
          f"{schedule.total_duration:.3f}s total  "
          f"bpm={bpm}  scale={scale}  root={root_hz:.1f}Hz")

    if not use_envelope:
        envelope_fac = None
        env_desc = "off (ExponentialEnvelope only)"
    elif env_type == "adsr":
        envelope_fac = adsr_envelope_factory(
            attack_time   = env_attack,
            decay_time    = env_decay,
            sustain_level = env_sustain,
            release_time  = env_release,
            peak_level    = env_peak,
        )
        env_desc = (f"adsr  A={env_attack}s D={env_decay}s "
                    f"S={env_sustain} R={env_release}s peak={env_peak}")
    else:
        knots = env_knots if env_knots is not None else _DEFAULT_KNOTS
        if env_type == "spline":
            envelope_fac = spline_envelope_factory(knots)
            env_desc = f"spline  knots={knots}"
        elif env_type == "monotone":
            envelope_fac = monotone_envelope_factory(knots)
            env_desc = f"monotone  knots={knots}"
        elif env_type == "linear":
            envelope_fac = piecewise_envelope_factory(knots)
            env_desc = f"linear  knots={knots}"
        else:
            raise ValueError(f"Unknown env_type: {env_type!r}")

    timbre = TimbrePreset(
        waveform_manifold = PhaseWarpedManifold(harmonic_warp_strength=0.18),
        drift_model       = SmoothSinusoidalDriftModel(
            base_rate_hz          = 0.06,
            max_offset_hz         = 0.025,
            harmonic_scaling_power= 1.0,
        ),
        amplitude_decay_tau = 0.22,
        shape_low    = 0.05,
        shape_high   = 0.65,
        shape_center = 0.08,
        shape_width  = 0.05,
        envelope_factory = envelope_fac,
    )
    print(f"Envelope: {env_desc}")

    projection_policy = ProjectionPolicy(
        output_sample_rate           = sample_rate,
        projection_half_support_seconds = 0.002,
        projection_kernel_steps      = 256,
    )

    builder  = LatticeBuilder(
        timbre            = timbre,
        projection_policy = projection_policy,
        witness_thresholds = WitnessThresholds(
            backward_phase_radians       = PI2,
            forward_phase_radians        = PI2,
            max_support_seconds          = 3.0,
            support_search_step_seconds  = 0.001,
            integration_steps_per_check  = 32,
        ),
        density_policy = DensityPolicy(
            oversampling_factor = 16.0,
            min_sample_rate     = 48_000.0,
            max_sample_rate     = 2_000_000.0,
            derivative_weight   = 0.5,
            support_weight      = 1.0,
        ),
    )
    renderer = SequenceRenderer(builder=builder, harmonic_lock=harmonic_lock)

    print("Rendering notes …")
    global_buf = renderer.render(schedule, carry_phase=carry_phase)
    print(f"  Adaptive samples: {len(global_buf)}")

    projector = AdaptiveProjector(projection_policy)
    return global_buf, projector, schedule.total_duration, int(sample_rate)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def save_tf_png(
    tf_renderer: AnalyticalTFRenderer,
    buf,
    t_end:  float,
    path:   str,
    dpi:    int = 150,
) -> None:
    image, t_ax, f_ax = tf_renderer.render_from_buffer(
        buf, 0.0, t_end,
        n_time   = 900,
        n_freq   = 600,
        log_freq = True,
    )
    tf_renderer.save_png(image, t_ax, f_ax, path,
                         title="Analytical TF  (phase-difference estimator)",
                         log_amplitude=True, dpi=dpi)
    print(f"Saved: {path}")


def save_phase_gradient_png(
    f_inst:  np.ndarray,
    t_axis:  np.ndarray,
    path:    str,
    dpi:     int = 150,
) -> None:
    import matplotlib.pyplot as plt

    active = f_inst > 0
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), dpi=dpi)

    # Top: raw f_inst time series (subsample for readability)
    step = max(1, len(t_axis) // 8000)
    axes[0].plot(t_axis[::step], f_inst[::step], lw=0.5, color="steelblue")
    axes[0].set_ylabel("Instantaneous frequency (Hz)")
    axes[0].set_title("Phase gradient field  f(t) = dφ/dt / 2π")
    axes[0].set_xlim(float(t_axis[0]), float(t_axis[-1]))

    # Bottom: 2-D histogram density (time × freq)
    if active.any():
        f_v  = f_inst[active]
        t_v  = t_axis[active]
        f0_v = float(np.percentile(f_v, 1))
        f1_v = float(np.percentile(f_v, 99))
        H, xe, ye = np.histogram2d(
            t_v, f_v, bins=[700, 400],
            range=[[float(t_axis[0]), float(t_axis[-1])], [f0_v, f1_v]],
        )
        axes[1].imshow(
            np.log1p(H.T), aspect="auto", origin="lower",
            extent=[float(t_axis[0]), float(t_axis[-1]), f0_v, f1_v],
            cmap="plasma", interpolation="bilinear",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_title("Phase gradient density — ray-world TF")

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_lissajous_png(
    L:       np.ndarray,
    R:       np.ndarray,
    path:    str,
    dpi:     int   = 150,
    max_pts: int   = 200_000,
) -> None:
    import matplotlib.pyplot as plt

    step = max(1, len(L) // max_pts)
    Ls, Rs = L[::step], R[::step]
    colours = np.linspace(0, 1, len(Ls))

    fig, ax = plt.subplots(figsize=(7, 7), dpi=dpi)
    sc = ax.scatter(Ls, Rs, c=colours, cmap="hsv", s=0.4, linewidths=0, alpha=0.55)
    plt.colorbar(sc, ax=ax, label="Time (normalised)")
    ax.set_aspect("equal")
    ax.set_xlabel("L  =  A·cos(φ)")
    ax.set_ylabel("R  =  A·sin(φ)")
    ax.set_title("Quadrature Lissajous — phase-space trajectory\n"
                 "(colour = time; each point = unique phase state)")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Companion sidecar (OpenGL-ready analytic stream)
# ---------------------------------------------------------------------------

_SIDECAR_CHANNELS = ["L", "R", "amplitude", "f_inst"]
_SIDECAR_STRIDE   = 16   # 4 × float32


def write_companion_sidecar(
    out_dir: str,
    L:       np.ndarray,
    R:       np.ndarray,
    f_inst:  np.ndarray,
    sr:      int,
    viz_fps: float = 120.0,
    prefix:  str   = "companion",
) -> None:
    """
    Write OpenGL-ready companion sidecar files for the analytic signal.

    Binary layout (float32 little-endian, 16 bytes / sample):

        offset  0 : L        — in-phase  (=mono real projection)
        offset  4 : R        — quadrature (90° rotated)
        offset  8 : A        — instantaneous amplitude  sqrt(L²+R²)
        offset 12 : f_inst   — instantaneous frequency (Hz)

    In a GL vertex shader, bind the VBO once and declare two attribs::

        // Lissajous / vectorscope XY
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 16, (void*)0);
        // amplitude + f_inst for colour/alpha mapping
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 16, (void*)8);

    Lissajous mode  :  gl_Position = vec4(L, R, 0, 1)
    Vectorscope mode:  gl_Position = vec4((L+R)*0.707, (L-R)*0.707, 0, 1)
                       (mid/side projection, standard broadcast orientation)

    Two rates are written:
        {prefix}_full.f32  — audio sample rate (for accurate frame sync)
        {prefix}_viz.f32   — downsampled to viz_fps  (lightweight stream for GPU upload)
        {prefix}_meta.json — metadata for the player (sample rates, sizes, attrib layout)
    """
    N    = len(L)
    L32  = L.astype(np.float32)
    R32  = R.astype(np.float32)
    A32  = np.sqrt(L32**2 + R32**2)
    f32  = f_inst.astype(np.float32)

    # Full-rate interleaved array  (N, 4) float32
    full = np.stack([L32, R32, A32, f32], axis=1)          # shape (N, 4), C-contiguous
    full_path = os.path.join(out_dir, f"{prefix}_full.f32")
    full.tofile(full_path)

    # Viz-rate: integer decimation
    viz_step = max(1, int(round(sr / viz_fps)))
    viz      = full[::viz_step]
    viz_path = os.path.join(out_dir, f"{prefix}_viz.f32")
    viz.tofile(viz_path)

    actual_viz_hz = sr / viz_step

    meta = {
        "version": 1,
        "channels": _SIDECAR_CHANNELS,
        "dtype": "float32",
        "byte_stride": _SIDECAR_STRIDE,
        "layout": "interleaved",
        "sample_rate": sr,
        "n_samples": N,
        "viz_rate_hz": actual_viz_hz,
        "viz_step": viz_step,
        "n_viz_samples": int(len(viz)),
        "duration_s": N / sr,
        "gl_attribs": {
            "xy_lissajous": {"index": 0, "size": 2, "type": "GL_FLOAT",
                             "stride": 16, "offset": 0,
                             "note": "L=X R=Y for Lissajous; (L+R)*0.707=X (L-R)*0.707=Y for vectorscope"},
            "amp_finst":    {"index": 1, "size": 2, "type": "GL_FLOAT",
                             "stride": 16, "offset": 8,
                             "note": "[0]=amplitude [1]=f_inst_hz"},
        },
        "vectorscope_glsl": "vec2 vs = vec2((L+R)*0.70710678, (L-R)*0.70710678);",
        "files": {
            "full_rate": os.path.basename(full_path),
            "viz_rate":  os.path.basename(viz_path),
            "meta":      f"{prefix}_meta.json",
        },
    }
    meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"  {os.path.basename(full_path):<36} {full.nbytes/1e6:6.1f} MB  "
          f"({N} samples @ {sr} Hz)")
    print(f"  {os.path.basename(viz_path):<36} {viz.nbytes/1e3:6.1f} kB  "
          f"({len(viz)} samples @ {actual_viz_hz:.0f} Hz)")
    print(f"  {os.path.basename(meta_path)}")


def write_hilbert_companion(
    out_dir: str,
    mono:    np.ndarray,
    sr:      int,
    viz_fps: float = 120.0,
) -> None:
    """
    Derive the analytic companion from the *mono* real projection via Hilbert
    transform and write its own sidecar.

    This simulates what a player would do in real time if it only had the
    mono WAV — no pre-computed sidecar.  Comparing ``hilbert_companion_*``
    against ``companion_*`` shows how much phase/amplitude information is
    recoverable from the collapsed real projection alone.

    f_inst is estimated from the derivative of the Hilbert-derived unwrapped
    phase; this is noisy near amplitude zeros.
    """
    from scipy.signal import hilbert as _hilbert

    analytic = _hilbert(mono.astype(np.float64))
    L_h      = np.real(analytic).astype(np.float32)
    R_h      = np.imag(analytic).astype(np.float32)

    # Instantaneous frequency from unwrapped phase derivative
    phase_h  = np.unwrap(np.angle(analytic))
    dt       = 1.0 / sr
    f_h      = np.gradient(phase_h, dt) / (2.0 * math.pi)
    f_h      = np.clip(f_h, 0.0, sr / 2.0).astype(np.float32)

    print("\n── Hilbert companion sidecar (real-time reconstruction simulation) ──")
    write_companion_sidecar(out_dir, L_h, R_h, f_h, sr, viz_fps,
                            prefix="hilbert_companion")
    print("  Note: phase folding from real projection may cause artefacts in R_h.")
    print("        Compare against companion_* to see information lost in mono.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Arpeggio demo — all projection worlds")
    ap.add_argument("--root-hz",      type=float, default=110.0,
                    help="Root frequency in Hz (default: A2=110)")
    ap.add_argument("--scale",        default="pentatonic_minor",
                    choices=sorted(MODAL_SCALES),
                    help="Modal scale name")
    ap.add_argument("--bpm",          type=float, default=120.0)
    ap.add_argument("--repeats",      type=int,   default=2,
                    help="Pattern repetitions")
    ap.add_argument("--partials",     type=int,   default=6,
                    help="Harmonic partials per note")
    ap.add_argument("--sample-rate",  type=float, default=48_000.0)
    ap.add_argument("--carry-phase",  action="store_true", default=True,
                    help="Phase-continuous note handoffs (default on)")
    ap.add_argument("--no-carry-phase", dest="carry_phase", action="store_false")
    ap.add_argument("--harmonic-lock", action="store_true", default=False)
    ap.add_argument("--out",          default=OUT)
    ap.add_argument("--dpi",          type=int,   default=150)
    # Envelope
    ap.add_argument("--no-envelope",  dest="use_envelope", action="store_false", default=True,
                    help="Disable complex envelope (use raw exponential decay only)")
    ap.add_argument("--env-type",     default="adsr",
                    choices=["adsr", "spline", "monotone", "linear"],
                    help="Envelope interpolation type (default: adsr)")
    # ADSR parameters — only used when --env-type adsr
    ap.add_argument("--env-attack",   type=float, default=0.008,  metavar="S",
                    help="[adsr] Attack time in seconds (default 0.008)")
    ap.add_argument("--env-decay",    type=float, default=0.04,   metavar="S",
                    help="[adsr] Decay time in seconds (default 0.04)")
    ap.add_argument("--env-sustain",  type=float, default=0.72,   metavar="0-1",
                    help="[adsr] Sustain level 0-1 (default 0.72)")
    ap.add_argument("--env-release",  type=float, default=0.06,   metavar="S",
                    help="[adsr] Release time in seconds (default 0.06)")
    ap.add_argument("--env-peak",     type=float, default=1.0,    metavar="0-1",
                    help="[adsr] Peak level (default 1.0)")
    # Knot-based parameters — used when --env-type is spline / monotone / linear
    ap.add_argument("--env-knots",    default=None, metavar="JSON",
                    help=("[spline/monotone/linear] JSON array of [t,v] knots with t in [0,1] "
                          "(normalized note duration).  Example: "
                          "\"[[0,0],[0.01,1],[0.1,0.7],[0.85,0.7],[1,0]]\""
                          "  Defaults to a built-in ADSR-like shape if omitted."))
    # Companion sidecar
    ap.add_argument("--sidecar",       dest="sidecar", action="store_true",  default=True,
                    help="Write OpenGL-ready companion sidecar files (default on)")
    ap.add_argument("--no-sidecar",    dest="sidecar", action="store_false",
                    help="Skip companion sidecar output")
    ap.add_argument("--viz-fps",       type=float, default=120.0,
                    help="Viz-rate decimated stream frame rate (default 120)")
    ap.add_argument("--hilbert-check", action="store_true", default=False,
                    help="Also write a Hilbert-derived companion from the mono signal "
                         "(simulates real-time analytic reconstruction)")
    ap.add_argument("--mood",          default=None,
                    choices=sorted(MOOD_PRESETS),
                    help=("Apply a named mood preset (overrides --scale, --bpm, and pattern). "
                          "Use --scale after --mood to override just the scale within the preset. "
                          f"Available: {', '.join(sorted(MOOD_PRESETS))}"))
    ap.add_argument("--list-moods",   action="store_true", default=False,
                    help="Print all mood presets with descriptions and exit.")
    args = ap.parse_args()

    if args.list_moods:
        print("\nAvailable mood presets:\n")
        for k, preset in sorted(MOOD_PRESETS.items()):
            mood_meta = SCALE_MOODS.get(preset.scale, {})
            adj = ", ".join(mood_meta.get("adjectives", [])[:4])
            print(f"  {k:<18}  {preset.scale:<22}  {preset.bpm:5.0f} bpm  {adj}")
            print(f"                     {preset.description}")
            if preset.progression in CHORD_PROGRESSIONS:
                prog = CHORD_PROGRESSIONS[preset.progression]
                print(f"                     chords: {' – '.join(prog['chords'])}")
            print()
        return

    # Apply mood preset if requested (before scale/bpm are consumed)
    mood_scale = args.scale
    mood_bpm   = args.bpm
    if args.mood is not None:
        preset = MOOD_PRESETS[args.mood]
        # --scale explicitly provided on CLI overrides mood's scale
        if args.scale == ap.get_default("scale"):
            mood_scale = preset.scale
        mood_bpm = preset.bpm
        print(f"Mood preset '{args.mood}': scale={mood_scale}  bpm={mood_bpm:.0f}  "
              f"progression={preset.progression}")
        print(f"  {preset.description}")

    # Parse knots JSON if provided
    env_knots = None
    if args.env_knots is not None:
        try:
            raw = json.loads(args.env_knots)
            env_knots = [(float(p[0]), float(p[1])) for p in raw]
        except Exception as exc:
            ap.error(f"--env-knots: invalid JSON — {exc}")

    os.makedirs(args.out, exist_ok=True)
    sr = int(args.sample_rate)

    # ---- render analytic buffer -------------------------------------------
    buf, projector, duration, sr = build_arpeggio(
        root_hz       = args.root_hz,
        scale         = mood_scale,
        bpm           = mood_bpm,
        repeats       = args.repeats,
        partial_count = args.partials,
        sample_rate   = args.sample_rate,
        carry_phase   = args.carry_phase,
        harmonic_lock = args.harmonic_lock,
        use_envelope  = args.use_envelope,
        env_type      = args.env_type,
        env_attack    = args.env_attack,
        env_decay     = args.env_decay,
        env_sustain   = args.env_sustain,
        env_release   = args.env_release,
        env_peak      = args.env_peak,
        env_knots     = env_knots,
    )

    # ---- three projections ------------------------------------------------
    print("\n── Projecting ──")

    print("  real (mono) …")
    mono = projector.project_real_numpy(buf, 0.0, duration)

    print("  quadrature (stereo) …")
    L, R = projector.project_quadrature_numpy(buf, 0.0, duration)

    print("  phase gradient …")
    f_inst, t_axis = projector.project_phase_gradient_numpy(buf, 0.0, duration)

    # ---- audio output -----------------------------------------------------
    print("\n── Audio ──")
    import scipy.io.wavfile as _wav

    peak = float(np.abs(mono).max()) or 1.0
    _wav.write(os.path.join(args.out, "mono.wav"),
               sr, (mono / peak).astype(np.float32))
    print(f"  mono.wav  ({len(mono)} samples @ {sr} Hz)")

    peak_lr = max(float(np.abs(L).max()), float(np.abs(R).max())) or 1.0
    stereo  = np.stack([(L / peak_lr).astype(np.float32),
                        (R / peak_lr).astype(np.float32)], axis=1)
    _wav.write(os.path.join(args.out, "stereo_quadrature.wav"), sr, stereo)
    print(f"  stereo_quadrature.wav  (L=cos R=sin, no phase folding)")

    # Sonify phase gradient: cumulative-trapezoid integrate f_inst → phase → cos
    if len(t_axis) > 1:
        dt_pg   = float(t_axis[1] - t_axis[0])
        phase_pg = np.zeros_like(f_inst)
        phase_pg[1:] = np.cumsum(0.5 * (f_inst[:-1] + f_inst[1:]) * dt_pg) * PI2
    else:
        phase_pg = np.zeros_like(f_inst)
    pg_audio = np.cos(phase_pg).astype(np.float32)
    _wav.write(os.path.join(args.out, "phase_gradient.wav"), sr, pg_audio)
    mean_f = float(f_inst[f_inst > 0].mean()) if (f_inst > 0).any() else 0.0
    print(f"  phase_gradient.wav  (mean f_inst={mean_f:.1f} Hz)")

    # ---- companion sidecar ------------------------------------------------
    if args.sidecar:
        print("\n\u2500\u2500 Companion sidecar \u2500\u2500")
        # Normalise to the same peak as the stereo file so the sidecar
        # XY coords are in [-1, +1] — direct clip space for the GL player.
        peak_lr = max(float(np.abs(L).max()), float(np.abs(R).max())) or 1.0
        write_companion_sidecar(
            args.out,
            L       / peak_lr,
            R       / peak_lr,
            f_inst,
            sr,
            viz_fps = args.viz_fps,
        )
        if args.hilbert_check:
            write_hilbert_companion(
                args.out,
                mono / (float(np.abs(mono).max()) or 1.0),
                sr,
                viz_fps = args.viz_fps,
            )
    print("\n── Images ──")

    tf_renderer = AnalyticalTFRenderer(freq_sigma_hz=10.0)
    save_tf_png(tf_renderer, buf, duration,
                os.path.join(args.out, "tf_analytic.png"), dpi=args.dpi)

    save_phase_gradient_png(f_inst, t_axis,
                            os.path.join(args.out, "phase_gradient.png"),
                            dpi=args.dpi)

    save_lissajous_png(L, R,
                       os.path.join(args.out, "lissajous.png"),
                       dpi=args.dpi)

    print(f"\nAll outputs in: {os.path.abspath(args.out)}/")
    print("  mono.wav                   — interference world")
    print("  stereo_quadrature.wav      — rotation world  (L=cos, R=sin)")
    print("  phase_gradient.wav         — ray world        (integrate f_inst)")
    print("  tf_analytic.png            — analytical TF ridges (no FFT)")
    print("  phase_gradient.png         — f(t) + density map")
    print("  lissajous.png              — phase-space trajectory")
    if args.sidecar:
        print("  companion_full.f32         — GL VBO: interleaved [L,R,A,f] float32 @ audio SR")
        print("  companion_viz.f32          — same decimated to viz_fps")
        print("  companion_meta.json        — attrib layout + file metadata")
    if args.sidecar and args.hilbert_check:
        print("  hilbert_companion_*.f32/json — same from Hilbert reconstruction (mono only)")


if __name__ == "__main__":
    main()
