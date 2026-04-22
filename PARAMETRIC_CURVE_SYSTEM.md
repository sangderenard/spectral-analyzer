# Parametric Curve System — Design Reference

**File:** `parametric_curve_editor.py`  
**Purpose:** Unified 1-D parametric curve engine: editor data model, bake pipeline (spline → analytic signal → damped complex oscillators), region effect DSL, time-warp gate engine, and sample-accurate evaluation under arbitrary note state history.

---

## 1. Core Concept

A `ParametricCurve` is **not a lookup table and not a polynomial**.  
It is a piecewise sum of **damped complex exponentials** (Matrix Pencil decomposition):

$$z^{(s)}(u) = \sum_{k=1}^{K} c_k^{(s)} \cdot e^{s_k^{(s)} \cdot u}, \quad s_k = \sigma_k + j\omega_k$$

where:
- $u \in [0,1]$ is the local normalized position within segment $s$
- $\sigma_k$ is decay rate (negative = decay, zero = pure oscillation, positive = growth)
- $\omega_k$ is instantaneous frequency (the Hilbert-quadrature component)
- $c_k$ is the complex amplitude + phase offset of the $k$-th mode
- Real part = the actual curve value; imaginary part = Hilbert quadrature (analytic signal)

This representation has **exact phase** within each segment. It can be evaluated at any $t$ without interpolation or lookup — just compute the sum of exponentials.

The curve is **designed in normalized space** $[0,1] \times [0,1]$ and mapped to real-world units at evaluation time.

---

## 2. Data Model

### 2.1 `ControlPoint`

```python
@dataclass
class ControlPoint:
    t:           float        # normalized x ∈ [0, 1]
    v:           float        # normalized y ∈ [0, 1]
    tension:     float = 0.5  # Catmull-Rom tangent scale: 0=linear, 1=loose
    break_after: bool  = False  # True → discontinuity (gap) after this point
```

Control points are the **primary segmentation anchors** — the Matrix Pencil bake always places segment boundaries at their `t` positions, plus adaptive curvature-based sub-boundaries within each interval.

`break_after=True` declares an intended discontinuity. The curve value may jump at this point. Phase stitching does NOT bridge these gaps (magnitude jump is preserved; only phase is stitched across smooth joins).

### 2.2 `TimeMarker`

```python
@dataclass
class TimeMarker:
    t:      float       # normalized x ∈ (0, 1)
    label:  str  = ""   # arbitrary region name
    pinned: bool = False
```

Markers divide the curve into **regions**. Dragging a marker proportionally rescales all control points in the two adjacent regions (shape is preserved, time is remapped). Region index 0 spans `[0, markers[0].t)`, region $k$ spans `[markers[k-1].t, markers[k].t)`, and the final region spans `[markers[-1].t, 1]`.

### 2.3 `RegionEffect`

Each region can have one of seven modes:

| Mode | Behavior |
|---|---|
| `normal` | Plain Catmull-Rom evaluation, no modification |
| `silence` | Force output to 0.0 in this region |
| `hold` | Lock output to the value at the left boundary (sample-and-hold) |
| `loop` | Tile `sub_curve` repeatedly within the region (loop_count times, or fill) |
| `mirror` | Tile `sub_curve` with alternating reversal (ping-pong tile) |
| `additive` | Add `sub_curve * lfo_depth` on top of the base curve, clamped [0,1] |
| `gate` | Binarise: `v > gate_threshold → 1.0`, else `0.0` |

`sub_curve` is itself a `ParametricCurve` — **nested curves are fully recursive**.

### 2.4 `ParametricCurve` Fields

```python
@dataclass
class ParametricCurve:
    points:              List[ControlPoint]       # sorted by t
    markers:             List[TimeMarker]         # sorted by t
    regions:             Dict[int, RegionEffect]  # region_index → effect
    v_lo:                float = 0.0              # real-world minimum
    v_hi:                float = 1.0              # real-world maximum
    slew_samples:        int   = 0                # one-pole IIR lag (0 = off)
    n_oscillators:       int   = 0                # 0 = auto-select K by SVD
    osc_energy_thresh:   float = 1e-3             # SVD energy cutoff for auto K
    piecewise_segments:  int   = 1                # total bake segment budget
    activation:          str   = "none"           # post-eval nonlinearity
    activation_drive:    float = 1.0              # activation temperature
    bake_resolution:     int   = 1024             # dense render sample count
    name:                str   = "curve"
    warp_coordinator:    TimeWarpCoordinator | None = None
```

### 2.5 Internal Bake Cache (not serialised)

```python
    _seg_poles:    torch.Tensor  # [S, K] complex128 — per-segment poles s_k
    _seg_residues: torch.Tensor  # [S, K] complex128 — per-segment residues c_k
    _seg_breaks:   torch.Tensor  # [S+1]  float64   — segment boundary t-values
```

The cache is invalidated whenever control points, markers, region effects, or bake parameters change. Activation and slew never invalidate the cache.

---

## 3. Bake Pipeline (`_ensure_baked`)

Three sequential stages convert the user-drawn curve into the oscillator representation.

### Stage 1 — Dense Spline Render

1. Sort control points by `t`.
2. Split into **Catmull-Rom chains** at any `break_after=True` point. Each chain is a continuous spline. Gaps between chains are filled with `gap_fill=0.0`.
3. Render all chains onto a dense `t_dense = linspace(0, 1, bake_resolution)` grid → `base [N]` float64.
4. Apply region effects in-place over the dense grid (silence, hold, gate, loop, mirror, additive). Nested `sub_curve` calls recurse through this same pipeline.

### Stage 2 — Segment Layout

Segments are placed at two levels:

1. **Forced breaks** at every control point `t` value (the user's declared complexity anchors).
2. **Curvature-adaptive sub-breaks** within each interval, allocated proportionally to curvature mass, until the total segment budget `S = max(len(ctrl_pts)+1, piecewise_segments)` is exhausted.

This is computed by `_curvature_segment_breaks(base, S, ctrl_t)` → `seg_breaks [S+1]` float64 Tensor.

### Stage 3 — Matrix Pencil per Segment

For each segment $s$:

1. Extract the dense slice `seg = base[i_lo : i_hi + 1]` (min 4 samples, zero-padded if shorter).
2. Compute the **analytic signal** via FFT Hilbert transform: `z_s = _analytic_signal(seg)` → complex128 Tensor.
3. Run **Matrix Pencil** decomposition: `(poles_s, residues_s) = _matrix_pencil_fit(z_s, ...)`.
   - Constructs the Hankel data matrix via `z.unfold(0, L+1, 1)`.
   - Computes economy SVD. Auto-selects $K$ from SVD energy cumsum (or uses `n_oscillators` if fixed).
   - Solves the Matrix Pencil eigenproblem via `torch.linalg.lstsq` + `torch.linalg.eigvals`.
   - Ratchets $K$ upward until per-sample RMSE ≤ `8·N·ε₆₄·seg_scale` (or $K$ = N//2-1).
   - **All operations are pure torch, no numpy.**

4. Zero-pad all segments' pole/residue vectors to `K_max` (the largest $K$ across all segments) → stacked `[S, K_max]` tensors.

---

## 4. Evaluation

### 4.1 `evaluate_normalized(t_norm)` — Real output, [0,1]

```
t_norm [M] float64
  → _eval_oscillators_piecewise(seg_poles, seg_residues, seg_breaks, t_norm)
  → .real   ← ONLY legitimate .real in the repo
  → .clamp(0, 1)
  → _apply_activation(out, mode, drive)   [optional]
  → _apply_slew(out, slew_samples)        [optional]
  → float64 [M]
```

`_eval_oscillators_piecewise`:
1. `seg_idx = searchsorted(seg_breaks[1:-1], t_norm)` → which segment each sample falls in.
2. `u = (t_norm - t0) / (t1 - t0)` → local normalized time within segment.
3. `Z = R * exp(P * u)` → [M, K] complex128.
4. `Z.sum(-1)` → [M] complex128.

**No Python loops at eval time.** One batched matrix operation.

### 4.2 `evaluate_complex(t_norm)` — Full analytic signal, complex128

Same path but **does not take `.real`**. Returns the full $z(u)$ including Hilbert quadrature. Used when the curve drives a complex carrier:

$$\text{output}(t) = E_\text{complex}(u(t)) \cdot e^{j\omega t}$$

The imaginary part carries instantaneous phase coherence across the note duration.

### 4.3 `evaluate(t_ax, dur)` — Real-world time + units

```
t_ax [N] seconds → t_norm = t_ax / dur → evaluate_normalized → v_lo + v_norm * (v_hi - v_lo)
```

### 4.4 `__call__(t, gate_history, warp)` — Universal interface

When `gate_history` is `None`: delegates to `evaluate_normalized`.  
When `gate_history` is provided: first runs `TimeWarpCoordinator.warp(t_abs, gates)` → `t_norm`, then evaluates.

---

## 5. Time Warp Engine (`TimeWarpCoordinator`)

The time warp answers: **"given absolute time $t$ and the note event history, what curve position $u \in [0,1]$ should be evaluated?"**

This decouples the curve shape from wall-clock time, enabling all performance effects without modifying the curve itself.

### 5.1 Policies

#### `retrigger_mode`
| Value | Behavior |
|---|---|
| `"retrigger"` | Curve resets to $u=0$ at every note-on. Default. |
| `"legato"` | Curve continues from wherever it was at the previous note-off. Phase accumulates across gates. |
| `"free"` | Gate history ignored. $u = t_\text{abs} / \text{curve\_duration}$. |

#### `release_mode`
| Value | Behavior |
|---|---|
| `"tail"` | Curve keeps advancing at natural rate after note-off. |
| `"freeze"` | Phase locks to the note-off position until next note-on. |
| `"reset"` | Phase snaps to $u=0$ immediately after note-off. |

#### `loop_mode`
| Value | Behavior |
|---|---|
| `"none"` | Clamp $u$ at 1.0 when gate exceeds curve duration. |
| `"loop"` | Wrap: $u = \phi \bmod 1.0$. |
| `"ping_pong"` | Bounce: $u = 1 - |\phi \bmod 2 - 1|$. |

#### `curve_duration`
`None` (default) → **auto-stretch** curve to fill exactly one gate span. Every note, regardless of duration, traverses the full curve once.  
`float` → fixed nominal duration in seconds; gates may be shorter or longer.

### 5.2 Warp Computation (vectorised)

```
t_abs [M] float64  +  gates: list[GateEvent]
  → g_idx = searchsorted(t_on_arr, t_abs) - 1          # which gate owns each sample
  → t_in_gate = (t_abs - t_on_m).clamp(0)              # time elapsed in current gate
  → t_raw = phase_offset[g_idx] + t_in_gate / cdur      # raw phase (may exceed 1)
  → apply release_mode policy
  → apply loop_mode policy
  → t_norm [M] float64 ∈ [0, 1]
```

O(G) Python setup (G = number of gates), then fully vectorised over M samples.

---

## 6. Post-Eval Nonlinearities

Applied after oscillator evaluation, before returning to caller. Never invalidate the bake cache.

| `activation` | Formula |
|---|---|
| `"none"` | identity |
| `"tanh"` | $\tanh(\text{drive} \cdot x)$ (normalized back to [0,1]) |
| `"sigmoid"` | $\sigma(\text{drive} \cdot (2x-1))$ (normalized back to [0,1]) |
| `"softplus"` | $\ln(1 + e^{\text{drive} \cdot x}) / \ln(1 + e^\text{drive})$ |
| `"elu"` | ELU with $\alpha = \text{drive}$ |

**Slew:** One-pole IIR low-pass over `slew_samples` samples. Applied last, after activation.

---

## 7. The Sample Evaluation Formula (Complete)

Given system state at sample $n$:

```
state = {
    current_state: ATTACK | SUSTAIN | RELEASE | TRILL | DEAD,
    t_abs:         absolute time in seconds,
    gate_history:  list[GateEvent],          # all note_on/note_off so far
    curve:         ParametricCurve,
    warp:          TimeWarpCoordinator,
}
```

The evaluated sample value is:

$$\boxed{
v[n] = \text{clamp}\!\left(\text{slew}\!\left(\text{act}\!\left(\operatorname{Re}\!\left[\sum_{k=1}^{K} c_k^{(s(u))} \cdot e^{s_k^{(s(u))} \cdot u[n]}\right]\right)\right), \; v_\text{lo}, \; v_\text{hi}\right)
}$$

where:

$$u[n] = \text{warp}\!\left(t_\text{abs}[n],\; \text{gate\_history}\right) \in [0,1]$$

$$s(u) = \text{argmax}_s \;\bigl\{t_s \leq u < t_{s+1}\bigr\} \quad \text{(segment index)}$$

$$u_\text{local} = \frac{u - t_s}{t_{s+1} - t_s} \in [0,1] \quad \text{(local segment coordinate)}$$

No interpolation. No lookup. No Python loops at evaluation time.

---

## 8. State Machine — Envelope Behavior

The curve alone describes shape. The **state machine** determines which curve position is evaluated and how segments stitch together across performance events.

```
IDLE
 │ note_on
 ▼
ATTACK ────────────────────────────────────────────► STACCATO (early note_off)
 │ t_local ≥ t_attack                                  │ jump to RELEASE at
 ▼                                                      │ current curve value
SUSTAIN LOOP ◄──────────────────────────────────────────┘
 │ t_local %= loop_duration      (retrigger/legato/loop_mode from TimeWarpCoordinator)
 │ note_off
 ▼
RELEASE
 │ t_local ≥ t_release
 ▼
DEAD ──────────────────────────────────────────────► IDLE


TRILL (while SUSTAIN, trill_flag set):
  blend = cos²(π · trill_hz · t)
  out = curve_A(u) · blend + curve_B(u) · (1 - blend)
  Two ParametricCurves additively blended at trill rate.


LEGATO (note_on while already ATTACK or SUSTAIN):
  φ_stitch = arg(z_current) - arg(z_next_segment_start)
  Rotate all c_k of incoming segment by φ_stitch.
  No phase discontinuity. No amplitude jump unless deliberately designed.
```

---

## 9. Phase Stitching at Segment Boundaries

After baking, a forward pass applies phase continuity across smooth joins:

For each boundary $i$ → $i+1$:
- Evaluate outgoing value: $z_\text{out} = z^{(i)}(1)$
- Evaluate incoming value: $z_\text{in} = z^{(i+1)}(0)$
- If boundary is **smooth** (not `break_after`):
  - Apply: $\phi = z_\text{out} / z_\text{in}$ (full complex rotation + magnitude match)
  - Multiply all $c_k^{(i+1)}$ by $\phi$
- If boundary is a **declared discontinuity** (`break_after=True`):
  - Apply phase-only stitch: $\phi = e^{j(\arg z_\text{out} - \arg z_\text{in})}$
  - Magnitude jump is preserved (it is the user's intended discontinuity)

Propagate: corrections for segment $i+2$ are relative to already-stitched segment $i+1$.

**Note:** Phase stitching is a planned post-bake pass. The current `_ensure_baked` does not yet run it. Segments are currently C0-real only at boundaries.

---

## 10. Factory Defaults

```python
default_envelope(name)  # attack → sustain → release, 4 control points
default_chirp(name)     # frequency modulation curve, monotone shape
default_blank(name)     # single segment, flat at 0.5
```

---

## 11. What This Is Not

- **Not a polynomial spline at eval time.** The Catmull-Rom spline exists only in Stage 1 of the bake. At eval time, only the oscillator sum is used.
- **Not sample-rate dependent.** The curve is defined in normalized $[0,1]$ space and stretched by `TimeWarpCoordinator` to any gate duration at runtime. No pre-rendered buffer.
- **Not a per-note envelope generator.** It is a parametric function evaluated at the current state. Multiple simultaneous notes each maintain their own `TimeWarpCoordinator` state and query the same shared `ParametricCurve`.

---

## 12. File Structure Quick Reference

| Symbol | Location | Purpose |
|---|---|---|
| `ControlPoint` | line ~600 | Knot in the spline |
| `TimeMarker` | line ~624 | Region divider |
| `RegionEffect` | line ~667 | Per-region DSL |
| `GateEvent` | line ~727 | Note-on/off record |
| `TimeWarpCoordinator` | line ~740 | t_abs → t_norm mapping |
| `ParametricCurve` | line ~865 | Main object |
| `_curvature_segment_breaks` | line ~431 | Adaptive segment layout |
| `_analytic_signal` | line ~223 | FFT Hilbert transform |
| `_matrix_pencil_fit` | line ~249 | Damped oscillator decomposition |
| `_eval_oscillators_piecewise` | line ~558 | Vectorised eval |
| `_apply_activation` | line ~364 | Post-eval nonlinearity |
| `default_envelope` | line ~1328 | Factory |
| `default_chirp` | line ~1342 | Factory |
| `default_blank` | line ~1352 | Factory |
