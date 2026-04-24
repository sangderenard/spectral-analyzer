#!/usr/bin/env python3
"""Legacy synthesis layer for the analytic driver graph migration."""
from __future__ import annotations

from analytic_shared import *  # noqa: F401,F403
import analytic_shared as _analytic_shared
import analytic_model as _analytic_model
import analytic_routing as _analytic_routing
import analytic_score as _analytic_score

def _import_all_from(module):
    globals().update({
        k: v for k, v in vars(module).items()
        if k != "_import_all_from" and not (k.startswith('__') and k.endswith('__'))
    })

_import_all_from(_analytic_model)
_import_all_from(_analytic_routing)
_import_all_from(_analytic_score)
_import_all_from(_analytic_shared)

_GRANULAR_SEED_EDITOR = object()

_VOICE_ROLE_PRESETS: dict = {
    "air": {
        "harmonic_mode": "inharmonic",
        "env_attack_s":   0.08,
        "env_decay_s":    0.2,
        "env_sustain":    0.85,
        "env_release_s":  0.5,
        "loop_enabled":   True,
        "loop_start":     0.15,
        "loop_end":       0.75,
        "fm_index":       0.8,
    },
    "transient": {
        "harmonic_mode": "harmonic",
        "env_attack_s":   0.001,
        "env_decay_s":    0.06,
        "env_sustain":    0.0,
        "env_release_s":  0.08,
        "loop_enabled":   False,
        "chirp.chirp_type":    "exponential",
        "chirp.f_delta_start": 220.0,
        "chirp.f_delta_end":   0.0,
        "chirp.tau":           0.015,
        "fm_index":       3.0,
    },
    "body": {
        "harmonic_mode": "harmonic",
        "env_attack_s":   0.04,
        "env_decay_s":    0.12,
        "env_sustain":    0.9,
        "env_release_s":  0.8,
        "loop_enabled":   True,
        "loop_start":     0.10,
        "loop_end":       0.85,
        "fm_index":       0.25,
    },
}

def _list_audio_devices(iscapture: bool) -> list[str]:
    try:
        if not pygame.get_init():
            return []
        from pygame._sdl2.audio import get_audio_device_names
        return [str(x) for x in get_audio_device_names(bool(iscapture))]
    except Exception:
        return []


def _probe_default_sounddevice(iscapture: bool, requested_channels: int,
                               requested_rate: int = 48000) -> tuple[str, int, int]:
    try:
        import sounddevice as sd
        devsel = sd.default.device
        if isinstance(devsel, (list, tuple)):
            dev_idx = int(devsel[0] if iscapture else devsel[1])
        else:
            dev_idx = int(devsel)
        info = sd.query_devices(dev_idx)
        name = str(info.get("name", ""))
        max_ch_key = "max_input_channels" if iscapture else "max_output_channels"
        channels = int(info.get(max_ch_key, 0))
        rate = int(float(info.get("default_samplerate", requested_rate) or requested_rate))
        return name, max(0, channels), max(0, rate)
    except Exception:
        return "", 0, 0


def _probe_audio_device(name: str, iscapture: bool, requested_channels: int,
                        requested_rate: int = 48000) -> tuple[str, int, int]:
    if not str(name or "").strip():
        return _probe_default_sounddevice(iscapture, requested_channels, requested_rate)
    try:
        if not pygame.get_init():
            return "", 0, 0
        from pygame._sdl2.audio import (
            AudioDevice, AUDIO_F32, AUDIO_ALLOW_ANY_CHANGE,
        )
        names = _list_audio_devices(iscapture)
        devname = str(name or "").strip()
        if not devname:
            if not names:
                return "", 0, 0
            devname = names[0]

        def _probe_cb(_dev, mv):
            if not iscapture:
                try:
                    mv[:] = b"\x00" * len(mv)
                except Exception:
                    pass

        dev = AudioDevice(
            devicename=devname,
            iscapture=bool(iscapture),
            frequency=max(8000, int(requested_rate)),
            audioformat=AUDIO_F32,
            numchannels=max(1, int(requested_channels)),
            chunksize=512,
            allowed_changes=AUDIO_ALLOW_ANY_CHANGE,
            callback=_probe_cb,
        )
        actual = max(0, int(getattr(dev, "numchannels", 0)))
        actual_rate = max(0, int(getattr(dev, "frequency", 0)))
        actual_name = str(getattr(dev, "devicename", devname))
        dev.close()
        return actual_name, actual, actual_rate
    except Exception:
        return str(name or "").strip(), 0, 0


def _refresh_system_audio_report(sysdev: "SystemAudioDevice") -> None:
    sysdev._reported_output_devices = _list_audio_devices(False)
    sysdev._reported_input_devices = _list_audio_devices(True)
    out_name, out_ch, out_rate = _probe_audio_device(
        sysdev.output_device_name, False,
        max(1, sysdev.output_channels),
        requested_rate=max(8000, int(sysdev.export_sample_rate)),
    )
    in_req = max(1, sysdev.input_channels) if sysdev.input_channels > 0 else 2
    in_name, in_ch, in_rate = _probe_audio_device(
        sysdev.input_device_name, True, in_req,
        requested_rate=max(8000, int(sysdev.export_sample_rate)),
    )
    sysdev._reported_output_name = out_name
    sysdev._reported_input_name = in_name
    sysdev._reported_output_hw_channels = out_ch
    sysdev._reported_input_hw_channels = in_ch
    sysdev._reported_output_hw_rate = out_rate
    sysdev._reported_input_hw_rate = in_rate


def _prepare_output_bus_for_device(out_bus: np.ndarray, src_sr: int,
                                   dst_sr: int, dst_channels: int) -> np.ndarray:
    """Resample and channel-map a float bus for an SDL audio device."""
    bus = np.asarray(out_bus, dtype=np.float32)
    if bus.ndim == 1:
        bus = bus[:, None]
    if bus.shape[1] < 1:
        bus = np.zeros((len(bus), 1), dtype=np.float32)
    if dst_sr != src_sr:
        cols = []
        for ci in range(bus.shape[1]):
            cols.append(_resample_audio(bus[:, ci], src_sr, dst_sr))
        bus = np.column_stack(cols).astype(np.float32, copy=False)
    dst_ch = max(1, int(dst_channels))
    if bus.shape[1] < dst_ch:
        if bus.shape[1] == 1:
            bus = np.repeat(bus, dst_ch, axis=1)
        else:
            reps = (dst_ch + bus.shape[1] - 1) // bus.shape[1]
            bus = np.tile(bus, (1, reps))[:, :dst_ch]
    elif bus.shape[1] > dst_ch:
        bus = bus[:, :dst_ch]
    return np.clip(bus, -1.0, 1.0).astype(np.float32, copy=False)


def _apply_voice_role_preset(voice: "AnalyticVoice", role: str) -> None:
    """Apply parameter overrides for the given voice_role to *voice* in-place."""
    overrides = _VOICE_ROLE_PRESETS.get(role)
    if overrides is None:
        return
    for attr, val in overrides.items():
        if "." in attr:
            parts = attr.split(".", 1)
            sub = getattr(voice, parts[0], None)
            if sub is not None:
                setattr(sub, parts[1], val)
        else:
            if hasattr(voice, attr):
                setattr(voice, attr, val)


def _ensure_granular(voice: "AnalyticVoice") -> "object":
    """Return voice.granular, creating a default GrainPopulationSpec if needed."""
    if voice.granular is None and _HAS_GRANULAR:
        voice.granular = _GrainPopulationSpec(center_frequency_hz=voice.freq_hz)
    return voice.granular


def _resample_audio(arr: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resample a 1-D float array from src_sr to dst_sr using polyphase FIR.
    Returns the same dtype as the input.  No-ops when src_sr == dst_sr."""
    if src_sr == dst_sr or len(arr) == 0:
        return arr
    g = math.gcd(int(dst_sr), int(src_sr))
    up   = int(dst_sr) // g
    down = int(src_sr) // g
    return _scipy_resample_poly(arr, up, down).astype(arr.dtype)
def _snap_to_phase_boundary(
    t_norm:       float,
    voice:        "AnalyticVoice",
    patch:        "AnalyticPatch",
    phase_cycles: "np.ndarray | None",
) -> float:
    """Snap normalized time to the nearest complete-revolution boundary.

    Loop segments must begin and end on integer-cycle boundaries so that
    the analytic phase is phase-continuous at the splice point.  The
    ``phase_cycles`` array holds the cumulative cycle count at every
    sample (computed from the chirp-resolved instantaneous frequency,
    excluding FM modulation which is a small overlay).
    """
    if phase_cycles is None or len(phase_cycles) == 0:
        return max(0.0, min(1.0, t_norm))
    n = len(phase_cycles)
    i_raw = int(round(t_norm * (n - 1)))
    i_raw = max(0, min(n - 1, i_raw))

    # Revolution crossings: samples where floor(phase_cycles) increments.
    # Search within ±10 % of total length around the raw position.
    half = max(int(n * 0.10), 4)
    lo = max(0, i_raw - half)
    hi = min(n - 1, i_raw + half)

    window = phase_cycles[lo : hi + 1]
    floor_diff = np.floor(window[1:]) - np.floor(window[:-1])
    crossing_rel = np.where(floor_diff != 0)[0]   # offsets within [lo, hi-1]

    if len(crossing_rel) == 0:
        # No crossing in window — interpolate to nearest integer cycle.
        target = round(float(phase_cycles[i_raw]))
        target = max(1.0, float(target))
        if target >= phase_cycles[-1]:
            return 1.0
        idx = int(np.searchsorted(phase_cycles, target))
        return float(np.clip(idx / n, 0.0, 1.0))

    crossing_abs = crossing_rel + lo
    nearest_idx = int(crossing_abs[np.argmin(np.abs(crossing_abs - i_raw))])
    return float(np.clip(nearest_idx / n, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Synthesis helpers
# ---------------------------------------------------------------------------

def _get_nested_attr(obj: Any, path: str) -> Any:
    """Traverse a dot-separated attribute path, returning None on any missing step."""
    cur = obj
    for part in path.split("."):
        if cur is None:
            return None
        cur = getattr(cur, part, None)
    return cur


def _set_nested_attr(obj: Any, path: str, float_val: float, knob: KnobSpec) -> None:
    """Apply *float_val* to *obj* at the dot-path described by *knob*."""
    parts = path.split(".")
    target = obj
    for p in parts[:-1]:
        sub = getattr(target, p, None)
        if sub is None:
            if p in ("fm", "am"):
                setattr(target, p, ModRouting())
            elif p == "granular":
                _ensure_granular(target)
            sub = getattr(target, p)
        target = sub
    attr = parts[-1]
    if knob.dtype == "choice":
        idx = max(0, min(len(knob.choices) - 1, round(float_val)))
        setattr(target, attr, knob.choices[idx])
    elif knob.dtype == "int":
        setattr(target, attr, int(round(float_val)))
    elif knob.dtype == "bool":
        setattr(target, attr, bool(round(float_val)))
    else:
        setattr(target, attr, float(float_val))


def _lfo_signal(lfo: LFODefinition, t: np.ndarray) -> np.ndarray:
    ph = 2.0 * np.pi * lfo.rate_hz * t + lfo.phase_offset
    if lfo.shape == "Sine":
        return lfo.depth * np.sin(ph)
    elif lfo.shape == "Triangle":
        return lfo.depth * (2.0 * np.abs(2.0 * (ph / (2 * np.pi) % 1.0) - 1.0) - 1.0)
    elif lfo.shape == "Sawtooth":
        return lfo.depth * (2.0 * (ph / (2 * np.pi) % 1.0) - 1.0)
    else:  # Square
        return lfo.depth * np.sign(np.sin(ph))


def _compute_envelope(voice: AnalyticVoice, n: int, duration: float) -> np.ndarray:
    if voice.piecewise_env is not None:
        t_ax = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
        vals = voice.piecewise_env.curve.evaluate_normalized(t_ax).abs().to(torch.float64)
        return vals.detach().cpu().numpy().astype(np.float64, copy=False)
    knots = voice.active_knots()
    ts = np.array([k[0] * duration for k in knots], dtype=np.float64)
    vs = np.array([k[1]            for k in knots], dtype=np.float64)
    t_ax = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    return np.interp(t_ax, ts, vs)


def _compute_chirp_deviation_series(
    voice: AnalyticVoice,
    n: int,
    duration: float,
    *,
    t_axis_s: "np.ndarray | None" = None,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if t_axis_s is None:
        t_axis_s = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    else:
        t_axis_s = np.asarray(t_axis_s, dtype=np.float64)
    piecewise = getattr(voice, "piecewise_env", None)
    piecewise_chirp = getattr(piecewise, "chirp_curve", None) if piecewise is not None else None
    piecewise_delta = np.zeros(len(t_axis_s), dtype=np.float64)
    if piecewise_chirp is not None:
        if duration > 0.0:
            t_norm = np.clip(np.maximum(t_axis_s, 0.0) / duration, 0.0, 1.0)
        else:
            t_norm = np.zeros(len(t_axis_s), dtype=np.float64)
        t_tensor = torch.as_tensor(t_norm, dtype=torch.float64)
        piecewise_raw = piecewise_chirp.evaluate_normalized(t_tensor).real.clamp(0.0, 1.0)
        piecewise_delta = (
            piecewise_chirp.to_physical(piecewise_raw)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
    return piecewise_delta + _compute_knob_chirp_deviation_series(
        voice,
        n,
        duration,
        t_axis_s=t_axis_s,
    )


def _compute_knob_chirp_deviation_series(
    voice: AnalyticVoice,
    n: int,
    duration: float,
    *,
    t_axis_s: "np.ndarray | None" = None,
) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if t_axis_s is None:
        t_axis_s = np.linspace(0.0, duration, n, endpoint=False, dtype=np.float64)
    else:
        t_axis_s = np.asarray(t_axis_s, dtype=np.float64)
    chirp = getattr(voice, "chirp", None)
    if chirp is None:
        return np.zeros(len(t_axis_s), dtype=np.float64)
    ct = str(getattr(chirp, "chirp_type", "none") or "none")
    if ct == "linear":
        if len(t_axis_s) == 1:
            return np.array([float(getattr(chirp, "f_delta_start", 0.0))], dtype=np.float64)
        return np.linspace(
            float(getattr(chirp, "f_delta_start", 0.0)),
            float(getattr(chirp, "f_delta_end", 0.0)),
            len(t_axis_s),
            dtype=np.float64,
        )
    if ct == "exponential":
        tau = max(float(getattr(chirp, "tau", 0.5)), 1e-9)
        dec = np.exp(-np.maximum(t_axis_s, 0.0) / tau)
        return (
            float(getattr(chirp, "f_delta_start", 0.0)) * dec
            + float(getattr(chirp, "f_delta_end", 0.0)) * (1.0 - dec)
        ).astype(np.float64, copy=False)
    if ct == "power" and duration > 0.0:
        tau_n = (np.maximum(t_axis_s, 0.0) / duration) ** max(float(getattr(chirp, "chirp_power", 1.0)), 1e-3)
        return (
            float(getattr(chirp, "f_delta_start", 0.0)) * (1.0 - tau_n)
            + float(getattr(chirp, "f_delta_end", 0.0)) * tau_n
        ).astype(np.float64, copy=False)
    return np.zeros(len(t_axis_s), dtype=np.float64)


def _compute_chirp_frequency_series(voice: AnalyticVoice, n: int, duration: float) -> np.ndarray:
    return float(voice.freq_hz) + _compute_chirp_deviation_series(voice, n, duration)


def _sample_curve_from_series(
    values: np.ndarray,
    *,
    name: str,
    v_lo: float = 0.0,
    v_hi: float = 1.0,
    n_points: int = 24,
) -> ParametricCurve:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return _pc_default_blank(name)
    if arr.size == 1:
        arr = np.repeat(arr, 2)
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo <= 1e-12:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    curve = ParametricCurve(name=name, v_lo=v_lo, v_hi=v_hi)
    idxs = np.linspace(0, arr.size - 1, max(2, n_points), dtype=int)
    seen: set[int] = set()
    for idx in idxs.tolist():
        if idx in seen:
            continue
        seen.add(idx)
        t = 0.0 if arr.size <= 1 else float(idx) / float(arr.size - 1)
        curve.add_point(t, float(np.clip(norm[idx], 0.0, 1.0)))
    return curve


def _load_detected_piecewise_envelope(path: str) -> PiecewiseVoiceEnvelope | None:
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None
    lower = path.lower()
    try:
        if lower.endswith(".json"):
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict) and raw.get("curve"):
                pw = PiecewiseVoiceEnvelope.from_dict(raw)
                if pw is not None:
                    pw.source_path = path
                    pw.detected_envelope_path = path
                    return pw
            if isinstance(raw, dict) and "curve" in raw and "rule_tree" in raw:
                curve = ParametricCurve.from_dict(dict(raw.get("curve", {})))
                chirp_curve = ParametricCurve.from_dict(dict(raw.get("chirp", {}))) if raw.get("chirp") else _pc_default_chirp(f"{curve.name}_chirp")
                rule_tree = EnvelopeRuleTree.from_dict(dict(raw.get("rule_tree", {}))) if raw.get("rule_tree") else EnvelopeRuleTree.default()
                return PiecewiseVoiceEnvelope(
                    curve=curve,
                    chirp_curve=chirp_curve,
                    signal_curve=_pc_default_blank(f"{curve.name}_signal"),
                    rule_tree=rule_tree,
                    source_path=path,
                    detected_envelope_path=path,
                )
            curve = ParametricCurve.from_dict(raw)
            return PiecewiseVoiceEnvelope(
                curve=curve,
                chirp_curve=_pc_default_chirp(f"{curve.name}_chirp"),
                signal_curve=_pc_default_blank(f"{curve.name}_signal"),
                rule_tree=EnvelopeRuleTree.default(),
                source_path=path,
                detected_envelope_path=path,
            )
        if lower.endswith(".npz"):
            analysis_dir = os.path.dirname(os.path.dirname(path)) if os.path.basename(path).startswith("fb_envelopes") else os.path.dirname(path)
            loaded = FilterBankDecomposition.load_envelopes(analysis_dir)
            if not loaded:
                return None
            mags, phases, _hops = loaded
            mag_stack = [np.asarray(m, dtype=np.float64).reshape(-1) for m in mags if m is not None]
            if not mag_stack:
                return None
            width = max(len(m) for m in mag_stack)
            t_dst = np.linspace(0.0, 1.0, width, endpoint=True, dtype=np.float64)
            resampled_mag = []
            for mag in mag_stack:
                t_src = np.linspace(0.0, 1.0, len(mag), endpoint=True, dtype=np.float64)
                resampled_mag.append(np.interp(t_dst, t_src, mag))
            avg_mag = np.mean(np.stack(resampled_mag, axis=0), axis=0)
            amp_curve = _sample_curve_from_series(avg_mag, name=os.path.splitext(os.path.basename(path))[0], n_points=32)
            chirp_curve = _pc_default_chirp(f"{amp_curve.name}_chirp")
            phase_stack = [np.asarray(p, dtype=np.float64).reshape(-1) for p in phases if p is not None]
            if phase_stack:
                resampled_phase = []
                for phase in phase_stack:
                    t_src = np.linspace(0.0, 1.0, len(phase), endpoint=True, dtype=np.float64)
                    resampled_phase.append(np.interp(t_dst, t_src, phase))
                avg_phase = np.mean(np.stack(resampled_phase, axis=0), axis=0)
                phase_delta = np.diff(np.unwrap(avg_phase), prepend=avg_phase[:1])
                chirp_curve = _sample_curve_from_series(phase_delta, name=f"{amp_curve.name}_chirp", v_lo=-200.0, v_hi=200.0, n_points=24)
            return PiecewiseVoiceEnvelope(
                curve=amp_curve,
                chirp_curve=chirp_curve,
                signal_curve=_pc_default_blank(f"{amp_curve.name}_signal"),
                rule_tree=EnvelopeRuleTree.default(),
                source_path=path,
                detected_envelope_path=path,
            )
    except Exception:
        return None
    return None


def _detected_envelope_artifact_paths(root_dir: str) -> list[str]:
    root_dir = os.path.abspath(root_dir)
    found: list[str] = []
    seen: set[str] = set()
    for cur_root, _dirs, files in os.walk(root_dir):
        if len(found) >= 64:
            break
        if "analysis_inventory.json" in files:
            inv_path = os.path.join(cur_root, "analysis_inventory.json")
            try:
                with open(inv_path, "r", encoding="utf-8") as fh:
                    inv = AnalysisInventory.from_dict(json.load(fh))
                for ds in inv.datasets:
                    for art in ds.artifacts:
                        if art.kind != "envelopes":
                            continue
                        art_path = art.path
                        if not os.path.isabs(art_path):
                            art_path = os.path.join(cur_root, art_path)
                        art_path = os.path.abspath(art_path)
                        if art_path not in seen and os.path.isfile(art_path):
                            seen.add(art_path)
                            found.append(art_path)
            except Exception:
                pass
        for fn in files:
            if not (fn.endswith(".json") or fn.endswith(".npz")):
                continue
            if fn.startswith("fb_envelopes") or fn.endswith("_envelope.json") or "piecewise" in fn.lower():
                fp = os.path.abspath(os.path.join(cur_root, fn))
                if fp not in seen:
                    seen.add(fp)
                    found.append(fp)
    return sorted(found)


def _inst_freq_from_csig(csig: np.ndarray, sr: float) -> np.ndarray:
    """Extract instantaneous frequency (Hz) from a complex analytic signal.

    Uses the discrete phase-derivative estimator:
        f_inst[n] = angle(z[n] * conj(z[n-1])) * sr / (2π)
    The first sample is assumed equal to the second to avoid an index-zero edge.
    """
    if len(csig) < 2:
        return np.zeros(len(csig), dtype=np.float64)
    phase_diff = np.angle(csig[1:] * np.conj(csig[:-1]))
    f = phase_diff * (sr / (2.0 * math.pi))
    return np.concatenate(([f[0]], f))
def _apply_performer_transforms_to_src(
    performer_parent_map: "dict[str, list[PerformerPlacement]]",
    voice_sigs: "dict[str, np.ndarray]",
    Src: "np.ndarray",
    ki: "dict[str, int]",
    n_ext: int,
    sr: int,
) -> None:
    """Inject performer-transformed voice signals into *Src* in-place.

    For every voice that has PerformerPlacement entries (those excluded from the
    normal ``Src`` population), synthesize the ensemble contribution:

      1. Take the raw synthesised voice signal.
      2. For each PerformerPlacement:
           a. Apply geometric + humanization delay (integer-sample circular shift,
              zeroing the pre-roll region so causality is preserved).
           b. Apply phase offset (complex rotation of the analytic signal).
           c. Apply gain_db (amplitude scale).
      3. Sum performer copies and average by performer count (preserves loudness
         regardless of section size).
      4. Write the result into ``Src[ki[voice_key]]``.

    This implements the "dispatch to performers" step: the solved score (NoteSchedule)
    was handed to the most holistic available target (PerformerPlacement).  When no
    performers exist, this function is a no-op and voices reach Src via the normal
    un-transformed path.
    """
    for vk, placements in performer_parent_map.items():
        if vk not in ki or vk not in voice_sigs:
            continue
        raw = np.asarray(voice_sigs[vk], dtype=np.complex128)
        if len(raw) < n_ext:
            raw = np.pad(raw, (0, n_ext - len(raw)))
        else:
            raw = raw[:n_ext]

        acc = np.zeros(n_ext, dtype=np.complex128)
        for pf in placements:
            delay_s = (float(getattr(pf, "geometric_delay_ms", 0.0))
                       + float(getattr(pf, "humanization_ms", 0.0))) * 1e-3
            delay_n = int(round(delay_s * sr))
            sig = raw.copy()
            if delay_n > 0:
                sig = np.roll(sig, delay_n)
                sig[:delay_n] = 0.0
            phase = float(getattr(pf, "phase_offset_rad", 0.0))
            if phase:
                sig = sig * complex(math.cos(phase), math.sin(phase))
            gain_db = float(getattr(pf, "gain_db", 0.0))
            if gain_db:
                sig = sig * (10.0 ** (gain_db / 20.0))
            acc += sig

        n_pl = len(placements)
        if n_pl > 1:
            acc /= n_pl
        Src[ki[vk]] = acc
def _make_note_temp_patch(
    parent: "AnalyticPatch",
    note_voices: "list[AnalyticVoice]",
    duration_s: float,
    event_hz: float,
    note_keys: "list[str]",
    group_voice_keys: "list[str]",
    *,
    shared_modules: "list[AnalyticModule] | None" = None,
) -> "AnalyticPatch":
    """Build a lightweight per-note patch that shares read-only structures by reference.

    Only the module *state* needs isolation: ``_sm_state``, ``_sm_out_cache``,
    ``_sm_aux_state``, and ``_sm_log_text`` are the only fields that
    ``_synthesize_patch`` mutates on a module.  Everything else (routing, LFOs,
    controls, mixers, param_nodes, system_audio) is read-only during synthesis
    and can be shared safely.

    When *shared_modules* is provided those module objects are used directly
    (their mutable state slots are snapshotted/restored by the caller).
    Otherwise, fall back to a shallow copy with fresh state dicts.
    """
    tp = AnalyticPatch()
    tp.duration               = duration_s
    tp.preview_sr             = parent.preview_sr
    tp.voices                 = note_voices
    # Read-only — share by reference
    tp.lfos                   = parent.lfos
    tp.controls               = parent.controls
    tp.routing                = parent.routing
    tp.mixers                 = parent.mixers
    tp.param_nodes            = parent.param_nodes
    tp.system_audio           = parent.system_audio
    tp.tuning                 = parent.tuning
    tp.projection_mode        = parent.projection_mode
    tp.projection_rotation_hz = parent.projection_rotation_hz
    tp.normalize_output       = False
    tp.performer_phase_mode   = parent.performer_phase_mode
    tp.seq_tonic_hz           = parent.seq_tonic_hz
    tp._seq_note_hz           = float(event_hz)
    # Modules: shallow-copy list, reset mutable state slots so notes don't
    # cross-contaminate.  Scene caches live inside ``_sm_aux_state`` and are
    # persisted separately by the caller if desired.
    if shared_modules is not None:
        tp.modules = shared_modules
    else:
        _fresh: list[AnalyticModule] = []
        for m in parent.modules:
            mc = copy.copy(m)           # shallow — shares sm_params, sm_items etc.
            mc._sm_state     = {}
            mc._sm_out_cache = {}
            mc._sm_aux_state = {}
            mc._sm_log_text  = ""
            _fresh.append(mc)
        tp.modules = _fresh
    tp.parts = _copy_matching_parts_for_voice_keys(
        parent, group_voice_keys, note_keys)
    return tp
def _apply_loop_tiling(sig: np.ndarray, voice: "AnalyticVoice", n: int) -> np.ndarray:
    """Tile [loop_start, loop_end] to fill [loop_end, n) using the complex signal.

    Loop endpoints are snapped to integer phase-cycle boundaries by
    _snap_to_phase_boundary, so exp(i*phase) is continuous at every wrap.
    A short cosine-squared crossfade at each wrap boundary uses the analytic
    complex values directly to erase any sub-sample amplitude residual.
    """
    ls_n = max(0, min(n - 2, int(round(voice.loop_start * n))))
    le_n = max(ls_n + 2, min(n, int(round(voice.loop_end * n))))
    loop_len = le_n - ls_n
    tail_len = n - le_n
    if loop_len < 2 or tail_len <= 0:
        return sig
    out = sig.copy()
    body = sig[ls_n:le_n]
    reps = math.ceil(tail_len / loop_len)
    out[le_n:] = np.tile(body, reps)[:tail_len]
    # Complex crossfade at each wrap: blend tail-of-outgoing with head-of-incoming.
    # Both sides are the same body looped, so this smooths any floating-point seam.
    xfade_n = min(loop_len // 8, 32)
    if xfade_n > 1:
        t_fade   = np.linspace(0.0, math.pi / 2.0, xfade_n, dtype=np.float64)
        fade_out = np.cos(t_fade) ** 2
        fade_in  = np.sin(t_fade) ** 2
        for rep in range(reps):
            wrap = le_n + rep * loop_len
            head = wrap
            tail = wrap - xfade_n
            if tail < le_n or head + xfade_n > n:
                continue
            out[tail:wrap] = out[tail:wrap] * fade_out + out[head:head + xfade_n] * fade_in
    return out


def _synthesize_voice(
    voice: AnalyticVoice,
    patch:   AnalyticPatch,
    lfo_map: dict,
    p_map:   dict,
    t_offset: float = 0.0,    # seconds before nominal t=0 to start synthesis (pre-roll)
    n_samples: int  = 0,      # if >0, override the default sr*duration sample count
    param_overrides: "dict | None" = None,  # {attr: float64 time series} from param routing
    voice_signal_map: "dict | None" = None,  # {voice_key: complex128 array} for voice-to-voice FM/AM
    param_series: "dict | None" = None,  # {param_node_key: float64 series} for ParamNode FM/AM (H4)
    granular_rng_seed: object = _GRANULAR_SEED_EDITOR,  # sentinel→spec.editor_seed; None→random; int→fixed
    granular_seed_offset: int = 0,  # added to editor_seed when using sentinel (for animation)
) -> np.ndarray:
    sr  = patch.preview_sr
    dur = patch.duration
    n   = n_samples if n_samples > 0 else int(sr * dur)

    if voice.piecewise_env is not None:
        po = param_overrides or {}
        freq_hz = float(np.mean(po["freq_hz"][:n])) if "freq_hz" in po and len(po["freq_hz"]) >= n else float(voice.freq_hz)
        gain = float(np.mean(po["amplitude"][:n])) if "amplitude" in po and len(po["amplitude"]) >= n else float(voice.amplitude)
        gate_history = [GateEvent(t_on=0.0, t_off=float(dur), velocity=1.0)]
        env_engine = ParametricCurveEngine(
            voice.piecewise_env.curve,
            voice.piecewise_env.rule_tree,
            chirp_curve=voice.piecewise_env.chirp_curve,
            max_cache=8,
        )
        chirp_engine = ParametricCurveEngine(
            voice.piecewise_env.chirp_curve,
            voice.piecewise_env.rule_tree,
            max_cache=8,
        )
        env_fn = env_engine.interpret(gate_history, force_rebuild=True)
        chirp_fn = chirp_engine.interpret(gate_history, force_rebuild=True)
        def _base_piecewise_chirp(t_abs: "torch.Tensor | np.ndarray | Any") -> np.ndarray:
            if isinstance(t_abs, torch.Tensor):
                t_np = t_abs.detach().cpu().numpy().astype(np.float64, copy=False)
            else:
                t_np = np.asarray(t_abs, dtype=np.float64)
            return _compute_chirp_deviation_series(voice, len(t_np), float(dur), t_axis_s=t_np)
        audio, _amp_env, _chirp_env, _osc = render_piecewise_audio(
            env_fn=env_fn,
            chirp_fn=chirp_fn,
            gate_history=gate_history,
            amp_curve=voice.piecewise_env.curve,
            chirp_curve=voice.piecewise_env.chirp_curve,
            freq_hz=freq_hz,
            gain=gain,
            dur=float(dur),
            sr=int(sr),
            oversample=4,
            base_chirp_hz=_base_piecewise_chirp,
        )
        if audio is None:
            raise RuntimeError(
                f"render_piecewise_audio returned None for voice {voice.key!r}; "
                "check piecewise_env curve and chirp_curve definitions."
            )
        result = np.asarray(audio, dtype=np.complex128).reshape(-1)
        if voice.pre_delay > 0.0:
            silence_n = min(len(result), int(round(voice.pre_delay * float(sr))))
            result[:silence_n] = 0.0
        if len(result) >= n:
            return result[:n]
        out = np.zeros(n, dtype=np.complex128)
        out[:len(result)] = result
        return out

    # --- Granular emission branch ---
    if voice.emission_mode == "granular" and _HAS_GRANULAR:
        gspec = _ensure_granular(voice)
        if gspec is not None:
            import copy as _copy
            gspec_use = _copy.copy(gspec)
            # H2 fix: propagate parent voice manifold settings into the grain spec
            # so that warp, harmonic count, and brightness are not lost in granular mode.
            if voice.manifold_type in ("harmonic", "harmonic_warp"):
                # Map harmonic content to grain_manifold_mix (0=sine,1=harmonic)
                # Override only if the user hasn't explicitly deviated from 0.
                if gspec_use.grain_manifold_mix < 1e-6:
                    gspec_use = _copy.copy(gspec_use)
                    gspec_use.grain_manifold_mix = 1.0
            # Apply scalar param overrides to granular spec fields
            if param_overrides:
                import dataclasses as _dc
                gran_fields = {f.name for f in _dc.fields(gspec_use)}
                for attr, arr in param_overrides.items():
                    # "granular.foo" or bare "foo" both map to spec field "foo"
                    gran_attr = attr[len("granular."):] if attr.startswith("granular.") else attr
                    if gran_attr in gran_fields:
                        setattr(gspec_use, gran_attr, float(np.mean(arr)))
            # Gap 5: build parent phase callable so grains can phase-lock
            _p0  = float(voice.phase_origin)
            _f0  = float(gspec_use.center_frequency_hz)
            _ct  = getattr(voice.chirp, "chirp_type", "none")
            _cfd_start = float(getattr(voice.chirp, "f_delta_start", 0.0))
            _cfd_end   = float(getattr(voice.chirp, "f_delta_end",   0.0))
            _cdur      = max(float(dur), 1e-9)
            if _ct == "linear":
                _chirp_rate = (_cfd_end - _cfd_start) / _cdur
                def _phase_at(t: float, _p=_p0, _f=_f0, _cs=_cfd_start, _cr=_chirp_rate) -> float:
                    fi = _f + _cs + _cr * t
                    return _p + 2.0 * math.pi * (fi * t)
            else:
                def _phase_at(t: float, _p=_p0, _f=_f0) -> float:
                    return _p + 2.0 * math.pi * _f * t

            driver = _GranularClusterDriver(gspec_use, sr=float(sr), parent_phase_at=_phase_at,
                                            rng_seed=(int(gspec_use.editor_seed) + granular_seed_offset
                                                      if granular_rng_seed is _GRANULAR_SEED_EDITOR
                                                      else granular_rng_seed))
            raw = driver.synthesize(dur)
            env = _compute_envelope(voice, len(raw), dur)
            result = (raw * env).astype(np.complex128)
            # Apply pre_delay: silence the leading samples up to pre_delay seconds
            if voice.pre_delay > 0.0:
                silence_n = min(len(result), int(round(voice.pre_delay * float(sr))))
                result[:silence_n] = 0.0
            if len(result) >= n:
                return result[:n]
            out = np.zeros(n, dtype=np.complex128)
            out[:len(result)] = result
            return out
    # t axis: starts at t_offset (negative for pre-roll), advances at 1/sr per sample
    t = (torch.arange(n, dtype=torch.float64) / sr) + t_offset

    # Apply param_overrides to base frequency/amplitude before chirp/FM
    po = param_overrides or {}
    _po_freq = po.get("freq_hz")
    if _po_freq is not None and len(_po_freq) >= n:
        f_inst = torch.as_tensor(_po_freq[:n], dtype=torch.float64)
    else:
        f_inst = torch.full((n,), voice.freq_hz, dtype=torch.float64)
    ct = voice.chirp.chirp_type
    if ct == "linear":
        f_inst = f_inst + torch.linspace(voice.chirp.f_delta_start, voice.chirp.f_delta_end, n, dtype=torch.float64)
    elif ct == "exponential" and voice.chirp.tau > 0:
        decay   = torch.exp(-t / voice.chirp.tau)
        f_inst = f_inst + voice.chirp.f_delta_start * decay + voice.chirp.f_delta_end * (1 - decay)
    elif ct == "power" and dur > 0:
        tau_n = (t / dur) ** max(voice.chirp.chirp_power, 1e-3)
        f_inst = f_inst + voice.chirp.f_delta_start * (1.0 - tau_n) + voice.chirp.f_delta_end * tau_n

    if voice.fm and voice.fm.source_key:
        sk = voice.fm.source_key
        if sk in lfo_map:
            lfo = lfo_map[sk]
            ph = 2.0 * math.pi * lfo.rate_hz * t + lfo.phase_offset
            if lfo.shape == "Sine":
                mod = lfo.depth * torch.sin(ph)
            elif lfo.shape == "Triangle":
                mod = lfo.depth * (2.0 * torch.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0)
            elif lfo.shape == "Sawtooth":
                mod = lfo.depth * (2.0 * (ph / (2 * math.pi) % 1.0) - 1.0)
            else:  # Square
                mod = lfo.depth * torch.sign(torch.sin(ph))
        elif voice_signal_map is not None and sk in voice_signal_map:
            # Use the source voice's synthesized complex signal: extract
            # instantaneous frequency (normalised to [-0.5, 0.5] * Nyquist)
            src_csig = voice_signal_map[sk]
            src_t = torch.as_tensor(src_csig[:n], dtype=torch.complex128)
            phase_diff = torch.angle(src_t[1:] * src_t[:-1].conj())
            f_mod = phase_diff * (float(patch.preview_sr) / (2.0 * math.pi))
            f_mod = torch.cat([f_mod[:1], f_mod])
            f_mid = float(p_map[sk].freq_hz) if sk in p_map else float(torch.mean(torch.abs(f_mod)).item())
            mod = f_mod / max(f_mid, 1.0)  # normalise so depth_hz is in sensible units
        elif param_series is not None and sk in param_series:
            # H4 fix: ParamNode output as FM modulator (already a float64 series)
            ps = param_series[sk]
            if len(ps) >= n:
                mod = torch.as_tensor(ps[:n], dtype=torch.float64)
            else:
                mod = torch.nn.functional.pad(torch.as_tensor(ps[:n], dtype=torch.float64), (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = torch.cos(2.0 * math.pi * p_map[sk].freq_hz * t)
        else:
            mod = torch.zeros(n, dtype=torch.float64)
        f_inst = f_inst + voice.fm.depth_hz * mod

    phase = torch.cumsum(2.0 * math.pi * f_inst / sr, dim=0) + voice.phase_origin

    _po_amp = po.get("amplitude")
    if _po_amp is not None and len(_po_amp) >= n:
        amp = torch.as_tensor(_po_amp[:n], dtype=torch.float64)
    else:
        amp = torch.full((n,), voice.amplitude, dtype=torch.float64)
    if voice.am and voice.am.source_key:
        sk = voice.am.source_key
        if sk in lfo_map:
            lfo = lfo_map[sk]
            ph = 2.0 * math.pi * lfo.rate_hz * t + lfo.phase_offset
            if lfo.shape == "Sine":
                mod = lfo.depth * torch.sin(ph)
            elif lfo.shape == "Triangle":
                mod = lfo.depth * (2.0 * torch.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0)
            elif lfo.shape == "Sawtooth":
                mod = lfo.depth * (2.0 * (ph / (2 * math.pi) % 1.0) - 1.0)
            else:  # Square
                mod = lfo.depth * torch.sign(torch.sin(ph))
        elif voice_signal_map is not None and sk in voice_signal_map:
            # Use magnitude envelope of the source voice's synthesized signal
            src_csig = voice_signal_map[sk]
            src_t = torch.as_tensor(src_csig[:n], dtype=torch.complex128)
            mod = torch.abs(src_t).to(torch.float64)
            peak = float(torch.max(mod).item())
            if peak > 1e-12:
                mod = mod / peak
        elif param_series is not None and sk in param_series:
            # H4 fix: ParamNode output as AM modulator (already a float64 series)
            ps = param_series[sk]
            if len(ps) >= n:
                mod = torch.as_tensor(ps[:n], dtype=torch.float64)
            else:
                mod = torch.nn.functional.pad(torch.as_tensor(ps[:n], dtype=torch.float64), (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = torch.cos(2.0 * math.pi * p_map[sk].freq_hz * t)
        else:
            mod = torch.zeros(n, dtype=torch.float64)
        amp = amp * (1.0 + voice.am.depth_amp * mod)

    env = amp

    # --- manifold synthesis ---
    mt = voice.manifold_type
    if mt in ("harmonic", "harmonic_warp") and voice.harmonic_count > 1:
        sig = torch.zeros(n, dtype=torch.complex128)
        hc  = max(1, voice.harmonic_count)
        bri = voice.harmonic_brightness
        warp = voice.harmonic_warp_strength
        for k in range(1, hc + 1):
            h_ratio = k + warp * (k - 1)  # warp=0 → exact integer multiples
            h_amp   = 1.0 / (k ** bri) if bri > 0 else 1.0
            # C2 fix: k-th partial starts at k * phase_origin so all harmonics
            # are constructive at t=0 even when ratios are non-integer (warp > 0).
            h_phase = torch.cumsum(2.0 * math.pi * (f_inst * h_ratio) / sr, dim=0) + (k * voice.phase_origin) % (2.0 * math.pi)
            sig = sig + h_amp * torch.exp(1j * h_phase)
        # Normalise so amplitude 1 still means peak ~1 for a single harmonic baseline
        norm = sum(1.0 / (k ** bri) if bri > 0 else 1.0 for k in range(1, hc + 1))
        sig = sig / norm
        if voice.loop_enabled:
            sig = _apply_loop_tiling(sig, voice, n)
        out = env * sig
    else:
        sig = torch.exp(1j * phase)
        if voice.loop_enabled:
            sig = _apply_loop_tiling(sig, voice, n)
        out = env * sig

    # --- pre-delay: zero-pad the onset ---
    # The pre_delay is always relative to t=0; with a pre-roll (t_offset < 0),
    # the voice should be silent up to t = max(0, pre_delay), i.e. the first
    # abs(t_offset) samples are pre-roll so silence only applies in [0, pre_delay).
    if voice.pre_delay > 0.0:
        # Number of samples that sit before t=0 (the pre-roll prefix)
        preroll_n = max(0, int(round(abs(min(0.0, t_offset)) * sr)))
        # Silence from t=0 up to pre_delay (offset by the pre-roll prefix)
        silence_end = preroll_n + min(n, int(round(voice.pre_delay * sr)))
        out[preroll_n:silence_end] = 0.0

    return out


def _subbatch(
    fn: "Callable[[torch.Tensor], torch.Tensor]",
    items: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Run fn on items in chunks of chunk_size along dim 0, cat results along dim 0."""
    return torch.cat([fn(chunk) for chunk in items.split(chunk_size)], dim=0)


def _synthesize_voice_batched(
    voice: AnalyticVoice,
    patch:   AnalyticPatch,
    lfo_map: dict,
    p_map:   dict,
    phase_origins: "torch.Tensor",  # (B,) — one phase offset per slot
    t_offset: float = 0.0,
    n_samples: int  = 0,
    param_overrides: "dict | None" = None,
    voice_signal_map: "dict | None" = None,
    param_series: "dict | None" = None,
    granular_seed_offset: int = 0,
) -> "torch.Tensor":  # (B, n) complex128
    """Standard-path voice synthesis for B phase origins in one vectorised pass.

    Only handles the standard torch path.  Callers must verify that
    voice.piecewise_env is None and voice.emission_mode != 'granular' before
    calling; those paths do not use phase_origin and should use _synthesize_voice
    with a post-hoc rotation instead.
    """
    sr  = patch.preview_sr
    dur = patch.duration
    n   = n_samples if n_samples > 0 else int(sr * dur)
    B   = int(phase_origins.shape[0])

    t = (torch.arange(n, dtype=torch.float64) / sr) + t_offset

    po = param_overrides or {}
    _po_freq = po.get("freq_hz")
    if _po_freq is not None and len(_po_freq) >= n:
        f_inst = torch.as_tensor(_po_freq[:n], dtype=torch.float64)
    else:
        f_inst = torch.full((n,), voice.freq_hz, dtype=torch.float64)

    ct = voice.chirp.chirp_type
    if ct == "linear":
        f_inst = f_inst + torch.linspace(voice.chirp.f_delta_start, voice.chirp.f_delta_end, n, dtype=torch.float64)
    elif ct == "exponential" and voice.chirp.tau > 0:
        decay  = torch.exp(-t / voice.chirp.tau)
        f_inst = f_inst + voice.chirp.f_delta_start * decay + voice.chirp.f_delta_end * (1 - decay)
    elif ct == "power" and dur > 0:
        tau_n  = (t / dur) ** max(voice.chirp.chirp_power, 1e-3)
        f_inst = f_inst + voice.chirp.f_delta_start * (1.0 - tau_n) + voice.chirp.f_delta_end * tau_n

    if voice.fm and voice.fm.source_key:
        sk = voice.fm.source_key
        if sk in lfo_map:
            lfo = lfo_map[sk]
            ph = 2.0 * math.pi * lfo.rate_hz * t + lfo.phase_offset
            if lfo.shape == "Sine":
                mod = lfo.depth * torch.sin(ph)
            elif lfo.shape == "Triangle":
                mod = lfo.depth * (2.0 * torch.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0)
            elif lfo.shape == "Sawtooth":
                mod = lfo.depth * (2.0 * (ph / (2 * math.pi) % 1.0) - 1.0)
            else:
                mod = lfo.depth * torch.sign(torch.sin(ph))
        elif voice_signal_map is not None and sk in voice_signal_map:
            src_csig = voice_signal_map[sk]
            src_t = torch.as_tensor(src_csig[:n], dtype=torch.complex128)
            phase_diff = torch.angle(src_t[1:] * src_t[:-1].conj())
            f_mod = phase_diff * (float(patch.preview_sr) / (2.0 * math.pi))
            f_mod = torch.cat([f_mod[:1], f_mod])
            f_mid = float(p_map[sk].freq_hz) if sk in p_map else float(torch.mean(torch.abs(f_mod)).item())
            mod = f_mod / max(f_mid, 1.0)
        elif param_series is not None and sk in param_series:
            ps = param_series[sk]
            mod = torch.as_tensor(ps[:n], dtype=torch.float64) if len(ps) >= n else \
                  torch.nn.functional.pad(torch.as_tensor(ps, dtype=torch.float64), (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = torch.cos(2.0 * math.pi * p_map[sk].freq_hz * t)
        else:
            mod = torch.zeros(n, dtype=torch.float64)
        f_inst = f_inst + voice.fm.depth_hz * mod

    # Cumulative phase — common to all slots  (n,)
    phase_base = torch.cumsum(2.0 * math.pi * f_inst / sr, dim=0)

    _po_amp = po.get("amplitude")
    if _po_amp is not None and len(_po_amp) >= n:
        amp = torch.as_tensor(_po_amp[:n], dtype=torch.float64)
    else:
        amp = torch.full((n,), voice.amplitude, dtype=torch.float64)

    if voice.am and voice.am.source_key:
        sk = voice.am.source_key
        if sk in lfo_map:
            lfo = lfo_map[sk]
            ph = 2.0 * math.pi * lfo.rate_hz * t + lfo.phase_offset
            if lfo.shape == "Sine":
                mod = lfo.depth * torch.sin(ph)
            elif lfo.shape == "Triangle":
                mod = lfo.depth * (2.0 * torch.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0)
            elif lfo.shape == "Sawtooth":
                mod = lfo.depth * (2.0 * (ph / (2 * math.pi) % 1.0) - 1.0)
            else:
                mod = lfo.depth * torch.sign(torch.sin(ph))
        elif voice_signal_map is not None and sk in voice_signal_map:
            src_csig = voice_signal_map[sk]
            src_t = torch.as_tensor(src_csig[:n], dtype=torch.complex128)
            mod = torch.abs(src_t).to(torch.float64)
            peak = float(torch.max(mod).item())
            if peak > 1e-12:
                mod = mod / peak
        elif param_series is not None and sk in param_series:
            ps = param_series[sk]
            mod = torch.as_tensor(ps[:n], dtype=torch.float64) if len(ps) >= n else \
                  torch.nn.functional.pad(torch.as_tensor(ps, dtype=torch.float64), (0, n - len(ps)))
        elif sk in p_map and sk != voice.key:
            mod = torch.cos(2.0 * math.pi * p_map[sk].freq_hz * t)
        else:
            mod = torch.zeros(n, dtype=torch.float64)
        amp = amp * (1.0 + voice.am.depth_amp * mod)

    env = amp  # (n,)

    mt = voice.manifold_type
    if mt in ("harmonic", "harmonic_warp") and voice.harmonic_count > 1:
        hc   = max(1, voice.harmonic_count)
        bri  = voice.harmonic_brightness
        warp = voice.harmonic_warp_strength
        norm = sum(1.0 / (k ** bri) if bri > 0 else 1.0 for k in range(1, hc + 1))
        sig  = torch.zeros(B, n, dtype=torch.complex128)
        for k in range(1, hc + 1):
            h_ratio      = k + warp * (k - 1)
            h_amp        = 1.0 / (k ** bri) if bri > 0 else 1.0
            h_phase_base = torch.cumsum(2.0 * math.pi * (f_inst * h_ratio) / sr, dim=0)  # (n,)
            k_origins    = (k * phase_origins) % (2.0 * math.pi)                         # (B,)
            h_phase      = h_phase_base[None, :] + k_origins[:, None]                    # (B, n)
            sig = sig + h_amp * torch.exp(1j * h_phase)
        sig = sig / norm
    else:
        phase = phase_base[None, :] + phase_origins[:, None]  # (B, n)
        sig   = torch.exp(1j * phase)

    if voice.loop_enabled:
        sig_np = sig.detach().cpu().numpy()  # (B, n)
        for b in range(B):
            sig_np[b] = _apply_loop_tiling(sig_np[b], voice, n)
        sig = torch.as_tensor(sig_np, dtype=torch.complex128)

    out = env[None, :] * sig  # (B, n)

    if voice.pre_delay > 0.0:
        preroll_n   = max(0, int(round(abs(min(0.0, t_offset)) * sr)))
        silence_end = preroll_n + min(n, int(round(voice.pre_delay * sr)))
        out[:, preroll_n:silence_end] = 0.0

    return out  # (B, n) complex128


def _synthesize_lfo_csig(
    lfo: LFODefinition,
    n: int,
    sr: float,
    t_offset: float = 0.0,  # seconds before t=0 to start (pre-roll)
) -> np.ndarray:
    """Return an LFO as a complex128 analytic (Hilbert) signal.

    The imaginary part is the Hilbert transform of the real waveform so that:
    - ``extractor="magnitude"`` returns the true envelope, not |real|.
    - ``extractor="phase"`` returns the smooth analytic phase, not a degenerate
      square wave (M2 fix: previous version had imag=0 which caused
      angle(real + 0j) = 0 or π — a square wave rather than a smooth ramp).

    Sine LFOs have a natural analytic form: re + i·im = A·e^{iωt}.
    For other waveforms the analytic signal is approximated via the Hilbert
    transform of the real waveform using scipy.signal.hilbert when available,
    falling back to real-only (original behaviour) if scipy is absent.
    """
    t = (np.arange(n, dtype=np.float64) / max(sr, 1.0)) + t_offset
    real = _lfo_signal(lfo, t)  # float64 real waveform
    shape = getattr(lfo, "shape", "Sine")
    if shape == "Sine":
        # For a pure sine the analytic signal is exact: e^{i*(ωt + φ)}
        omega = 2.0 * math.pi * float(lfo.rate_hz)
        phi   = float(getattr(lfo, "phase_offset", 0.0))
        depth = float(getattr(lfo, "depth", 1.0))
        csig  = depth * np.exp(1j * (omega * t + phi))
        return csig.astype(np.complex128)
    # Non-sine shapes: attempt Hilbert lift
    try:
        from scipy.signal import hilbert as _hilbert
        analytic = _hilbert(real)
        return analytic.astype(np.complex128)
    except Exception:
        # Fallback: real-only (pre-M2 behaviour); phase extractor will be degenerate
        return real.astype(np.complex128)


def _synthesize_lfo_channel_csig(ch: dict, n: int, sr: float,
                                  t_offset: float = 0.0) -> np.ndarray:
    """Synthesize one LFO channel dict to a complex analytic signal (length n).

    ch keys: rate_hz, amplitude, phase_offset, shape, tension,
             resample (ZOH decimation factor), slew_order (1|2), slew (0-1).
    """
    rate_hz      = float(ch.get("rate_hz",      1.0))
    amplitude    = float(ch.get("amplitude",    1.0))
    phase_offset = float(ch.get("phase_offset", 0.0))
    shape        = ch.get("shape", "Sine")
    tension      = max(1e-3, float(ch.get("tension",    1.0)))
    resample     = max(1,    int(  ch.get("resample",   1)))
    slew_order   = max(1, min(2, int(ch.get("slew_order", 1))))
    slew_val     = float(np.clip(ch.get("slew", 0.0), 0.0, 0.9999))

    t  = (np.arange(n, dtype=np.float64) / max(sr, 1.0)) + t_offset
    ph = 2.0 * math.pi * rate_hz * t + phase_offset

    # Raw waveform (always real)
    if shape == "Sine":
        raw = np.sin(ph)
    elif shape == "Triangle":
        raw = 2.0 * np.abs(2.0 * (ph / (2 * math.pi) % 1.0) - 1.0) - 1.0
    elif shape == "Sawtooth":
        raw = 2.0 * (ph / (2 * math.pi) % 1.0) - 1.0
    else:  # Square
        raw = np.sign(np.sin(ph))

    # Tension shaping: sign(x)|x|^t
    shaped = np.sign(raw) * np.abs(raw) ** tension

    # Resample — zero-order hold (sample-and-hold decimation)
    if resample > 1:
        decimated = shaped[::resample]
        shaped = np.repeat(decimated, resample)[:n]
        if len(shaped) < n:
            pad = np.full(n - len(shaped), shaped[-1] if len(shaped) else 0.0)
            shaped = np.concatenate([shaped, pad])

    # Slew — exponential IIR low-pass, 1st or 2nd order
    # alpha=1 → passthrough; alpha→0 → DC (no response)
    # slew_val^2 gives a gentle nonlinear mapping so small values have effect
    if slew_val > 1e-6:
        alpha = (1.0 - slew_val) ** 2
        alpha = max(1e-6, alpha)
        try:
            from scipy.signal import lfilter as _lf
            b = [alpha]
            a = [1.0, -(1.0 - alpha)]
            shaped = _lf(b, a, shaped)
            if slew_order == 2:
                shaped = _lf(b, a, shaped)
        except Exception:
            y = float(shaped[0])
            out = np.empty(n)
            for k in range(n):
                y += alpha * (shaped[k] - y)
                out[k] = y
            if slew_order == 2:
                y2 = out[0]
                out2 = np.empty(n)
                for k in range(n):
                    y2 += alpha * (out[k] - y2)
                    out2[k] = y2
                out = out2
            shaped = out

    try:
        from scipy.signal import hilbert as _hilbert
        return (amplitude * _hilbert(shaped)).astype(np.complex128)
    except Exception:
        return (amplitude * shaped).astype(np.complex128)


def _sm_plugin_dir() -> str:
    """Return absolute path to the sm_plugins folder (sibling of this file)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "sm_plugins")


def _load_sm_plugin(plugin_name: str):
    """Import and return the plugin module for *plugin_name* (stem, no .py).

    Returns None if the plugin cannot be found or imported.
    """
    if not plugin_name:
        return None
    plugin_path = os.path.join(_sm_plugin_dir(), f"{plugin_name}.py")
    if not os.path.isfile(plugin_path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"sm_plugin_{plugin_name}", plugin_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return None
    return mod


def _sm_plugin_list() -> list:
    """Return sorted list of plugin name stems available in sm_plugins/."""
    d = _sm_plugin_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(d)
        if f.endswith(".py") and not f.startswith("_")
    )


def _sm_plugin_state_vars(plugin) -> list[str]:
    """Return normalized state/output variable names declared by a plugin."""
    vars_raw = getattr(plugin, "STATE_VARS", []) if plugin is not None else []
    vars_out: list[str] = []
    for v in vars_raw:
        s = str(v).strip()
        if s and s not in vars_out:
            vars_out.append(s)
    return vars_out


def _sm_plugin_output_vars(plugin) -> list[str]:
    """Return normalized output variable names declared by a plugin."""
    vars_raw = getattr(plugin, "OUTPUT_VARS", None) if plugin is not None else None
    if vars_raw is None:
        return _sm_plugin_state_vars(plugin)
    vars_out: list[str] = []
    for v in vars_raw:
        s = str(v).strip()
        if s and s not in vars_out:
            vars_out.append(s)
    return vars_out


def _sm_plugin_item_names(plugin, n_items: int) -> list[str]:
    """Return item names for a plugin, honoring optional naming helpers."""
    n = max(1, int(n_items))
    if plugin is None:
        return [f"m{i}" for i in range(n)]
    item_names_fn = getattr(plugin, "item_names", None)
    if callable(item_names_fn):
        try:
            names = [str(x).strip() for x in item_names_fn(n)]
            names = [x for x in names if x]
            if len(names) == n and len(set(names)) == n:
                return names
        except Exception:
            pass
    prefix = str(getattr(plugin, "ITEM_PREFIX", "m")).strip() or "m"
    return [f"{prefix}{i}" for i in range(n)]


def _sm_plugin_param_specs(plugin) -> list[dict]:
    """Return normalized plugin parameter specs for state-machine modules."""
    specs = getattr(plugin, "PARAM_SPECS", []) if plugin is not None else []
    out: list[dict] = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        name = str(spec.get("name", "")).strip()
        if not name:
            continue
        dtype = str(spec.get("dtype", "float")).strip() or "float"
        out.append({
            "name": name,
            "label": str(spec.get("label", name)),
            "dtype": dtype,
            "default": spec.get("default", 0.0 if dtype != "choice" else ""),
            "low": float(spec.get("low", 0.0)),
            "high": float(spec.get("high", 1.0)),
            "fmt": str(spec.get("fmt", ".3g")),
            "is_log": bool(spec.get("is_log", False)),
            "choices": [str(c) for c in spec.get("choices", [])],
            "group": str(spec.get("group", "Plugin")),
        })
    return out


def _sm_plugin_log_text(log_payload: object) -> str:
    """Normalize optional plugin-provided log payloads into display text."""
    if log_payload is None:
        return ""
    if isinstance(log_payload, str):
        return log_payload
    if isinstance(log_payload, (list, tuple)):
        lines = [str(x) for x in log_payload if x is not None]
        return "\n".join(lines)
    return str(log_payload)


def _sm_plugin_default_params(plugin) -> dict[str, object]:
    """Return default parameter values for a state-machine plugin."""
    return {spec["name"]: spec["default"] for spec in _sm_plugin_param_specs(plugin)}


def _sm_to_complex(arr) -> np.ndarray:
    """Convert a real float64 array (or torch Tensor) to complex128 for routing."""
    if hasattr(arr, "detach"):          # torch tensor
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr, dtype=np.float64).astype(np.complex128)


def _sm_wrap(arr, use_torch: bool):
    """Return arr as a torch Tensor if use_torch and torch is available."""
    if use_torch:
        try:
            import torch
            if isinstance(arr, np.ndarray):
                return torch.from_numpy(arr)
        except ImportError:
            pass
    return arr


def _sm_unwrap(val) -> np.ndarray:
    """Convert torch Tensor or ndarray to ndarray preserving complex dtype."""
    if hasattr(val, "detach"):
        val = val.detach().cpu().numpy()
    arr = np.asarray(val)
    if np.iscomplexobj(arr):
        return np.asarray(arr, dtype=np.complex128)
    return np.asarray(arr, dtype=np.float64)


def _place_signal(z: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray,
                  distance: np.ndarray, width: np.ndarray,
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Spatialize one complex analytic signal into a (ch1_out, ch2_out) pair.

    Everything stays complex throughout.  The per-ear rotation mixes Re and Im
    of the input before producing each output — no component is discarded early.

    azimuth  [-1, 1]  left=−1, center=0, right=+1
    elevation[-1, 1]  down=−1, up=+1  (secondary tilt)
    distance [0, 1]   amplitude fall-off
    width    [0, 1]   angular spread between ears
    """
    dist_gain = 1.0 / (1.0 + np.clip(distance, 0.0, 1.0) * 4.0)
    z_s = z * dist_gain

    # Map azimuth to a center rotation angle; width spreads the two ears apart.
    # theta=0 → phase unchanged (Re dominant), theta=π/2 → Im dominant.
    # Equal-power panning arises naturally from the projection's Re read.
    az   = np.clip(azimuth,  -1.0, 1.0)
    el   = np.clip(elevation, -1.0, 1.0)
    half_w = np.clip(width, 0.0, 1.0) * (math.pi / 4.0)

    theta_c = (az + 1.0) * (math.pi / 4.0)        # 0 at full-left, π/2 at full-right
    el_tilt  = el * (math.pi / 8.0)                # ±π/8 elevation modifier

    theta_ch1 = theta_c - half_w + el_tilt
    theta_ch2 = theta_c + half_w - el_tilt

    ch1_out = z_s * np.exp(1j * theta_ch1)
    ch2_out = z_s * np.exp(1j * theta_ch2)
    return ch1_out, ch2_out


def _apply_projection(csig: np.ndarray, mode: str, rotation_hz: float,
                      sr: float) -> tuple[np.ndarray, np.ndarray]:
    """Project a complex analytic signal to a stereo (L, R) float32 pair.

    *rotation_hz* continuously rotates the analytic projection plane:
        z_rot(t) = csig(t) * exp(i * 2π * rotation_hz * t)
    This is a pure phase-rotation (frequency shift of rotation_hz Hz) applied
    before the spatial mode decode — all modes share this single knob.

    Modes
    -----
    mono              L = R = Re(z_rot)
    stereo_quadrature L = Re(z_rot), R = Im(z_rot)   (classic analytic stereo)
    stereo_ms         L = Re+Im, R = Re-Im            (mid–side)
    lissajous         amplitude-pan from instantaneous phase angle
    """
    n = len(csig)
    if n == 0:
        z = np.zeros(1, dtype=np.float32)
        return z, z
    t = np.arange(n, dtype=np.float64) / max(sr, 1.0)
    if rotation_hz != 0.0:
        csig = csig * np.exp(1j * (2.0 * math.pi * rotation_hz * t))
    re = np.asarray(csig.real, dtype=np.float32)
    im = np.asarray(csig.imag, dtype=np.float32)
    if mode == "stereo_quadrature":
        return re, im
    elif mode == "stereo_ms":
        return (re + im).astype(np.float32), (re - im).astype(np.float32)
    elif mode == "lissajous":
        phase = np.angle(csig).astype(np.float32)           # [-π, π]
        pan   = phase / math.pi                              # [-1, 1]
        env   = np.abs(csig).astype(np.float32)
        l_gain = np.clip(1.0 - pan, 0.0, 2.0).astype(np.float32) * 0.5
        r_gain = np.clip(1.0 + pan, 0.0, 2.0).astype(np.float32) * 0.5
        return env * l_gain, env * r_gain
    else:  # "mono" and fallback
        return re, re.copy()


def _auto_mix_signal_keys(patch: "AnalyticPatch") -> list[str]:
    """Return the signal-producing node keys eligible for legacy auto-mix."""
    return (
        [v.key for v in patch.voices] +
        [l.key for l in patch.lfos] +
        [m.key for m in patch.modules
         if m.module_type not in ("interaural",)
         and not (m.module_type == "lfo" and m.lfo_channels)] +
        [m.lfo_ch_key(i)
         for m in patch.modules if m.module_type == "lfo" and m.lfo_channels
         for i in range(len(m.lfo_channels))]
    )


def _working_routing_graph_for_synthesis(patch: "AnalyticPatch") -> RoutingGraph:
    """Return a non-mutating routing graph for preview / render solves.

    Legacy patches with an entirely empty signal-routing graph still need a
    default source->mix path so they remain audible. Once the user has created
    any explicit signal edges, missing edges stay missing; deleted routes are
    not silently reintroduced during preview or rendering.
    """
    if getattr(patch, "routers", None):
        g = RoutingGraph()
        base = RoutingGraph.from_dict(patch.routing.to_dict())
        for nk in base.node_keys():
            g.add_node(
                nk,
                node_type=base.get_node_type(nk),
                source_router_types=(
                    base.get_node_source_router_types(nk)
                    if nk in base.node_source_router_types else None
                ),
                sink_router_types=(
                    base.get_node_sink_router_types(nk)
                    if nk in base.node_sink_router_types else None
                ),
            )
        g.edges.extend(copy.copy(e) for e in base.edges)
        g.param_edges.extend(copy.copy(pe) for pe in base.param_edges)
        g.meta_edges.extend(copy.copy(me) for me in getattr(base, "meta_edges", []))
        g.feedback = copy.deepcopy(base.feedback)
        g.latency_compensation = bool(getattr(base, "latency_compensation", False))
        # New-model patches: merge deployed router graphs into one synthesis
        # graph so the render path can consume the same router instances the
        # editor exposes.  Feedback policy remains graph-global for now.
        for ri, router in enumerate(patch.routers):
            rg = RoutingGraph.from_dict(router.graph.to_dict())
            if ri == 0 and not base.edges and not base.param_edges:
                g.feedback = copy.deepcopy(rg.feedback)
                g.latency_compensation = bool(getattr(rg, "latency_compensation", False))
            for nk in rg.node_keys():
                g.add_node(
                    nk,
                    node_type=rg.get_node_type(nk),
                    source_router_types=(
                        rg.get_node_source_router_types(nk)
                        if nk in rg.node_source_router_types else None
                    ),
                    sink_router_types=(
                        rg.get_node_sink_router_types(nk)
                        if nk in rg.node_sink_router_types else None
                    ),
                )
            g.edges.extend(copy.copy(e) for e in rg.edges)
            g.param_edges.extend(copy.copy(pe) for pe in rg.param_edges)
            g.meta_edges.extend(copy.copy(me) for me in getattr(rg, "meta_edges", []))
    else:
        g = RoutingGraph.from_dict(patch.routing.to_dict())
    mixer_keys = [m.key for m in patch.mixers]
    default_mix_key = mixer_keys[0] if mixer_keys else "__mix__"
    auto_signal_keys = _auto_mix_signal_keys(patch)
    if not g.edges:
        g.ensure_defaults(auto_signal_keys, mix_key=default_mix_key)
    _sanitize_system_io_edges(g, patch)
    g.prune()
    return g


def _synthesize_voice_sources(
    patch: "AnalyticPatch",
    lfo_map: dict,
    p_map: dict,
    *,
    n_samples: int,
    t_offset_map: dict[str, float] | None = None,
    voice_param_overrides: dict | None = None,
    param_series: dict | None = None,
    file_render: bool = False,
    granular_seed_offset: int = 0,
) -> dict[str, np.ndarray]:
    import torch
    device = torch.device("cpu")
    sr = float(patch.preview_sr)

    # Determine muted voice keys (solo exclusion)
    solo_key = getattr(patch, "solo_key", None)
    muted_keys: set[str] = set()
    for v in patch.voices:
        if v.muted or (solo_key is not None and v.key != solo_key):
            muted_keys.add(v.key)

    # Flat performer list from placement solver (may be empty)
    all_performers = [
        pf
        for pt in getattr(patch, "parts", [])
        for ch in getattr(pt, "chairs", [])
        for pf in getattr(ch, "performers", [])
    ]

    note_hz = float(patch._seq_note_hz) if getattr(patch, "_seq_note_hz", 0.0) > 0 else float(patch.seq_tonic_hz)

    cfg, performers, driver_list = build_driver_config(
        patch, all_performers, device, sr,
        note_hz=note_hz,
        muted_voice_keys=muted_keys,
    )

    voices = list(patch.voices)
    voice_sigs: dict[str, np.ndarray] = {}

    # Zero-fill all voices (muted or absent from driver_list)
    for v in voices:
        voice_sigs[v.key] = np.zeros(n_samples, dtype=np.complex128)

    if cfg.D == 0:
        return voice_sigs

    # Build initial state; apply t_offset (pre-roll) to t_pos per driver
    state = init_driver_state(cfg)
    if t_offset_map:
        for d, (p_idx, v_idx) in enumerate(driver_list):
            vk = voices[v_idx].key
            offset_s = float((t_offset_map or {}).get(vk, 0.0))
            if offset_s != 0.0:
                state.t_pos[d] = state.t_pos[d] + offset_s

    driver_out, voice_out, _ = multi_level_driver_step(cfg, state, n_samples, sr)

    # voice_out: (V, T) — accumulated per voice across all its drivers
    for vi, v in enumerate(voices):
        if v.key not in muted_keys:
            voice_sigs[v.key] = voice_out[vi].numpy()

    return voice_sigs


def _build_sequence_driver_config(
    patch: "AnalyticPatch",
    play_groups: "list[tuple]",
    sr: float,
    device,
) -> "tuple[DriverConfig, list, DriverState]":
    """Build a DriverConfig with one driver slot per (note_event × voice).

    Each driver encodes note start time by initialising ``t_pos = -start_time``
    so that the envelope phase-zero aligns exactly with the note's onset sample.
    ``pre_delay_samples`` is left at 0 — the negative ``t_pos`` already gates
    output via the ``sample_global >= pre_delay_samples`` mask in
    ``driver_synthesis_step``.

    Returns ``(cfg, patch.voices, initial_state)``.
    """
    import copy as _copy
    from performer_engine import CHIRP_NONE as _CHIRP_NONE_PE

    voices = list(patch.voices)
    V = len(voices)
    voice_key_to_idx = {v.key: i for i, v in enumerate(voices)}
    root_hz = float(patch.seq_tonic_hz)

    # Determine padded H and K across all voices
    H_max = 1
    K_max = 5
    for v in voices:
        h_r, _ = _build_harmonics(v)
        H_max = max(H_max, len(h_r))
        K_max = max(K_max, len(v.active_knots()))

    f0_list, amplitude_list, phase_origin_list = [], [], []
    pre_delay_samp_list, note_dur_list, active_list = [], [], []
    chirp_type_list, chirp_fs_list, chirp_fe_list = [], [], []
    chirp_tau_list, chirp_pow_list = [], []
    h_ratios_list, h_amps_list, n_harmonics_list = [], [], []
    env_t_list, env_v_list, env_n_list = [], [], []
    voice_idx_list, instrument_idx_list = [], []
    fm_src_list, fm_depth_list = [], []
    am_src_list, am_depth_list = [], []
    t_pos_init_list: list[float] = []   # initial t_pos per driver

    perf_idx = 0  # monotone instrument_idx counter across all note×voice slots

    for grp_sched, grp_voices in play_groups:
        prev_hz: "float | None" = None
        for event in grp_sched.events:
            start_time = float(event.start_time)
            for src_v in grp_voices:
                vi = voice_key_to_idx.get(src_v.key)
                if vi is None:
                    continue

                v = _copy.copy(src_v)
                v.amplitude = float(src_v.amplitude) * float(event.velocity)

                # Resolve frequency for this note event
                if getattr(event, "_exact_pitch", False):
                    resolved_hz = float(event.fundamental_hz)
                else:
                    resolved_hz = _resolve_voice_hz(src_v, patch.tuning, event.fundamental_hz)
                    _role = getattr(src_v, "seq_role", "melody")
                    if _role == "bass":
                        resolved_hz *= (2.0 ** patch.seq_bass_octave)
                    elif _role == "root":
                        resolved_hz = patch.seq_tonic_hz * (2.0 ** patch.seq_root_octave)
                    elif _role == "stab":
                        resolved_hz *= (2.0 ** patch.seq_stab_octave)

                v.freq_hz = resolved_hz
                v.pre_delay = 0.0

                # Portamento chirp between consecutive notes in the same group
                if (patch.seq_portamento_s > 0 and prev_hz is not None
                        and abs(prev_hz - resolved_hz) > 0.5):
                    v.chirp = ChirpSpec(
                        chirp_type    = "exponential",
                        f_delta_start = prev_hz - resolved_hz,
                        f_delta_end   = 0.0,
                        tau           = max(patch.seq_portamento_s, 0.001),
                    )
                else:
                    v.chirp = ChirpSpec()

                chirp = getattr(v, "chirp", None)
                f0 = _resolve_f0(v, resolved_hz, root_hz)
                h_r, h_a = _build_harmonics(v)
                n_h = len(h_r)
                env_t_secs, env_v = _build_env_knots(v, event.duration_s)
                n_k = len(env_t_secs)
                last_v = env_v[-1] if env_v else 0.0

                f0_list.append(f0)
                amplitude_list.append(float(v.amplitude))
                phase_origin_list.append(float(v.phase_origin))
                pre_delay_samp_list.append(0)          # gated by t_pos < 0
                note_dur_list.append(float(event.duration_s))
                active_list.append(True)

                chirp_code = _CHIRP_CODE.get(
                    getattr(chirp, "chirp_type", "none"), CHIRP_NONE)
                chirp_type_list.append(chirp_code)
                chirp_fs_list.append(float(getattr(chirp, "f_delta_start", 0.0)) if chirp else 0.0)
                chirp_fe_list.append(float(getattr(chirp, "f_delta_end",   0.0)) if chirp else 0.0)
                chirp_tau_list.append(float(getattr(chirp, "tau",           0.5)) if chirp else 0.5)
                chirp_pow_list.append(float(getattr(chirp, "chirp_power",   1.0)) if chirp else 1.0)

                h_ratios_list.append(h_r + [0.0] * (H_max - n_h))
                h_amps_list.append(h_a   + [0.0] * (H_max - n_h))
                n_harmonics_list.append(n_h)

                env_t_list.append(env_t_secs + [1e30]   * (K_max - n_k))
                env_v_list.append(env_v      + [last_v] * (K_max - n_k))
                env_n_list.append(n_k)

                voice_idx_list.append(vi)
                instrument_idx_list.append(perf_idx)

                fm = getattr(v, "fm", None)
                fm_vi = (voice_key_to_idx.get(fm.source_key, -1)
                         if fm and getattr(fm, "source_key", "") else -1)
                fm_src_list.append(fm_vi)
                fm_depth_list.append(float(fm.depth_hz) if fm else 0.0)

                am = getattr(v, "am", None)
                am_vi = (voice_key_to_idx.get(am.source_key, -1)
                         if am and getattr(am, "source_key", "") else -1)
                am_src_list.append(am_vi)
                am_depth_list.append(float(am.depth_amp) if am else 0.0)

                # Negative t_pos so envelope onset aligns with note start sample
                t_pos_init_list.append(-start_time)
                perf_idx += 1

            # Track the last resolved hz for portamento (first voice governs)
            if grp_voices:
                if getattr(event, "_exact_pitch", False):
                    prev_hz = float(event.fundamental_hz)
                else:
                    prev_hz = _resolve_voice_hz(
                        grp_voices[0], patch.tuning, event.fundamental_hz)

    D = len(f0_list)
    if D == 0:
        from patch_to_driver import _empty_config as _ec
        empty = _ec(V, device)
        return empty, voices, init_driver_state(empty)

    packed_env, param_env_row, packed_chirp, param_chirp_row = _build_parametric_driver_batches(
        voices,
        list(zip(instrument_idx_list, voice_idx_list)),
        device,
    )

    def _ft(lst):  return torch.tensor(lst, dtype=torch.float64, device=device)
    def _it(lst):  return torch.tensor(lst, dtype=torch.int64,   device=device)
    def _bt(lst):  return torch.tensor(lst, dtype=torch.bool,    device=device)
    def _ft2(lst): return torch.tensor(lst, dtype=torch.float64, device=device)

    cfg = DriverConfig(
        D=D, H=H_max, K=K_max, V=V, device=device,
        f0                = _ft(f0_list),
        amplitude         = _ft(amplitude_list),
        phase_origin      = _ft(phase_origin_list),
        pre_delay_samples = _it(pre_delay_samp_list),
        note_duration     = _ft(note_dur_list),
        active            = _bt(active_list),
        chirp_type        = _it(chirp_type_list),
        chirp_f_start     = _ft(chirp_fs_list),
        chirp_f_end       = _ft(chirp_fe_list),
        chirp_tau         = _ft(chirp_tau_list),
        chirp_power       = _ft(chirp_pow_list),
        h_ratios          = _ft2(h_ratios_list),
        h_amps            = _ft2(h_amps_list),
        n_harmonics       = _it(n_harmonics_list),
        env_t             = _ft2(env_t_list),
        env_v             = _ft2(env_v_list),
        env_n             = _it(env_n_list),
        voice_idx         = _it(voice_idx_list),
        instrument_idx    = _it(instrument_idx_list),
        fm_source_voice   = _it(fm_src_list),
        fm_depth_hz       = _ft(fm_depth_list),
        am_source_voice   = _it(am_src_list),
        am_depth          = _ft(am_depth_list),
        parametric_env_packed = packed_env,
        parametric_env_row    = param_env_row,
        parametric_chirp_packed = packed_chirp,
        parametric_chirp_row    = param_chirp_row,
    )

    # Seed state: t_pos = -start_time per driver so envelope zero-phase aligns
    # with note onset. Phase accumulator seeded from phase_origin as usual.
    state = init_driver_state(cfg)
    state.t_pos = torch.tensor(t_pos_init_list, dtype=torch.float64, device=device)

    return cfg, voices, state


def _synthesize_sequence_full_batch(
    patch: "AnalyticPatch",
    play_groups: "list[tuple]",
    total_n: int,
    *,
    file_render: bool = False,
    out_channels: int = 2,
    persistent_aux: "dict[str, dict] | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray, dict]":
    """Sample-wise causal step solver for full-sequence synthesis.

    Signal chain executed once per sample (all D drivers batched):

      1. Advance all D drivers by 1 sample via ``driver_synthesis_step``
         (FM cross-voice resolved by ``multi_level_driver_step`` topo sort).
         Atmospheric FM offset from the previous sample is injected here.
      2. Routing solve: x = M @ src  (M = (I − W)⁻¹ pre-computed once).
      3. Step all SM modules (room/body physics) on the 1-sample signals.
         SM state (CavityScene, stream state, body states) persists across
         samples — IR tails and resonance accumulate correctly.
      4. Extract ``feedback_pressure`` from each SM item → FM offset for
         the next sample's driver advance.  This is the sympathetic chirp
         coupling: atmosphere → driver → voice.

    Stereo projection is applied once, after the loop completes.

    Returns
    -------
    (left, right, output_channels, updated_persistent_aux)
    """
    import copy as _copy, time as _time, io, contextlib

    sr = float(patch.preview_sr)
    if persistent_aux is None:
        persistent_aux = {}

    device = torch.device("cpu")

    # ── 1. Build full-sequence DriverConfig ────────────────────────────────────
    cfg, voices, drv_state = _build_sequence_driver_config(
        patch, play_groups, sr, device)

    if cfg.D == 0:
        z = np.zeros(total_n)
        return z, z, np.zeros((total_n, out_channels)), dict(persistent_aux)

    f0_base = cfg.f0.clone()   # save nominal per-driver frequencies

    # ── 2. Compile torch routing solve once ───────────────────────────────────
    g = _working_routing_graph_for_synthesis(patch)
    node_keys = _patch_node_keys(patch)
    for _k in g.node_keys():
        if _k not in node_keys:
            node_keys.append(_k)
    N = len(node_keys)
    ki = {k: i for i, k in enumerate(node_keys)}
    global_decay = float(getattr(patch.routing, "global_decay", 1.0))
    compiled_router = CompiledRouter(
        node_keys,
        list(g.edges),
        sr,
        device,
        global_decay=global_decay,
        batch_size=1,
        max_iterations=max(1, int(getattr(g.feedback, "max_iterations", 64))),
        convergence_eps=1e-10,
        infinity_threshold=1e6,
    )

    # ── 3. Pre-compute LFO signals (deterministic, no feedback) ───────────────
    lfo_sigs: dict[str, np.ndarray] = {}
    for lfo in patch.lfos:
        lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, total_n, sr)

    # ── 4. Initialise SM modules, seed room/body caches ───────────────────────
    sm_mods: list = []
    sm_plugs: list = []
    for m in patch.modules:
        if m.module_type != "state_machine" or m.muted:
            sm_mods.append(None)
            sm_plugs.append(None)
            continue
        mc = _copy.copy(m)
        mc._sm_state     = {}
        mc._sm_out_cache = {}   # {out_key: 1-element complex128 array}
        mc._sm_log_text  = ""
        mc._sm_aux_state = dict(persistent_aux.get(m.key, {}))
        sm_mods.append(mc)
        plug = _load_sm_plugin(mc.sm_plugin)
        sm_plugs.append(plug if (plug and callable(getattr(plug, "step", None))) else None)

    # Per-SM-module: list of (src_key, slot) tuples.
    # slot = edge.item_slot if set, else src_key.  The SM receives each
    # signal under `slot` so driver/instrument identity is preserved.
    sm_input_src: list[list[tuple]] = []
    for mc in sm_mods:
        if mc is None:
            sm_input_src.append([])
            continue
        sm_input_src.append([
            (e.src_key, e.item_slot if e.item_slot else e.src_key)
            for e in g.edges
            if e.dst_key == mc.key and e.src_key in ki
        ])

    # Voice index in DriverConfig's V dimension → node key mapping
    voice_list = voices   # list of VoiceDefinition objects, indexed by v_idx

    # ── 5. Sample-wise step loop ───────────────────────────────────────────────
    X_full = np.zeros((N, total_n), dtype=np.complex128)
    # Atmospheric FM offset per driver (Hz); fed back from SM feedback_pressure
    atm_fm_hz = torch.zeros(cfg.D, dtype=torch.float64, device=device)

    t0_wall = _time.monotonic()
    log_interval = max(1, total_n // 20)

    for t in range(total_n):
        # 5a. Inject atmospheric FM and advance all D drivers by 1 sample.
        #     multi_level_driver_step resolves FM/AM cross-voice topo ordering.
        cfg.f0 = f0_base + atm_fm_hz
        _d_out, v_out_t, drv_state = multi_level_driver_step(cfg, drv_state, 1, sr)
        # v_out_t: (V, 1) complex128 — per-voice signal for this sample

        # 5b. Populate Src vector: voices + LFOs + SM cached outputs
        Src_vec = np.zeros(N, dtype=np.complex128)
        for vi, v in enumerate(voice_list):
            if v.key in ki:
                Src_vec[ki[v.key]] = v_out_t[vi, 0].item()
        for lkey, lsig in lfo_sigs.items():
            if lkey in ki:
                Src_vec[ki[lkey]] = lsig[t]
        for mc in sm_mods:
            if mc is None:
                continue
            for ok, cached in mc._sm_out_cache.items():
                if ok in ki and len(cached) > 0:
                    Src_vec[ki[ok]] = cached[0]

        # 5c. Routing solve — single matrix-vector multiply (M pre-computed)
        X_t = compiled_router.step(
            torch.as_tensor(Src_vec, dtype=torch.complex128, device=device)
        )
        X_vec = np.asarray(X_t.detach().cpu().numpy(), dtype=np.complex128)
        X_full[:, t] = X_vec

        # 5d. Step each SM module on this 1-sample X; update atmospheric FM
        atm_fm_hz.zero_()
        for idx, (mc, plug) in enumerate(zip(sm_mods, sm_plugs)):
            if mc is None or plug is None:
                continue
            sm_inputs = {
                slot: _sm_wrap(
                    np.array([X_vec[ki[src_key]]], dtype=np.complex128),
                    mc.sm_use_torch)
                for src_key, slot in sm_input_src[idx]
            }
            _sm_stdout = io.StringIO()
            _sm_stderr = io.StringIO()
            try:
                with contextlib.redirect_stdout(_sm_stdout), \
                     contextlib.redirect_stderr(_sm_stderr):
                    _sm_traj = plug.step(
                        sm_inputs, dict(mc._sm_state), 1.0 / sr,
                        n_items=mc.sm_n_items,
                        use_torch=mc.sm_use_torch,
                        params=dict(mc.sm_params),
                        plugin_state=dict(mc._sm_aux_state or {}),
                    )
            except Exception:
                continue

            if not (isinstance(_sm_traj, dict) and "outputs" in _sm_traj):
                continue

            sm_out     = _sm_traj.get("outputs", {})
            mc._sm_state     = _sm_traj.get("state", {})
            mc._sm_aux_state = dict(_sm_traj.get("plugin_state", {}) or {})

            # Cache 1-sample outputs for the next iteration's Src population
            new_cache: dict = {}
            for item in mc.sm_items:
                for var in mc.sm_vars:
                    ok = mc.sm_out_key(item, var)
                    traj_val = sm_out.get(item, {}).get(var)
                    if traj_val is not None:
                        arr = np.atleast_1d(
                            np.asarray(_sm_unwrap(traj_val), dtype=np.complex128))
                        new_cache[ok] = arr[:1]
            mc._sm_out_cache = new_cache

            # Extract feedback_pressure per item → FM offset for each driver.
            # Instantaneous frequency from complex: ω = angle(z) * sr / 2π.
            # instrument_idx maps each driver slot to the SM item that owns it.
            for i, item in enumerate(mc.sm_items):
                fp = sm_out.get(item, {}).get("feedback_pressure")
                if fp is None:
                    continue
                fp_arr = np.atleast_1d(
                    np.asarray(_sm_unwrap(fp), dtype=np.complex128))
                if len(fp_arr) == 0:
                    continue
                fb_hz = float(np.angle(fp_arr[0])) * (sr / (2.0 * math.pi))
                mask = cfg.instrument_idx == i
                if mask.any():
                    atm_fm_hz[mask] += fb_hz

        if (t + 1) % log_interval == 0 or t == total_n - 1:
            elapsed = _time.monotonic() - t0_wall
            rate = (t + 1) / max(elapsed, 1e-6)
            eta  = (total_n - t - 1) / max(rate, 1e-6)
            print(f"  Step solver: {t+1}/{total_n} samples "
                  f"({100*(t+1)/total_n:.0f}%)  "
                  f"elapsed {elapsed:.1f}s  ETA {eta:.1f}s")

    # ── 6. Stereo projection — once, after the loop ────────────────────────────
    active_mixer_keys = [m.key for m in patch.mixers if m.projection_active]
    sys_out_keys = [k for k in _system_output_keys(patch) if k in ki]
    has_system_routing = bool(sys_out_keys) and any(
        e.dst_key in set(sys_out_keys) for e in g.edges
    )

    if has_system_routing:
        out_bus = np.column_stack([
            X_full[ki[k], :total_n].real for k in sys_out_keys
        ])
        if patch.normalize_output and out_bus.size:
            peak = float(np.max(np.abs(out_bus)))
            if peak > 1e-9:
                out_bus /= peak
        left  = out_bus[:, 0].astype(np.float64)
        right = (out_bus[:, 1] if out_bus.shape[1] > 1 else out_bus[:, 0]).astype(np.float64)
    elif active_mixer_keys:
        mix = sum(X_full[ki[mk]] for mk in active_mixer_keys if mk in ki)
        if patch.normalize_output:
            peak = float(np.max(np.abs(mix)))
            if peak > 1e-9:
                mix /= peak
        left, right = _apply_projection(mix, patch.projection_mode,
                                        patch.projection_rotation_hz, sr)
        out_bus = np.column_stack([left, right])
    else:
        out_bus = np.zeros((total_n, max(2, out_channels)))
        left  = out_bus[:, 0]
        right = out_bus[:, 1]

    if out_bus.ndim == 2 and out_bus.shape[1] < out_channels:
        out_bus = np.pad(out_bus, ((0, 0), (0, out_channels - out_bus.shape[1])))
    elif out_bus.ndim == 1:
        out_bus = np.column_stack([left, right])

    # Harvest updated SM aux state (cavity/room scene caches)
    updated_aux: dict = dict(persistent_aux)
    for mc in sm_mods:
        if mc is not None and mc.key and getattr(mc, "_sm_aux_state", None):
            updated_aux[mc.key] = dict(mc._sm_aux_state)

    elapsed_total = _time.monotonic() - t0_wall
    print(f"  Step-solver complete: {total_n} samples, "
          f"{cfg.D} drivers in {elapsed_total:.1f}s")

    return (
        np.asarray(left),
        np.asarray(right),
        out_bus.astype(np.float64),
        updated_aux,
    )


def _synthesize_patch(
    patch: AnalyticPatch,
    *,
    granular_seed_offset: int = 0,
    file_render: bool = False,
    _return_mixer_sigs: bool = False,
    _return_sidecar: bool = False,
    _return_output_channels: bool = False,
    _prebuilt_voice_sigs: "dict[str, np.ndarray] | None" = None,
) -> tuple:
    """Return (left, right) float32 stereo after routing + projection.

    Routing model (N nodes = voices + LFOs + __mix__):
        x = src + W @ x  →  x = (I − W)⁻¹ · src          (instantaneous)
        x(t) = src(t) + decay · W @ x(t − d)               (delayed)

    When no routing edges exist, falls back to direct voice sum (backward-compat).
    The '__mix__' node output is the stereo output before projection.

    *_prebuilt_voice_sigs*: when provided, skip ``_synthesize_voice_sources`` and
    use these pre-assembled full-timeline buffers directly.  This is the entry
    point for the batched-sequence pipeline where all notes have been pre-rendered
    and accumulated into per-voice complex128 arrays before the routing solve.

    When *_return_sidecar=True* the return value gains a trailing SidecarBus:
        (left, right)                           default
        (left, right, output_channels)          _return_output_channels=True
        (left, right, sidecar)                  _return_sidecar=True
        (left, right, mixer_outs)               _return_mixer_sigs=True
        (left, right, mixer_outs, sidecar)      mixer + sidecar
    """
    lfo_map = {l.key: l for l in patch.lfos}
    p_map   = {p.key: p for p in patch.voices}
    sr      = float(patch.preview_sr)
    n       = int(patch.preview_sr * patch.duration)

    # 1. Synthesize independent sources for each node.
    # Build param overrides from the PREVIOUS frame's cached param series so that
    # voices can be synthesised with modulation in a single solve pass.
    # One-buffer latency is perceptually invisible at audio buffer sizes.
    _cached_ps = patch._param_series_cache   # {} on first frame
    _iau_mod_keys_set: set = {m.key for m in patch.modules if m.module_type == "interaural"}
    _iau_ch_to_mod_map: dict = {}
    for _m in patch.modules:
        if _m.module_type == "interaural":
            _iau_ch_to_mod_map[_m.ch1_key()] = _m
            _iau_ch_to_mod_map[_m.ch2_key()] = _m
            _iau_ch_to_mod_map[_m.key]       = _m
    _voice_param_ov: dict = {}
    _module_param_ov: dict = {}
    _routing_param_ov: dict = {}
    _mixer_keys = {m.key for m in patch.mixers}
    if patch.param_nodes and _cached_ps:
        for _pn in patch.param_nodes:
            if _pn.key not in _cached_ps:
                continue
            for _tgt in _pn.targets:
                _vk = _tgt.get("voice_key", "")
                _at = _tgt.get("attr", "")
                if not (_vk and _at):
                    continue
                if _vk in _iau_ch_to_mod_map:
                    _mm = _iau_ch_to_mod_map[_vk]
                    _module_param_ov.setdefault(_mm.key, {})[_at] = _cached_ps[_pn.key]
                elif _vk in _mixer_keys:
                    _routing_param_ov[_at] = _cached_ps[_pn.key]
                elif _vk not in _iau_mod_keys_set:
                    _voice_param_ov.setdefault(_vk, {})[_at] = _cached_ps[_pn.key]

    voice_sigs = (
        _prebuilt_voice_sigs
        if _prebuilt_voice_sigs is not None
        else _synthesize_voice_sources(
            patch,
            lfo_map,
            p_map,
            n_samples=n,
            voice_param_overrides=_voice_param_ov,
            param_series=_cached_ps or None,
            file_render=file_render,
            granular_seed_offset=granular_seed_offset,
        )
    )

    lfo_sigs: dict[str, np.ndarray] = {}
    for lfo in patch.lfos:
        lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, n, sr)

    sys_in_sigs: dict[str, np.ndarray] = {}
    _sys_inputs = _system_input_keys(patch)
    _sys_bufs = list(getattr(patch.system_audio, "_input_buffers", []))
    for _i, _k in enumerate(_sys_inputs):
        if _i < len(_sys_bufs):
            _buf = np.asarray(_sys_bufs[_i], dtype=np.float32).reshape(-1)
            if len(_buf) >= n:
                _arr = _buf[-n:]
            elif len(_buf) > 0:
                _arr = np.pad(_buf, (0, n - len(_buf)), mode="constant")
            else:
                _arr = np.zeros(n, dtype=np.float32)
        else:
            _arr = np.zeros(n, dtype=np.float32)
        sys_in_sigs[_k] = _arr.astype(np.complex128)

    # Module signals — LFO type reuses the LFO synthesiser; passthrough starts
    # at zero (its signal comes entirely from routing edges).
    # Multi-channel LFO: main key gets zero; each channel key gets its own signal.
    module_sigs: dict[str, np.ndarray] = {}
    for mod in patch.modules:
        if mod.muted:
            module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)
            if mod.module_type == "lfo" and mod.lfo_channels:
                for i in range(len(mod.lfo_channels)):
                    module_sigs[mod.lfo_ch_key(i)] = np.zeros(n, dtype=np.complex128)
        elif mod.module_type == "lfo":
            if mod.lfo_channels:
                module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)
                for i, ch in enumerate(mod.lfo_channels):
                    module_sigs[mod.lfo_ch_key(i)] = _synthesize_lfo_channel_csig(ch, n, sr)
            else:
                _tmp = LFODefinition(key=mod.key, label=mod.label,
                                     rate_hz=mod.rate_hz, shape=mod.shape,
                                     phase_offset=mod.phase_offset, depth=mod.depth)
                module_sigs[mod.key] = _synthesize_lfo_csig(_tmp, n, sr)
        elif mod.module_type == "state_machine":
            module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)
            # State machine output nodes: seed Src from last-frame cache so that
            # downstream routing sees these values in the single solve pass.
            for ok in mod.sm_out_keys():
                cached_sig = mod._sm_out_cache.get(ok)
                if cached_sig is not None:
                    Tpad = n
                    c = cached_sig[:Tpad] if len(cached_sig) >= Tpad else np.pad(
                        cached_sig, (0, Tpad - len(cached_sig)), mode="edge")
                    module_sigs[ok] = c.astype(np.complex128)
                else:
                    module_sigs[ok] = np.zeros(n, dtype=np.complex128)
        else:  # passthrough / pitch_quantizer — zero independent source
            module_sigs[mod.key] = np.zeros(n, dtype=np.complex128)

    # Control slider signals — constant DC at the slider's current scaled value.
    # These are routing nodes so any ParamNode wired to them is driven by the
    # slider position.  They are NOT auto-routed to the mix bus.
    ctrl_sigs: dict[str, np.ndarray] = {}
    for cs in patch.controls:
        for sl in cs.sliders:
            ctrl_sigs[sl.key] = np.full(n, sl.scaled_value(), dtype=np.complex128)

    # ---------------------------------------------------------------------------
    # SidecarBus: populate initial per-source channels from raw synthesis outputs.
    # Channels are registered from native-dtype arrays — no forced dtype change.
    # The bus is cheap to build (no extra synthesis); it references already-computed
    # arrays. The "routed" channel is added after the routing solve below.
    # ---------------------------------------------------------------------------
    sidecar = SidecarBus()
    # Voices: envelope (float64), amplitude |z|, phase arg(z)
    _voice_map = {v.key: v for v in patch.voices}
    for vkey, vsig in voice_sigs.items():
        voice = _voice_map.get(vkey)
        if voice is not None:
            env_curve = _compute_envelope(voice, len(vsig), patch.duration)
            sidecar.put(vkey, "envelope", env_curve)
        sidecar.put(vkey, "amplitude", np.abs(vsig))
        sidecar.put(vkey, "phase",     np.angle(vsig))
    # LFOs: amplitude and phase from the analytic signal
    for lkey, lsig in lfo_sigs.items():
        sidecar.put(lkey, "amplitude", np.abs(lsig))
        sidecar.put(lkey, "phase",     np.angle(lsig))
    for skey, ssig in sys_in_sigs.items():
        sidecar.put(skey, "amplitude", np.abs(ssig))
        sidecar.put(skey, "phase",     np.angle(ssig))
    # Modules: amplitude and phase
    for mkey, msig in module_sigs.items():
        sidecar.put(mkey, "amplitude", np.abs(msig))
        sidecar.put(mkey, "phase",     np.angle(msig))
    # Control sliders: scalar value broadcast to an array
    for cs in patch.controls:
        for sl in cs.sliders:
            sidecar.put(sl.key, "value",
                        np.full(n, sl.scaled_value(), dtype=np.float64))

    g = _working_routing_graph_for_synthesis(patch)
    if _routing_param_ov:
        for _at, _arr in _routing_param_ov.items():
            _parsed = _parse_routing_edge_attr(_at)
            if _parsed is None:
                continue
            _kind, _src, _dst = _parsed
            _scalar = float(np.mean(np.asarray(_arr, dtype=np.float64)))
            if _kind == "mix":
                g.set_weight(_src, _dst, _scalar)
            elif _kind == "angle":
                g.set_angle_rad(_src, _dst, _scalar)
            elif _kind == "delay":
                g.set_delay_s(_src, _dst, _scalar)
        g.prune()

    # 2. Build N-node routing system (voices + lfos + modules + ctrl-sliders + mixers + param_nodes)
    node_keys = _patch_node_keys(patch)
    N  = len(node_keys)
    ki = {k: i for i, k in enumerate(node_keys)}

    # --- Optional latency compensation (pre-roll synthesis) ---
    # When enabled, each source is pre-synthesised starting before t=0 by the
    # amount needed for its signal to arrive at every destination on time.
    # The tone generator evaluates at negative t exactly — no approximation.
    lead_map: dict[str, int] = {}
    max_lead: int = 0
    if g.latency_compensation:
        lead_map = compute_latency_compensation(g.edges, node_keys, sr)
        max_lead = max(lead_map.values(), default=0)

    n_ext = n + max_lead   # extended buffer length including pre-roll

    # Virtual patch nodes:
    # __patch_tonic__ = tonal center (scale root) — always seq_tonic_hz
    # __patch_seq__   = current note Hz — _seq_note_hz if set, else seq_tonic_hz
    # Both carry real=Hz, imag=0.  They are note-domain sources, not audio.
    _patch_tonic_val = complex(patch.seq_tonic_hz, 0.0)
    _patch_seq_hz    = patch._seq_note_hz if patch._seq_note_hz > 0 else patch.seq_tonic_hz
    _patch_seq_val   = complex(_patch_seq_hz, 0.0)
    patch_virtual_sigs: dict = {
        "__patch_tonic__": np.full(n_ext, _patch_tonic_val, dtype=np.complex128),
        "__patch_seq__":   np.full(n_ext, _patch_seq_val, dtype=np.complex128),
    }
    for _vk, _vsig in patch_virtual_sigs.items():
        sidecar.put(_vk, "value", _vsig.real.astype(np.float64))

    # Re-synthesise with pre-roll if needed
    if max_lead > 0:
        voice_sigs = _synthesize_voice_sources(
            patch,
            lfo_map,
            p_map,
            n_samples=n_ext,
            t_offset_map={k: -(lead_map.get(k, 0) / sr) for k in p_map.keys()},
            voice_param_overrides=_voice_param_ov,
            param_series=_cached_ps or None,
            file_render=file_render,
            granular_seed_offset=granular_seed_offset,
        )
        lfo_sigs = {}
        for lfo in patch.lfos:
            lead_n = lead_map.get(lfo.key, 0)
            lead_s = lead_n / sr
            lfo_sigs[lfo.key] = _synthesize_lfo_csig(lfo, n_ext, sr, t_offset=-lead_s)

        # Re-synthesise module signals with pre-roll
        module_sigs = {}
        for mod in patch.modules:
            if mod.muted:
                module_sigs[mod.key] = np.zeros(n_ext, dtype=np.complex128)
                if mod.module_type == "lfo" and mod.lfo_channels:
                    for i in range(len(mod.lfo_channels)):
                        module_sigs[mod.lfo_ch_key(i)] = np.zeros(n_ext, dtype=np.complex128)
            elif mod.module_type == "lfo":
                if mod.lfo_channels:
                    module_sigs[mod.key] = np.zeros(n_ext, dtype=np.complex128)
                    for i, ch in enumerate(mod.lfo_channels):
                        lead_s = lead_map.get(mod.lfo_ch_key(i), 0) / sr
                        module_sigs[mod.lfo_ch_key(i)] = _synthesize_lfo_channel_csig(
                            ch, n_ext, sr, t_offset=-lead_s)
                else:
                    lead_n = lead_map.get(mod.key, 0)
                    lead_s = lead_n / sr
                    _tmp = LFODefinition(key=mod.key, label=mod.label,
                                         rate_hz=mod.rate_hz, shape=mod.shape,
                                         phase_offset=mod.phase_offset, depth=mod.depth)
                    module_sigs[mod.key] = _synthesize_lfo_csig(_tmp, n_ext, sr, t_offset=-lead_s)
            else:
                module_sigs[mod.key] = np.zeros(n_ext, dtype=np.complex128)

        # DC control sliders don't change with pre-roll
        ctrl_sigs = {}
        for cs in patch.controls:
            for sl in cs.sliders:
                ctrl_sigs[sl.key] = np.full(n_ext, sl.scaled_value(), dtype=np.complex128)
        sys_in_sigs = {}
        for _i, _k in enumerate(_sys_inputs):
            if _i < len(_sys_bufs):
                _buf = np.asarray(_sys_bufs[_i], dtype=np.float32).reshape(-1)
                if len(_buf) >= n_ext:
                    _arr = _buf[-n_ext:]
                elif len(_buf) > 0:
                    _arr = np.pad(_buf, (0, n_ext - len(_buf)), mode="constant")
                else:
                    _arr = np.zeros(n_ext, dtype=np.float32)
            else:
                _arr = np.zeros(n_ext, dtype=np.float32)
            sys_in_sigs[_k] = _arr.astype(np.complex128)

    # Source matrix: Src[i, :] = node i's independent signal.
    # Mixer nodes have NO independent source -- driven purely by routing edges.
    # pitch_quantizer modules start at zero and are post-processed after solve.
    Src = np.zeros((N, n_ext), dtype=np.complex128)
    for key, sig in voice_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in lfo_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in module_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in ctrl_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in sys_in_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]
    for key, sig in patch_virtual_sigs.items():
        if key in ki:
            Src[ki[key]] = sig[:n_ext]

    # Build node_transforms for nonlinear modules (e.g. pitch_quantizer).
    # Each transform is a callable(complex128 array) -> complex128 array that
    # is applied *inside* the routing solve (two-pass) so all downstream nodes
    # see the correctly transformed signal during the same solve.
    _node_transforms: dict = {}
    _dt_q = 1.0 / float(patch.preview_sr) if patch.preview_sr > 0 else 0.0
    for _qmod in patch.modules:
        if _qmod.module_type != "pitch_quantizer" or _qmod.muted:
            continue
        if _qmod.key not in ki:
            continue
        _qhandle = make_quantizer_handle(_qmod, patch.tuning)
        def _make_qtransform(_h=_qhandle, _dt=_dt_q):
            def _qtransform(row: "np.ndarray") -> "np.ndarray":
                # row is complex128: real = Hz input, imag = quadrature (preserved).
                hz_in = np.asarray(row.real, dtype=np.float64)
                hz_out = _h.process_series(hz_in, hz_in, domain="hz", dt=_dt)
                # Only the real (Hz) component is quantized; imaginary stays intact.
                return hz_out.astype(np.complex128) + 1j * row.imag
            return _qtransform
        _node_transforms[_qmod.key] = _make_qtransform()

    # Build coupled_transforms for interaural modules.
    # Each module couples its ch1+ch2 rows: both accumulate routed inputs
    # inside the solve, then _place_signal maps them to spatialized outputs.
    # In mono (nothing wired to ch2), X[ch2_idx] is naturally zero from the
    # solve — no copy/doubling needed.
    def _build_iau_coupled(param_overrides: dict) -> dict:
        """Return coupled_transforms dict for all active interaural modules.

        param_overrides: {mod.key: {attr: float64 series}} for time-varying params.
        """
        _ct: dict = {}
        for _im in patch.modules:
            if _im.module_type != "interaural" or _im.muted:
                continue
            _c1, _c2 = _im.ch1_key(), _im.ch2_key()
            if _c1 not in ki or _c2 not in ki:
                continue
            _ov = param_overrides.get(_im.key, {})
            def _make_iau_fn(_m=_im, _ov=_ov, _c1=_c1, _c2=_c2):
                def _bc(v_scalar, series, T):
                    if series is not None:
                        a = np.asarray(series, dtype=np.float64)
                        return np.pad(a, (0, max(0, T - len(a))), mode="edge")[:T]
                    return np.full(T, v_scalar, dtype=np.float64)
                def _iau_fn(rows):
                    inp1 = rows[_c1]
                    inp2 = rows[_c2]
                    T = inp1.shape[0]
                    az1  = _bc(_m.iau_azimuth,    _ov.get("iau_azimuth"),    T)
                    el1  = _bc(_m.iau_elevation,  _ov.get("iau_elevation"),  T)
                    dst1 = _bc(_m.iau_distance,   _ov.get("iau_distance"),   T)
                    wid1 = _bc(_m.iau_width,      _ov.get("iau_width"),      T)
                    az2  = _bc(_m.iau_azimuth_ch2,   _ov.get("iau_azimuth_ch2"),   T)
                    el2  = _bc(_m.iau_elevation_ch2, _ov.get("iau_elevation_ch2"), T)
                    dst2 = _bc(_m.iau_distance_ch2,  _ov.get("iau_distance_ch2"),  T)
                    wid2 = _bc(_m.iau_width_ch2,     _ov.get("iau_width_ch2"),     T)
                    out1_a, out2_a = _place_signal(inp1, az1, el1, dst1, wid1)
                    out1_b, out2_b = _place_signal(inp2, az2, el2, dst2, wid2)
                    return {_c1: out1_a + out1_b, _c2: out2_a + out2_b}
                return _iau_fn
            _ct[(_c1, _c2)] = _make_iau_fn()
        return _ct

    _coupled_transforms = _build_iau_coupled(_module_param_ov)

    # 3. Solve the routing system — interaural spatial transforms run inside
    # the convergence loop, so feedback loops through spatial modules are correct.
    fb = g.feedback
    global_decay = max(0.0, 1.0 - float(fb.decay)) if fb.enabled else 1.0
    X, _ = solve_routing_with_ringdown(
        Src, g.edges, node_keys, sr, global_decay, fb,
        node_transforms=_node_transforms or None,
        coupled_transforms=_coupled_transforms or None,
    )


    # 3a-ii. LFO channel scale post-solve.
    # Each multi-channel LFO channel's Src already holds amplitude*lfo_waveform.
    # After the solve: X[ch_key] = lfo_src + sum(routed_inputs).
    # We apply per-channel scale to the routed portion only:
    #   new = lfo_src + scale * (X[ch_key] - lfo_src)
    # Then delta-update downstream nodes.
    for _lmod in patch.modules:
        if _lmod.module_type != "lfo" or not _lmod.lfo_channels or _lmod.muted:
            continue
        for _li, _lch in enumerate(_lmod.lfo_channels):
            _lck = _lmod.lfo_ch_key(_li)
            if _lck not in ki:
                continue
            _scale  = float(_lch.get("scale", 0.0))
            _lsig   = module_sigs.get(_lck)
            if _lsig is None:
                continue
            _T       = X.shape[1]
            _lsig_T  = _lsig[:_T]
            _old_val = X[ki[_lck]].copy()
            _new_val = _lsig_T + _scale * (_old_val - _lsig_T)
            _delta   = _new_val - _old_val
            for _e in g.edges:
                if _e.src_key != _lck or _e.delay_s != 0.0:
                    continue
                _di = ki.get(_e.dst_key)
                if _di is None:
                    continue
                _cw = complex(
                    _e.weight * global_decay * math.cos(_e.angle_rad),
                    _e.weight * global_decay * math.sin(_e.angle_rad),
                )
                X[_di] += _cw * _delta
            X[ki[_lck]] = _new_val

    # Populate sidecar "routed" channel for every node: the full complex routed
    # output (post-solve, pre-projection).  This is the definitive signal state
    # at each node and enables retroactive inspection of envelope, mix balance,
    # modulation depth, etc. for any node in the graph.
    _X_n = min(n, X.shape[1])
    for _k in node_keys:
        if _k in ki:
            sidecar.put(_k, "routed", X[ki[_k], :_X_n].copy())

    # 3b. Update param-series cache for the NEXT frame.
    # Extract each param node's output from the just-solved X and store it on
    # the patch.  Next frame's voice synthesis will consume these values before
    # the solve, so there is exactly one solve per frame (one-buffer latency).
    def extract_param_series(csig: np.ndarray, extractor: str) -> np.ndarray:
        if extractor == "real":
            return csig.real.astype(np.float64)
        if extractor == "imag":
            return csig.imag.astype(np.float64)
        if extractor == "phase":
            return np.angle(csig)
        if extractor == "energy":
            return (csig.real ** 2 + csig.imag ** 2).astype(np.float64)
        if extractor == "rms":
            mag = np.abs(csig)
            kernel = np.ones(128) / 128.0
            return np.convolve(mag, kernel, mode="same").astype(np.float64)
        # default: magnitude
        return np.abs(csig).astype(np.float64)

    if patch.param_nodes:
        _new_cache: dict = {}
        X_cols = X.shape[1]
        for pn in patch.param_nodes:
            if pn.key not in ki:
                continue
            raw = extract_param_series(X[ki[pn.key], :min(n, X_cols)], pn.extractor)
            lo, hi = float(pn.low), float(pn.high)
            span = hi - lo
            is_driven = any(pe.dst_key == pn.key for pe in patch.routing.param_edges)
            if not is_driven:
                _new_cache[pn.key] = np.full(len(raw), np.clip(float(pn.default_value), lo, hi))
                continue
            if span > 0:
                rmin, rmax = float(raw.min()), float(raw.max())
                rspan = rmax - rmin
                if rspan > 1e-12:
                    raw = (raw - rmin) / rspan * span + lo
                else:
                    raw = np.full_like(raw, lo + span * 0.5)
            _new_cache[pn.key] = np.clip(raw, lo, hi)
        patch._param_series_cache = _new_cache

    # 3c. State machine post-solve step.
    # Each state_machine module reads its accumulated routing inputs from X,
    # calls the plugin's step() with those inputs + current state, writes the
    # output trajectories back to X for its output nodes, and caches them for
    # the next frame's Src pre-population.  Delta-propagation covers zero-delay
    # downstream edges.
    _T_sm = X.shape[1]
    for _smmod in patch.modules:
        if _smmod.module_type != "state_machine" or _smmod.muted:
            continue
        _smmod._sm_log_text = ""
        if not _smmod.sm_items or not _smmod.sm_vars or not _smmod.sm_plugin:
            continue
        _sm_plug = _load_sm_plugin(_smmod.sm_plugin)
        if _sm_plug is None or not callable(getattr(_sm_plug, "step", None)):
            continue
        # Gather inputs: all signals that have edges targeting the main node.
        # When an edge has item_slot set, the SM receives the signal under that
        # name so driver/instrument identity is preserved through the boundary.
        _sm_inputs: dict = {}
        for _e in g.edges:
            if _e.dst_key == _smmod.key and _e.src_key in ki:
                _slot = _e.item_slot if _e.item_slot else _e.src_key
                _sm_inputs[_slot] = _sm_wrap(
                    X[ki[_e.src_key], :_T_sm].copy(), _smmod.sm_use_torch)
        _sm_dt = 1.0 / max(sr, 1.0)
        _sm_state_in = dict(_smmod._sm_state)
        _sm_stdout = io.StringIO()
        _sm_stderr = io.StringIO()
        try:
            with contextlib.redirect_stdout(_sm_stdout), contextlib.redirect_stderr(_sm_stderr):
                _sm_traj = _sm_plug.step(
                    _sm_inputs, _sm_state_in, _sm_dt,
                    n_items=_smmod.sm_n_items,
                    use_torch=_smmod.sm_use_torch,
                    params=dict(_smmod.sm_params),
                    plugin_state=dict(getattr(_smmod, "_sm_aux_state", {}) or {}),
                )
        except Exception:
            _captured = []
            _stdout_txt = _sm_stdout.getvalue()
            _stderr_txt = _sm_stderr.getvalue()
            if _stdout_txt:
                _captured.append(_stdout_txt.rstrip())
            if _stderr_txt:
                _captured.append(_stderr_txt.rstrip())
            _captured.append(traceback.format_exc().rstrip())
            _smmod._sm_log_text = "\n".join(x for x in _captured if x)
            continue
        if isinstance(_sm_traj, dict) and "outputs" in _sm_traj:
            _sm_outputs = _sm_traj.get("outputs", {}) or {}
            _sm_state_out = _sm_traj.get("state", {}) or {}
            _sm_log_extra = _sm_plugin_log_text(_sm_traj.get("log"))
            _sm_aux_state_out = _sm_traj.get("plugin_state", {}) or {}
        else:
            _sm_outputs = _sm_traj or {}
            _sm_state_out = {}
            _sm_log_extra = _sm_plugin_log_text(_sm_traj.get("log")) if isinstance(_sm_traj, dict) else ""
            _sm_aux_state_out = {}
        _sm_log_parts = []
        _stdout_txt = _sm_stdout.getvalue()
        _stderr_txt = _sm_stderr.getvalue()
        if _stdout_txt:
            _sm_log_parts.append(_stdout_txt.rstrip())
        if _stderr_txt:
            _sm_log_parts.append(_stderr_txt.rstrip())
        if _sm_log_extra:
            _sm_log_parts.append(_sm_log_extra.rstrip())
        _smmod._sm_log_text = "\n".join(x for x in _sm_log_parts if x)
        # Write trajectories back to X output nodes + cache + update state
        _new_sm_state: dict = {}
        for _item in _smmod.sm_items:
            _new_sm_state[_item] = {}
            for _var in _smmod.sm_vars:
                _ok = _smmod.sm_out_key(_item, _var)
                if _ok not in ki:
                    continue
                _traj_val = _sm_outputs.get(_item, {}).get(_var)
                if _traj_val is None:
                    continue
                _traj_arr = _sm_unwrap(_traj_val)
                # Pad/trim to match X columns
                if len(_traj_arr) < _T_sm:
                    _traj_arr = np.pad(_traj_arr, (0, _T_sm - len(_traj_arr)), mode="edge")
                _csig = _traj_arr[:_T_sm].astype(np.complex128)
                _old  = X[ki[_ok]].copy()
                _delta = _csig - _old
                # Delta-propagate to zero-delay downstream edges
                for _e2 in g.edges:
                    if _e2.src_key != _ok or _e2.delay_s != 0.0:
                        continue
                    _di = ki.get(_e2.dst_key)
                    if _di is None:
                        continue
                    _cw2 = complex(
                        _e2.weight * global_decay * math.cos(_e2.angle_rad),
                        _e2.weight * global_decay * math.sin(_e2.angle_rad),
                    )
                    X[_di] += _cw2 * _delta
                X[ki[_ok]] = _csig
                _smmod._sm_out_cache[_ok] = _csig[:n].copy()
            for _state_var in _smmod.sm_state_vars:
                _state_val = _sm_state_out.get(_item, {}).get(_state_var, None)
                if _state_val is not None:
                    _new_sm_state[_item][_state_var] = float(_state_val)
                    continue
                _traj_val = _sm_outputs.get(_item, {}).get(_state_var)
                if _traj_val is None:
                    _traj_val = _sm_traj.get(_item, {}).get(_state_var) if isinstance(_sm_traj, dict) else None
                if _traj_val is None:
                    _new_sm_state[_item][_state_var] = float(
                        _sm_state_in.get(_item, {}).get(_state_var, 0.0)
                    )
                    continue
                _traj_arr = _sm_unwrap(_traj_val)
                if len(_traj_arr) > 0:
                    _new_sm_state[_item][_state_var] = float(_traj_arr[-1])
        _smmod._sm_state = _new_sm_state
        _smmod._sm_aux_state = dict(_sm_aux_state_out)

    # 4. Resolve the main PCM output bus.
    # If the user has routed signal into the built-in system output channels,
    # those channels define the main output directly. Otherwise, preserve the
    # legacy active-mixer projection path.
    active_mixer_keys = [m.key for m in patch.mixers if m.projection_active]
    sys_out_keys = [k for k in _system_output_keys(patch) if k in ki]
    has_system_routing = bool(sys_out_keys) and any(
        e.dst_key in set(sys_out_keys) for e in g.edges
    )
    if has_system_routing:
        out_channels = np.column_stack([
            X[ki[_k], :n].real.astype(np.float32) for _k in sys_out_keys
        ]).astype(np.float32, copy=False)
        if patch.normalize_output and out_channels.size:
            peak = float(np.max(np.abs(out_channels)))
            if peak > 1e-9:
                out_channels /= peak
        left = out_channels[:, 0]
        right = out_channels[:, 1] if out_channels.shape[1] > 1 else out_channels[:, 0]
    elif active_mixer_keys:
        mix = sum(X[ki[mk]] for mk in active_mixer_keys if mk in ki)
        if patch.normalize_output:
            peak = float(np.max(np.abs(mix)))
            if peak > 1e-9:
                mix /= peak
        left, right = _apply_projection(mix, patch.projection_mode,
                                        patch.projection_rotation_hz, sr)
        out_channels = np.column_stack([left, right]).astype(np.float32, copy=False)
    else:
        out_channels = np.zeros((n, max(2, len(sys_out_keys) or 2)), dtype=np.float32)
        left = out_channels[:, 0]
        right = out_channels[:, 1]
    out = (left, right)
    if _return_mixer_sigs:
        _mixer_outs: dict = {}
        for _m in patch.mixers:
            if _m.export_to_file and _m.key in ki:
                _m_sig = X[ki[_m.key]][:n].copy()
                if patch.normalize_output:
                    _pk = float(np.max(np.abs(_m_sig)))
                    if _pk > 1e-9:
                        _m_sig = _m_sig / _pk
                _ml, _mr = _apply_projection(_m_sig, patch.projection_mode,
                                             patch.projection_rotation_hz, sr)
                _mixer_outs[_m.key] = (_ml, _mr)
        ret: list = [*out]
        if _return_output_channels:
            ret.append(out_channels)
        ret.append(_mixer_outs)
        if _return_sidecar:
            ret.append(sidecar)
        return tuple(ret)
    if _return_output_channels and _return_sidecar:
        return (*out, out_channels, sidecar)
    if _return_output_channels:
        return (*out, out_channels)
    if _return_sidecar:
        return (*out, sidecar)
    return out
