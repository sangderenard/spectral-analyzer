"""tests/test_patch_to_driver.py — Unit tests for patch_to_driver.build_driver_config.

Tests run without pygame / OpenGL by using the stubs already in conftest.py.
"""

from __future__ import annotations

import math
import sys
import os
import types

# ── Stub heavy deps before any project import ─────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _stub(name: str) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m


for _dep in [
    "pygame", "pygame.locals", "pygame.mixer", "pygame.font",
    "OpenGL", "OpenGL.GL", "OpenGL.GL.shaders",
    "scipy", "scipy.signal",
    "bass_viewer", "plot_widget",
]:
    _stub(_dep)

# stub pygame.locals constants
import pygame.locals as _pgl
for _attr in ["QUIT", "KEYDOWN", "KEYUP", "MOUSEBUTTONDOWN", "MOUSEBUTTONUP",
               "MOUSEMOTION", "K_ESCAPE", "K_SPACE", "K_LEFT", "K_RIGHT",
               "K_UP", "K_DOWN", "K_LCTRL", "K_RCTRL", "K_LSHIFT", "K_RSHIFT",
               "KMOD_CTRL", "KMOD_SHIFT", "RESIZABLE", "OPENGL", "DOUBLEBUF",
               "NOFRAME", "FULLSCREEN"]:
    if not hasattr(_pgl, _attr):
        setattr(_pgl, _attr, 0)

import pytest
import torch

from patch_to_driver import build_driver_config, _build_harmonics, _build_env_knots


# ── Minimal patch/voice stubs ─────────────────────────────────────────────────

class _Chirp:
    chirp_type    = "none"
    f_delta_start = 0.0
    f_delta_end   = 0.0
    tau           = 0.5
    chirp_power   = 1.0


class _Voice:
    def __init__(self, key="v0", freq_hz=440.0, amplitude=1.0, muted=False,
                 label="Voice",
                 manifold_type="pure", harmonic_count=1, harmonic_brightness=1.0,
                 harmonic_warp_strength=0.0, note_tracking="note",
                 semitone_offset=0.0, phase_origin=0.0, pre_delay=0.0,
                 env_type="adsr", body_type="direct", fm=None, am=None):
        self.key                  = key
        self.label                = label
        self.freq_hz              = freq_hz
        self.amplitude            = amplitude
        self.muted                = muted
        self.manifold_type        = manifold_type
        self.harmonic_count       = harmonic_count
        self.harmonic_brightness  = harmonic_brightness
        self.harmonic_warp_strength = harmonic_warp_strength
        self.note_tracking        = note_tracking
        self.semitone_offset      = semitone_offset
        self.phase_origin         = phase_origin
        self.pre_delay            = pre_delay
        self.env_type             = env_type
        self.body_type            = body_type
        self.chirp                = _Chirp()
        self.fm                   = fm
        self.am                   = am

    def active_knots(self):
        # default ADSR as normalized fractions (duration=1.0)
        return [[0.0, 0.0], [0.005, 1.0], [0.045, 0.75], [0.92, 0.75], [1.0, 0.0]]


class _Patch:
    def __init__(self, voices=None, duration=1.0, seq_tonic_hz=440.0):
        self.voices       = voices or []
        self.duration     = duration
        self.seq_tonic_hz = seq_tonic_hz


class _Performer:
    def __init__(self, key="p0", source_voice_keys=None, chair_key="default",
                 body_type="direct", x=0.0, y=0.0,
                 geometric_delay_ms=0.0, humanization_ms=0.0,
                 phase_offset_rad=0.0, gain_db=0.0):
        self.key               = key
        self.source_voice_keys = source_voice_keys or []
        self.chair_key         = chair_key
        self.body_type         = body_type
        self.x                 = x
        self.y                 = y
        self.geometric_delay_ms = geometric_delay_ms
        self.humanization_ms   = humanization_ms
        self.phase_offset_rad  = phase_offset_rad
        self.gain_db           = gain_db


# ── Helper ────────────────────────────────────────────────────────────────────

_DEV = torch.device("cpu")
_SR  = 48000.0


def _build(patch, performers, note_hz=None):
    cfg, _performers, _driver_list = build_driver_config(patch, performers, _DEV, _SR, note_hz=note_hz)
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# _build_harmonics
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildHarmonics:
    def test_pure_manifold_single_harmonic(self):
        v = _Voice(manifold_type="pure", harmonic_count=4)
        ratios, amps = _build_harmonics(v)
        assert ratios[0] == pytest.approx(1.0)
        assert amps[0]   == pytest.approx(1.0)
        # slots beyond first are padded 0
        assert all(r == 0.0 for r in ratios[1:])
        assert all(a == 0.0 for a in amps[1:])

    def test_harmonic_manifold_normalised(self):
        v = _Voice(manifold_type="harmonic", harmonic_count=4,
                   harmonic_brightness=1.0)
        ratios, amps = _build_harmonics(v)
        assert ratios == pytest.approx([1.0, 2.0, 3.0, 4.0])
        # amps sum to 1
        assert sum(amps) == pytest.approx(1.0)
        # rolloff: amp[k] ∝ 1/k
        assert amps[0] > amps[1] > amps[2] > amps[3]

    def test_harmonic_zero_brightness_flat(self):
        v = _Voice(manifold_type="harmonic", harmonic_count=3,
                   harmonic_brightness=0.0)
        ratios, amps = _build_harmonics(v)
        # with brightness=0 every raw amp is 1.0 → normalised to 1/3
        assert amps == pytest.approx([1 / 3, 1 / 3, 1 / 3])

    def test_harmonic_warp_increases_ratios(self):
        v = _Voice(manifold_type="harmonic_warp", harmonic_count=4,
                   harmonic_warp_strength=0.5)
        ratios, _ = _build_harmonics(v)
        # first ratio always 1; subsequent ratios > k (stretched)
        assert ratios[0] == pytest.approx(1.0)
        assert ratios[1] > 2.0
        assert ratios[2] > 3.0

    def test_length_matches_harmonic_count(self):
        for H in (1, 4, 8, 16):
            v = _Voice(manifold_type="harmonic", harmonic_count=H)
            ratios, amps = _build_harmonics(v)
            assert len(ratios) == H
            assert len(amps)   == H


# ═══════════════════════════════════════════════════════════════════════════════
# _build_env_knots
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildEnvKnots:
    def test_time_scaled_by_duration(self):
        v   = _Voice()
        dur = 2.0
        t, vals = _build_env_knots(v, dur)
        knots = v.active_knots()
        for i, k in enumerate(knots):
            assert t[i] == pytest.approx(k[0] * dur)
            assert vals[i] == pytest.approx(k[1])

    def test_starts_at_zero_ends_at_zero(self):
        v = _Voice()
        t, vals = _build_env_knots(v, 1.0)
        assert t[0]    == pytest.approx(0.0)
        assert vals[0] == pytest.approx(0.0)
        assert vals[-1] == pytest.approx(0.0)

    def test_peak_value_present(self):
        v = _Voice()
        _, vals = _build_env_knots(v, 1.0)
        assert max(vals) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# build_driver_config — shape correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildDriverConfigShape:
    def test_single_voice_no_performers_synthetic_fallback(self):
        patch = _Patch(voices=[_Voice(key="v0", freq_hz=440.0)])
        cfg = _build(patch, [])
        assert cfg.D == 1
        assert cfg.V == 1

    def test_two_voices_two_performers(self):
        voices = [_Voice(key="v0"), _Voice(key="v1")]
        perfs  = [
            _Performer(key="p0", source_voice_keys=["v0"]),
            _Performer(key="p1", source_voice_keys=["v1"]),
        ]
        patch = _Patch(voices=voices)
        cfg = _build(patch, perfs)
        assert cfg.D == 2
        assert cfg.V == 2

    def test_one_performer_covering_all_voices(self):
        voices = [_Voice(key="v0"), _Voice(key="v1"), _Voice(key="v2")]
        # empty source_voice_keys → expands to all non-muted
        perfs = [_Performer(key="p0", source_voice_keys=[])]
        patch = _Patch(voices=voices)
        cfg = _build(patch, perfs)
        assert cfg.D == 3

    def test_muted_voice_skipped(self):
        voices = [_Voice(key="v0"), _Voice(key="v1", muted=True)]
        patch  = _Patch(voices=voices)
        cfg = _build(patch, [])
        assert cfg.D == 1

    def test_empty_patch_returns_zero_drivers(self):
        patch = _Patch(voices=[])
        cfg = _build(patch, [])
        assert cfg.D == 0

    def test_all_voices_muted_returns_zero_drivers(self):
        voices = [_Voice(key="v0", muted=True), _Voice(key="v1", muted=True)]
        patch  = _Patch(voices=voices)
        cfg = _build(patch, [])
        assert cfg.D == 0

    def test_tensor_shapes_consistent(self):
        voices = [_Voice(key="v0"), _Voice(key="v1")]
        patch  = _Patch(voices=voices)
        cfg = _build(patch, [])
        D, H, K, V = cfg.D, cfg.H, cfg.K, cfg.V
        assert cfg.f0.shape             == (D,)
        assert cfg.amplitude.shape      == (D,)
        assert cfg.h_ratios.shape       == (D, H)
        assert cfg.h_amps.shape         == (D, H)
        assert cfg.env_t.shape          == (D, K)
        assert cfg.env_v.shape          == (D, K)
        assert cfg.voice_idx.shape      == (D,)
        assert cfg.instrument_idx.shape == (D,)


# ═══════════════════════════════════════════════════════════════════════════════
# build_driver_config — value correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildDriverConfigValues:
    def test_f0_note_tracking(self):
        """note_tracking='note' uses note_hz + semitone_offset."""
        v = _Voice(key="v0", note_tracking="note", semitone_offset=12.0)
        patch = _Patch(voices=[v], seq_tonic_hz=220.0)
        cfg = _build(patch, [], note_hz=440.0)
        expected = 440.0 * (2.0 ** (12.0 / 12.0))  # 880 Hz
        assert cfg.f0[0].item() == pytest.approx(expected)

    def test_f0_root_tracking(self):
        """note_tracking='root' always uses seq_tonic_hz."""
        v = _Voice(key="v0", note_tracking="root", semitone_offset=0.0)
        patch = _Patch(voices=[v], seq_tonic_hz=220.0)
        cfg = _build(patch, [], note_hz=440.0)
        assert cfg.f0[0].item() == pytest.approx(220.0)

    def test_f0_free_tracking(self):
        """note_tracking='free' uses voice.freq_hz directly."""
        v = _Voice(key="v0", note_tracking="free", freq_hz=880.0, semitone_offset=0.0)
        patch = _Patch(voices=[v], seq_tonic_hz=220.0)
        cfg = _build(patch, [], note_hz=440.0)
        assert cfg.f0[0].item() == pytest.approx(880.0)

    def test_default_note_hz_is_tonic(self):
        """When note_hz is None the tonic is used for 'note' tracking."""
        v = _Voice(key="v0", note_tracking="note", semitone_offset=0.0)
        patch = _Patch(voices=[v], seq_tonic_hz=660.0)
        cfg = _build(patch, [], note_hz=None)
        assert cfg.f0[0].item() == pytest.approx(660.0)

    def test_amplitude_is_voice_amplitude(self):
        """DriverConfig.amplitude = voice.amplitude (no performer gain applied)."""
        v    = _Voice(key="v0", amplitude=0.5)
        perf = _Performer(key="p0", source_voice_keys=["v0"], gain_db=6.0)
        patch = _Patch(voices=[v])
        cfg = _build(patch, [perf])
        assert cfg.amplitude[0].item() == pytest.approx(0.5)

    def test_performer_gain_not_in_amplitude(self):
        """PerformerConfig.gain carries gain_db, not DriverConfig.amplitude."""
        v    = _Voice(key="v0", amplitude=1.0)
        perf = _Performer(key="p0", source_voice_keys=["v0"], gain_db=20.0)
        patch = _Patch(voices=[v])
        cfg = _build(patch, [perf])
        # amplitude unchanged
        assert cfg.amplitude[0].item() == pytest.approx(1.0)

    def test_pre_delay_accumulates_sources(self):
        """pre_delay_samples = round((voice.pre_delay + geo_ms + human_ms) * sr)."""
        v    = _Voice(key="v0", pre_delay=0.010)  # 10 ms
        perf = _Performer(key="p0", source_voice_keys=["v0"],
                          geometric_delay_ms=5.0, humanization_ms=3.0)
        patch = _Patch(voices=[v])
        cfg = _build(patch, [perf])
        expected = int(round((0.010 + 0.005 + 0.003) * _SR))
        assert cfg.pre_delay_samples[0].item() == expected

    def test_env_t_scaled_to_duration(self):
        v     = _Voice(key="v0")
        dur   = 2.5
        patch = _Patch(voices=[v], duration=dur)
        cfg = _build(patch, [])
        knots  = v.active_knots()
        for i, k in enumerate(knots):
            assert cfg.env_t[0, i].item() == pytest.approx(k[0] * dur)

    def test_env_padded_with_sentinel(self):
        """Slots beyond the last real knot carry sentinel time 1e30."""
        v     = _Voice(key="v0")
        patch = _Patch(voices=[v])
        cfg = _build(patch, [])
        n_k = cfg.env_n[0].item()
        # first padded position (if any)
        if cfg.K > n_k:
            assert cfg.env_t[0, n_k].item() == pytest.approx(1e30)

    def test_harmonic_h_amps_normalised(self):
        v = _Voice(key="v0", manifold_type="harmonic", harmonic_count=4,
                   harmonic_brightness=1.0)
        patch = _Patch(voices=[v])
        cfg = _build(patch, [])
        # Real harmonic slots
        n_h = int(cfg.n_harmonics[0].item())
        amps = cfg.h_amps[0, :n_h]
        assert amps.sum().item() == pytest.approx(1.0)

    def test_voice_idx_maps_correctly(self):
        voices = [_Voice(key="v0"), _Voice(key="v1")]
        perfs  = [
            _Performer(key="p0", source_voice_keys=["v1"]),
        ]
        patch = _Patch(voices=voices)
        cfg = _build(patch, perfs)
        assert int(cfg.voice_idx[0].item()) == 1  # v1 is index 1

    def test_performer_idx_maps_correctly(self):
        voices = [_Voice(key="v0"), _Voice(key="v1")]
        perfs  = [
            _Performer(key="p0", source_voice_keys=["v0"], chair_key="A"),
            _Performer(key="p1", source_voice_keys=["v1"], chair_key="B"),
        ]
        patch = _Patch(voices=voices)
        cfg = _build(patch, perfs)
        pidxs = sorted(cfg.instrument_idx.tolist())
        assert pidxs == [0, 1]

    def test_fm_wiring_maps_source_voice(self):
        class _FM:
            source_key = "v1"
            depth_hz   = 5.0
            depth_amp  = 0.0

        v0 = _Voice(key="v0", fm=_FM())
        v1 = _Voice(key="v1")
        perfs = [_Performer(key="p0", source_voice_keys=["v0", "v1"])]
        patch = _Patch(voices=[v0, v1])
        cfg = _build(patch, perfs)
        # Driver slot for v0 should have fm_source_voice pointing to v1 (index 1)
        slot_v0 = (cfg.voice_idx == 0).nonzero(as_tuple=True)[0][0].item()
        assert cfg.fm_source_voice[slot_v0].item() == 1
        assert cfg.fm_depth_hz[slot_v0].item() == pytest.approx(5.0)

    def test_am_wiring_maps_source_voice(self):
        class _AM:
            source_key = "v0"
            depth_hz   = 0.0
            depth_amp  = 0.3

        v0 = _Voice(key="v0")
        v1 = _Voice(key="v1", am=_AM())
        perfs = [_Performer(key="p0", source_voice_keys=["v0", "v1"])]
        patch = _Patch(voices=[v0, v1])
        cfg = _build(patch, perfs)
        slot_v1 = (cfg.voice_idx == 1).nonzero(as_tuple=True)[0][0].item()
        assert cfg.am_source_voice[slot_v1].item() == 0
        assert cfg.am_depth[slot_v1].item() == pytest.approx(0.3)

    def test_no_fm_am_wiring_gives_minus_one(self):
        v = _Voice(key="v0")
        patch = _Patch(voices=[v])
        cfg = _build(patch, [])
        assert cfg.fm_source_voice[0].item() == -1
        assert cfg.am_source_voice[0].item() == -1

    def test_dangling_fm_source_key_ignored(self):
        """FM source_key pointing to a non-existent voice → -1."""
        class _FM:
            source_key = "ghost"
            depth_hz   = 5.0
            depth_amp  = 0.0

        v = _Voice(key="v0", fm=_FM())
        patch = _Patch(voices=[v])
        cfg = _build(patch, [])
        assert cfg.fm_source_voice[0].item() == -1

    def test_chirp_type_codes(self):
        from performer_engine import CHIRP_LINEAR, CHIRP_EXPONENTIAL, CHIRP_POWER, CHIRP_NONE

        for ctype, expected in [
            ("none", CHIRP_NONE), ("linear", CHIRP_LINEAR),
            ("exponential", CHIRP_EXPONENTIAL), ("power", CHIRP_POWER),
        ]:
            v = _Voice(key="v0")
            v.chirp.chirp_type = ctype
            patch = _Patch(voices=[v])
            cfg = _build(patch, [])
            assert cfg.chirp_type[0].item() == expected

    def test_active_all_true_by_default(self):
        voices = [_Voice(key="v0"), _Voice(key="v1")]
        patch  = _Patch(voices=voices)
        cfg = _build(patch, [])
        assert cfg.active.all()


# ═══════════════════════════════════════════════════════════════════════════════
# Round-trip: build_driver_config → driver_synthesis_step produces valid output
# ═══════════════════════════════════════════════════════════════════════════════

class TestEndToEndSynthStep:
    """Smoke-test: the produced config actually runs through the performer engine."""

    def test_single_voice_step_shape(self):
        from performer_engine import init_driver_state, driver_synthesis_step

        v     = _Voice(key="v0", freq_hz=440.0, manifold_type="pure")
        patch = _Patch(voices=[v], duration=1.0)
        cfg = _build(patch, [])
        state  = init_driver_state(cfg)

        chunk_T = 512
        d_out, v_out, new_state = driver_synthesis_step(cfg, state, chunk_T, _SR)

        assert d_out.shape == (cfg.D, chunk_T)
        assert v_out.shape == (cfg.V, chunk_T)

    def test_harmonic_voice_step_finite(self):
        from performer_engine import init_driver_state, driver_synthesis_step

        v = _Voice(key="v0", freq_hz=220.0, manifold_type="harmonic",
                   harmonic_count=4, harmonic_brightness=1.0)
        patch = _Patch(voices=[v], duration=2.0)
        cfg = _build(patch, [])
        state  = init_driver_state(cfg)

        d_out, _, _ = driver_synthesis_step(cfg, state, 1024, _SR)
        assert torch.isfinite(d_out).all()

    def test_fm_wired_step_runs(self):
        """Two-voice patch with FM — multi_level_driver_step should succeed."""
        from performer_engine import init_driver_state, multi_level_driver_step

        class _FM:
            source_key = "v_carrier"
            depth_hz   = 10.0
            depth_amp  = 0.0

        v_mod     = _Voice(key="v_mod",     freq_hz=6.0)
        v_carrier = _Voice(key="v_carrier", freq_hz=440.0, fm=_FM())
        perfs = [_Performer(key="p0", source_voice_keys=["v_mod", "v_carrier"])]
        patch = _Patch(voices=[v_mod, v_carrier], duration=1.0)
        cfg = _build(patch, perfs)
        state  = init_driver_state(cfg)

        d_out, v_out, _ = multi_level_driver_step(cfg, state, 256, _SR)
        assert torch.isfinite(d_out).all()
        assert torch.isfinite(v_out).all()

    def test_state_advances_t_pos(self):
        from performer_engine import init_driver_state, driver_synthesis_step

        v     = _Voice(key="v0")
        patch = _Patch(voices=[v])
        cfg = _build(patch, [])
        state  = init_driver_state(cfg)

        chunk_T = 480
        _, _, new_state = driver_synthesis_step(cfg, state, chunk_T, _SR)
        expected_dt = chunk_T / _SR
        assert new_state.t_pos[0].item() == pytest.approx(expected_dt)
