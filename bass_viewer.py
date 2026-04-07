#!/usr/bin/env python3
"""
Spectrogram Viewer — OpenGL/Pygame data-texture navigator with per-field control.

Loads CQT data directly from cqt_data.npz (memory-mapped) and builds
viewport-sized GPU textures on the fly via a reduction-gather algorithm.
Only the visible region at screen resolution is uploaded each frame,
so arbitrarily large datasets are handled without hitting memory or
pixel-count limits.

Usage
-----
    python bass_viewer.py <analysis_dir> [audio_file]

Controls
--------
    Space           Play / pause
    Left / Right    Scroll time axis (hold Shift for fast)
    Mouse wheel     Zoom in / out (at cursor)
    Left drag       Pan
    Right drag      Rectangle selection → crop to region
    Middle click    Seek to time position
    Home / End      Jump to start / end
    R               Reset zoom to fit
    Tab             Cycle through data-field tabs
    F               Toggle frequency-axis labels
    T               Toggle time-axis labels
    X               Lock zoom to time axis only
    Y               Lock zoom to frequency axis only
    S               Play viewport TF synthesis
    Escape          Quit

Dependencies
------------
pip install numpy pygame PyOpenGL Pillow scipy
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP,
    MOUSEMOTION, MOUSEWHEEL, OPENGL, QUIT, RESIZABLE, VIDEORESIZE,
    K_SPACE, K_ESCAPE, K_LEFT, K_RIGHT, K_HOME, K_END, K_TAB,
    K_r, K_f, K_t, K_x, K_y, K_s, K_a, K_d,
    K_LSHIFT, K_RSHIFT,
)
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders
from PIL import Image, ImageDraw, ImageFont
from scipy.io import wavfile

from plot_widget import PlotWidget, PlotSeries


# ---------------------------------------------------------------------------
# Glyph atlas — tiny texture with all characters needed for axis labels
# ---------------------------------------------------------------------------

_GLYPH_CHARS = "0123456789.-+ AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz#♭"
_GLYPH_FONT_SIZE = 14
_GLYPH_PAD = 1


class GlyphAtlas:
    """A single small texture containing pre-rendered character glyphs.

    At render time, axis labels are composed by emitting one textured
    quad per character — no PIL work, no large textures, works at any zoom.
    """

    def __init__(self) -> None:
        self.tex_id: int = 0
        self.atlas_w: int = 0
        self.atlas_h: int = 0
        self.glyph_map: dict[str, tuple[float, float, float, float, int, int]] = {}
        # glyph_map[ch] = (u0, v0, u1, v1, px_w, px_h)

    def build(self) -> None:
        """Render glyphs into an atlas image and upload as a GL texture."""
        font = _load_font(_GLYPH_FONT_SIZE)
        metrics: list[tuple[str, int, int]] = []
        for ch in _GLYPH_CHARS:
            bbox = font.getbbox(ch)
            w = bbox[2] - bbox[0] + 2 * _GLYPH_PAD
            h = bbox[3] - bbox[1] + 2 * _GLYPH_PAD
            metrics.append((ch, w, h))

        max_h = max(h for _, _, h in metrics)
        # Pack in a single row
        total_w = sum(w for _, w, _ in metrics)
        atlas_w = total_w
        atlas_h = max_h

        img = Image.new("RGBA", (atlas_w, atlas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        x_cursor = 0
        for ch, gw, gh in metrics:
            bbox = font.getbbox(ch)
            ox = -bbox[0] + _GLYPH_PAD
            oy = -bbox[1] + _GLYPH_PAD
            draw.text((x_cursor + ox, oy), ch,
                      fill=(255, 255, 255, 255), font=font)
            u0 = x_cursor / atlas_w
            v0 = 0.0
            u1 = (x_cursor + gw) / atlas_w
            v1 = gh / atlas_h
            self.glyph_map[ch] = (u0, v0, u1, v1, gw, gh)
            x_cursor += gw

        self.atlas_w = atlas_w
        self.atlas_h = atlas_h

        raw = img.tobytes("raw", "RGBA")
        if self.tex_id:
            glDeleteTextures([self.tex_id])
        self.tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlas_w, atlas_h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)

    def measure(self, text: str) -> tuple[int, int]:
        """Return (width, height) in pixels for a string."""
        w, h = 0, 0
        for ch in text:
            g = self.glyph_map.get(ch)
            if g is None:
                continue
            w += g[4]
            h = max(h, g[5])
        return w, h

    def draw_string(self, text: str, x: float, y: float,
                    win_w: int, win_h: int, *,
                    anchor_x: float = 0.0,
                    anchor_y: float = 0.5,
                    color: tuple[float, float, float, float] = (0.78, 0.78, 0.78, 1.0)) -> None:
        """Emit textured quads for *text* at pixel position (x, y).

        anchor_x: 0=left, 0.5=center, 1=right
        anchor_y: 0=top, 0.5=middle, 1=bottom
        """
        tw, th = self.measure(text)
        if tw == 0:
            return
        x -= tw * anchor_x
        y -= th * anchor_y

        glColor4f(*color)
        glBindTexture(GL_TEXTURE_2D, self.tex_id)
        glBegin(GL_QUADS)
        cx = x
        for ch in text:
            g = self.glyph_map.get(ch)
            if g is None:
                continue
            u0, v0, u1, v1, gw, gh = g
            # pixel -> NDC
            nx0 = 2.0 * cx / win_w - 1.0
            ny1 = 1.0 - 2.0 * y / win_h
            nx1 = 2.0 * (cx + gw) / win_w - 1.0
            ny0 = 1.0 - 2.0 * (y + gh) / win_h
            glTexCoord2f(u0, v0); glVertex2f(nx0, ny1)
            glTexCoord2f(u1, v0); glVertex2f(nx1, ny1)
            glTexCoord2f(u1, v1); glVertex2f(nx1, ny0)
            glTexCoord2f(u0, v1); glVertex2f(nx0, ny0)
            cx += gw
        glEnd()


# ---------------------------------------------------------------------------
# Shader sources
# ---------------------------------------------------------------------------

VERT_SRC = """
#version 330 compatibility
out vec2 vUV;
void main() {
    gl_Position = gl_Vertex;
    vUV = gl_MultiTexCoord0.xy;
}
"""

FRAG_SRC = """
#version 330 compatibility
#define MAX_FIELDS 16

in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uTex0, uTex1, uTex2, uTex3;
uniform sampler2D uTex4, uTex5, uTex6, uTex7;

uniform int   uFieldCount;
uniform ivec4 uFieldDesc[MAX_FIELDS];   // (tex_unit, tex_chan, target, norm_mode)
uniform vec4  uFieldParams[MAX_FIELDS]; // (gamma, scale, 0, 0)

uniform vec4  uOutGamma;               // per-display-channel post-agg gamma
uniform vec4  uOutScale;               // per-display-channel post-agg scale

vec4 sampleTex(int u) {
    if (u == 0) return texture(uTex0, vUV);
    if (u == 1) return texture(uTex1, vUV);
    if (u == 2) return texture(uTex2, vUV);
    if (u == 3) return texture(uTex3, vUV);
    if (u == 4) return texture(uTex4, vUV);
    if (u == 5) return texture(uTex5, vUV);
    if (u == 6) return texture(uTex6, vUV);
    if (u == 7) return texture(uTex7, vUV);
    return vec4(0.0);
}

float extractCh(vec4 t, int ch) {
    if (ch == 0) return t.r;
    if (ch == 1) return t.g;
    if (ch == 2) return t.b;
    return t.a;
}

float applyNorm(float v, int mode) {
    if (mode == 1) {
        // dB: 10*log10 then remap [-60,0] -> [0,1]
        float db = 10.0 * log(max(v, 1e-12)) / log(10.0);
        return clamp((db + 60.0) / 60.0, 0.0, 1.0);
    }
    // mode 0 (linear) and mode 2 (rank_order — pre-ranked on CPU)
    return clamp(v, 0.0, 1.0);
}

void main() {
    vec4 accum  = vec4(0.0);
    vec4 counts = vec4(0.0);

    for (int i = 0; i < MAX_FIELDS; i++) {
        if (i >= uFieldCount) break;

        ivec4 d = uFieldDesc[i];
        int texU   = d.x;
        int texCh  = d.y;
        int target = d.z;
        int nMode  = d.w;
        if (target < 0) continue;

        vec4  p     = uFieldParams[i];
        float gamma = p.x;
        float scale = p.y;

        float raw    = extractCh(sampleTex(texU), texCh);
        float normed = applyNorm(raw, nMode);
        float val    = pow(normed, gamma) * scale;

        if      (target == 0) { accum.r += val; counts.r += 1.0; }
        else if (target == 1) { accum.g += val; counts.g += 1.0; }
        else if (target == 2) { accum.b += val; counts.b += 1.0; }
        else if (target == 3) { accum.a += val; counts.a += 1.0; }
        else if (target == 4) { accum.r += val; counts.r += 1.0;   // W = R+G+B
                                 accum.g += val; counts.g += 1.0;
                                 accum.b += val; counts.b += 1.0; }
        else if (target == 5) { accum.g += val; counts.g += 1.0;   // C = G+B
                                 accum.b += val; counts.b += 1.0; }
        else if (target == 6) { accum.r += val; counts.r += 1.0;   // M = R+B
                                 accum.b += val; counts.b += 1.0; }
        else if (target == 7) { accum.r += val; counts.r += 1.0;   // Y = R+G
                                 accum.g += val; counts.g += 1.0; }
    }

    // Average when multiple fields target the same channel
    if (counts.r > 1.0) accum.r /= counts.r;
    if (counts.g > 1.0) accum.g /= counts.g;
    if (counts.b > 1.0) accum.b /= counts.b;
    if (counts.a > 1.0) accum.a /= counts.a;

    // Global post-aggregation normalization
    accum.r = pow(accum.r, uOutGamma.x) * uOutScale.x;
    accum.g = pow(accum.g, uOutGamma.y) * uOutScale.y;
    accum.b = pow(accum.b, uOutGamma.z) * uOutScale.z;
    accum.a = pow(accum.a, uOutGamma.w) * uOutScale.w;

    // Default alpha to 1 if no field targets it
    if (counts.a == 0.0) accum.a = 1.0;

    fragColor = clamp(accum, 0.0, 1.0);
}
"""



# ---------------------------------------------------------------------------
# Audio player
# ---------------------------------------------------------------------------

class AudioPlayer:
    def __init__(self, path: str) -> None:
        self.path = path
        sr, data = wavfile.read(path)
        if np.issubdtype(data.dtype, np.floating):
            data = (data * 32767).clip(-32768, 32767).astype(np.int16)
        elif data.dtype == np.int32:
            data = (data >> 16).astype(np.int16)
        elif data.dtype != np.int16:
            data = data.astype(np.int16)
        if data.ndim == 1:
            data = np.column_stack([data, data])
        self.sr = sr
        self.data = data
        self.duration = len(data) / sr
        self.position = 0.0
        self.playing = False
        self._play_wall: float = 0.0
        self._channel: pygame.mixer.Channel | None = None
        self._sound: pygame.mixer.Sound | None = None

    def play(self, start_s: float | None = None) -> None:
        if start_s is not None:
            self.position = max(0.0, min(start_s, self.duration))
        self.stop()
        start_sample = int(self.position * self.sr)
        if start_sample >= len(self.data):
            return
        buf = self.data[start_sample:]
        self._sound = pygame.sndarray.make_sound(buf.copy())
        self._channel = self._sound.play()
        self._play_wall = time.time()
        self.playing = True

    def stop(self) -> None:
        if self.playing:
            self.position = self.get_position()
            if self._channel is not None:
                self._channel.stop()
            self.playing = False
        self._channel = None
        self._sound = None

    def toggle(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def get_position(self) -> float:
        if self.playing:
            pos = self.position + (time.time() - self._play_wall)
            if pos >= self.duration:
                self.playing = False
                return self.duration
            return pos
        return self.position

    def seek(self, t: float) -> None:
        t = max(0.0, min(t, self.duration))
        was_playing = self.playing
        self.stop()
        self.position = t
        if was_playing:
            self.play()


# ---------------------------------------------------------------------------
# Data field model
# ---------------------------------------------------------------------------

FIELD_DISPLAY = {
    "real_left": "Re L",
    "imag_left": "Im L",
    "real_right": "Re R",
    "imag_right": "Im R",
    "mag_left": "Mag L",
    "phase_left": "\u03a6 L",
    "mag_right": "Mag R",
    "phase_right": "\u03a6 R",
    "mag_similarity": "Mag Sim",
    "phase_similarity": "\u03a6 Sim",
    "onset": "Onset",
}

NORM_MODES = ["linear", "dB", "rank_order"]
TARGET_LABELS = ["none", "R", "G", "B", "A", "W", "C", "M", "Y"]
TARGET_VALUES = [-1, 0, 1, 2, 3, 4, 5, 6, 7]

# Virtual texture channel layouts — same channel packing as bass_plot.py
_TEXTURE_LAYOUTS: dict[str, dict[str, str | None]] = {
    "cqt_left_complex":  {"R": "real_left",  "G": "imag_left",  "B": None,              "A": None},
    "cqt_right_complex": {"R": "real_right", "G": "imag_right", "B": None,              "A": None},
    "cqt_left_polar":    {"R": "mag_left",   "G": "phase_left", "B": None,              "A": None},
    "cqt_right_polar":   {"R": "mag_right",  "G": "phase_right","B": None,              "A": None},
    "cqt_mag_stereo":    {"R": "mag_left",   "G": "mag_right",  "B": "mag_similarity",  "A": None},
    "cqt_phase_stereo":  {"R": "phase_left", "G": "phase_right","B": "phase_similarity","A": None},
    "onset":             {"R": "onset",       "G": None,         "B": None,              "A": None},
}


@dataclass
class FieldSource:
    name: str
    display_name: str
    tex_name: str
    tex_channel: int  # 0=R 1=G 2=B 3=A


@dataclass
class FieldConfig:
    target: int = -1       # display channel: 0=R 1=G 2=B 3=A, -1=disabled
    norm_mode: int = 0     # 0=linear 1=dB 2=rank_order
    gamma: float = 1.0
    scale: float = 1.0
    use_global: bool = True


@dataclass
class GlobalDefaults:
    norm_mode: int = 0
    gamma: float = 1.0
    scale: float = 1.0
    out_gamma: float = 1.0
    out_scale: float = 1.0


# ---------------------------------------------------------------------------
# Subplot tiling — multi-plot support within each display mode
# ---------------------------------------------------------------------------

class TilingStrategy(Enum):
    COLUMNS = "columns"
    ROWS = "rows"
    SQUARE = "square"


@dataclass
class SubPlot:
    """A named, togglable rendering slot within the display area."""
    key: str
    label: str
    visible: bool = True


def _tile_rects(
    sx: int, sy: int, sw: int, sh: int,
    n: int, strategy: TilingStrategy, gap: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Compute (x, y, w, h) cell rects for *n* visible subplots."""
    if n <= 0:
        return []
    if n == 1:
        return [(sx, sy, sw, sh)]

    if strategy == TilingStrategy.COLUMNS:
        cols, rows = n, 1
    elif strategy == TilingStrategy.ROWS:
        cols, rows = 1, n
    else:  # SQUARE
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

    cell_w = max(1, (sw - gap * (cols - 1)) // cols)
    cell_h = max(1, (sh - gap * (rows - 1)) // rows)

    rects: list[tuple[int, int, int, int]] = []
    for i in range(n):
        c = i % cols
        r = i // cols
        cx = sx + c * (cell_w + gap)
        cy = sy + r * (cell_h + gap)
        rects.append((cx, cy, cell_w, cell_h))
    return rects


# ---------------------------------------------------------------------------
# Filter bank decomposition — arbitrary crossover with perfect reconstruction
# ---------------------------------------------------------------------------

_FILTER_TYPES = ["Linkwitz-Riley 4", "Butterworth 4", "Butterworth 8"]

_RESAMPLE_ENGINES = ["scipy", "soxr"]
_RESAMPLE_INTERPS = ["sinc", "linear", "nearest", "cubic"]
_RESAMPLE_TAPS = ["8", "16", "32", "64", "128", "256"]
_RESAMPLE_PRECISIONS = ["16-bit", "24-bit", "32-bit float"]

# Musically meaningful BPO values (all multiples of 12)
_BPO_SNAPS = [12, 24, 36, 48, 60, 72, 96, 120, 240, 360, 600, 1200]

_NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]


def _midi_to_freq(midi: int) -> float:
    """MIDI note number → frequency (A4 = 440 Hz, MIDI 69)."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def _freq_to_note_name(freq: float) -> str:
    """Return the nearest Western note name + octave for *freq* Hz."""
    if freq <= 0:
        return "—"
    midi = 69 + 12 * math.log2(freq / 440.0)
    midi_r = round(midi)
    name = _NOTE_NAMES[midi_r % 12]
    octave = (midi_r // 12) - 1
    return f"{name}{octave}"


def _snap_to_note(freq: float) -> float:
    """Snap *freq* to the nearest equal-temperament semitone (A440)."""
    if freq <= 0:
        return freq
    midi = 69 + 12 * math.log2(freq / 440.0)
    return _midi_to_freq(round(midi))


def _snap_bpo(val: float) -> int:
    """Snap a raw BPO value to the nearest entry in _BPO_SNAPS."""
    best = _BPO_SNAPS[0]
    best_dist = abs(val - best)
    for s in _BPO_SNAPS[1:]:
        d = abs(val - s)
        if d < best_dist:
            best, best_dist = s, d
    return best


@dataclass
class BandDef:
    """One band in a crossover filter bank."""
    fmin: float    # 0.0 means from DC
    fmax: float    # >= Nyquist means to Nyquist
    label: str = ""
    enabled: bool = True

    def auto_label(self, sr: int) -> str:
        nyq = sr / 2.0
        lo = "DC" if self.fmin <= 0.0 else f"{self.fmin:.0f}"
        hi = "Nyq" if self.fmax >= nyq else f"{self.fmax:.0f}"
        return f"{lo}–{hi} Hz"


class FilterBankDecomposition:
    """Apply a set of crossover filters to a mono signal, producing
    perfectly-reconstructing subbands.

    Crossover frequencies define the splits; adjacent bands share a
    crossover point.  Linkwitz-Riley 4th-order crossovers sum flat.
    """

    def __init__(self, sr: int, filter_type: str = "Linkwitz-Riley 4") -> None:
        self.sr: int = sr
        self.filter_type: str = filter_type
        self.bands: list[BandDef] = []
        self.subbands: list[np.ndarray] = []    # filtered signal per band
        self._source_signal: np.ndarray | None = None

    @staticmethod
    def bands_from_crossovers(crossovers: list[float], sr: int) -> list[BandDef]:
        """Build band definitions from sorted crossover frequencies."""
        xo = sorted(set(crossovers))
        bands: list[BandDef] = []
        prev = 0.0
        for freq in xo:
            bands.append(BandDef(fmin=prev, fmax=freq))
            prev = freq
        bands.append(BandDef(fmin=prev, fmax=float(sr / 2)))
        for b in bands:
            b.label = b.auto_label(sr)
        return bands

    def compute(self, signal: np.ndarray, bands: list[BandDef]) -> None:
        """Apply crossover filters and store subbands."""
        from scipy.signal import butter, sosfilt

        self._source_signal = signal.copy()
        self.bands = list(bands)
        self.subbands = []
        nyq = self.sr / 2.0

        if self.filter_type.startswith("Linkwitz-Riley"):
            order = int(self.filter_type.split()[-1])
            half_order = order // 2
            self._compute_lr(signal, nyq, half_order)
        else:
            order = int(self.filter_type.split()[-1])
            self._compute_butterworth(signal, nyq, order)

    def _compute_lr(self, signal: np.ndarray, nyq: float,
                    half_order: int) -> None:
        """Linkwitz-Riley crossover: cascade two Butterworth filters of
        half the order, then square the response.  Adjacent LP+HP sum flat."""
        from scipy.signal import butter, sosfilt

        residual = signal.copy()
        for band in self.bands:
            if not band.enabled:
                self.subbands.append(np.zeros_like(signal))
                continue
            flo = band.fmin
            fhi = band.fmax
            if flo <= 0.0 and fhi >= nyq:
                # Full band
                self.subbands.append(residual.copy())
                continue

            if flo <= 0.0:
                # Lowpass
                wn = min(fhi / nyq, 0.9999)
                sos = butter(half_order, wn, btype="low", output="sos")
                filtered = sosfilt(sos, sosfilt(sos, signal))
            elif fhi >= nyq:
                # Highpass
                wn = max(flo / nyq, 0.0001)
                sos = butter(half_order, wn, btype="high", output="sos")
                filtered = sosfilt(sos, sosfilt(sos, signal))
            else:
                # Bandpass = LP(fhi) cascaded twice, minus LP(flo) cascaded twice
                wn_lo = max(flo / nyq, 0.0001)
                wn_hi = min(fhi / nyq, 0.9999)
                sos_lo = butter(half_order, wn_lo, btype="high", output="sos")
                sos_hi = butter(half_order, wn_hi, btype="low", output="sos")
                filtered = sosfilt(sos_hi, sosfilt(sos_hi,
                            sosfilt(sos_lo, sosfilt(sos_lo, signal))))
            self.subbands.append(filtered.astype(np.float32))

    def _compute_butterworth(self, signal: np.ndarray, nyq: float,
                             order: int) -> None:
        """Standard Butterworth crossover (not guaranteed flat sum)."""
        from scipy.signal import butter, sosfilt

        for band in self.bands:
            if not band.enabled:
                self.subbands.append(np.zeros_like(signal))
                continue
            flo = band.fmin
            fhi = band.fmax
            if flo <= 0.0 and fhi >= nyq:
                self.subbands.append(signal.copy())
                continue
            if flo <= 0.0:
                wn = min(fhi / nyq, 0.9999)
                sos = butter(order, wn, btype="low", output="sos")
            elif fhi >= nyq:
                wn = max(flo / nyq, 0.0001)
                sos = butter(order, wn, btype="high", output="sos")
            else:
                wn = [max(flo / nyq, 0.0001), min(fhi / nyq, 0.9999)]
                sos = butter(order, wn, btype="band", output="sos")
            self.subbands.append(sosfilt(sos, signal).astype(np.float32))

    def reconstruction_error(self) -> float:
        """RMS error between sum-of-bands and original signal (0.0 = perfect)."""
        if self._source_signal is None or not self.subbands:
            return float("inf")
        recon = sum(self.subbands)
        err = self._source_signal - recon
        return float(np.sqrt(np.mean(err ** 2)))

    def reconstruct(self, band_mask: list[bool] | None = None) -> np.ndarray:
        """Sum subbands to reconstruct time-domain audio.

        *band_mask* selects which bands to include (all if *None*).
        With LR crossovers the sum is exact (perfect reconstruction).
        """
        if not self.subbands:
            return np.zeros(0, dtype=np.float32)
        if band_mask is None:
            band_mask = [b.enabled for b in self.bands]
        out = np.zeros_like(self.subbands[0])
        for include, sub in zip(band_mask, self.subbands):
            if include:
                out += sub
        return out

    # ------------------------------------------------------------------
    # CQT ↔ filterbank band mapping
    # ------------------------------------------------------------------
    @staticmethod
    def bins_for_band(band: "BandDef",
                      cqt_freqs: np.ndarray) -> np.ndarray:
        """Return indices of CQT bins whose center frequencies fall within
        *band*'s [fmin, fmax) range.

        Each CQT bin is a bandpass filter centered at ``cqt_freqs[k]`` with
        bandwidth ``freq[k] / Q``.  A bin belongs to a filterbank band when
        its center frequency sits inside that band.  This gives a 1:1
        mapping between the two representations.
        """
        mask = np.ones(len(cqt_freqs), dtype=bool)
        if band.fmin > 0:
            mask &= cqt_freqs >= band.fmin
        if band.fmax < cqt_freqs[-1] * 1.01:  # allow tiny overshoot at top
            mask &= cqt_freqs < band.fmax
        return np.where(mask)[0]

    @staticmethod
    def band_map(bands: list["BandDef"],
                 cqt_freqs: np.ndarray) -> list[np.ndarray]:
        """Build a complete mapping: for each band, the array of CQT bin
        indices that belong to it.

        ``sum(len(m) for m in result)`` will equal ``len(cqt_freqs)`` when
        the bands tile the full frequency axis without gaps.
        """
        return [FilterBankDecomposition.bins_for_band(b, cqt_freqs)
                for b in bands]

    def save(self, analysis_dir: str, sr: int) -> None:
        """Save subbands into the analysis folder as WAV files + metadata."""
        fb_dir = os.path.join(analysis_dir, "filterbank")
        os.makedirs(fb_dir, exist_ok=True)
        meta = {
            "filter_type": self.filter_type,
            "sr": sr,
            "n_bands": len(self.bands),
            "bands": [],
        }
        for i, (band, sub) in enumerate(zip(self.bands, self.subbands)):
            fname = f"band_{i:02d}.wav"
            fpath = os.path.join(fb_dir, fname)
            # Scale float32 to int16
            peak = np.abs(sub).max()
            if peak > 0:
                scaled = (sub / peak * 32767).astype(np.int16)
            else:
                scaled = np.zeros(len(sub), dtype=np.int16)
            wavfile.write(fpath, sr, scaled)
            meta["bands"].append({
                "index": i,
                "fmin": band.fmin,
                "fmax": band.fmax,
                "label": band.label,
                "file": fname,
                "peak_amplitude": float(peak),
            })
        import json
        with open(os.path.join(fb_dir, "filterbank_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def load(analysis_dir: str) -> "FilterBankDecomposition | None":
        """Load a previously saved filter bank decomposition."""
        import json
        fb_dir = os.path.join(analysis_dir, "filterbank")
        meta_path = os.path.join(fb_dir, "filterbank_meta.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        fb = FilterBankDecomposition(meta["sr"], meta.get("filter_type", "Linkwitz-Riley 4"))
        for bm in meta["bands"]:
            fb.bands.append(BandDef(
                fmin=bm["fmin"], fmax=bm["fmax"], label=bm.get("label", "")))
            fpath = os.path.join(fb_dir, bm["file"])
            if os.path.isfile(fpath):
                sr_wav, data = wavfile.read(fpath)
                sub = data.astype(np.float32) / 32767.0 * bm.get("peak_amplitude", 1.0)
                fb.subbands.append(sub)
            else:
                fb.subbands.append(np.zeros(0, dtype=np.float32))
        return fb

    def graduate_band(self, band_idx: int, output_dir: str, sr: int) -> str:
        """Export a single subband as a standalone WAV file in output_dir.
        Returns the written file path."""
        band = self.bands[band_idx]
        sub = self.subbands[band_idx]
        safe_label = band.label.replace(" ", "_").replace("/", "-")
        fname = f"fb_{safe_label}.wav"
        fpath = os.path.join(output_dir, fname)
        peak = np.abs(sub).max()
        if peak > 0:
            scaled = (sub / peak * 32767).astype(np.int16)
        else:
            scaled = np.zeros(len(sub), dtype=np.int16)
        wavfile.write(fpath, sr, scaled)
        return fpath


def _build_field_sources(has_onset: bool) -> list[FieldSource]:
    """Build the list of data fields from the virtual texture layouts."""
    seen: set[str] = set()
    sources: list[FieldSource] = []
    for tex_name, layout in _TEXTURE_LAYOUTS.items():
        if tex_name == "onset" and not has_onset:
            continue
        for ch_idx, ch_key in enumerate("RGBA"):
            field_name = layout.get(ch_key)
            if field_name is None or field_name in seen:
                continue
            seen.add(field_name)
            sources.append(FieldSource(
                name=field_name,
                display_name=FIELD_DISPLAY.get(field_name, field_name),
                tex_name=tex_name,
                tex_channel=ch_idx,
            ))
    return sources


# ---------------------------------------------------------------------------
# Menu bar — top-of-window pull-down menu system
# ---------------------------------------------------------------------------

@dataclass
class MenuItem:
    """One item inside a pull-down menu.

    *label*
        Display text.  ``"-"`` creates a visual separator.
    *action*
        Callable invoked when clicked.  ``None`` for separators / parent menus.
    *shortcut*
        Optional keyboard hint string shown right-aligned (display only).
    *toggle_state*
        Callable returning ``bool`` for check-mark items, or ``None``.
    *enabled*
        Callable returning ``bool``, or a plain ``bool``.
    *children*
        Nested ``list[MenuItem]`` for a submenu.
    *radio_group*
        Optional string tag.  All items in the same menu that share the
        same *radio_group* value form a mutually-exclusive set: clicking
        one deactivates the others.  The active item shows a bullet
        (``\u25cf``) instead of a checkmark.  Use *toggle_state* to
        report which is currently selected.
    """
    label: str
    action: Any = None            # () -> None
    shortcut: str = ""
    toggle_state: Any = None      # () -> bool | None
    enabled: Any = True           # () -> bool | bool
    children: list["MenuItem"] | None = None
    radio_group: str | None = None

    @property
    def is_separator(self) -> bool:
        return self.label == "-"

    @property
    def is_enabled(self) -> bool:
        if callable(self.enabled):
            return self.enabled()
        return bool(self.enabled)

    @property
    def is_checked(self) -> bool | None:
        if self.toggle_state is None:
            return None
        return self.toggle_state()

    @property
    def has_submenu(self) -> bool:
        return bool(self.children)


@dataclass
class _MenuColumn:
    """Runtime state for one open pull-down column."""
    title: str
    items: list[MenuItem]
    rect: pygame.Rect                   # bounding rect in screen coords
    item_rects: list[pygame.Rect]       # per-item rects in screen coords
    hover_idx: int = -1


class MenuBar:
    """Top-of-window pull-down menu bar.

    Usage::

        menu = MenuBar()
        menu.add_menu("File", [
            MenuItem("Open…", action=self._open_file, shortcut="Ctrl+O"),
            MenuItem("-"),
            MenuItem("Quit", action=self._quit, shortcut="Esc"),
        ])
        menu.add_menu("View", [...])

    In the run loop, call ``menu.handle_event(event)`` **before** other
    handlers (it returns ``True`` when consumed).  Call ``menu.render()``
    to get a ``pygame.Surface`` for the bar, and ``menu.render_dropdown()``
    for any open column.  Both may return ``None``.

    Rendering contract
    ------------------
    * The bar surface spans the full window width and ``MenuBar.BAR_H`` px.
    * Dropdown surfaces are positioned absolutely in screen space — query
      ``menu.dropdown_screen_rect`` for GL quad placement.
    """

    BAR_H = 24
    ITEM_H = 24
    SEP_H = 9
    MIN_COL_W = 160
    SHORTCUT_PAD = 40

    # colours
    _BAR_BG = (38, 38, 38, 230)
    _BAR_HOVER = (60, 60, 80, 255)
    _BAR_TEXT = (200, 200, 200)
    _DD_BG = (34, 34, 42, 240)
    _DD_HOVER = (55, 64, 90, 255)
    _DD_TEXT = (220, 220, 220)
    _DD_DISABLED = (100, 100, 100)
    _DD_SHORTCUT = (130, 130, 150)
    _DD_SEP = (60, 60, 70)
    _DD_CHECK = (120, 200, 120)
    _DD_BORDER = (80, 80, 100)
    _DD_SUBMENU_ARROW = (160, 160, 180)

    def __init__(self) -> None:
        self._menus: list[tuple[str, list[MenuItem]]] = []
        self._title_rects: list[pygame.Rect] = []
        self._open_idx: int = -1               # which top-level is open
        self._columns: list[_MenuColumn] = []   # stack (supports submenu depth)
        self._bar_hover_idx: int = -1
        self._armed: bool = False               # True while bar is "sticky"
        self.font: pygame.font.Font | None = None
        self.visible: bool = True

    # ---- Public API -------------------------------------------------------

    def add_menu(self, title: str, items: list[MenuItem]) -> None:
        """Append a new top-level menu."""
        self._menus.append((title, items))

    def insert_menu(self, index: int, title: str,
                    items: list[MenuItem]) -> None:
        """Insert a top-level menu at *index*."""
        self._menus.insert(index, (title, items))

    def set_items(self, title: str, items: list[MenuItem]) -> None:
        """Replace the items of an existing top-level menu by *title*."""
        for i, (t, _) in enumerate(self._menus):
            if t == title:
                self._menus[i] = (title, items)
                return
        self.add_menu(title, items)

    @property
    def is_open(self) -> bool:
        return self._open_idx >= 0

    @property
    def bar_height(self) -> int:
        return self.BAR_H if self.visible else 0

    # ---- Rendering --------------------------------------------------------

    def _ensure_font(self) -> None:
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("consolas", 13)

    def _layout_titles(self, win_w: int) -> None:
        """Recompute title rects for the current window width."""
        self._ensure_font()
        x = 0
        rects: list[pygame.Rect] = []
        for title, _ in self._menus:
            tw = self.font.size(title)[0] + 20
            rects.append(pygame.Rect(x, 0, tw, self.BAR_H))
            x += tw
        self._title_rects = rects

    def render(self, win_w: int) -> pygame.Surface | None:
        """Return a surface for the menu bar (full window width)."""
        if not self.visible:
            return None
        self._ensure_font()
        self._layout_titles(win_w)
        surf = pygame.Surface((win_w, self.BAR_H), pygame.SRCALPHA)
        surf.fill(self._BAR_BG)

        for i, (title, _) in enumerate(self._menus):
            rect = self._title_rects[i]
            if i == self._open_idx or i == self._bar_hover_idx:
                pygame.draw.rect(surf, self._BAR_HOVER, rect)
            txt = self.font.render(title, True, self._BAR_TEXT)
            surf.blit(txt, (rect.x + 10, rect.y + (self.BAR_H - txt.get_height()) // 2))

        # subtle bottom border
        pygame.draw.line(surf, self._DD_BORDER, (0, self.BAR_H - 1),
                         (win_w, self.BAR_H - 1))
        return surf

    def render_dropdown(self) -> pygame.Surface | None:
        """Return a surface covering all open dropdown columns."""
        if not self._columns:
            return None
        # Compute bounding rect of all columns
        rects = [c.rect for c in self._columns]
        x0 = min(r.x for r in rects)
        y0 = min(r.y for r in rects)
        x1 = max(r.x + r.w for r in rects)
        y1 = max(r.y + r.h for r in rects)
        bw, bh = x1 - x0, y1 - y0
        surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        # Blit each column into the combined surface
        for col in self._columns:
            cs = self._render_column(col)
            surf.blit(cs, (col.rect.x - x0, col.rect.y - y0))
        self._dropdown_bounding = pygame.Rect(x0, y0, bw, bh)
        return surf

    @property
    def dropdown_screen_rect(self) -> pygame.Rect | None:
        """Screen-space bounding rect of all open dropdown columns."""
        if not self._columns:
            return None
        return getattr(self, '_dropdown_bounding', self._columns[-1].rect)

    def _render_column(self, col: _MenuColumn) -> pygame.Surface:
        self._ensure_font()
        r = col.rect
        surf = pygame.Surface((r.w, r.h), pygame.SRCALPHA)
        surf.fill(self._DD_BG)
        for i, item in enumerate(col.items):
            ir = col.item_rects[i]
            # item_rects are in screen coords; offset to surface-local
            ly = ir.y - r.y
            lx = 0
            local_r = pygame.Rect(lx, ly, ir.w, ir.h)

            if item.is_separator:
                mid = ly + self.SEP_H // 2
                pygame.draw.line(surf, self._DD_SEP, (8, mid), (r.w - 8, mid))
                continue

            enabled = item.is_enabled
            if i == col.hover_idx and enabled:
                pygame.draw.rect(surf, self._DD_HOVER, local_r)

            text_col = self._DD_TEXT if enabled else self._DD_DISABLED
            x_text = 8

            # check / radio mark for toggle items
            checked = item.is_checked
            if checked is not None:
                if item.radio_group is not None:
                    mark = "\u25cf " if checked else "   "
                else:
                    mark = "\u2713 " if checked else "   "
                msf = self.font.render(mark, True, self._DD_CHECK if checked else text_col)
                surf.blit(msf, (x_text, ly + (ir.h - msf.get_height()) // 2))
                x_text += msf.get_width()

            label_sf = self.font.render(item.label, True, text_col)
            surf.blit(label_sf, (x_text, ly + (ir.h - label_sf.get_height()) // 2))

            # shortcut hint (right-aligned)
            if item.shortcut:
                sc_sf = self.font.render(item.shortcut, True, self._DD_SHORTCUT)
                surf.blit(sc_sf, (r.w - sc_sf.get_width() - 10,
                                  ly + (ir.h - sc_sf.get_height()) // 2))

            # submenu arrow
            if item.has_submenu:
                arr = self.font.render("\u25b6", True, self._DD_SUBMENU_ARROW)
                surf.blit(arr, (r.w - arr.get_width() - 6,
                                ly + (ir.h - arr.get_height()) // 2))

        pygame.draw.rect(surf, self._DD_BORDER,
                         pygame.Rect(0, 0, r.w, r.h), 1)
        return surf

    # ---- Column geometry helpers ------------------------------------------

    def _build_column(self, items: list[MenuItem], anchor_x: int,
                      anchor_y: int, title: str = "") -> _MenuColumn:
        """Create a _MenuColumn positioned at *(anchor_x, anchor_y)*.

        Clamps the column so it stays within the current window bounds.
        """
        self._ensure_font()
        col_w = self.MIN_COL_W
        for item in items:
            if item.is_separator:
                continue
            lw = self.font.size(item.label)[0] + 24
            if item.is_checked is not None or item.radio_group is not None:
                lw += self.font.size("\u25cf ")[0]
            if item.shortcut:
                lw += self.font.size(item.shortcut)[0] + self.SHORTCUT_PAD
            if item.has_submenu:
                lw += 20
            col_w = max(col_w, lw)

        total_h = sum(self.SEP_H if it.is_separator else self.ITEM_H
                       for it in items)

        # --- clamp to window bounds ---
        try:
            win_w, win_h = pygame.display.get_surface().get_size()
        except Exception:
            win_w, win_h = 1920, 1080
        # Horizontal: keep within window
        if anchor_x + col_w > win_w:
            anchor_x = max(0, win_w - col_w)
        # Vertical: if it'd extend past the bottom, shift it up
        if anchor_y + total_h > win_h:
            anchor_y = max(0, win_h - total_h)

        y = anchor_y
        rects: list[pygame.Rect] = []
        for item in items:
            h = self.SEP_H if item.is_separator else self.ITEM_H
            rects.append(pygame.Rect(anchor_x, y, col_w, h))
            y += h

        col_rect = pygame.Rect(anchor_x, anchor_y, col_w, total_h)
        return _MenuColumn(title=title, items=items, rect=col_rect,
                           item_rects=rects)

    def _open_top_menu(self, idx: int) -> None:
        """Open top-level menu *idx*, closing whatever was open."""
        self._close_all()
        if idx < 0 or idx >= len(self._menus):
            return
        title, items = self._menus[idx]
        r = self._title_rects[idx]
        col = self._build_column(items, r.x, self.BAR_H, title)
        self._columns = [col]
        self._open_idx = idx
        self._armed = True

    def _close_all(self) -> None:
        self._columns.clear()
        self._open_idx = -1
        self._armed = False

    # ---- Event handling ---------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Process *event*.  Returns ``True`` if consumed."""
        if not self.visible:
            return False

        if event.type == MOUSEMOTION:
            mx, my = event.pos
            # Track hover over bar titles
            self._bar_hover_idx = -1
            for i, r in enumerate(self._title_rects):
                if r.collidepoint(mx, my):
                    self._bar_hover_idx = i
                    # If another menu is already open, slide into this one
                    if self._armed and i != self._open_idx:
                        self._open_top_menu(i)
                    break

            # Track hover inside open columns (from deepest to shallowest)
            if self._columns:
                hit_any = False
                for depth in range(len(self._columns) - 1, -1, -1):
                    col = self._columns[depth]
                    if not col.rect.collidepoint(mx, my):
                        continue
                    hit_any = True
                    col.hover_idx = -1
                    for j, ir in enumerate(col.item_rects):
                        if ir.collidepoint(mx, my):
                            item = col.items[j]
                            if not item.is_separator:
                                col.hover_idx = j
                                # Auto-open submenu on hover
                                if item.has_submenu and item.is_enabled:
                                    self._open_submenu(j)
                            break
                    # If mouse is in a parent column, collapse any deeper
                    if depth < len(self._columns) - 1:
                        while len(self._columns) > depth + 1:
                            self._columns.pop()
                    break  # only process the deepest hit

                # If mouse is outside all columns and bar, clear deepest hover
                if not hit_any and my >= self.BAR_H:
                    self._columns[-1].hover_idx = -1

            # Consume motion only if over bar or open column
            if my < self.BAR_H:
                return True
            if self._columns:
                for c in self._columns:
                    if c.rect.collidepoint(mx, my):
                        return True
            return False

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Click on bar title
            for i, r in enumerate(self._title_rects):
                if r.collidepoint(mx, my):
                    if self._open_idx == i:
                        self._close_all()
                    else:
                        self._open_top_menu(i)
                    return True

            # Click inside open dropdown (check all columns, deepest first)
            if self._columns:
                for depth in range(len(self._columns) - 1, -1, -1):
                    col = self._columns[depth]
                    if not col.rect.collidepoint(mx, my):
                        continue
                    for j, ir in enumerate(col.item_rects):
                        if ir.collidepoint(mx, my):
                            item = col.items[j]
                            if item.is_separator or not item.is_enabled:
                                return True
                            if item.has_submenu:
                                self._open_submenu(j)
                                return True
                            # Fire action
                            if item.action:
                                self._close_all()
                                item.action()
                            else:
                                self._close_all()
                            return True
                    # Clicked inside column rect but not on any item
                    return True

            # Click outside — close everything
            if self._armed:
                self._close_all()
                return True

            # Click in bar region but not on a title — consume anyway
            if my < self.BAR_H:
                return True

        if event.type == MOUSEBUTTONUP and event.button == 1:
            mx, my = event.pos
            if my < self.BAR_H:
                return True
            if self._columns:
                for c in self._columns:
                    if c.rect.collidepoint(mx, my):
                        return True

        if event.type == MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if my < self.BAR_H:
                return True
            if self._columns:
                for c in self._columns:
                    if c.rect.collidepoint(mx, my):
                        return True

        if event.type == KEYDOWN and self._armed:
            key = event.key
            if key == K_ESCAPE:
                self._close_all()
                return True
            if key == K_LEFT:
                self._navigate_bar(-1)
                return True
            if key == K_RIGHT:
                self._navigate_bar(1)
                return True
            if self._columns:
                col = self._columns[-1]
                if key == pygame.K_UP:
                    col.hover_idx = self._next_enabled(col, col.hover_idx, -1)
                    return True
                if key == pygame.K_DOWN:
                    col.hover_idx = self._next_enabled(col, col.hover_idx, 1)
                    return True
                if key == pygame.K_RETURN:
                    if 0 <= col.hover_idx < len(col.items):
                        item = col.items[col.hover_idx]
                        if item.has_submenu and item.is_enabled:
                            self._open_submenu(col.hover_idx)
                        elif item.action and item.is_enabled:
                            self._close_all()
                            item.action()
                    return True

        return False

    # ---- Keyboard navigation helpers --------------------------------------

    def _navigate_bar(self, direction: int) -> None:
        n = len(self._menus)
        if n == 0:
            return
        idx = (self._open_idx + direction) % n
        self._open_top_menu(idx)

    @staticmethod
    def _next_enabled(col: _MenuColumn, current: int,
                      direction: int) -> int:
        n = len(col.items)
        if n == 0:
            return -1
        idx = current
        for _ in range(n):
            idx = (idx + direction) % n
            item = col.items[idx]
            if not item.is_separator and item.is_enabled:
                return idx
        return current

    def _open_submenu(self, parent_idx: int) -> None:
        col = self._columns[-1]
        item = col.items[parent_idx]
        if not item.children:
            return
        # Avoid re-opening the same submenu
        if (len(self._columns) > 1
                and self._columns[-1].title == item.label):
            return
        # Pop any existing deeper submenu
        while len(self._columns) > 1:
            popped = self._columns[-1]
            # keep current level
            if popped is col:
                break
            self._columns.pop()
        ir = col.item_rects[parent_idx]
        sub = self._build_column(item.children, ir.right, ir.y, item.label)
        self._columns.append(sub)


# ---------------------------------------------------------------------------
# Scrollable item list — reusable scrolling row list with selection
# ---------------------------------------------------------------------------

_TGT_COLORS: dict[str, tuple[int, int, int]] = {
    "R": (255, 80, 80), "G": (80, 255, 80),
    "B": (80, 80, 255), "A": (200, 200, 200),
    "W": (255, 255, 255), "C": (80, 255, 255),
    "M": (255, 80, 255), "Y": (255, 255, 80),
    "none": (80, 80, 80),
}


@dataclass
class ListItem:
    """One row in a :class:`ScrollableItemList`."""
    key: str              # unique id
    display_name: str     # shown label
    tag: str = "none"     # short right-side tag text (target channel, etc.)
    tag_color_key: str = "none"   # lookup in _TGT_COLORS


class ScrollableItemList:
    """Generic scrollable list widget rendered onto a parent surface.

    Supports arbitrary item count, scroll offset, single selection, and
    per-item target-colour tag.  Designed to be embedded inside a larger
    panel surface — all coordinates are surface-local.

    The ``▲ more`` / ``▼ more`` indicators are clickable buttons that
    scroll by one page.  Mouse-wheel scrolling works anywhere in the list.
    """

    ROW_H = 22
    MORE_H = 18          # height of a clickable more-indicator row
    PAD = 6

    def __init__(self, title: str, max_visible: int = 8) -> None:
        self.title = title
        self.items: list[ListItem] = []
        self.selected_idx: int = 0
        self.scroll_offset: int = 0
        self.max_visible = max_visible
        self._row_rects: list[pygame.Rect] = []  # surface-local rects
        self._more_up_rect: pygame.Rect | None = None
        self._more_down_rect: pygame.Rect | None = None
        self._rendered_h: int = 0                 # height consumed in last render

    @property
    def needs_scroll(self) -> bool:
        return len(self.items) > self.max_visible

    @property
    def visible_count(self) -> int:
        return min(len(self.items), self.max_visible)

    def clamp(self) -> None:
        """Ensure selection and scroll offset are within bounds."""
        n = len(self.items)
        if n == 0:
            self.selected_idx = 0
            self.scroll_offset = 0
            return
        self.selected_idx = max(0, min(self.selected_idx, n - 1))
        max_off = max(0, n - self.max_visible)
        self.scroll_offset = max(0, min(self.scroll_offset, max_off))
        # Ensure selection is visible
        if self.selected_idx < self.scroll_offset:
            self.scroll_offset = self.selected_idx
        if self.selected_idx >= self.scroll_offset + self.max_visible:
            self.scroll_offset = self.selected_idx - self.max_visible + 1

    def render(self, surf: pygame.Surface, font: pygame.font.Font,
               x: int, y: int, w: int) -> int:
        """Draw the list onto *surf* starting at *(x, y)*.

        Returns the total height consumed (including title).
        """
        self._row_rects = []
        self._more_up_rect = None
        self._more_down_rect = None
        pad = self.PAD
        start_y = y

        # Title
        ttl = font.render(f"\u2500\u2500 {self.title} \u2500\u2500",
                           True, (180, 180, 180))
        surf.blit(ttl, (pad, y + 2))
        y += self.ROW_H + 2

        if not self.items:
            dim = font.render("(empty)", True, (100, 100, 100))
            surf.blit(dim, (pad + 4, y + 3))
            y += self.ROW_H + 2
            self._rendered_h = y - start_y
            return self._rendered_h

        self.clamp()
        vis_start = self.scroll_offset
        vis_end = min(len(self.items), vis_start + self.max_visible)

        # Scroll-up button
        if vis_start > 0:
            btn = pygame.Rect(pad, y, w - 2 * pad, self.MORE_H)
            pygame.draw.rect(surf, (40, 40, 55), btn)
            pygame.draw.rect(surf, (60, 60, 80), btn, 1)
            arr = font.render(f"\u25b2 more ({vis_start})", True, (140, 140, 170))
            surf.blit(arr, (w // 2 - arr.get_width() // 2, y + 1))
            self._more_up_rect = btn
            y += self.MORE_H + 2

        for idx in range(vis_start, vis_end):
            item = self.items[idx]
            rect = pygame.Rect(pad, y, w - 2 * pad, self.ROW_H)
            if idx == self.selected_idx:
                pygame.draw.rect(surf, (50, 50, 70), rect)
            pygame.draw.rect(surf, (60, 60, 80), rect, 1)
            txt = font.render(item.display_name, True, (200, 200, 200))
            surf.blit(txt, (pad + 4, y + 3))
            color = _TGT_COLORS.get(item.tag_color_key, (80, 80, 80))
            arrow = font.render(f"\u2192 {item.tag}", True, color)
            surf.blit(arrow, (w - 60, y + 3))
            self._row_rects.append(rect)
            y += self.ROW_H + 2

        # Scroll-down button
        remaining = len(self.items) - vis_end
        if remaining > 0:
            btn = pygame.Rect(pad, y, w - 2 * pad, self.MORE_H)
            pygame.draw.rect(surf, (40, 40, 55), btn)
            pygame.draw.rect(surf, (60, 60, 80), btn, 1)
            arr = font.render(f"\u25bc more ({remaining})", True, (140, 140, 170))
            surf.blit(arr, (w // 2 - arr.get_width() // 2, y + 1))
            self._more_down_rect = btn
            y += self.MORE_H + 2

        self._rendered_h = y - start_y
        return self._rendered_h

    def handle_click(self, lx: int, ly: int) -> int | None:
        """Test click at surface-local *(lx, ly)*.

        Returns the **absolute** item index if a row was hit, else ``None``.
        Clicking the ``▲ more`` / ``▼ more`` buttons scrolls by one page
        and returns ``-1`` (consumed but no selection change).
        """
        # More-up button
        if self._more_up_rect and self._more_up_rect.collidepoint(lx, ly):
            self.scroll_offset = max(0, self.scroll_offset - self.max_visible)
            return -1
        # More-down button
        if self._more_down_rect and self._more_down_rect.collidepoint(lx, ly):
            max_off = max(0, len(self.items) - self.max_visible)
            self.scroll_offset = min(max_off,
                                     self.scroll_offset + self.max_visible)
            return -1
        # Row items
        for vis_i, rect in enumerate(self._row_rects):
            if rect.collidepoint(lx, ly):
                idx = self.scroll_offset + vis_i
                self.selected_idx = idx
                return idx
        return None

    def handle_scroll(self, direction: int) -> bool:
        """Scroll by *direction* rows (-1 = up, +1 = down).

        Returns True if the list consumed the event.
        """
        if not self.needs_scroll:
            return False
        self.scroll_offset = max(
            0, min(self.scroll_offset + direction,
                   len(self.items) - self.max_visible))
        return True


# ---------------------------------------------------------------------------
# Synthesis product — one output from the synthesizer
# ---------------------------------------------------------------------------

PHASE_MODES = ["original", "griffin_lim", "zero_phase"]
PHASE_LABELS = {
    "original":    "Original Phase",
    "griffin_lim":  "Griffin-Lim",
    "zero_phase":  "Zero Phase",
}


@dataclass
class SynthJob:
    """One synthesis job in the async queue."""
    job_id: int
    label: str
    status: str = "queued"          # queued | running | done | error
    source_type: str = "cqt"       # cqt | filterbank | mixed
    params: dict = field(default_factory=dict)
    result_stereo: np.ndarray | None = None   # (N, 2) int16
    duration: float = 0.0
    view_t0: float = 0.0
    view_t1: float = 0.0
    error_msg: str = ""
    submitted_at: float = 0.0
    completed_at: float = 0.0


# ---------------------------------------------------------------------------
# Panel base class — common interface for dockable side panels
# ---------------------------------------------------------------------------

class Panel:
    """Abstract base for a dockable side panel.

    Subclasses must implement :meth:`render` and :meth:`handle_event`.
    """

    PANEL_W = 280
    _top_offset: int = 0   # set by the viewer (e.g. menu-bar height)

    def __init__(self, *, title: str = "Panel",
                 side: str = "right") -> None:
        self.title = title
        self.side: str = side          # "left" or "right"
        self.visible: bool = True
        self.font: pygame.font.Font | None = None

    def _ensure_font(self) -> None:
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("consolas", 13)

    @property
    def panel_rect(self) -> pygame.Rect:
        sw = pygame.display.get_surface().get_width()
        sh = pygame.display.get_surface().get_height()
        top = self._top_offset
        if self.side == "left":
            return pygame.Rect(0, top, self.PANEL_W, sh - top)
        return pygame.Rect(sw - self.PANEL_W, top, self.PANEL_W, sh - top)

    def render(self) -> pygame.Surface | None:
        raise NotImplementedError

    def handle_event(self, event: pygame.event.Event) -> bool:
        raise NotImplementedError

    def render_dropdown_overlay(self) -> pygame.Surface | None:
        return None


# ---------------------------------------------------------------------------
# Curve helpers for the plot widget
# ---------------------------------------------------------------------------

def _apply_curve(xs: np.ndarray, norm_mode: int,
                 gamma: float, scale: float) -> np.ndarray:
    """Replicate the GLSL applyNorm → pow(v,gamma)*scale pipeline on CPU."""
    if norm_mode == 1:  # dB
        safe = np.maximum(xs, 1e-12)
        db = 10.0 * np.log10(safe)
        normed = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
    else:  # linear (0) or rank_order (2) — passthrough
        normed = np.clip(xs, 0.0, 1.0)
    return np.clip(np.power(normed, gamma) * scale, 0.0, 1.0).astype(np.float32)


def _jitter_color(base: tuple[int, int, int],
                  idx: int) -> tuple[int, int, int]:
    """Shift brightness for the *idx*-th duplicate on the same target."""
    if idx == 0:
        return base
    shift = ((-1) ** idx) * (18 * ((idx + 1) // 2))
    return (max(0, min(255, base[0] + shift)),
            max(0, min(255, base[1] + shift)),
            max(0, min(255, base[2] + shift)))


# ---------------------------------------------------------------------------
# GUI Panel — per-field tab interface
# ---------------------------------------------------------------------------

class FieldTabPanel(Panel):
    """Panel with a tab per data field + global defaults + output controls.

    Contains two :class:`ScrollableItemList` sections:
    *  **Data Fields** (top) — the analysis products loaded from disk.
    *  **Synthesis** (bottom) — outputs produced by viewport synthesis.
    """

    PANEL_W = 280
    ROW_H = 22
    PAD = 6

    def __init__(self, fields: list[FieldSource],
                 configs: list[FieldConfig],
                 global_defaults: GlobalDefaults,
                 *, side: str = "right") -> None:
        super().__init__(title="Fields", side=side)
        self.fields = fields
        self.configs = configs
        self.global_defaults = global_defaults
        self.selected_idx = 0
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        self._dragging: str | None = None
        self._item_map: dict[str, tuple[pygame.Rect, Any]] = {}

        # Data-field item list (top section)
        self.data_list = ScrollableItemList("Data Fields", max_visible=10)
        self._sync_data_list()

        # Plot widget (bottom section — shows normalised data curves)
        self.plot = PlotWidget()
        self.plot.title = "Data Curves"
        self.plot.y_min = 0.0
        self.plot.y_max = 1.0
        self._plot_h: int = 140  # pixel height of the plot box

    def _sync_data_list(self) -> None:
        """Rebuild data_list items from fields/configs."""
        items: list[ListItem] = []
        for i, fs in enumerate(self.fields):
            cfg = self.configs[i]
            tgt_text = TARGET_LABELS[TARGET_VALUES.index(cfg.target)]
            items.append(ListItem(
                key=fs.name,
                display_name=fs.display_name,
                tag=tgt_text,
                tag_color_key=tgt_text,
            ))
        self.data_list.items = items
        self.data_list.selected_idx = self.selected_idx

    _CURVE_XS = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    def _rebuild_plot_series(self) -> None:
        """Recompute transfer-curve series from current field configs."""
        xs = self._CURVE_XS
        gd = self.global_defaults
        target_seen: dict[int, int] = {}  # target → count seen so far
        series: list[PlotSeries] = []

        for i, (fs, cfg) in enumerate(zip(self.fields, self.configs)):
            if cfg.target < 0:
                continue
            norm_mode = cfg.norm_mode if not cfg.use_global else gd.norm_mode
            gamma = cfg.gamma if not cfg.use_global else gd.gamma
            scale = cfg.scale if not cfg.use_global else gd.scale

            ys = _apply_curve(xs, norm_mode, gamma, scale)

            tgt_label = TARGET_LABELS[TARGET_VALUES.index(cfg.target)]
            base_color = _TGT_COLORS.get(tgt_label, (180, 180, 180))
            n = target_seen.get(cfg.target, 0)
            target_seen[cfg.target] = n + 1
            color = _jitter_color(base_color, n)

            series.append(PlotSeries(
                key=fs.name,
                label=fs.display_name,
                color=color,
                line=True,
                data_x=xs,
                data_y=ys,
            ))

        self.plot.series = series

    # ---- Rendering --------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._item_map = {}

        # Keep list selections in sync
        self.data_list.selected_idx = self.selected_idx
        self._sync_data_list()

        # We render into a list of sections, summing height as we go,
        # then blit everything onto the final surface.

        # --- Upper half: data field list + controls ---
        upper_rows: list[tuple[str, Any]] = []

        # Selected field controls
        sel = self.selected_idx
        sel_fs = self.fields[sel]
        sel_cfg = self.configs[sel]
        upper_rows.append(("label", f"\u2500\u2500 {sel_fs.display_name} \u2500\u2500"))

        tgt_label = TARGET_LABELS[TARGET_VALUES.index(sel_cfg.target)]
        upper_rows.append(("dropdown", ("f_target", tgt_label, TARGET_LABELS)))
        upper_rows.append(("toggle", ("f_custom", "Custom", not sel_cfg.use_global)))

        gd = self.global_defaults
        eff_norm = NORM_MODES[sel_cfg.norm_mode if not sel_cfg.use_global
                              else gd.norm_mode]
        eff_gamma = sel_cfg.gamma if not sel_cfg.use_global else gd.gamma
        eff_scale = sel_cfg.scale if not sel_cfg.use_global else gd.scale

        upper_rows.append(("dropdown", ("f_norm", eff_norm, NORM_MODES)))
        upper_rows.append(("slider", ("f_gamma", eff_gamma, 0.1, 5.0, "{:.2f}")))
        upper_rows.append(("slider", ("f_scale", eff_scale, 0.01, 10.0, "{:.2f}")))

        upper_rows.append(("label", ""))

        # Global defaults
        upper_rows.append(("label", "\u2500\u2500 Global Defaults \u2500\u2500"))
        upper_rows.append(("dropdown", ("g_norm",
                                         NORM_MODES[gd.norm_mode], NORM_MODES)))
        upper_rows.append(("slider", ("g_gamma", gd.gamma, 0.1, 5.0, "{:.2f}")))
        upper_rows.append(("slider", ("g_scale", gd.scale, 0.01, 10.0, "{:.2f}")))

        upper_rows.append(("label", ""))

        # Output post-aggregation
        upper_rows.append(("label", "\u2500\u2500 Output \u2500\u2500"))
        upper_rows.append(("slider", ("o_gamma", gd.out_gamma, 0.1, 5.0, "{:.2f}")))
        upper_rows.append(("slider", ("o_scale", gd.out_scale, 0.01, 10.0, "{:.2f}")))

        # --- Pre-compute total height ---
        # Data list
        data_list_h = self.data_list.ROW_H + 2  # title
        if self.data_list.items:
            data_list_h += self.data_list.visible_count * (self.ROW_H + 2)
            if self.data_list.scroll_offset > 0:
                data_list_h += self.data_list.MORE_H + 2
            vis_end = self.data_list.scroll_offset + self.data_list.max_visible
            if vis_end < len(self.data_list.items):
                data_list_h += self.data_list.MORE_H + 2
        else:
            data_list_h += self.ROW_H + 2

        upper_controls_h = sum(self.ROW_H + 2 for _ in upper_rows)

        plot_h = self._plot_h

        total_h = self.PAD + data_list_h + upper_controls_h + plot_h + self.PAD

        surf = pygame.Surface((w, total_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))

        y = self.PAD
        field_uses_global = sel_cfg.use_global

        # --- Render data field list ---
        data_h = self.data_list.render(surf, font, 0, y, w)
        y += data_h

        # --- Render field controls ---
        for rtype, rdata in upper_rows:
            if rtype == "label":
                txt = font.render(rdata, True, (180, 180, 180))
                surf.blit(txt, (self.PAD, y + 2))

            elif rtype == "dropdown":
                key, current, opts = rdata
                greyed = (key == "f_norm" and field_uses_global)
                prefix = {"f_target": "Tgt:", "f_norm": "Nrm:",
                          "g_norm": "Nrm:"}.get(key, key[:4])
                lbl_col = (100, 100, 100) if greyed else (160, 160, 160)
                lbl = font.render(prefix, True, lbl_col)
                surf.blit(lbl, (self.PAD, y + 2))
                btn_x = 60
                btn_w = w - btn_x - self.PAD
                btn_rect = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
                if greyed:
                    bg = (40, 40, 50)
                elif self._active_dropdown == key:
                    bg = (80, 80, 120)
                else:
                    bg = (60, 60, 80)
                pygame.draw.rect(surf, bg, btn_rect)
                pygame.draw.rect(surf, (100, 100, 120), btn_rect, 1)
                tc = (120, 120, 120) if greyed else (220, 220, 220)
                txt = font.render(current, True, tc)
                surf.blit(txt, (btn_x + 4, y + 3))
                if not greyed:
                    self._item_map[key] = (btn_rect, opts)

            elif rtype == "toggle":
                key, label_text, checked = rdata
                rect = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
                box = pygame.Rect(self.PAD + 2, y + 4, 14, 14)
                pygame.draw.rect(surf, (80, 80, 100), box, 1)
                if checked:
                    pygame.draw.line(surf, (120, 200, 120),
                                    (box.x + 2, box.y + 7),
                                    (box.x + 5, box.y + 11), 2)
                    pygame.draw.line(surf, (120, 200, 120),
                                    (box.x + 5, box.y + 11),
                                    (box.x + 12, box.y + 2), 2)
                txt = font.render(label_text, True, (200, 200, 200))
                surf.blit(txt, (self.PAD + 22, y + 3))
                self._item_map[key] = (rect, None)

            elif rtype == "slider":
                key, val, vmin, vmax, fmt = rdata
                greyed = (key.startswith("f_") and field_uses_global)
                BTN_W = 16
                lbl_end = self.PAD
                val_space = 50

                # Minus button
                minus_r = pygame.Rect(lbl_end, y + 2, BTN_W, self.ROW_H - 4)
                mc = (35, 35, 45) if greyed else (50, 50, 65)
                pygame.draw.rect(surf, mc, minus_r)
                pygame.draw.rect(surf, (80, 80, 100), minus_r, 1)
                mt = font.render("\u2212", True,
                                 (80, 80, 90) if greyed else (180, 180, 200))
                surf.blit(mt, (minus_r.x + (BTN_W - mt.get_width()) // 2,
                               y + 2))

                # Track
                track_x = lbl_end + BTN_W + 2
                track_end = w - self.PAD - val_space - BTN_W - 2
                track_w = max(track_end - track_x, 10)
                tc = (35, 35, 40) if greyed else (50, 50, 60)
                pygame.draw.rect(surf, tc,
                                 pygame.Rect(track_x, y + 8, track_w, 6))
                frac = (val - vmin) / max(vmax - vmin, 1e-9)
                frac = max(0.0, min(1.0, frac))
                thumb_x = track_x + int(frac * track_w)
                thc = (70, 80, 100) if greyed else (120, 140, 200)
                pygame.draw.rect(surf, thc,
                                 pygame.Rect(thumb_x - 5, y + 3, 10, 16))

                # Plus button
                plus_r = pygame.Rect(track_end + 2, y + 2, BTN_W,
                                     self.ROW_H - 4)
                pc = (35, 35, 45) if greyed else (50, 50, 65)
                pygame.draw.rect(surf, pc, plus_r)
                pygame.draw.rect(surf, (80, 80, 100), plus_r, 1)
                pt = font.render("+", True,
                                 (80, 80, 90) if greyed else (180, 180, 200))
                surf.blit(pt, (plus_r.x + (BTN_W - pt.get_width()) // 2,
                               y + 2))

                # Value label
                vc = (100, 100, 100) if greyed else (200, 200, 200)
                vtxt = font.render(fmt.format(val), True, vc)
                surf.blit(vtxt, (plus_r.x + BTN_W + 4, y + 2))

                if not greyed:
                    self._item_map[f"{key}_minus"] = (minus_r, None)
                    self._item_map[f"{key}_plus"] = (plus_r, None)
                    self._item_map[key] = (
                        pygame.Rect(track_x, y, track_w, self.ROW_H),
                        (vmin, vmax))

            y += self.ROW_H + 2

        # --- Rebuild & render plot widget ---
        self._rebuild_plot_series()
        self.plot.render(surf, self.PAD, y, w - 2 * self.PAD, self._plot_h,
                         font=font)

        return surf

    # ---- Event handling ---------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False

        pr = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not pr.collidepoint(mx, my):
                if self._active_dropdown:
                    self._active_dropdown = None
                return False

            lx, ly = mx - pr.x, my - pr.y

            # Open dropdown item selection
            if self._active_dropdown and self._dropdown_rect.collidepoint(lx, ly):
                for i, ir in enumerate(self._dropdown_item_rects):
                    if ir.collidepoint(lx, ly):
                        self._select_dropdown(self._active_dropdown,
                                              self._dropdown_opts[i])
                        break
                self._active_dropdown = None
                return True

            if self._active_dropdown:
                self._active_dropdown = None

            # Data field rows
            hit = self.data_list.handle_click(lx, ly)
            if hit is not None:
                if hit >= 0:
                    self.selected_idx = hit
                return True

            # Toggle
            if "f_custom" in self._item_map:
                rect, _ = self._item_map["f_custom"]
                if rect.collidepoint(lx, ly):
                    cfg = self.configs[self.selected_idx]
                    if cfg.use_global:
                        gd = self.global_defaults
                        cfg.norm_mode = gd.norm_mode
                        cfg.gamma = gd.gamma
                        cfg.scale = gd.scale
                    cfg.use_global = not cfg.use_global
                    return True

            # Dropdowns
            for key in ("f_target", "f_norm", "g_norm"):
                if key in self._item_map:
                    rect, opts = self._item_map[key]
                    if rect.collidepoint(lx, ly):
                        self._open_dropdown(key, opts, rect, pr)
                        return True

            # Slider +/- buttons
            for key in ("f_gamma", "f_scale",
                        "g_gamma", "g_scale",
                        "o_gamma", "o_scale"):
                minus_k = f"{key}_minus"
                plus_k = f"{key}_plus"
                if minus_k in self._item_map:
                    rect, _ = self._item_map[minus_k]
                    if rect.collidepoint(lx, ly):
                        self._nudge_slider(key, -1)
                        return True
                if plus_k in self._item_map:
                    rect, _ = self._item_map[plus_k]
                    if rect.collidepoint(lx, ly):
                        self._nudge_slider(key, +1)
                        return True

            # Slider track drag
            for key in ("f_gamma", "f_scale",
                        "g_gamma", "g_scale",
                        "o_gamma", "o_scale"):
                if key in self._item_map:
                    rect, (vmin, vmax) = self._item_map[key]
                    if rect.collidepoint(lx, ly):
                        self._dragging = key
                        self._update_slider(key, lx, rect, vmin, vmax)
                        return True

            return True  # consumed (in panel area)

        if event.type == MOUSEBUTTONUP and event.button == 1:
            if self._dragging:
                self._dragging = None
                return True
            mx, my = event.pos
            if pr.collidepoint(mx, my):
                return True

        if event.type == MOUSEMOTION:
            if self._dragging:
                if not event.buttons[0]:
                    self._dragging = None
                else:
                    mx, my = event.pos
                    lx = mx - pr.x
                    key = self._dragging
                    if key in self._item_map:
                        rect, (vmin, vmax) = self._item_map[key]
                        self._update_slider(key, lx, rect, vmin, vmax)
                return True
            mx, my = event.pos
            if pr.collidepoint(mx, my):
                return True

        if event.type == MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if pr.collidepoint(mx, my):
                self.data_list.handle_scroll(-event.y)
                return True

        return False

    def _open_dropdown(self, key: str, opts: list[str],
                       btn_rect: pygame.Rect,
                       panel_rect: pygame.Rect) -> None:
        self._active_dropdown = key
        self._dropdown_opts = opts
        x = btn_rect.x
        y = btn_rect.y + btn_rect.h
        item_h = self.ROW_H
        total_h = item_h * len(opts)
        self._dropdown_rect = pygame.Rect(x, y, btn_rect.w, total_h)
        self._dropdown_item_rects = [
            pygame.Rect(x, y + i * item_h, btn_rect.w, item_h)
            for i in range(len(opts))
        ]

    def _select_dropdown(self, key: str, value: str) -> None:
        if key == "f_target":
            self.configs[self.selected_idx].target = \
                TARGET_VALUES[TARGET_LABELS.index(value)]
        elif key == "f_norm":
            self.configs[self.selected_idx].norm_mode = \
                NORM_MODES.index(value)
        elif key == "g_norm":
            self.global_defaults.norm_mode = NORM_MODES.index(value)

    def _update_slider(self, key: str, lx: int,
                       rect: pygame.Rect,
                       vmin: float, vmax: float) -> None:
        frac = max(0.0, min(1.0, (lx - rect.x) / rect.w))
        val = round(vmin + frac * (vmax - vmin), 2)
        self._apply_slider_value(key, val)

    def _apply_slider_value(self, key: str, val: float) -> None:
        cfg = self.configs[self.selected_idx]
        gd = self.global_defaults
        if key == "f_gamma":
            cfg.gamma = val
        elif key == "f_scale":
            cfg.scale = val
        elif key == "g_gamma":
            gd.gamma = val
        elif key == "g_scale":
            gd.scale = val
        elif key == "o_gamma":
            gd.out_gamma = val
        elif key == "o_scale":
            gd.out_scale = val

    def _nudge_slider(self, key: str, direction: int) -> None:
        """Move a slider by one logical step. direction: +1 or -1."""
        # Determine current value and range from _item_map
        if key not in self._item_map:
            return
        _, (vmin, vmax) = self._item_map[key]
        cfg = self.configs[self.selected_idx]
        gd = self.global_defaults
        current = {
            "f_gamma": cfg.gamma, "f_scale": cfg.scale,
            "g_gamma": gd.gamma, "g_scale": gd.scale,
            "o_gamma": gd.out_gamma, "o_scale": gd.out_scale,
        }.get(key, 0.0)
        step = (vmax - vmin) / 50.0
        val = round(max(vmin, min(vmax, current + direction * step)), 2)
        self._apply_slider_value(key, val)

    def render_dropdown_overlay(self) -> pygame.Surface | None:
        if not self._active_dropdown or not self._dropdown_opts:
            return None
        self._ensure_font()
        font = self.font
        rect = self._dropdown_rect
        surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        surf.fill((40, 40, 50, 240))
        for i, opt in enumerate(self._dropdown_opts):
            ir = self._dropdown_item_rects[i]
            iy = ir.y - rect.y
            txt = font.render(opt, True, (220, 220, 220))
            surf.blit(txt, (4, iy + 3))
            if i < len(self._dropdown_opts) - 1:
                pygame.draw.line(surf, (60, 60, 70),
                                 (0, iy + self.ROW_H - 1),
                                 (rect.w, iy + self.ROW_H - 1))
        pygame.draw.rect(surf, (100, 100, 130),
                         pygame.Rect(0, 0, rect.w, rect.h), 1)
        return surf


# ---------------------------------------------------------------------------
# Synthesis panel — synthesis products list + controls
# ---------------------------------------------------------------------------

_SYNTH_SOURCE_TYPES = ["cqt", "filterbank", "mixed"]
_SYNTH_SOURCE_LABELS = {"cqt": "CQT Viewport", "filterbank": "Filterbank",
                         "mixed": "Mixed (band iCQT)"}
_STATUS_COLORS = {
    "queued": (120, 120, 140),
    "running": (80, 180, 255),
    "done": (80, 220, 100),
    "error": (255, 80, 80),
}


class SynthesisPanel(Panel):
    """Async synthesis job queue panel.

    Holds a list of :class:`SynthJob` entries.  Jobs are executed on a
    background thread; completed jobs can be played back instantly.
    """

    ROW_H = 22
    BTN_W = 40
    PAD = 6

    def __init__(self, *, side: str = "right") -> None:
        super().__init__(title="Synthesis", side=side)
        self.jobs: list[SynthJob] = []
        self.job_list = ScrollableItemList("Synthesis Jobs", max_visible=10)
        self._next_id: int = 1
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        # Callbacks set by the viewer
        self.on_play: Any = None          # (SynthJob) -> None
        self.on_stop: Any = None          # () -> None
        self._playing_job_id: int | None = None
        # Per-row button rects (surface-local)
        self._play_rects: dict[int, pygame.Rect] = {}
        self._del_rects: dict[int, pygame.Rect] = {}

    # -- job management ----------------------------------------------------

    def submit(self, job: SynthJob) -> None:
        job.job_id = self._next_id
        self._next_id += 1
        job.submitted_at = time.time()
        with self._lock:
            self.jobs.append(job)
        self._ensure_worker()

    def _ensure_worker(self) -> None:
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while True:
            job: SynthJob | None = None
            with self._lock:
                for j in self.jobs:
                    if j.status == "queued":
                        j.status = "running"
                        job = j
                        break
            if job is None:
                return  # nothing left — thread exits
            try:
                self._execute(job)
                job.status = "done"
                job.completed_at = time.time()
            except Exception as exc:
                job.status = "error"
                job.error_msg = str(exc)
                job.completed_at = time.time()

    def _execute(self, job: SynthJob) -> None:
        p = job.params
        synth: ViewportSynthPlayer = p["synth"]
        if job.source_type == "cqt":
            result = synth.synthesise_to_array(
                p["raw_arrays"], p["freqs"], p["times"],
                p["view_x0"], p["view_x1"],
                p["view_y0"], p["view_y1"],
                p["n_frames"], p["n_bins"],
                p["t_start"], p["t_dur"], p["sr"],
                p["is_stereo"], p.get("normalize", True))
            if result is None:
                raise ValueError("Empty viewport")
            stereo, dur, vt0, vt1 = result
            job.result_stereo = stereo
            job.duration = dur
            job.view_t0 = vt0
            job.view_t1 = vt1
        elif job.source_type == "mixed":
            cqt = p["cqt"]
            freqs = p["freqs"]
            n_total = p["n_total_bins"]
            band_bin_map = p["band_bin_map"]
            band_mask = p.get("band_mask")
            sr = p["sr"]
            sig = synth.recombine_bands_icqt(
                cqt, freqs, n_total, band_bin_map, band_mask)
            # Normalise to int16 stereo
            peak = max(np.abs(sig).max(), 1e-12)
            sig = sig / peak * 0.8
            i16 = (sig * 32767).clip(-32768, 32767).astype(np.int16)
            stereo = np.column_stack([i16, i16])
            dur = len(sig) / sr
            job.result_stereo = stereo
            job.duration = dur
        elif job.source_type == "filterbank":
            fb: FilterBankDecomposition = p["filterbank"]
            band_mask = p.get("band_mask")
            sr = p["sr"]
            sig = fb.reconstruct(band_mask).astype(np.float64)
            peak = max(np.abs(sig).max(), 1e-12)
            sig = sig / peak * 0.8
            i16 = (sig * 32767).clip(-32768, 32767).astype(np.int16)
            stereo = np.column_stack([i16, i16])
            dur = len(sig) / sr
            job.result_stereo = stereo
            job.duration = dur

    def remove_job(self, job_id: int) -> None:
        with self._lock:
            self.jobs = [j for j in self.jobs if j.job_id != job_id]

    # -- list sync ---------------------------------------------------------

    def _sync_list(self) -> None:
        items: list[ListItem] = []
        with self._lock:
            for j in self.jobs:
                tag = j.status
                color_key = {"done": "G", "running": "B",
                             "error": "R", "queued": "none"}.get(j.status, "none")
                items.append(ListItem(
                    key=str(j.job_id),
                    display_name=j.label,
                    tag=tag,
                    tag_color_key=color_key,
                ))
        self.job_list.items = items

    # -- render ------------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._sync_list()
        self._play_rects.clear()
        self._del_rects.clear()

        # Pre-compute height
        list_h = self.job_list.ROW_H + 2
        n_items = len(self.job_list.items)
        if n_items:
            list_h += self.job_list.visible_count * (self.ROW_H + 2)
            if self.job_list.scroll_offset > 0:
                list_h += self.job_list.MORE_H + 2
            vis_end = self.job_list.scroll_offset + self.job_list.max_visible
            if vis_end < n_items:
                list_h += self.job_list.MORE_H + 2
        else:
            list_h += self.ROW_H + 2

        # Extra row for per-job action buttons
        btn_row_h = 0
        sel_job = self._selected_job()
        if sel_job is not None:
            btn_row_h = self.ROW_H + 4

        total_h = self.PAD + list_h + btn_row_h + self.PAD
        surf = pygame.Surface((w, total_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))

        y_after_list = self.job_list.render(surf, font, 0, self.PAD, w)
        y_cur = self.PAD + y_after_list

        # Action buttons for selected job
        if sel_job is not None:
            pad = self.PAD
            bx = pad
            # Play / Stop button
            if sel_job.status == "done":
                is_playing = (self._playing_job_id == sel_job.job_id)
                play_label = "Stop" if is_playing else "Play"
                play_color = (200, 80, 80) if is_playing else (60, 160, 80)
                pr = pygame.Rect(bx, y_cur, self.BTN_W, self.ROW_H)
                pygame.draw.rect(surf, play_color, pr)
                pygame.draw.rect(surf, (200, 200, 200), pr, 1)
                lbl = font.render(play_label, True, (255, 255, 255))
                surf.blit(lbl, (bx + 4, y_cur + 3))
                self._play_rects[sel_job.job_id] = pr
                bx += self.BTN_W + 4
            # Duration label
            if sel_job.duration > 0:
                dur_txt = f"{sel_job.duration:.2f}s"
                dtl = font.render(dur_txt, True, (160, 160, 160))
                surf.blit(dtl, (bx, y_cur + 3))
                bx += dtl.get_width() + 8
            # Status / error
            if sel_job.status == "error":
                etl = font.render(sel_job.error_msg[:30], True, (255, 80, 80))
                surf.blit(etl, (bx, y_cur + 3))
            # Delete button (right-aligned)
            del_w = 26
            del_x = w - pad - del_w
            dr = pygame.Rect(del_x, y_cur, del_w, self.ROW_H)
            pygame.draw.rect(surf, (120, 40, 40), dr)
            pygame.draw.rect(surf, (200, 200, 200), dr, 1)
            xl = font.render("X", True, (255, 255, 255))
            surf.blit(xl, (del_x + 7, y_cur + 3))
            self._del_rects[sel_job.job_id] = dr

        return surf

    def _selected_job(self) -> SynthJob | None:
        idx = self.job_list.selected_idx
        with self._lock:
            if 0 <= idx < len(self.jobs):
                return self.jobs[idx]
        return None

    # -- events ------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        pr = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not pr.collidepoint(mx, my):
                return False
            lx, ly = mx - pr.x, my - pr.y

            # Check action buttons first
            for jid, rect in self._play_rects.items():
                if rect.collidepoint(lx, ly):
                    self._on_play_click(jid)
                    return True
            for jid, rect in self._del_rects.items():
                if rect.collidepoint(lx, ly):
                    self.remove_job(jid)
                    return True

            hit = self.job_list.handle_click(lx, ly)
            if hit is not None:
                return True
            return True

        if event.type == MOUSEBUTTONUP and event.button == 1:
            mx, my = event.pos
            if pr.collidepoint(mx, my):
                return True

        if event.type == MOUSEMOTION:
            mx, my = event.pos
            if pr.collidepoint(mx, my):
                return True

        if event.type == MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if pr.collidepoint(mx, my):
                self.job_list.handle_scroll(-event.y)
                return True

        return False

    def _on_play_click(self, job_id: int) -> None:
        if self._playing_job_id == job_id:
            # Stop
            self._playing_job_id = None
            if self.on_stop:
                self.on_stop()
            return
        with self._lock:
            job = next((j for j in self.jobs if j.job_id == job_id), None)
        if job is None or job.result_stereo is None:
            return
        self._playing_job_id = job_id
        if self.on_play:
            self.on_play(job)


# ---------------------------------------------------------------------------
# Source panel — WAV browser, analysis settings, filter bank, folder mgmt
# ---------------------------------------------------------------------------

_CHANNEL_MODES = ["Stereo", "L", "R", "Mono Mix"]
_ANALYSIS_MODES = ["CQT", "Filter Bank", "Both"]


class SourcePanel(Panel):
    """Dockable panel for WAV source management, analysis dispatch,
    filter bank configuration, and analysis folder browsing.

    Sections (top to bottom):
      1. WAV file list (from input/)
      2. Channel mode + analysis type selection
      3. CQT settings (collapsible)
      4. Filter bank crossover editor (collapsible)
      5. Resampling preferences
      6. Analyze / Apply buttons
      7. Analysis folder list + actions
    """

    PANEL_W = 280
    ROW_H = 22
    PAD = 6

    def __init__(self, *, side: str = "left",
                 input_dir: str = "",
                 output_root: str = "") -> None:
        super().__init__(title="Source", side=side)
        self.input_dir: str = input_dir or os.getcwd()
        self.output_root: str = output_root or self.input_dir

        # WAV file list
        self.wav_list = ScrollableItemList("WAV Files", max_visible=6)
        self._wav_paths: list[str] = []
        self._refresh_wavs()

        # Channel mode
        self.channel_mode_idx: int = 0

        # Analysis mode
        self.analysis_mode_idx: int = 0  # index into _ANALYSIS_MODES

        # CQT settings
        self.hop_length: int = 512
        self.bins_per_octave: int = 1200
        self.cqt_fmin: float = 16.35  # C0
        self.cqt_fmax: float = 19912.13  # D#9 (snapped)
        self._cqt_expanded: bool = True

        # Filter bank settings
        self._fb_expanded: bool = True
        self.fb_crossovers: list[float] = [
            _snap_to_note(200.0),   # ~G3
            _snap_to_note(800.0),   # ~G#5
            _snap_to_note(4000.0),  # ~B7
        ]
        self.fb_filter_type_idx: int = 0  # index into _FILTER_TYPES
        self._fb_decomp: FilterBankDecomposition | None = None

        # Resampling preferences
        self._rs_expanded: bool = True
        self.resample_engine_idx: int = 0  # index into _RESAMPLE_ENGINES
        self.resample_interp_idx: int = 0  # index into _RESAMPLE_INTERPS
        self.resample_taps_idx: int = 3    # index into _RESAMPLE_TAPS (64)
        self.resample_precision_idx: int = 2  # index into _RESAMPLE_PRECISIONS

        # Active / A / B
        self.active_folder: str | None = None
        self.a_folder: str | None = None
        self.b_folder: str | None = None

        # Analysis folder list
        self.folder_list = ScrollableItemList("Analysis Folders", max_visible=6)
        self._folder_paths: list[str] = []
        self._refresh_folders()

        # UI state
        self._item_map: dict[str, tuple[pygame.Rect, Any]] = {}
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        self._dragging: str | None = None
        self._analyzing: bool = False
        self._analysis_thread: threading.Thread | None = None
        self._analysis_error: str | None = None
        self._fb_editing_idx: int = -1  # which crossover is being edited

        # Callbacks
        self.on_set_active: Any = None
        self.on_set_a: Any = None
        self.on_set_b: Any = None
        self.on_unload: Any = None
        self.on_fb_computed: Any = None  # (FilterBankDecomposition) -> None

    # ---- Scanning ---------------------------------------------------------

    def _refresh_wavs(self) -> None:
        self._wav_paths = []
        items: list[ListItem] = []
        if os.path.isdir(self.input_dir):
            for fn in sorted(os.listdir(self.input_dir)):
                if fn.lower().endswith(".wav"):
                    fp = os.path.join(self.input_dir, fn)
                    self._wav_paths.append(fp)
                    items.append(ListItem(key=fp, display_name=fn, tag="wav"))
        self.wav_list.items = items

    def _refresh_folders(self) -> None:
        """Scan output_root and its parent for analysis directories.

        Each folder is listed once; if it contains a ``versions.json``
        manifest, individual version entries are shown as sub-items that
        can be individually deleted.
        """
        self._folder_paths = []
        self._folder_versions: dict[str, list[tuple[str, str]]] = {}
        items: list[ListItem] = []
        roots = [self.output_root]
        parent = os.path.dirname(self.output_root)
        if parent and parent != self.output_root:
            roots.append(parent)
        seen: set[str] = set()
        for root in roots:
            if not os.path.isdir(root):
                continue
            for fn in sorted(os.listdir(root)):
                fp = os.path.join(root, fn)
                if fp in seen:
                    continue
                if not os.path.isdir(fp):
                    continue
                # Detect both versioned and legacy CQT files
                import glob as _glob
                has_cqt = (os.path.isfile(os.path.join(fp, "cqt_data.npz")) or
                           bool(_glob.glob(os.path.join(fp, "cqt_data_*.npz"))))
                has_fb = os.path.isdir(os.path.join(fp, "filterbank"))
                if not (has_cqt or has_fb):
                    continue
                seen.add(fp)
                self._folder_paths.append(fp)
                tag = ""
                if fp == self.active_folder:
                    tag = "active"
                elif fp == self.a_folder:
                    tag = "A"
                elif fp == self.b_folder:
                    tag = "B"
                suffix = ""
                if has_cqt and has_fb:
                    suffix = " [CQT+FB]"
                elif has_cqt:
                    suffix = " [CQT]"
                elif has_fb:
                    suffix = " [FB]"

                # Check for versions manifest
                manifest_path = os.path.join(fp, "versions.json")
                versions: list[tuple[str, str]] = []
                if os.path.isfile(manifest_path):
                    try:
                        import json as _json
                        with open(manifest_path, encoding="utf-8") as mf:
                            manifest = _json.load(mf)
                        for vhash, vinfo in manifest.get("versions", {}).items():
                            short = vhash[:8]
                            settings = vinfo.get("settings", {})
                            desc = (f"h{settings.get('hop_length', '?')}"
                                    f" b{settings.get('bins_per_octave', '?')}"
                                    f" {settings.get('cqt_fmin', '?')}"
                                    f"-{settings.get('cqt_fmax', '?')}Hz")
                            versions.append((vhash, desc))
                    except Exception:
                        pass
                self._folder_versions[fp] = versions

                n_ver = len(versions)
                ver_hint = f" ({n_ver}v)" if n_ver > 1 else ""
                items.append(ListItem(
                    key=fp, display_name=fn + suffix + ver_hint,
                    tag=tag or "-"))

                # Add sub-items for each version
                for vhash, desc in versions:
                    items.append(ListItem(
                        key=f"{fp}::{vhash}",
                        display_name=f"  #{vhash[:6]} {desc}",
                        tag="ver",
                        tag_color_key="none",
                    ))
                    self._folder_paths.append(f"{fp}::{vhash}")

        self.folder_list.items = items

    def set_input_dir(self, path: str) -> None:
        self.input_dir = path
        self.output_root = path
        self._refresh_wavs()
        self._refresh_folders()

    # ---- Analysis ---------------------------------------------------------

    def _run_analysis(self) -> None:
        idx = self.wav_list.selected_idx
        if idx < 0 or idx >= len(self._wav_paths):
            return
        wav_path = self._wav_paths[idx]
        self._analyzing = True
        self._analysis_error = None
        mode = _ANALYSIS_MODES[self.analysis_mode_idx]

        def _worker() -> None:
            try:
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                outdir = os.path.join(self.output_root,
                                      f"{base_name}_analysis")
                os.makedirs(outdir, exist_ok=True)

                # CQT analysis
                if mode in ("CQT", "Both"):
                    import subprocess, sys
                    cmd = [
                        sys.executable, "-m", "bass_analysis",
                        wav_path,
                        "--hop-length", str(self.hop_length),
                        "--bins-per-octave", str(self.bins_per_octave),
                        "--cqt-fmin", str(self.cqt_fmin),
                        "--cqt-fmax", str(self.cqt_fmax),
                        "--outdir", outdir,
                    ]
                    subprocess.check_call(cmd,
                                          cwd=os.path.dirname(
                                              os.path.abspath(__file__)))

                # Filter bank decomposition
                if mode in ("Filter Bank", "Both"):
                    sr_wav, raw = wavfile.read(wav_path)
                    if raw.ndim == 2:
                        mono = raw.mean(axis=1).astype(np.float32) / 32768.0
                    else:
                        mono = raw.astype(np.float32) / 32768.0
                    ftype = _FILTER_TYPES[self.fb_filter_type_idx]
                    fb = FilterBankDecomposition(sr_wav, ftype)
                    bands = FilterBankDecomposition.bands_from_crossovers(
                        self.fb_crossovers, sr_wav)
                    fb.compute(mono, bands)
                    fb.save(outdir, sr_wav)
                    self._fb_decomp = fb
                    if self.on_fb_computed:
                        self.on_fb_computed(fb)

                self._refresh_folders()
            except Exception as exc:
                self._analysis_error = str(exc)
            finally:
                self._analyzing = False

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        self._analysis_thread = t

    def _graduate_band(self, band_idx: int) -> None:
        """Export a filter bank band as a standalone WAV in the input dir."""
        if not self._fb_decomp or band_idx >= len(self._fb_decomp.subbands):
            return
        try:
            path = self._fb_decomp.graduate_band(
                band_idx, self.input_dir, self._fb_decomp.sr)
            self._refresh_wavs()
        except Exception:
            pass

    # ---- Rendering helpers ------------------------------------------------

    def _render_section_header(self, surf: pygame.Surface, font: Any,
                               y: int, w: int, label: str,
                               key: str, expanded: bool) -> int:
        """Draw a collapsible section header. Returns new y."""
        arrow = "\u25bc" if expanded else "\u25b6"
        txt = font.render(f" {arrow} {label}", True, (180, 200, 220))
        rect = pygame.Rect(0, y, w, self.ROW_H)
        pygame.draw.rect(surf, (40, 45, 55), rect)
        pygame.draw.line(surf, (60, 65, 75), (0, y + self.ROW_H - 1),
                         (w, y + self.ROW_H - 1))
        surf.blit(txt, (2, y + 3))
        self._item_map[key] = (rect, None)
        return y + self.ROW_H

    def _render_slider(self, surf: pygame.Surface, font: Any,
                       y: int, w: int, key: str, prefix: str,
                       val: float, vmin: float, vmax: float,
                       fmt: str = "{:.0f}",
                       note_label: bool = False,
                       log: bool = False) -> int:
        BTN_W = 16  # width of +/- buttons
        lbl = font.render(prefix, True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 2))
        lbl_end = 50
        val_space = 80 if note_label else 50

        # Minus button
        minus_r = pygame.Rect(lbl_end, y + 2, BTN_W, self.ROW_H - 4)
        pygame.draw.rect(surf, (50, 50, 65), minus_r)
        pygame.draw.rect(surf, (80, 80, 100), minus_r, 1)
        mt = font.render("\u2212", True, (180, 180, 200))
        surf.blit(mt, (minus_r.x + (BTN_W - mt.get_width()) // 2, y + 2))
        self._item_map[f"{key}_minus"] = (minus_r, None)

        # Track
        track_x = lbl_end + BTN_W + 2
        track_end = w - self.PAD - val_space - BTN_W - 2
        track_w = max(track_end - track_x, 10)
        pygame.draw.rect(surf, (50, 50, 60),
                         pygame.Rect(track_x, y + 8, track_w, 6))
        if log and vmin > 0 and vmax > vmin:
            frac = (math.log(val) - math.log(vmin)) / (
                math.log(vmax) - math.log(vmin))
        else:
            frac = (val - vmin) / max(vmax - vmin, 1e-9)
        frac = max(0.0, min(1.0, frac))
        thumb_x = track_x + int(frac * track_w)
        pygame.draw.rect(surf, (120, 140, 200),
                         pygame.Rect(thumb_x - 5, y + 3, 10, 16))

        # Plus button
        plus_r = pygame.Rect(track_end + 2, y + 2, BTN_W, self.ROW_H - 4)
        pygame.draw.rect(surf, (50, 50, 65), plus_r)
        pygame.draw.rect(surf, (80, 80, 100), plus_r, 1)
        pt = font.render("+", True, (180, 180, 200))
        surf.blit(pt, (plus_r.x + (BTN_W - pt.get_width()) // 2, y + 2))
        self._item_map[f"{key}_plus"] = (plus_r, None)

        # Value label
        if note_label:
            vtxt = font.render(
                f"{fmt.format(val)} {_freq_to_note_name(val)}",
                True, (200, 200, 200))
        else:
            vtxt = font.render(fmt.format(val), True, (200, 200, 200))
        surf.blit(vtxt, (plus_r.x + BTN_W + 4, y + 2))
        self._item_map[key] = (
            pygame.Rect(track_x, y, track_w, self.ROW_H),
            (vmin, vmax, log))
        return y + self.ROW_H + 2

    def _render_dropdown_btn(self, surf: pygame.Surface, font: Any,
                             y: int, w: int, key: str, prefix: str,
                             options: list[str], sel_idx: int) -> int:
        lbl = font.render(prefix, True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 2))
        btn_x = 60
        btn_w = w - btn_x - self.PAD
        btn_rect = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
        bg = (80, 80, 120) if self._active_dropdown == key else (60, 60, 80)
        pygame.draw.rect(surf, bg, btn_rect)
        pygame.draw.rect(surf, (100, 100, 120), btn_rect, 1)
        txt = font.render(options[sel_idx], True, (220, 220, 220))
        surf.blit(txt, (btn_x + 4, y + 3))
        self._item_map[key] = (btn_rect, options)
        return y + self.ROW_H + 2

    # ---- Main render ------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._item_map = {}
        self._refresh_folders()

        # Pre-compute total height (rough estimate, will clip via surface)
        est_h = 1200  # generous
        surf = pygame.Surface((w, est_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))
        y = self.PAD

        # === WAV file list ===
        y += self.wav_list.render(surf, font, 0, y, w)

        # === Channel mode ===
        y = self._render_dropdown_btn(surf, font, y, w, "chan_mode",
                                      "Chan:", _CHANNEL_MODES,
                                      self.channel_mode_idx)

        # === Analysis mode ===
        y = self._render_dropdown_btn(surf, font, y, w, "analysis_mode",
                                      "Mode:", _ANALYSIS_MODES,
                                      self.analysis_mode_idx)

        # === CQT settings (collapsible) ===
        mode = _ANALYSIS_MODES[self.analysis_mode_idx]
        if mode in ("CQT", "Both"):
            y = self._render_section_header(surf, font, y, w,
                                            "CQT Settings", "cqt_hdr",
                                            self._cqt_expanded)
            if self._cqt_expanded:
                y = self._render_slider(surf, font, y, w, "hop", "Hop:",
                                        float(self.hop_length), 64.0, 2048.0,
                                        log=True)
                y = self._render_slider(surf, font, y, w, "bpo", "BPO:",
                                        float(self.bins_per_octave), 12.0, 1200.0,
                                        log=True)
                y = self._render_slider(surf, font, y, w, "fmin", "fMin:",
                                        self.cqt_fmin, 1.0, 100.0, "{:.1f}",
                                        note_label=True, log=True)
                y = self._render_slider(surf, font, y, w, "fmax", "fMax:",
                                        self.cqt_fmax, 1000.0, 22050.0,
                                        note_label=True, log=True)

        # === Filter bank settings (collapsible) ===
        if mode in ("Filter Bank", "Both"):
            y = self._render_section_header(surf, font, y, w,
                                            "Filter Bank", "fb_hdr",
                                            self._fb_expanded)
            if self._fb_expanded:
                # Filter type
                y = self._render_dropdown_btn(surf, font, y, w,
                                              "fb_ftype", "Type:",
                                              _FILTER_TYPES,
                                              self.fb_filter_type_idx)

                # Crossover frequencies
                lbl = font.render("Crossovers:", True, (160, 160, 160))
                surf.blit(lbl, (self.PAD, y + 2))
                y += self.ROW_H

                for i, xo in enumerate(self.fb_crossovers):
                    key = f"xo_{i}"
                    y = self._render_slider(surf, font, y, w, key,
                                            f" X{i + 1}:",
                                            xo, 20.0, 20000.0,
                                            note_label=True, log=True)

                # Add / Remove crossover buttons
                btn_w2 = (w - 3 * self.PAD) // 2
                add_r = pygame.Rect(self.PAD, y, btn_w2, self.ROW_H)
                rem_r = pygame.Rect(self.PAD * 2 + btn_w2, y, btn_w2, self.ROW_H)
                pygame.draw.rect(surf, (40, 70, 40), add_r)
                pygame.draw.rect(surf, (70, 100, 70), add_r, 1)
                at = font.render("+ Add XO", True, (160, 230, 160))
                surf.blit(at, (add_r.x + add_r.w // 2 - at.get_width() // 2,
                               y + 3))
                self._item_map["fb_add_xo"] = (add_r, None)

                pygame.draw.rect(surf, (70, 40, 40), rem_r)
                pygame.draw.rect(surf, (100, 70, 70), rem_r, 1)
                rt = font.render("- Remove", True, (230, 160, 160))
                surf.blit(rt, (rem_r.x + rem_r.w // 2 - rt.get_width() // 2,
                               y + 3))
                self._item_map["fb_rem_xo"] = (rem_r, None)
                y += self.ROW_H + 2

                # Resulting bands preview
                sr_preview = 44100
                bands = FilterBankDecomposition.bands_from_crossovers(
                    self.fb_crossovers, sr_preview)
                lbl2 = font.render(f"Bands: {len(bands)}", True,
                                   (140, 160, 180))
                surf.blit(lbl2, (self.PAD, y + 2))
                y += self.ROW_H

                for bi, band in enumerate(bands):
                    bl = font.render(f"  {band.auto_label(sr_preview)}",
                                     True, (120, 140, 160))
                    surf.blit(bl, (self.PAD, y + 1))
                    # Graduate button
                    grad_key = f"grad_{bi}"
                    gr = pygame.Rect(w - 50 - self.PAD, y, 50, self.ROW_H - 2)
                    has_fb = (self._fb_decomp is not None and
                              bi < len(self._fb_decomp.subbands))
                    gc = (50, 70, 90) if has_fb else (40, 40, 45)
                    pygame.draw.rect(surf, gc, gr)
                    pygame.draw.rect(surf, (80, 100, 120), gr, 1)
                    gt = font.render("\u2197 WAV", True,
                                     (160, 200, 230) if has_fb else (80, 80, 90))
                    surf.blit(gt, (gr.x + 4, y + 2))
                    self._item_map[grad_key] = (gr, bi)
                    y += self.ROW_H

                # Reconstruction error (if computed)
                if self._fb_decomp and self._fb_decomp.subbands:
                    err = self._fb_decomp.reconstruction_error()
                    err_txt = f"Recon err: {err:.6f} RMS"
                    ec = (120, 230, 120) if err < 0.01 else (230, 180, 120)
                    et = font.render(err_txt, True, ec)
                    surf.blit(et, (self.PAD, y + 2))
                    y += self.ROW_H

        # === Resampling preferences ===
        y += 2
        y = self._render_section_header(surf, font, y, w,
                                        "Resampling", "rs_hdr",
                                        self._rs_expanded)
        if self._rs_expanded:
            y = self._render_dropdown_btn(surf, font, y, w, "rs_engine",
                                          "Engine:", _RESAMPLE_ENGINES,
                                          self.resample_engine_idx)
            y = self._render_dropdown_btn(surf, font, y, w, "rs_interp",
                                          "Interp:", _RESAMPLE_INTERPS,
                                          self.resample_interp_idx)
            y = self._render_dropdown_btn(surf, font, y, w, "rs_taps",
                                          "Taps:", _RESAMPLE_TAPS,
                                          self.resample_taps_idx)
            y = self._render_dropdown_btn(surf, font, y, w, "rs_prec",
                                          "Prec:", _RESAMPLE_PRECISIONS,
                                          self.resample_precision_idx)

        # === Analyze button ===
        btn_rect = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
        if self._analyzing:
            pygame.draw.rect(surf, (80, 80, 40), btn_rect)
            txt = font.render("Analyzing ...", True, (220, 220, 180))
        else:
            pygame.draw.rect(surf, (40, 80, 40), btn_rect)
            txt = font.render("Analyze", True, (180, 255, 180))
        pygame.draw.rect(surf, (80, 120, 80), btn_rect, 1)
        surf.blit(txt, (btn_rect.x + btn_rect.w // 2 - txt.get_width() // 2,
                         y + 3))
        self._item_map["analyze_btn"] = (btn_rect, None)
        y += self.ROW_H + 2

        # Error display
        if self._analysis_error:
            err_txt = font.render(self._analysis_error[:38], True,
                                  (255, 120, 120))
            surf.blit(err_txt, (self.PAD, y))
            y += self.ROW_H

        y += self.PAD

        # === Analysis folder list ===
        y += self.folder_list.render(surf, font, 0, y, w)

        # === Folder action buttons ===
        sel_idx = self.folder_list.selected_idx
        has_sel = 0 <= sel_idx < len(self._folder_paths)
        btn_labels = [("Set Active", "set_active"),
                      ("Set A", "set_a"),
                      ("Set B", "set_b"),
                      ("Unload", "unload"),
                      ("Del", "delete_folder")]
        bw = (w - 2 * self.PAD - (len(btn_labels) - 1) * 2) // len(btn_labels)
        bx = self.PAD
        for label_text, bkey in btn_labels:
            br = pygame.Rect(bx, y, bw, self.ROW_H)
            if bkey == "delete_folder":
                bg_c = (80, 40, 40) if has_sel else (50, 35, 35)
                fg_c = (255, 160, 160) if has_sel else (120, 80, 80)
            elif bkey == "set_active":
                bg_c = (40, 60, 80) if has_sel else (30, 40, 50)
                fg_c = (160, 200, 255) if has_sel else (80, 100, 120)
            elif bkey == "set_a":
                bg_c = (60, 40, 80) if has_sel else (40, 30, 50)
                fg_c = (200, 160, 255) if has_sel else (100, 80, 120)
            elif bkey == "unload":
                bg_c = (70, 60, 30) if has_sel else (45, 40, 25)
                fg_c = (240, 220, 120) if has_sel else (110, 100, 60)
            else:
                bg_c = (80, 60, 40) if has_sel else (50, 40, 30)
                fg_c = (255, 200, 160) if has_sel else (120, 100, 80)
            pygame.draw.rect(surf, bg_c, br)
            pygame.draw.rect(surf, (100, 100, 120), br, 1)
            bt = font.render(label_text, True, fg_c)
            surf.blit(bt, (bx + bw // 2 - bt.get_width() // 2, y + 3))
            self._item_map[bkey] = (br, None)
            bx += bw + 2
        y += self.ROW_H + 4

        # Crop surface to actual content height
        final = pygame.Surface((w, y), pygame.SRCALPHA)
        final.blit(surf, (0, 0))
        return final

    # ---- Event handling ---------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        pr = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos
            if not pr.collidepoint(mx, my):
                if self._active_dropdown:
                    self._active_dropdown = None
                return False
            lx, ly = mx - pr.x, my - pr.y

            # Dropdown selection
            if self._active_dropdown and self._dropdown_rect.collidepoint(lx, ly):
                for i, ir in enumerate(self._dropdown_item_rects):
                    if ir.collidepoint(lx, ly):
                        self._select_dropdown(self._active_dropdown,
                                              self._dropdown_opts[i])
                        break
                self._active_dropdown = None
                return True
            if self._active_dropdown:
                self._active_dropdown = None

            # WAV list click
            if self.wav_list.handle_click(lx, ly) is not None:
                return True

            # Folder list click
            if self.folder_list.handle_click(lx, ly) is not None:
                return True

            # Section headers (collapse/expand)
            if "cqt_hdr" in self._item_map:
                rect, _ = self._item_map["cqt_hdr"]
                if rect.collidepoint(lx, ly):
                    self._cqt_expanded = not self._cqt_expanded
                    return True
            if "fb_hdr" in self._item_map:
                rect, _ = self._item_map["fb_hdr"]
                if rect.collidepoint(lx, ly):
                    self._fb_expanded = not self._fb_expanded
                    return True
            if "rs_hdr" in self._item_map:
                rect, _ = self._item_map["rs_hdr"]
                if rect.collidepoint(lx, ly):
                    self._rs_expanded = not self._rs_expanded
                    return True

            # Dropdowns
            for key in ("chan_mode", "analysis_mode", "fb_ftype",
                        "rs_engine", "rs_interp", "rs_taps", "rs_prec"):
                if key in self._item_map:
                    rect, opts = self._item_map[key]
                    if rect.collidepoint(lx, ly):
                        self._open_dropdown(key, opts, rect, pr)
                        return True

            # Slider +/- buttons and track drag
            slider_keys = ["hop", "bpo", "fmin", "fmax"]
            slider_keys += [f"xo_{i}" for i in range(len(self.fb_crossovers))]
            for skey in slider_keys:
                minus_k = f"{skey}_minus"
                plus_k = f"{skey}_plus"
                if minus_k in self._item_map:
                    rect, _ = self._item_map[minus_k]
                    if rect.collidepoint(lx, ly):
                        self._nudge_slider(skey, -1)
                        return True
                if plus_k in self._item_map:
                    rect, _ = self._item_map[plus_k]
                    if rect.collidepoint(lx, ly):
                        self._nudge_slider(skey, +1)
                        return True
                if skey in self._item_map:
                    rect, (vmin, vmax, is_log) = self._item_map[skey]
                    if rect.collidepoint(lx, ly):
                        self._dragging = skey
                        self._update_slider(skey, lx, rect, vmin, vmax, is_log)
                        return True

            # Analyze button
            if "analyze_btn" in self._item_map:
                rect, _ = self._item_map["analyze_btn"]
                if rect.collidepoint(lx, ly) and not self._analyzing:
                    self._run_analysis()
                    return True

            # Add / Remove crossover
            if "fb_add_xo" in self._item_map:
                rect, _ = self._item_map["fb_add_xo"]
                if rect.collidepoint(lx, ly):
                    top = max(self.fb_crossovers) if self.fb_crossovers else 200.0
                    self.fb_crossovers.append(min(top * 2, 20000.0))
                    self.fb_crossovers.sort()
                    return True
            if "fb_rem_xo" in self._item_map:
                rect, _ = self._item_map["fb_rem_xo"]
                if rect.collidepoint(lx, ly) and self.fb_crossovers:
                    self.fb_crossovers.pop()
                    return True

            # Graduate band buttons
            for key, val in list(self._item_map.items()):
                if key.startswith("grad_"):
                    rect, bi = val
                    if rect.collidepoint(lx, ly):
                        self._graduate_band(bi)
                        return True

            # Folder action buttons
            for bkey in ("set_active", "set_a", "set_b",
                        "unload", "delete_folder"):
                if bkey in self._item_map:
                    rect, _ = self._item_map[bkey]
                    if rect.collidepoint(lx, ly):
                        self._handle_folder_action(bkey)
                        return True

            return True

        if event.type == MOUSEBUTTONUP and event.button == 1:
            if self._dragging:
                self._dragging = None
                return True
            mx, my = event.pos
            if pr.collidepoint(mx, my):
                return True

        if event.type == MOUSEMOTION:
            if self._dragging:
                if not event.buttons[0]:
                    self._dragging = None
                else:
                    mx, my = event.pos
                    lx = mx - pr.x
                    key = self._dragging
                    if key in self._item_map:
                        rect, (vmin, vmax, is_log) = self._item_map[key]
                        self._update_slider(key, lx, rect, vmin, vmax, is_log)
                return True
            mx, my = event.pos
            if pr.collidepoint(mx, my):
                return True

        if event.type == MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if pr.collidepoint(mx, my):
                ly = my - pr.y
                half = pr.h // 2
                if ly < half:
                    self.wav_list.handle_scroll(-event.y)
                else:
                    self.folder_list.handle_scroll(-event.y)
                return True

        return False

    def _open_dropdown(self, key: str, opts: list[str],
                       btn_rect: pygame.Rect,
                       panel_rect: pygame.Rect) -> None:
        self._active_dropdown = key
        self._dropdown_opts = opts
        x = btn_rect.x
        y = btn_rect.y + btn_rect.h
        item_h = self.ROW_H
        total_h = item_h * len(opts)
        self._dropdown_rect = pygame.Rect(x, y, btn_rect.w, total_h)
        self._dropdown_item_rects = [
            pygame.Rect(x, y + i * item_h, btn_rect.w, item_h)
            for i in range(len(opts))
        ]

    def _select_dropdown(self, key: str, value: str) -> None:
        if key == "chan_mode":
            self.channel_mode_idx = _CHANNEL_MODES.index(value)
        elif key == "analysis_mode":
            self.analysis_mode_idx = _ANALYSIS_MODES.index(value)
        elif key == "fb_ftype":
            self.fb_filter_type_idx = _FILTER_TYPES.index(value)
        elif key == "rs_engine":
            self.resample_engine_idx = _RESAMPLE_ENGINES.index(value)
        elif key == "rs_interp":
            self.resample_interp_idx = _RESAMPLE_INTERPS.index(value)
        elif key == "rs_taps":
            self.resample_taps_idx = _RESAMPLE_TAPS.index(value)
        elif key == "rs_prec":
            self.resample_precision_idx = _RESAMPLE_PRECISIONS.index(value)

    def _update_slider(self, key: str, lx: int,
                       rect: pygame.Rect,
                       vmin: float, vmax: float,
                       log: bool = False) -> None:
        frac = max(0.0, min(1.0, (lx - rect.x) / rect.w))
        if log and vmin > 0 and vmax > vmin:
            val = math.exp(math.log(vmin) + frac * (
                math.log(vmax) - math.log(vmin)))
        else:
            val = vmin + frac * (vmax - vmin)
        self._apply_slider_value(key, val)

    def _apply_slider_value(self, key: str, val: float) -> None:
        """Clamp + snap a raw slider value and store it."""
        if key == "hop":
            raw = max(64, min(2048, int(val)))
            self.hop_length = 2 ** round(math.log2(raw))
        elif key == "bpo":
            self.bins_per_octave = _snap_bpo(val)
        elif key == "fmin":
            self.cqt_fmin = round(_snap_to_note(max(1.0, min(100.0, val))), 2)
        elif key == "fmax":
            self.cqt_fmax = round(_snap_to_note(max(1000.0, min(22050.0, val))), 2)
        elif key.startswith("xo_"):
            idx = int(key[3:])
            if 0 <= idx < len(self.fb_crossovers):
                self.fb_crossovers[idx] = round(
                    _snap_to_note(max(20.0, min(20000.0, val))), 2)
                self.fb_crossovers.sort()

    def _nudge_slider(self, key: str, direction: int) -> None:
        """Move a slider by one logical step. direction: +1 or -1."""
        if key == "hop":
            exp = round(math.log2(self.hop_length)) + direction
            self.hop_length = max(64, min(2048, 2 ** exp))
        elif key == "bpo":
            idx = _BPO_SNAPS.index(self.bins_per_octave) if \
                self.bins_per_octave in _BPO_SNAPS else 0
            idx = max(0, min(len(_BPO_SNAPS) - 1, idx + direction))
            self.bins_per_octave = _BPO_SNAPS[idx]
        elif key == "fmin":
            midi = 69 + 12 * math.log2(self.cqt_fmin / 440.0)
            midi = round(midi) + direction
            self.cqt_fmin = round(max(1.0, min(100.0,
                                  _midi_to_freq(midi))), 2)
        elif key == "fmax":
            midi = 69 + 12 * math.log2(self.cqt_fmax / 440.0)
            midi = round(midi) + direction
            self.cqt_fmax = round(max(1000.0, min(22050.0,
                                  _midi_to_freq(midi))), 2)
        elif key.startswith("xo_"):
            idx = int(key[3:])
            if 0 <= idx < len(self.fb_crossovers):
                midi = 69 + 12 * math.log2(
                    self.fb_crossovers[idx] / 440.0)
                midi = round(midi) + direction
                self.fb_crossovers[idx] = round(max(20.0, min(20000.0,
                    _midi_to_freq(midi))), 2)
                self.fb_crossovers.sort()

    def _handle_folder_action(self, action: str) -> None:
        idx = self.folder_list.selected_idx
        if idx < 0 or idx >= len(self._folder_paths):
            return
        raw = self._folder_paths[idx]

        # Detect version sub-item  (key = "{folder}::{hash}")
        if "::" in raw:
            folder_path, version_hash = raw.rsplit("::", 1)
        else:
            folder_path, version_hash = raw, ""

        if action == "unload":
            if self.on_unload:
                self.on_unload()
            return
        if action == "set_active":
            self.active_folder = folder_path
            if self.on_set_active:
                self.on_set_active(folder_path, version_hash)
        elif action == "set_a":
            self.a_folder = folder_path
            if self.on_set_a:
                self.on_set_a(folder_path, version_hash)
        elif action == "set_b":
            self.b_folder = folder_path
            if self.on_set_b:
                self.on_set_b(folder_path, version_hash)
        elif action == "delete_folder":
            if version_hash:
                # Delete a single version via the manifest system
                from bass_analysis import delete_version
                try:
                    folder_removed = delete_version(folder_path, version_hash)
                except Exception:
                    folder_removed = False
                if folder_removed:
                    if folder_path == self.active_folder:
                        self.active_folder = None
                    if folder_path == self.a_folder:
                        self.a_folder = None
                    if folder_path == self.b_folder:
                        self.b_folder = None
            else:
                # Delete entire folder
                import shutil
                try:
                    shutil.rmtree(folder_path)
                except Exception:
                    pass
                if folder_path == self.active_folder:
                    self.active_folder = None
                if folder_path == self.a_folder:
                    self.a_folder = None
                if folder_path == self.b_folder:
                    self.b_folder = None
            self._refresh_folders()

    def render_dropdown_overlay(self) -> pygame.Surface | None:
        if not self._active_dropdown or not self._dropdown_opts:
            return None
        self._ensure_font()
        font = self.font
        rect = self._dropdown_rect
        surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        surf.fill((40, 40, 50, 240))
        for i, opt in enumerate(self._dropdown_opts):
            ir = self._dropdown_item_rects[i]
            iy = ir.y - rect.y
            txt = font.render(opt, True, (220, 220, 220))
            surf.blit(txt, (4, iy + 3))
            if i < len(self._dropdown_opts) - 1:
                pygame.draw.line(surf, (60, 60, 70),
                                 (0, iy + self.ROW_H - 1),
                                 (rect.w, iy + self.ROW_H - 1))
        pygame.draw.rect(surf, (100, 100, 130),
                         pygame.Rect(0, 0, rect.w, rect.h), 1)
        return surf


# ---------------------------------------------------------------------------
# Panel dock — manages left / right panel slots
# ---------------------------------------------------------------------------

class PanelDock:
    """Holds an ordered registry of :class:`Panel` instances and tracks
    which one is assigned to the left and right screen edges.

    * ``registry``  — all available panels (key → Panel).
    * ``left_key``  — key of panel currently docked left (or *None*).
    * ``right_key`` — key of panel currently docked right (or *None*).
    """

    def __init__(self) -> None:
        self.registry: dict[str, Panel] = {}
        self.left_key: str | None = None
        self.right_key: str | None = None

    def register(self, key: str, panel: Panel) -> None:
        self.registry[key] = panel

    @property
    def left(self) -> Panel | None:
        p = self.registry.get(self.left_key) if self.left_key else None
        if p:
            p.side = "left"
        return p

    @property
    def right(self) -> Panel | None:
        p = self.registry.get(self.right_key) if self.right_key else None
        if p:
            p.side = "right"
        return p

    def set_left(self, key: str | None) -> None:
        if key == self.right_key:
            self.right_key = None
        self.left_key = key
        if key and key in self.registry:
            self.registry[key].side = "left"

    def set_right(self, key: str | None) -> None:
        if key == self.left_key:
            self.left_key = None
        self.right_key = key
        if key and key in self.registry:
            self.registry[key].side = "right"

    @property
    def left_w(self) -> int:
        p = self.left
        return Panel.PANEL_W if (p and p.visible) else 0

    @property
    def right_w(self) -> int:
        p = self.right
        return Panel.PANEL_W if (p and p.visible) else 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        for p in (self.right, self.left):
            if p and p.visible and p.handle_event(event):
                return True
        return False

    def panels_visible(self) -> list[Panel]:
        out: list[Panel] = []
        for p in (self.left, self.right):
            if p and p.visible:
                out.append(p)
        return out


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("consola.ttf", "cour.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _nice_step(span: float, n_px: int, min_spacing: int = 80) -> float:
    """Choose a round tick spacing for *span* across *n_px* pixels."""
    target_count = max(1, n_px // min_spacing)
    raw = span / target_count
    if raw <= 0:
        return 1.0
    mag = 10 ** math.floor(math.log10(raw))
    for nice in (1.0, 2.0, 5.0, 10.0):
        step = nice * mag
        if span / step <= target_count * 1.5:
            return step
    return mag * 10.0


# ---------------------------------------------------------------------------
# Viewport gather / reduce
# ---------------------------------------------------------------------------

EPS = 1e-12


def _reduce_2d(data: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reduce a 2-D array to (out_h, out_w) by block-mean averaging.

    When the data is smaller than the output in either dimension, that
    dimension is left unchanged (GL_LINEAR handles upsampling).
    """
    h, w = data.shape
    result = data.astype(np.float64)

    if h > out_h:
        y_idx = np.linspace(0, h, out_h + 1, dtype=np.intp)
        row_sums = np.add.reduceat(result, y_idx[:-1], axis=0)
        y_sizes = np.diff(y_idx).astype(np.float64)
        result = row_sums / y_sizes[:, None]
    elif h < out_h:
        result = result[np.linspace(0, h - 1, out_h, dtype=np.intp)]

    if w > out_w:
        x_idx = np.linspace(0, w, out_w + 1, dtype=np.intp)
        col_sums = np.add.reduceat(result, x_idx[:-1], axis=1)
        x_sizes = np.diff(x_idx).astype(np.float64)
        result = col_sums / x_sizes[None, :]
    elif w < out_w:
        result = result[:, np.linspace(0, w - 1, out_w, dtype=np.intp)]

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
# Mipmap cache — multi-resolution pyramid with disk persistence
# ---------------------------------------------------------------------------

_MIPMAP_MIN_DIM = 64
_SCAN_CHUNK = 32


class MipmapCache:
    """Multi-resolution pyramid for fast viewport gathering at any zoom.

    Builds reduced copies of every data field at power-of-2 scale factors.
    The coarsest level is built first (instant at ~64 px); finer levels are
    built progressively in a background thread.  The entire pyramid is
    cached to disk so subsequent opens load into RAM with no recomputation.
    """

    CACHE_FILE = "mipmap_cache.npz"

    def __init__(self, analysis_dir: str, raw_arrays: dict[str, np.ndarray],
                 n_bins: int, n_frames: int,
                 settings_hash: str = "") -> None:
        self.analysis_dir = analysis_dir
        self.raw_arrays = raw_arrays
        self.n_bins = n_bins
        self.n_frames = n_frames
        self.settings_hash = settings_hash
        self.levels: dict[int, dict[str, np.ndarray]] = {}
        self.max_level = self._compute_max_level()
        self._lock = threading.Lock()
        self._build_thread: threading.Thread | None = None
        self._built_all = False

    def _compute_max_level(self) -> int:
        level = 0
        h, w = self.n_bins, self.n_frames
        while h > _MIPMAP_MIN_DIM and w > _MIPMAP_MIN_DIM:
            h //= 2
            w //= 2
            level += 1
        return max(1, level)

    @property
    def cache_path(self) -> str:
        if self.settings_hash:
            return os.path.join(self.analysis_dir,
                                f"mipmap_cache_{self.settings_hash}.npz")
        return os.path.join(self.analysis_dir, self.CACHE_FILE)

    # ---- Disk persistence ------------------------------------------------

    def load_from_disk(self) -> bool:
        path = self.cache_path
        if not os.path.isfile(path):
            return False
        try:
            data = np.load(path, allow_pickle=False)
            if "_meta" not in data:
                data.close()
                return False
            meta = data["_meta"]
            if int(meta[0]) != self.n_bins or int(meta[1]) != self.n_frames:
                data.close()
                return False
            for key in data.files:
                if key.startswith("_"):
                    continue
                idx = key.rfind("_L")
                if idx < 0:
                    continue
                field_name = key[:idx]
                level = int(key[idx + 2:])
                if level not in self.levels:
                    self.levels[level] = {}
                self.levels[level][field_name] = np.array(data[key],
                                                          dtype=np.float32)
            data.close()
            self._built_all = bool(self.levels)
            print(f"  Loaded mipmap cache ({len(self.levels)} levels)")
            return True
        except Exception as exc:
            print(f"  Mipmap cache load failed: {exc}")
            return False

    def save_to_disk(self) -> None:
        save_dict: dict[str, np.ndarray] = {
            "_meta": np.array([self.n_bins, self.n_frames]),
        }
        with self._lock:
            for level, fields in self.levels.items():
                for field_name, arr in fields.items():
                    save_dict[f"{field_name}_L{level}"] = arr
        tmp_path = self.cache_path + ".tmp"
        # np.savez auto-appends .npz to the filename
        actual_tmp = tmp_path + ".npz"
        try:
            np.savez(tmp_path, **save_dict)
            os.replace(actual_tmp, self.cache_path)
            print(f"  Mipmap cache saved ({len(self.levels)} levels)")
        except Exception as exc:
            print(f"  Mipmap cache save failed: {exc}")
            try:
                os.remove(actual_tmp)
            except OSError:
                pass

    # ---- Level building --------------------------------------------------

    def build_level(self, level: int) -> None:
        factor = 2 ** level
        out_h = max(1, self.n_bins // factor)
        out_w = max(1, self.n_frames // factor)
        lvl: dict[str, np.ndarray] = {}
        for name, arr in self.raw_arrays.items():
            lvl[name] = _reduce_2d(arr, out_h, out_w)
        for side in ("left", "right"):
            rk, ik = f"real_{side}", f"imag_{side}"
            mk = f"mag_{side}"
            if rk in lvl and ik in lvl:
                r, i = lvl[rk], lvl[ik]
                lvl[mk] = np.sqrt(r * r + i * i).astype(np.float32)
        if "mag_left" in lvl and "mag_right" in lvl:
            ml, mr = lvl["mag_left"], lvl["mag_right"]
            peak = np.maximum(ml, mr)
            lvl["mag_similarity"] = np.where(
                peak < EPS, 1.0, np.minimum(ml, mr) / peak,
            ).astype(np.float32)
        if "phase_left" in lvl and "phase_right" in lvl:
            pl, pr = lvl["phase_left"], lvl["phase_right"]
            lvl["phase_similarity"] = (
                (1.0 + np.cos(pl - pr)) / 2.0
            ).astype(np.float32)
        with self._lock:
            self.levels[level] = lvl

    def start_background_build(self) -> None:
        self._build_thread = threading.Thread(
            target=self._build_all_levels, daemon=True)
        self._build_thread.start()

    def _build_all_levels(self) -> None:
        for level in range(self.max_level, 0, -1):
            with self._lock:
                if level in self.levels:
                    continue
            print(f"  Building mipmap level {level} ...")
            self.build_level(level)
            self.save_to_disk()
        self._built_all = True

    # ---- Query -----------------------------------------------------------

    def best_level(self, view_bins: float, view_frames: float,
                   out_h: int, out_w: int) -> int:
        dpx = min(view_frames / max(1, out_w),
                  view_bins / max(1, out_h))
        if dpx <= 1:
            return 0
        target = int(math.floor(math.log2(dpx)))
        target = min(target, self.max_level)
        with self._lock:
            while target > 0 and target not in self.levels:
                target -= 1
        return target

    def get_field(self, field_name: str, level: int,
                  y0: float, y1: float, x0: float, x1: float,
                  out_h: int, out_w: int) -> np.ndarray | None:
        with self._lock:
            if level not in self.levels:
                return None
            lvl = self.levels[level]
            if field_name not in lvl:
                return None
            arr = lvl[field_name]
        factor = 2 ** level
        my0 = max(0, int(y0 / factor))
        my1 = min(arr.shape[0], int(math.ceil(y1 / factor)))
        mx0 = max(0, int(x0 / factor))
        mx1 = min(arr.shape[1], int(math.ceil(x1 / factor)))
        if my1 <= my0 or mx1 <= mx0:
            return None
        return _reduce_2d(arr[my0:my1, mx0:mx1], out_h, out_w)


# ---------------------------------------------------------------------------
# Viewport TF synthesis — invert visible spectrogram window to audio
# ---------------------------------------------------------------------------

class ViewportSynthPlayer:
    """Inverse-CQT viewport synthesizer with multi-rate reconstruction.

    Reconstructs a time-domain waveform from the visible CQT viewport
    using proper overlap-add inversion of the multi-rate CQT.  Each
    octave is inverted at its decimated sample rate using the same
    Hann-windowed sinusoidal basis used during analysis, then
    upsampled and summed to produce near-perfect reconstruction for
    the selected frequency band.

    When PyTorch + CUDA are available the heavy inner loops run on GPU
    with fully batched FFTs (10-50× faster for large viewports).
    Falls back transparently to NumPy on CPU.
    """

    # Try to import torch once at class load time
    _torch = None
    _has_cuda = False
    try:
        import torch as _torch_mod
        _torch = _torch_mod
        _has_cuda = _torch_mod.cuda.is_available()
    except ImportError:
        pass

    def __init__(self, sr: int, hop_length: int,
                 bins_per_octave: int) -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.playing = False
        self.position: float = 0.0
        self.duration: float = 0.0
        self._play_wall: float = 0.0
        self._channel: pygame.mixer.Channel | None = None
        self._sound: pygame.mixer.Sound | None = None
        self.view_t0: float = 0.0
        self.view_t1: float = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _fftconvolve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Real-valued FFT convolution (full output)."""
        n = len(a) + len(b) - 1
        fft_n = 1
        while fft_n < n:
            fft_n <<= 1
        return np.fft.irfft(np.fft.rfft(a, fft_n) *
                            np.fft.rfft(b, fft_n), fft_n)[:n]

    # ------------------------------------------------------------------
    # GPU-accelerated inverse CQT
    # ------------------------------------------------------------------
    def _inverse_cqt_channel_gpu(self, cqt_slice: np.ndarray,
                                 view_freqs: np.ndarray,
                                 y0: int, n_total_bins: int,
                                 n_view_frames: int) -> np.ndarray:
        """Batched GPU inverse CQT — mirrors the forward CQT structure."""
        torch = self._torch
        device = torch.device("cuda")

        bpo = self.bins_per_octave
        hop = self.hop_length
        sr = self.sr
        Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
        n_octaves = int(math.ceil(n_total_bins / bpo))

        out_samples = n_view_frames * hop
        output = torch.zeros(out_samples, dtype=torch.float64, device=device)

        # Build upsample anti-alias filter once on GPU
        max_octave = n_octaves - 1
        _up_filts: dict[int, torch.Tensor] = {}
        for octave in range(1, n_octaves):
            repeat = 1 << octave
            filt_len = 64 * repeat + 1
            filt_half = filt_len // 2
            ft = torch.arange(-filt_half, filt_half + 1,
                              dtype=torch.float64, device=device)
            sinc_v = torch.sinc(ft / repeat)
            kaiser_v = torch.from_numpy(
                np.kaiser(filt_len, 6.0)).to(device=device, dtype=torch.float64)
            aa = sinc_v * kaiser_v
            aa = aa / (aa.sum() * repeat)
            _up_filts[octave] = aa

        for octave in range(n_octaves):
            gbin_start = max(0, n_total_bins - (octave + 1) * bpo)
            gbin_end = n_total_bins - octave * bpo
            vis_start = max(gbin_start, y0)
            vis_end = min(gbin_end, y0 + cqt_slice.shape[0])
            if vis_end <= vis_start:
                continue

            local_start = vis_start - y0
            local_end = vis_end - y0
            repeat = 1 << octave

            oct_cqt = torch.from_numpy(
                np.asarray(cqt_slice[local_start:local_end, ::repeat],
                           dtype=np.complex128)).to(device)
            oct_freqs_np = np.asarray(view_freqs[local_start:local_end],
                                      dtype=np.float64)
            n_oct_bins, n_oct_frames = oct_cqt.shape
            oct_sr = sr >> octave

            # Build dense kernel matrix: (n_oct_bins, max_klen)
            kernel_lengths = np.ceil(Q * oct_sr / oct_freqs_np).astype(int)
            max_klen = int(kernel_lengths.max())
            if max_klen < 1:
                continue

            basis = np.zeros((n_oct_bins, max_klen), dtype=np.complex128)
            win_sq_arr = np.zeros((n_oct_bins, max_klen), dtype=np.float64)
            for i in range(n_oct_bins):
                klen = kernel_lengths[i]
                if klen < 1:
                    continue
                t = np.arange(klen, dtype=np.float64)
                win = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / klen)
                basis[i, :klen] = win * np.exp(
                    2j * np.pi * oct_freqs_np[i] * t / oct_sr)
                win_sq_arr[i, :klen] = win * win

            # Build coefficient impulse train matrix: (n_oct_bins, train_len)
            train_len = (n_oct_frames - 1) * hop + 1
            conv_len = train_len + max_klen - 1
            fft_n = 1
            while fft_n < conv_len:
                fft_n <<= 1

            # Move kernels to GPU and FFT
            basis_gpu = torch.from_numpy(basis).to(device)
            basis_fft = torch.fft.fft(basis_gpu, n=fft_n, dim=1)
            del basis_gpu

            # Build impulse trains on GPU
            trains_gpu = torch.zeros((n_oct_bins, train_len),
                                     dtype=torch.complex128, device=device)
            hop_indices = torch.arange(0, train_len, hop, device=device)
            n_hop = min(len(hop_indices), n_oct_frames)
            trains_gpu[:, ::hop] = oct_cqt[:, :n_hop]
            del oct_cqt

            trains_fft = torch.fft.fft(trains_gpu, n=fft_n, dim=1)
            del trains_gpu

            # Batched convolution: all bins at once
            conv_result = torch.fft.ifft(
                trains_fft * basis_fft, dim=1).real[:, :conv_len]
            del trains_fft, basis_fft

            # Sum across all bins → octave output
            oct_output = conv_result.sum(dim=0)
            del conv_result

            # OLA normalization: build norm envelope on GPU
            win_sq_gpu = torch.from_numpy(win_sq_arr).to(device)
            norm_trains = torch.zeros((n_oct_bins, train_len),
                                      dtype=torch.float64, device=device)
            norm_trains[:, ::hop] = 1.0
            nfft_norm = fft_n

            norm_trains_fft = torch.fft.rfft(norm_trains, n=nfft_norm, dim=1)
            win_sq_fft = torch.fft.rfft(win_sq_gpu, n=nfft_norm, dim=1)
            del win_sq_gpu, norm_trains

            norm_conv = torch.fft.irfft(
                norm_trains_fft * win_sq_fft,
                n=nfft_norm, dim=1)[:, :conv_len]
            del norm_trains_fft, win_sq_fft

            oct_norm = norm_conv.sum(dim=0)
            del norm_conv

            safe_norm = torch.where(oct_norm > 1e-10, oct_norm,
                                    torch.ones_like(oct_norm))
            oct_output = oct_output / safe_norm
            del oct_norm, safe_norm

            # Upsample to full sample rate
            if octave > 0:
                oct_out_len = oct_output.shape[0]
                up = torch.zeros(oct_out_len * repeat,
                                 dtype=torch.float64, device=device)
                up[::repeat] = oct_output * repeat
                aa = _up_filts[octave]
                filt_half = aa.shape[0] // 2
                up_3d = up.view(1, 1, -1)
                aa_3d = aa.flip(0).view(1, 1, -1)
                oct_output = torch.nn.functional.conv1d(
                    up_3d, aa_3d, padding=filt_half
                ).view(-1)
                del up, up_3d, aa_3d

            trim = min(oct_output.shape[0], out_samples)
            output[:trim] += oct_output[:trim]
            del oct_output

        return output.cpu().numpy()

    # ------------------------------------------------------------------
    def _inverse_cqt_channel(self, cqt_slice: np.ndarray,
                             view_freqs: np.ndarray,
                             y0: int, n_total_bins: int,
                             n_view_frames: int) -> np.ndarray:
        """Reconstruct one channel via multi-rate overlap-add inverse CQT.

        *cqt_slice* has shape ``(n_view_bins, n_view_frames)`` and holds
        the complex CQT coefficients for the visible viewport.
        """
        bpo = self.bins_per_octave
        hop = self.hop_length
        sr = self.sr
        Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
        n_octaves = int(math.ceil(n_total_bins / bpo))

        # Output at full sample rate
        out_samples = n_view_frames * hop
        output = np.zeros(out_samples, dtype=np.float64)

        for octave in range(n_octaves):
            # Global bin range for this octave
            gbin_start = max(0, n_total_bins - (octave + 1) * bpo)
            gbin_end = n_total_bins - octave * bpo

            # Intersect with viewport
            vis_start = max(gbin_start, y0)
            vis_end = min(gbin_end, y0 + cqt_slice.shape[0])
            if vis_end <= vis_start:
                continue

            local_start = vis_start - y0
            local_end = vis_end - y0

            # Undo frame duplication for this octave
            repeat = 1 << octave
            oct_cqt = np.asarray(cqt_slice[local_start:local_end, ::repeat],
                                 dtype=np.complex128)
            oct_freqs = np.asarray(view_freqs[local_start:local_end],
                                   dtype=np.float64)
            n_oct_bins, n_oct_frames = oct_cqt.shape
            oct_sr = sr >> octave

            # Per-bin overlap-add via FFT convolution
            train_len = (n_oct_frames - 1) * hop + 1
            oct_out_len = 0
            norm_cache: dict[int, np.ndarray] = {}

            for i in range(n_oct_bins):
                fk = oct_freqs[i]
                klen = int(math.ceil(Q * oct_sr / fk))
                if klen < 1:
                    continue

                # Build synthesis kernel (Hann-windowed complex sinusoid)
                t = np.arange(klen, dtype=np.float64)
                win = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / klen)
                basis = win * np.exp(2j * np.pi * fk * t / oct_sr)

                # Coefficient impulse train
                train = np.zeros(train_len, dtype=np.complex128)
                train[::hop] = oct_cqt[i, :]

                # Convolve (complex train × complex basis → take real part)
                conv_len = train_len + klen - 1
                fft_n = 1
                while fft_n < conv_len:
                    fft_n <<= 1
                conv = np.fft.ifft(
                    np.fft.fft(train, fft_n) * np.fft.fft(basis, fft_n)
                ).real[:conv_len]

                if oct_out_len == 0:
                    oct_out_len = conv_len
                    oct_output = np.zeros(oct_out_len, dtype=np.float64)
                    oct_norm = np.zeros(oct_out_len, dtype=np.float64)
                elif conv_len > oct_out_len:
                    oct_output = np.pad(oct_output,
                                        (0, conv_len - oct_out_len))
                    oct_norm = np.pad(oct_norm,
                                      (0, conv_len - oct_out_len))
                    oct_out_len = conv_len

                oct_output[:conv_len] += conv

                # Accumulate window energy for OLA normalization
                # (cache by kernel length since many bins share the same)
                if klen not in norm_cache:
                    win_sq = win * win
                    norm_train = np.zeros(train_len, dtype=np.float64)
                    norm_train[::hop] = 1.0
                    norm_cache[klen] = self._fftconvolve(norm_train, win_sq)
                nc = norm_cache[klen]
                oct_norm[:len(nc)] += nc[:oct_out_len]

            if oct_out_len == 0:
                continue

            # Normalise by window overlap
            safe_norm = np.where(oct_norm > 1e-10, oct_norm, 1.0)
            oct_output /= safe_norm

            # Upsample to full sample rate
            if octave > 0:
                # Zero-stuff + low-pass (polyphase resampling)
                up = np.zeros(oct_out_len * repeat, dtype=np.float64)
                up[::repeat] = oct_output * repeat
                # Simple windowed-sinc AA filter
                filt_len = 64 * repeat + 1
                filt_half = filt_len // 2
                ft = np.arange(-filt_half, filt_half + 1, dtype=np.float64)
                sinc = np.sinc(ft / repeat)
                kaiser = np.kaiser(filt_len, 6.0)
                aa = sinc * kaiser
                aa /= aa.sum() * repeat
                oct_output = np.convolve(up, aa, mode='same')
                oct_out_len = len(oct_output)

            trim = min(oct_out_len, out_samples)
            output[:trim] += oct_output[:trim]

        return output

    # ------------------------------------------------------------------
    # Band-selective iCQT: reconstruct only the CQT bins within each
    # filterbank band, producing per-band time-domain audio that is
    # equivalent to filterbank decomposition but built from CQT data.
    # ------------------------------------------------------------------
    def icqt_per_band(self, cqt: np.ndarray,
                      freqs: np.ndarray,
                      n_total_bins: int,
                      band_bin_map: list[np.ndarray]
                      ) -> list[np.ndarray]:
        """Run iCQT separately for each filterbank band's CQT bins.

        Parameters
        ----------
        cqt : complex array (n_total_bins, n_frames)
            Full complex CQT (one channel).
        freqs : array (n_total_bins,)
            Center frequencies per bin.
        n_total_bins : int
            Total bin count.
        band_bin_map : list of int arrays
            Per-band arrays of global bin indices, as returned by
            ``FilterBankDecomposition.band_map()``.

        Returns
        -------
        list of 1-D float64 arrays, one per band.
            Each is the time-domain reconstruction of that band's bins.
        """
        n_frames = cqt.shape[1]
        _icqt = (self._inverse_cqt_channel_gpu if self._has_cuda
                 else self._inverse_cqt_channel)
        results: list[np.ndarray] = []
        for bin_indices in band_bin_map:
            if len(bin_indices) == 0:
                results.append(np.zeros(n_frames * self.hop_length,
                                        dtype=np.float64))
                continue
            # Build a contiguous sub-CQT covering only this band's bins
            band_cqt = cqt[bin_indices, :]
            band_freqs = freqs[bin_indices]
            y0 = int(bin_indices[0])
            sig = _icqt(band_cqt, band_freqs, y0, n_total_bins, n_frames)
            results.append(sig)
        return results

    def recombine_bands_icqt(self, cqt: np.ndarray,
                             freqs: np.ndarray,
                             n_total_bins: int,
                             band_bin_map: list[np.ndarray],
                             band_mask: list[bool] | None = None
                             ) -> np.ndarray:
        """Reconstruct time-domain audio by iCQT-ing selected filterbank
        bands and summing them.

        This is the filterbank recombination path through iCQT: each band's
        CQT bins are inverted independently, then the per-band waveforms
        are added.  With all bands enabled this is identical to a full iCQT;
        with a subset it gives the filterbank-decomposed reconstruction.

        Parameters
        ----------
        cqt, freqs, n_total_bins, band_bin_map :
            Same as :meth:`icqt_per_band`.
        band_mask :
            Which bands to include (all if *None*).

        Returns
        -------
        1-D float64 array — reconstructed waveform.
        """
        per_band = self.icqt_per_band(cqt, freqs, n_total_bins,
                                       band_bin_map)
        if band_mask is None:
            band_mask = [True] * len(per_band)
        max_len = max((len(s) for s, m in zip(per_band, band_mask) if m),
                      default=0)
        output = np.zeros(max_len, dtype=np.float64)
        for sig, include in zip(per_band, band_mask):
            if include:
                n = min(len(sig), max_len)
                output[:n] += sig[:n]
        return output

    # ------------------------------------------------------------------
    def synthesise_to_array(self, raw_arrays: dict[str, np.ndarray],
                            freqs: np.ndarray, times: np.ndarray,
                            view_x0: float, view_x1: float,
                            view_y0: float, view_y1: float,
                            n_frames: int, n_bins: int,
                            t_start: float, t_dur: float,
                            sr: int, is_stereo: bool,
                            normalize: bool = True
                            ) -> tuple[np.ndarray, float, float, float] | None:
        """Run iCQT and return ``(stereo_i16, duration, view_t0, view_t1)``.

        Returns *None* if the viewport is empty.  The stereo array has shape
        ``(N, 2)`` dtype ``int16``, ready for ``pygame.sndarray.make_sound``.
        """
        x0 = max(0, int(math.floor(view_x0)))
        x1 = min(n_frames, int(math.ceil(view_x1)))
        y0 = max(0, int(math.floor(view_y0)))
        y1 = min(n_bins, int(math.ceil(view_y1)))
        if x1 <= x0 or y1 <= y0:
            return None

        vt0 = t_start + x0 / n_frames * t_dur
        vt1 = t_start + x1 / n_frames * t_dur
        dur = vt1 - vt0
        if dur <= 0:
            return None

        view_freqs = np.asarray(freqs[y0:y1], dtype=np.float64)

        # Build complex CQT slices
        real_l = np.asarray(raw_arrays["real_left"][y0:y1, x0:x1],
                            dtype=np.float64)
        imag_l = np.asarray(raw_arrays["imag_left"][y0:y1, x0:x1],
                            dtype=np.float64)
        cqt_l = real_l + 1j * imag_l
        n_vf = x1 - x0

        print(f"Inverse CQT: {y1 - y0} bins × {n_vf} frames, "
              f"{dur:.2f}s ...")
        t0 = time.time()

        # Dispatch to GPU or CPU
        _icqt = (self._inverse_cqt_channel_gpu if self._has_cuda
                 else self._inverse_cqt_channel)
        if self._has_cuda:
            print("  using GPU (CUDA)")
        sig_l = _icqt(cqt_l, view_freqs, y0, n_bins, n_vf)

        if is_stereo:
            real_r = np.asarray(raw_arrays["real_right"][y0:y1, x0:x1],
                                dtype=np.float64)
            imag_r = np.asarray(raw_arrays["imag_right"][y0:y1, x0:x1],
                                dtype=np.float64)
            sig_r = _icqt(real_r + 1j * imag_r,
                          view_freqs, y0, n_bins, n_vf)
        else:
            sig_r = sig_l

        elapsed = time.time() - t0
        print(f"  done in {elapsed:.2f}s")

        # Trim to exact duration
        n_samples = int(dur * sr)
        sig_l = sig_l[:n_samples]
        sig_r = sig_r[:n_samples]
        if len(sig_l) < n_samples:
            sig_l = np.pad(sig_l, (0, n_samples - len(sig_l)))
        if len(sig_r) < n_samples:
            sig_r = np.pad(sig_r, (0, n_samples - len(sig_r)))

        # Normalise to int16
        if normalize:
            peak = max(np.abs(sig_l).max(), np.abs(sig_r).max(), 1e-12)
            sig_l = sig_l / peak * 0.8
            sig_r = sig_r / peak * 0.8
        left_i16 = (sig_l * 32767).clip(-32768, 32767).astype(np.int16)
        right_i16 = (sig_r * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([left_i16, right_i16])
        return stereo, dur, vt0, vt1

    def play_array(self, stereo: np.ndarray, duration: float,
                   view_t0: float, view_t1: float) -> None:
        """Play a pre-synthesised stereo int16 array."""
        self.stop()
        self._sound = pygame.sndarray.make_sound(stereo)
        self._channel = self._sound.play()
        self._play_wall = time.time()
        self.duration = duration
        self.view_t0 = view_t0
        self.view_t1 = view_t1
        self.position = 0.0
        self.playing = True

    def synthesise_and_play(self, raw_arrays: dict[str, np.ndarray],
                            freqs: np.ndarray, times: np.ndarray,
                            view_x0: float, view_x1: float,
                            view_y0: float, view_y1: float,
                            n_frames: int, n_bins: int,
                            t_start: float, t_dur: float,
                            sr: int, is_stereo: bool,
                            normalize: bool = True) -> None:
        self.stop()
        result = self.synthesise_to_array(
            raw_arrays, freqs, times, view_x0, view_x1,
            view_y0, view_y1, n_frames, n_bins,
            t_start, t_dur, sr, is_stereo, normalize)
        if result is None:
            return
        stereo, dur, vt0, vt1 = result
        self.play_array(stereo, dur, vt0, vt1)

    def stop(self) -> None:
        if self.playing:
            self.position = self.get_position()
            if self._channel is not None:
                self._channel.stop()
            self.playing = False
        self._channel = None
        self._sound = None

    def get_position(self) -> float:
        if self.playing:
            pos = self.position + (time.time() - self._play_wall)
            if pos >= self.duration:
                self.playing = False
                return self.duration
            return pos
        return self.position

    def get_time(self) -> float:
        """Return the absolute time corresponding to the current position."""
        return self.view_t0 + self.get_position()


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

class SpectrogramViewer:
    """OpenGL viewer that loads npz directly and gathers viewport textures."""

    def __init__(self, analysis_dir: str | None = None,
                 audio_path: str | None = None) -> None:
        self.analysis_dir = analysis_dir
        self.audio_path = audio_path

        # Empty-state defaults (no data loaded)
        self._npz: Any = None
        self._raw_arrays: dict[str, np.ndarray] = {}
        self._has_onset: bool = False
        self._settings_hash: str = ""
        self.n_bins: int = 0
        self.n_frames: int = 0
        self.is_stereo: bool = False
        self.times: np.ndarray = np.zeros(0, dtype=np.float32)
        self.t_start: float = 0.0
        self.t_end: float = 0.0
        self.t_dur: float = 0.0
        self._freqs: np.ndarray = np.zeros(0, dtype=np.float32)
        self._semitone_mask: np.ndarray = np.zeros(0, dtype=bool)
        self._note_names: list[str] = []
        self._pad_influence: np.ndarray | None = None
        self._field_ranges: dict[str, tuple[float, float]] = {}
        self._mipmap: Any = None

        if analysis_dir is not None:
            self._load_npz(analysis_dir)

        # --- Field sources ---
        self.field_sources = _build_field_sources(self._has_onset)
        self.field_configs = [FieldConfig() for _ in self.field_sources]
        self.global_defaults = GlobalDefaults()

        for i, fs in enumerate(self.field_sources):
            if fs.name == "mag_left":
                self.field_configs[i].target = 0   # R
            elif fs.name == "mag_right":
                self.field_configs[i].target = 1   # G

        # Display state
        self.win_w = 1600
        self.win_h = 900
        self.view_x0: float = 0.0
        self.view_y0: float = 0.0
        self.view_x1: float = float(self.n_frames)
        self.view_y1: float = float(self.n_bins)

        self.show_freq_axis = True
        self.show_time_axis = True
        self.show_amp_axis = True
        self.display_mode: str = "tf"   # "tf" = time-frequency, "ts" = time-signal
        self.axis_margin_left = 200
        self.axis_margin_right = 200
        self.axis_margin_top = 32
        self.axis_margin_bottom = 32
        self.axis_margin_amp = 60

        # Subplot tiling
        self.subplots_tf: list[SubPlot] = [SubPlot("spectrogram", "Spectrogram")]
        self.subplots_ts: list[SubPlot] = [SubPlot("waveform", "Waveform")]
        self.tiling_strategy: TilingStrategy = TilingStrategy.COLUMNS

        self._rect_start: tuple[int, int] | None = None
        self._rect_end: tuple[int, int] | None = None
        self._is_panning = False
        self._pan_last: tuple[int, int] | None = None

        # GL handles
        self._program = 0
        self._tex_units: dict[str, int] = {}
        self._gui_tex = 0
        self._glyph_atlas = GlyphAtlas()

        # Double-buffer progressive scan state
        self._front_tex_ids: dict[str, int] = {}
        self._back_tex_ids: dict[str, int] = {}
        self._back_buffers: dict[str, np.ndarray] = {}
        self._scan_col = 0
        self._scan_total = 0
        self._scan_complete = True
        self._scan_key: tuple | None = None
        self._scan_params: tuple | None = None
        self._scan_needed: set[str] = set()

        # Mipmap cache
        if analysis_dir is not None:
            print("Initializing mipmap cache ...")
            self._mipmap = MipmapCache(
                analysis_dir, self._raw_arrays,
                self.n_bins, self.n_frames,
                settings_hash=getattr(self, '_settings_hash', ''))
            if not self._mipmap.load_from_disk():
                self._mipmap.build_level(self._mipmap.max_level)
                self._mipmap.save_to_disk()
                self._mipmap.start_background_build()
            print("  done")

        # Zoom axis lock: None=both, 'time'=time only, 'freq'=freq only
        self.zoom_lock: str | None = None

        self.player: AudioPlayer | None = None
        self.synth: ViewportSynthPlayer | None = None
        self.gui: FieldTabPanel | None = None
        self.synth_panel: SynthesisPanel | None = None
        self.file_panel: SourcePanel | None = None
        self.dock: PanelDock | None = None
        self.menu_bar: MenuBar | None = None

    @property
    def _loaded(self) -> bool:
        """True when analysis data is loaded and ready for display."""
        return self.n_frames > 0 and self.n_bins > 0

    def _load_npz(self, analysis_dir: str, version_hash: str = "") -> None:
        """Load CQT data from an analysis directory.

        If *version_hash* is given, load ``cqt_data_{hash}.npz``.
        Otherwise try the legacy ``cqt_data.npz``.
        """
        if version_hash:
            npz_path = os.path.join(analysis_dir,
                                    f"cqt_data_{version_hash}.npz")
        else:
            npz_path = os.path.join(analysis_dir, "cqt_data.npz")
        # Fallback: if the versioned file doesn't exist, check for
        # any cqt_data_*.npz and pick the newest.
        if not os.path.isfile(npz_path):
            import glob
            candidates = sorted(
                glob.glob(os.path.join(analysis_dir, "cqt_data_*.npz")),
                key=os.path.getmtime, reverse=True)
            if candidates:
                npz_path = candidates[0]
            else:
                npz_path = os.path.join(analysis_dir, "cqt_data.npz")
        print(f"Loading {npz_path} ...")
        self._npz = np.load(npz_path, mmap_mode="r", allow_pickle=True)
        self._settings_hash = str(
            self._npz["settings_hash"]) if "settings_hash" in self._npz else ""

        self._raw_arrays = {}
        for key in ("real_left", "imag_left", "real_right", "imag_right",
                     "phase_left", "phase_right"):
            self._raw_arrays[key] = self._npz[key]
        if "onset" in self._npz:
            self._raw_arrays["onset"] = self._npz["onset"]
        self._has_onset = "onset" in self._raw_arrays

        self.n_bins, self.n_frames = self._raw_arrays["real_left"].shape
        self.is_stereo = bool(self._npz["is_stereo"])

        self.times = self._npz["times"]
        self.t_start = float(self.times[0])
        self.t_end = float(self.times[-1])
        self.t_dur = self.t_end - self.t_start

        self._freqs = self._npz["freqs"]
        self._semitone_mask = self._npz["semitone_mask"]
        self._note_names = list(self._npz["note_names"])

        pi = self._npz["pad_influence"]
        self._pad_influence = pi if pi.size > 0 else None

        print(f"  {self.n_bins} bins x {self.n_frames} frames  "
              f"{'stereo' if self.is_stereo else 'mono'}")

        print("Computing normalization ranges ...")
        abs_maxes = []
        for key in ("real_left", "real_right", "imag_left", "imag_right"):
            arr = self._raw_arrays[key]
            hi = float(arr.max())
            lo = float(arr.min())
            abs_maxes.append(max(abs(hi), abs(lo)))
        ri_abs_max = max(abs_maxes)
        mag_max = float(np.sqrt(ri_abs_max ** 2 + ri_abs_max ** 2))

        self._field_ranges = {
            "real_left":        (-ri_abs_max, ri_abs_max),
            "imag_left":        (-ri_abs_max, ri_abs_max),
            "real_right":       (-ri_abs_max, ri_abs_max),
            "imag_right":       (-ri_abs_max, ri_abs_max),
            "mag_left":         (0.0, mag_max),
            "mag_right":        (0.0, mag_max),
            "phase_left":       (-float(np.pi), float(np.pi)),
            "phase_right":      (-float(np.pi), float(np.pi)),
            "mag_similarity":   (0.0, 1.0),
            "phase_similarity": (0.0, 1.0),
            "onset":            (0.0, 1.0),
        }
        print("  done")

    # ---- Derived-field computation on a slice -----------------------------

    def _get_field_slice(self, field_name: str,
                         y0: int, y1: int,
                         x0: int, x1: int) -> np.ndarray:
        """Return a (y1-y0, x1-x0) float32 slice for the given field."""
        if field_name in self._raw_arrays:
            return np.asarray(
                self._raw_arrays[field_name][y0:y1, x0:x1], dtype=np.float32)

        s = slice(y0, y1), slice(x0, x1)

        if field_name == "mag_left":
            r = np.asarray(self._raw_arrays["real_left"][s], dtype=np.float32)
            i = np.asarray(self._raw_arrays["imag_left"][s], dtype=np.float32)
            return np.sqrt(r * r + i * i)

        if field_name == "mag_right":
            r = np.asarray(self._raw_arrays["real_right"][s], dtype=np.float32)
            i = np.asarray(self._raw_arrays["imag_right"][s], dtype=np.float32)
            return np.sqrt(r * r + i * i)

        if field_name == "mag_similarity":
            mag_l = self._get_field_slice("mag_left", y0, y1, x0, x1)
            mag_r = self._get_field_slice("mag_right", y0, y1, x0, x1)
            peak = np.maximum(mag_l, mag_r)
            return np.where(peak < EPS, 1.0,
                            np.minimum(mag_l, mag_r) / peak).astype(np.float32)

        if field_name == "phase_similarity":
            pl = np.asarray(self._raw_arrays["phase_left"][s], dtype=np.float32)
            pr = np.asarray(self._raw_arrays["phase_right"][s], dtype=np.float32)
            return ((1.0 + np.cos(pl - pr)) / 2.0).astype(np.float32)

        return np.zeros((y1 - y0, x1 - x0), dtype=np.float32)

    def _normalize_field(self, data: np.ndarray,
                         field_name: str,
                         norm_mode: int = 0) -> np.ndarray:
        """Map raw field values to [0, 1] using global ranges.

        *norm_mode* 2 (rank_order) replaces min/max scaling with
        percentile-rank mapping so every value is its fractional rank
        in the current slice.
        """
        if norm_mode == 2:
            flat = data.ravel()
            n = flat.size
            if n == 0:
                return np.zeros_like(data, dtype=np.float32)
            order = flat.argsort().argsort()          # rank indices
            ranked = order.astype(np.float32) / max(n - 1, 1)
            return ranked.reshape(data.shape)
        vmin, vmax = self._field_ranges[field_name]
        span = vmax - vmin
        if span < EPS:
            return np.zeros_like(data, dtype=np.float32)
        return np.clip((data - vmin) / span, 0.0, 1.0).astype(np.float32)

    # ---- Mipmap-aware field access ----------------------------------------

    def _get_field_for_viewport(self, field_name: str,
                                y0: int, y1: int,
                                x0: int, x1: int,
                                out_h: int, out_w: int) -> np.ndarray:
        """Get reduced field data, accelerated by mipmap when possible."""
        view_h = y1 - y0
        view_w = x1 - x0
        level = self._mipmap.best_level(view_h, view_w, out_h, out_w)
        if level > 0:
            result = self._mipmap.get_field(
                field_name, level, y0, y1, x0, x1, out_h, out_w)
            if result is not None:
                return result
        raw = self._get_field_slice(field_name, y0, y1, x0, x1)
        return _reduce_2d(raw, out_h, out_w)

    # ---- Progressive viewport scan ----------------------------------------

    def _start_progressive_scan(self) -> None:
        """Begin a new progressive left-to-right viewport texture scan."""
        # Promote completed back -> front
        if self._scan_complete and self._back_tex_ids:
            for tex_id in self._front_tex_ids.values():
                glDeleteTextures([tex_id])
            self._front_tex_ids = dict(self._back_tex_ids)
            self._back_tex_ids = {}
            self._back_buffers = {}

        # Discard incomplete back buffer
        if not self._scan_complete:
            for tex_id in self._back_tex_ids.values():
                glDeleteTextures([tex_id])
            self._back_tex_ids = {}
            self._back_buffers = {}

        sx, sy, sw, sh = self._spec_rect()
        out_w = max(1, sw)
        out_h = max(1, sh)
        x0 = max(0, int(math.floor(self.view_x0)))
        x1 = min(self.n_frames, int(math.ceil(self.view_x1)))
        y0 = max(0, int(math.floor(self.view_y0)))
        y1 = min(self.n_bins, int(math.ceil(self.view_y1)))
        if x1 <= x0 or y1 <= y0:
            self._scan_complete = True
            return

        self._scan_params = (x0, x1, y0, y1, out_w, out_h)
        self._scan_col = 0
        self._scan_total = out_w
        self._scan_complete = False

        needed: set[str] = set()
        for i, fs in enumerate(self.field_sources):
            if self.field_configs[i].target >= 0:
                needed.add(fs.tex_name)
        self._scan_needed = needed

        self._tex_units = {}
        unit = 0
        for tex_name in sorted(needed):
            if unit >= 8:
                break
            self._tex_units[tex_name] = unit
            unit += 1

        for tex_name in needed:
            if tex_name not in self._tex_units:
                continue
            buf = np.zeros((out_h, out_w, 4), dtype=np.float32)
            self._back_buffers[tex_name] = buf
            tex_id = self._upload_texture_float(buf)
            self._back_tex_ids[tex_name] = tex_id

    def _tick_progressive_scan(self) -> None:
        """Advance the progressive scan by one time-budgeted batch."""
        if self._scan_complete or self._scan_params is None:
            return

        x0, x1, y0, y1, out_w, out_h = self._scan_params
        data_w = x1 - x0
        start_time = time.time()

        while self._scan_col < self._scan_total:
            col_start = self._scan_col
            col_end = min(col_start + _SCAN_CHUNK, self._scan_total)
            chunk_w = col_end - col_start

            chunk_x0_f = x0 + col_start * data_w / out_w
            chunk_x1_f = x0 + col_end * data_w / out_w
            cx0 = max(0, int(math.floor(chunk_x0_f)))
            cx1 = min(self.n_frames, int(math.ceil(chunk_x1_f)))
            if cx1 <= cx0:
                self._scan_col = col_end
                continue

            for tex_name in self._scan_needed:
                if tex_name not in self._tex_units:
                    continue
                layout = _TEXTURE_LAYOUTS.get(tex_name)
                if layout is None:
                    continue

                channels: list[np.ndarray] = []
                for ch_key in "RGBA":
                    field_name = layout.get(ch_key)
                    if field_name is None:
                        if ch_key == "A":
                            channels.append(
                                np.ones((out_h, chunk_w), dtype=np.float32))
                        else:
                            channels.append(
                                np.zeros((out_h, chunk_w), dtype=np.float32))
                    else:
                        data = self._get_field_for_viewport(
                            field_name, y0, y1, cx0, cx1, out_h, chunk_w)
                        # Determine effective norm_mode for this field
                        fnorm = 0
                        for fi, fs in enumerate(self.field_sources):
                            if fs.name == field_name:
                                cfg = self.field_configs[fi]
                                gd = self.global_defaults
                                fnorm = cfg.norm_mode if not cfg.use_global else gd.norm_mode
                                break
                        normed = self._normalize_field(data, field_name,
                                                       fnorm)
                        channels.append(normed)

                if self._pad_influence is not None:
                    pi_slice = np.asarray(
                        self._pad_influence[y0:y1, cx0:cx1],
                        dtype=np.float32)
                    alpha = _reduce_2d(1.0 - pi_slice, out_h, chunk_w)
                    channels[3] = np.clip(alpha, 0.0, 1.0).astype(
                        np.float32)

                chunk_rgba = np.stack(channels, axis=-1).astype(np.float32)
                self._back_buffers[tex_name][:, col_start:col_end, :] = \
                    chunk_rgba

                tex_id = self._back_tex_ids[tex_name]
                glBindTexture(GL_TEXTURE_2D, tex_id)
                raw_bytes = np.ascontiguousarray(chunk_rgba).tobytes()
                glTexSubImage2D(GL_TEXTURE_2D, 0, col_start, 0,
                                chunk_w, out_h, GL_RGBA, GL_FLOAT,
                                raw_bytes)

            self._scan_col = col_end

            if time.time() - start_time > 0.012:
                break

        if self._scan_col >= self._scan_total:
            self._scan_complete = True
            for tex_id in self._front_tex_ids.values():
                glDeleteTextures([tex_id])
            self._front_tex_ids = dict(self._back_tex_ids)
            self._back_tex_ids = {}
            self._back_buffers = {}

    def _viewport_gather_key(self) -> tuple:
        """Return a hashable key representing the current viewport state."""
        sx, sy, sw, sh = self._spec_rect()
        targets = tuple(c.target for c in self.field_configs)
        return (self.view_x0, self.view_x1, self.view_y0, self.view_y1,
                sw, sh, targets)

    # ---- Main loop --------------------------------------------------------

    def run(self) -> None:
        pygame.init()
        sr_hint = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100
        pygame.mixer.pre_init(frequency=sr_hint, size=-16, channels=2,
                              buffer=2048)
        pygame.mixer.init()
        pygame.display.set_mode(
            (self.win_w, self.win_h), DOUBLEBUF | OPENGL | RESIZABLE,
        )
        pygame.display.set_caption("Spectrogram Viewer")
        pygame.font.init()

        self._init_gl()
        self._glyph_atlas.build()
        self.gui = FieldTabPanel(self.field_sources, self.field_configs,
                                 self.global_defaults, side="right")

        self.dock = PanelDock()
        self.dock.register("fields", self.gui)
        self.synth_panel = SynthesisPanel(side="right")
        self.synth_panel.on_play = self._play_synth_job
        self.synth_panel.on_stop = self._stop_synth_job
        self.dock.register("synthesis", self.synth_panel)
        _app_root = os.path.dirname(os.path.abspath(__file__))
        input_dir = os.path.join(_app_root, "input")
        os.makedirs(input_dir, exist_ok=True)
        self.file_panel = SourcePanel(
            side="left", input_dir=input_dir, output_root=input_dir)
        self.file_panel.active_folder = self.analysis_dir
        self.file_panel.on_set_active = self._load_analysis_folder
        self.file_panel.on_unload = self._unload_data
        self.dock.register("files", self.file_panel)
        self.dock.right_key = "fields"
        self.dock.left_key = "files"

        self.menu_bar = MenuBar()
        self._build_menus()
        Panel._top_offset = self.menu_bar.bar_height

        if self.audio_path and os.path.isfile(self.audio_path):
            self.player = AudioPlayer(self.audio_path)
        if self._npz is not None:
            sr_val = int(self._npz["sr"]) if "sr" in self._npz else 44100
            hop_val = int(self._npz["hop_length"]) if "hop_length" in self._npz else 512
            bpo_val = int(self._npz["bins_per_octave"]) if "bins_per_octave" in self._npz else 1200
            self.synth = ViewportSynthPlayer(sr_val, hop_val, bpo_val)
        self.phase_mode: str = "original"
        self.normalize_loudness: bool = True

        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue

                if self.menu_bar and self.menu_bar.handle_event(event):
                    if event.type in (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION):
                        self._is_panning = False
                        self._pan_last = None
                        self._rect_start = None
                        self._rect_end = None
                    continue

                if self.dock and self.dock.handle_event(event):
                    if event.type in (MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION):
                        self._is_panning = False
                        self._pan_last = None
                        self._rect_start = None
                        self._rect_end = None
                    continue

                if event.type == KEYDOWN:
                    running = self._handle_key(event)
                elif event.type == MOUSEBUTTONDOWN:
                    self._handle_mouse_down(event)
                elif event.type == MOUSEBUTTONUP:
                    self._handle_mouse_up(event)
                elif event.type == MOUSEMOTION:
                    self._handle_mouse_motion(event)
                elif event.type == MOUSEWHEEL:
                    self._handle_wheel(event)
                elif event.type == VIDEORESIZE:
                    self.win_w, self.win_h = event.size
                    pygame.display.set_mode(
                        (self.win_w, self.win_h),
                        DOUBLEBUF | OPENGL | RESIZABLE,
                    )
                    self._init_gl()
                    self._glyph_atlas.build()

            # Keyboard-held scrolling
            if self._loaded:
                keys = pygame.key.get_pressed()
                speed = (self.view_x1 - self.view_x0) * 0.02
                if keys[K_LSHIFT] or keys[K_RSHIFT]:
                    speed *= 4
                if keys[K_LEFT]:
                    self._pan_view(-speed, 0)
                if keys[K_RIGHT]:
                    self._pan_view(speed, 0)

            # Auto-scroll to keep cursor visible
            if self._loaded and self.player and self.player.playing:
                cur_frame = self._time_to_frame(self.player.get_position())
                vw = self.view_x1 - self.view_x0
                if cur_frame < self.view_x0 or cur_frame > self.view_x1:
                    self.view_x0 = cur_frame - vw * 0.25
                    self.view_x1 = self.view_x0 + vw
                    self._clamp_view()

            # Progressive viewport texture scan (TF only)
            if self._loaded and self.display_mode == "tf":
                key = self._viewport_gather_key()
                if key != self._scan_key:
                    self._start_progressive_scan()
                    self._scan_key = key
                if not self._scan_complete:
                    self._tick_progressive_scan()

            self._render()
            pygame.display.flip()
            clock.tick(60)

        if self.synth:
            self.synth.stop()
        if self.player:
            self.player.stop()
        pygame.quit()

    # ---- Menu construction --------------------------------------------------

    def _build_menus(self) -> None:
        """Populate the menu bar with initial menus."""
        mb = self.menu_bar
        dock = self.dock

        # -- Helper: build radio items for a dock side --
        def _panel_radio(side: str) -> list[MenuItem]:
            setter = dock.set_left if side == "left" else dock.set_right
            getter = (lambda: dock.left_key) if side == "left" else (lambda: dock.right_key)
            items: list[MenuItem] = [
                MenuItem("None",
                         action=lambda s=setter: s(None),
                         toggle_state=lambda g=getter: g() is None,
                         radio_group=f"dock_{side}"),
            ]
            for key, panel in dock.registry.items():
                items.append(MenuItem(
                    panel.title,
                    action=lambda s=setter, k=key: s(k),
                    toggle_state=lambda g=getter, k=key: g() == k,
                    radio_group=f"dock_{side}",
                ))
            return items

        # -- Helper: build subplot toggle items for a mode --
        def _subplot_toggles(src: list[SubPlot]) -> list[MenuItem]:
            items: list[MenuItem] = []
            for sp in src:
                items.append(MenuItem(
                    sp.label,
                    action=lambda s=sp: setattr(s, "visible", not s.visible),
                    toggle_state=lambda s=sp: s.visible,
                ))
            return items

        mb.add_menu("View", [
            MenuItem("Reset Zoom", shortcut="R",
                     action=self._reset_zoom),
            MenuItem("-"),
            MenuItem("Display Mode", children=[
                MenuItem("Time-Frequency",
                         action=lambda: self._set_display_mode("tf"),
                         toggle_state=lambda: self.display_mode == "tf",
                         radio_group="display_mode"),
                MenuItem("Time-Signal",
                         action=lambda: self._set_display_mode("ts"),
                         toggle_state=lambda: self.display_mode == "ts",
                         radio_group="display_mode"),
            ]),
            MenuItem("-"),
            MenuItem("Tiling", children=[
                MenuItem("Columns",
                         action=lambda: setattr(self, "tiling_strategy",
                                                TilingStrategy.COLUMNS),
                         toggle_state=lambda: self.tiling_strategy == TilingStrategy.COLUMNS,
                         radio_group="tiling"),
                MenuItem("Rows",
                         action=lambda: setattr(self, "tiling_strategy",
                                                TilingStrategy.ROWS),
                         toggle_state=lambda: self.tiling_strategy == TilingStrategy.ROWS,
                         radio_group="tiling"),
                MenuItem("Smallest Square",
                         action=lambda: setattr(self, "tiling_strategy",
                                                TilingStrategy.SQUARE),
                         toggle_state=lambda: self.tiling_strategy == TilingStrategy.SQUARE,
                         radio_group="tiling"),
            ]),
            MenuItem("TF Subplots", children=_subplot_toggles(self.subplots_tf)),
            MenuItem("TS Subplots", children=_subplot_toggles(self.subplots_ts)),
            MenuItem("-"),
            MenuItem("Lock Zoom", children=[
                MenuItem("Both Axes",
                         action=lambda: setattr(self, "zoom_lock", None),
                         toggle_state=lambda: self.zoom_lock is None),
                MenuItem("Time Only", shortcut="X",
                         action=lambda: setattr(self, "zoom_lock", "time"),
                         toggle_state=lambda: self.zoom_lock == "time"),
                MenuItem("Frequency Only", shortcut="Y",
                         action=lambda: setattr(self, "zoom_lock", "freq"),
                         toggle_state=lambda: self.zoom_lock == "freq"),
            ]),
        ])

        mb.add_menu("Workspace", [
            MenuItem("TF Axes", children=[
                MenuItem("Frequency Axis", shortcut="F",
                         action=lambda: setattr(self, "show_freq_axis",
                                                not self.show_freq_axis),
                         toggle_state=lambda: self.show_freq_axis),
                MenuItem("Time Axis", shortcut="T",
                         action=lambda: setattr(self, "show_time_axis",
                                                not self.show_time_axis),
                         toggle_state=lambda: self.show_time_axis),
            ]),
            MenuItem("TS Axes", children=[
                MenuItem("Amplitude Axis", shortcut="A",
                         action=lambda: setattr(self, "show_amp_axis",
                                                not self.show_amp_axis),
                         toggle_state=lambda: self.show_amp_axis),
                MenuItem("Time Axis", shortcut="T",
                         action=lambda: setattr(self, "show_time_axis",
                                                not self.show_time_axis),
                         toggle_state=lambda: self.show_time_axis),
            ]),
            MenuItem("-"),
            MenuItem("Left Controls", children=_panel_radio("left")),
            MenuItem("Right Controls", children=_panel_radio("right")),
        ])

        mb.add_menu("Playback", [
            MenuItem("Play / Pause", shortcut="Space",
                     action=lambda: self.player.toggle() if self.player else None,
                     enabled=lambda: self.player is not None),
            MenuItem("-"),
            MenuItem("Jump to Start", shortcut="Home",
                     action=lambda: self.player.seek(self.t_start) if self.player else None,
                     enabled=lambda: self.player is not None),
            MenuItem("Jump to End", shortcut="End",
                     action=lambda: self.player.seek(self.t_end) if self.player else None,
                     enabled=lambda: self.player is not None),
            MenuItem("-"),
            MenuItem("Synthesize Viewport", shortcut="S",
                     action=self._toggle_synth,
                     enabled=lambda: self.synth is not None),
        ])

        mb.add_menu("Synthesis", [
            MenuItem("Original Phase",
                     action=lambda: setattr(self, "phase_mode", "original"),
                     toggle_state=lambda: self.phase_mode == "original",
                     radio_group="phase_mode"),
            MenuItem("Griffin-Lim",
                     action=lambda: setattr(self, "phase_mode", "griffin_lim"),
                     toggle_state=lambda: self.phase_mode == "griffin_lim",
                     radio_group="phase_mode"),
            MenuItem("Zero Phase",
                     action=lambda: setattr(self, "phase_mode", "zero_phase"),
                     toggle_state=lambda: self.phase_mode == "zero_phase",
                     radio_group="phase_mode"),
            MenuItem("-"),
            MenuItem("Normalize Loudness",
                     action=lambda: setattr(self, "normalize_loudness", not self.normalize_loudness),
                     toggle_state=lambda: self.normalize_loudness),
        ])

    def _reset_zoom(self) -> None:
        y_min, y_max = self._y_data_range()
        self.view_x0 = 0.0
        self.view_y0 = y_min
        self.view_x1 = float(self.n_frames)
        self.view_y1 = y_max

    def _set_display_mode(self, mode: str) -> None:
        """Switch display mode and reset Y viewport to the new range."""
        if mode == self.display_mode:
            return
        self.display_mode = mode
        y_min, y_max = self._y_data_range()
        self.view_y0 = y_min
        self.view_y1 = y_max

    # ---- Subplot helpers --------------------------------------------------

    def _active_subplots(self) -> list[SubPlot]:
        """Return visible subplots for the current display mode."""
        src = self.subplots_tf if self.display_mode == "tf" else self.subplots_ts
        return [sp for sp in src if sp.visible]

    def _render_subplot_content(self, subplot: SubPlot,
                                sx: int, sy: int, sw: int, sh: int) -> None:
        """Dispatch rendering for a single subplot into its cell rect."""
        if self.display_mode == "tf":
            if subplot.key == "spectrogram":
                self._render_tf(sx, sy, sw, sh)
        elif self.display_mode == "ts":
            if subplot.key == "waveform":
                self._render_ts(sx, sy, sw, sh)

    def _render_subplot_labels(self, subplots: list[SubPlot],
                               cells: list[tuple[int, int, int, int]]) -> None:
        """Draw a small label in the top-left corner of each cell when tiled."""
        if len(subplots) <= 1:
            return
        atlas = self._glyph_atlas
        if not atlas.tex_id:
            return
        ww, wh = self.win_w, self.win_h
        glEnable(GL_TEXTURE_2D)
        glUseProgram(0)
        for sp, (cx, cy, cw, ch) in zip(subplots, cells):
            atlas.draw_string(sp.label, float(cx + 4), float(cy + 4),
                              ww, wh, anchor_x=0.0, anchor_y=0.0)
        glDisable(GL_TEXTURE_2D)

    def _render_cell_borders(self, cells: list[tuple[int, int, int, int]]) -> None:
        """Draw thin borders around each subplot cell when there are multiple."""
        if len(cells) <= 1:
            return
        ww, wh = float(self.win_w), float(self.win_h)
        glLineWidth(1.0)
        glColor4f(0.35, 0.35, 0.35, 0.6)
        glBegin(GL_LINES)
        for cx, cy, cw, ch in cells:
            x0 = 2.0 * cx / ww - 1.0
            x1 = 2.0 * (cx + cw) / ww - 1.0
            y0 = 1.0 - 2.0 * (cy + ch) / wh
            y1 = 1.0 - 2.0 * cy / wh
            # four edges
            glVertex2f(x0, y0); glVertex2f(x1, y0)
            glVertex2f(x1, y0); glVertex2f(x1, y1)
            glVertex2f(x1, y1); glVertex2f(x0, y1)
            glVertex2f(x0, y1); glVertex2f(x0, y0)
        glEnd()

    def _load_analysis_folder(self, folder_path: str,
                              version_hash: str = "") -> None:
        """Reload the viewer with a different analysis folder."""
        # Check that at least some npz exists
        import glob as _glob
        if version_hash:
            target = os.path.join(folder_path,
                                  f"cqt_data_{version_hash}.npz")
        else:
            target = os.path.join(folder_path, "cqt_data.npz")
        if not os.path.isfile(target):
            # Try any versioned file as fallback
            candidates = _glob.glob(
                os.path.join(folder_path, "cqt_data_*.npz"))
            if not candidates and not os.path.isfile(
                    os.path.join(folder_path, "cqt_data.npz")):
                return
        # Resolve audio for the new folder
        try:
            audio = _resolve_audio(folder_path, None)
        except SystemExit:
            audio = None

        # Close old npz
        if self._npz is not None and hasattr(self._npz, 'close'):
            self._npz.close()

        # Re-init core data from the new folder
        self.analysis_dir = folder_path
        self.audio_path = audio
        self._load_npz(folder_path, version_hash=version_hash)

        # Rebuild mipmap
        self._mipmap = MipmapCache(
            folder_path, self._raw_arrays,
            self.n_bins, self.n_frames,
            settings_hash=self._settings_hash)
        if not self._mipmap.load_from_disk():
            self._mipmap.build_level(self._mipmap.max_level)
            self._mipmap.save_to_disk()
            self._mipmap.start_background_build()

        # Reset viewport
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_x1 = float(self.n_frames)
        self.view_y1 = float(self.n_bins)

        # Reset scan state
        self._front_tex_ids = {}
        self._back_tex_ids = {}
        self._back_buffers = {}
        self._scan_key = None
        self._scan_complete = True

        # Rebuild field sources
        self.field_sources = _build_field_sources(self._has_onset)
        self.field_configs = [FieldConfig() for _ in self.field_sources]
        for i, fs in enumerate(self.field_sources):
            if fs.name == "mag_left":
                self.field_configs[i].target = 0
            elif fs.name == "mag_right":
                self.field_configs[i].target = 1
        if self.gui:
            self.gui.fields = self.field_sources
            self.gui.configs = self.field_configs
            self.gui.selected_idx = 0
            self.gui._sync_data_list()

        # Rebuild player and synth
        if self.player:
            self.player.stop()
        if audio and os.path.isfile(audio):
            self.player = AudioPlayer(audio)
        else:
            self.player = None
        sr_val = int(self._npz["sr"]) if "sr" in self._npz else 44100
        hop_val = int(self._npz["hop_length"]) if "hop_length" in self._npz else 512
        bpo_val = int(self._npz["bins_per_octave"]) if "bins_per_octave" in self._npz else 1200
        self.synth = ViewportSynthPlayer(sr_val, hop_val, bpo_val)

        pygame.display.set_caption(
            f"Spectrogram Viewer \u2014 {os.path.basename(folder_path)}")
        print(f"Loaded analysis: {folder_path}")

    def _unload_data(self) -> None:
        """Release all loaded analysis data so files can be deleted."""
        # Close memory-mapped npz
        if self._npz is not None and hasattr(self._npz, 'close'):
            self._npz.close()
        self._npz = None
        self._raw_arrays = {}
        self._has_onset = False
        self._settings_hash = ""
        self.analysis_dir = None
        self.audio_path = None
        self.n_bins = 0
        self.n_frames = 0
        self.is_stereo = False
        self.times = np.zeros(0, dtype=np.float32)
        self.t_start = 0.0
        self.t_end = 0.0
        self.t_dur = 0.0
        self._freqs = np.zeros(0, dtype=np.float32)
        self._semitone_mask = np.zeros(0, dtype=bool)
        self._note_names = []
        self._pad_influence = None

        # Clear mipmap
        self._mipmap = None

        # Clear textures
        self._front_tex_ids = {}
        self._back_tex_ids = {}
        self._back_buffers = {}
        self._scan_key = None
        self._scan_complete = True

        # Stop audio
        if self.player:
            self.player.stop()
        self.player = None
        self.synth = None

        # Reset viewport
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_x1 = 1.0
        self.view_y1 = 1.0

        # Update panel state
        if self.file_panel:
            self.file_panel.active_folder = None

        pygame.display.set_caption("Spectrogram Viewer")
        print("Data unloaded.")

    def _submit_synth_job(self) -> None:
        if not self.synth or not self._loaded:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100
        label = (f"CQT {self.view_x0:.0f}-{self.view_x1:.0f} / "
                 f"{self.view_y0:.0f}-{self.view_y1:.0f}")
        job = SynthJob(
            job_id=0, label=label, source_type="cqt",
            params={
                "synth": self.synth,
                "raw_arrays": self._raw_arrays,
                "freqs": self._freqs,
                "times": self.times,
                "view_x0": self.view_x0, "view_x1": self.view_x1,
                "view_y0": self.view_y0, "view_y1": self.view_y1,
                "n_frames": self.n_frames, "n_bins": self.n_bins,
                "t_start": self.t_start, "t_dur": self.t_dur,
                "sr": sr_val, "is_stereo": self.is_stereo,
                "normalize": self.normalize_loudness,
            })
        if self.synth_panel:
            self.synth_panel.submit(job)

    def _play_synth_job(self, job: SynthJob) -> None:
        if self.synth and job.result_stereo is not None:
            self.synth.play_array(
                job.result_stereo, job.duration,
                job.view_t0, job.view_t1)

    def _stop_synth_job(self) -> None:
        if self.synth:
            self.synth.stop()

    def _toggle_synth(self) -> None:
        if self.synth and self.synth.playing:
            self.synth.stop()
            if self.synth_panel:
                self.synth_panel._playing_job_id = None
        else:
            self._submit_synth_job()

    # ---- Key handling -----------------------------------------------------

    def _handle_key(self, event: pygame.event.Event) -> bool:
        key = event.key
        if key == K_ESCAPE:
            return False
        if key == K_SPACE and self.player:
            self.player.toggle()
        elif key == K_HOME and self.player:
            self.player.seek(self.t_start)
        elif key == K_END and self.player:
            self.player.seek(self.t_end)
        elif key == K_r:
            self._reset_zoom()
        elif key == K_TAB and self.gui:
            self.gui.selected_idx = (
                (self.gui.selected_idx + 1) % len(self.field_sources))
        elif key == K_f:
            self.show_freq_axis = not self.show_freq_axis
        elif key == K_a:
            self.show_amp_axis = not self.show_amp_axis
        elif key == K_t:
            self.show_time_axis = not self.show_time_axis
        elif key == K_d:
            self._set_display_mode("ts" if self.display_mode == "tf" else "tf")
        elif key == K_x:
            self.zoom_lock = None if self.zoom_lock == "time" else "time"
        elif key == K_y:
            self.zoom_lock = None if self.zoom_lock == "freq" else "freq"
        elif key == K_s:
            self._toggle_synth()
        return True

    # ---- Mouse handling ---------------------------------------------------

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            self._is_panning = True
            self._pan_last = event.pos
        elif event.button == 2:
            # Clear any stuck panning state on middle click
            self._is_panning = False
            self._pan_last = None
            self._seek_to_screen_pos(event.pos)
        elif event.button == 3:
            self._is_panning = False
            self._pan_last = None
            self._rect_start = event.pos
            self._rect_end = event.pos

    def _handle_mouse_up(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            self._is_panning = False
            self._pan_last = None
        elif event.button == 3 and self._rect_start is not None:
            if self._rect_end is not None:
                self._zoom_to_rect(self._rect_start, self._rect_end)
            self._rect_start = None
            self._rect_end = None

    def _handle_mouse_motion(self, event: pygame.event.Event) -> None:
        if self._is_panning and self._pan_last is not None:
            # If left button no longer held (released outside window),
            # clear stuck panning state
            if not event.buttons[0]:
                self._is_panning = False
                self._pan_last = None
                return
            dx = event.pos[0] - self._pan_last[0]
            dy = event.pos[1] - self._pan_last[1]
            self._pan_last = event.pos
            spec = self._spec_rect()
            if spec[2] > 0 and spec[3] > 0:
                data_dx = -dx / spec[2] * (self.view_x1 - self.view_x0)
                data_dy = dy / spec[3] * (self.view_y1 - self.view_y0)
                self._pan_view(data_dx, data_dy)
        if self._rect_start is not None and event.buttons[2]:
            self._rect_end = event.pos

    def _handle_wheel(self, event: pygame.event.Event) -> None:
        mx, my = pygame.mouse.get_pos()
        factor = 1.15 ** event.y
        self._zoom_at_screen(mx, my, factor)

    # ---- Coordinate helpers -----------------------------------------------

    def _spec_rect(self) -> tuple[int, int, int, int]:
        if self.display_mode == "tf":
            use_freq = self.show_freq_axis
            ml = self.axis_margin_left if use_freq else 0
            mr = self.axis_margin_right if use_freq else 0
        else:
            use_amp = self.show_amp_axis
            ml = self.axis_margin_amp if use_amp else 0
            mr = self.axis_margin_amp if use_amp else 0
        mt = self.axis_margin_top if self.show_time_axis else 0
        mb = self.axis_margin_bottom if self.show_time_axis else 0
        menu_h = self.menu_bar.bar_height if self.menu_bar else 0
        left_w = self.dock.left_w if self.dock else 0
        right_w = self.dock.right_w if self.dock else 0
        x = ml + left_w
        y = mt + menu_h
        w = max(1, self.win_w - ml - mr - left_w - right_w)
        h = max(1, self.win_h - mt - mb - menu_h)
        return x, y, w, h

    def _y_data_range(self) -> tuple[float, float]:
        """Full Y data range for the current display mode."""
        if self.display_mode == "ts":
            return (-1.0, 1.0)
        return (0.0, float(self.n_bins))

    def _screen_to_data(self, sx: int, sy: int) -> tuple[float, float]:
        rx, ry, rw, rh = self._spec_rect()
        fx = (sx - rx) / rw
        fy = 1.0 - (sy - ry) / rh
        dx = self.view_x0 + fx * (self.view_x1 - self.view_x0)
        dy = self.view_y0 + fy * (self.view_y1 - self.view_y0)
        return dx, dy

    def _time_to_frame(self, t: float) -> float:
        if self.t_dur <= 0:
            return 0.0
        return (t - self.t_start) / self.t_dur * self.n_frames

    def _frame_to_time(self, f: float) -> float:
        return self.t_start + f / self.n_frames * self.t_dur

    def _pan_view(self, dx: float, dy: float) -> None:
        self.view_x0 += dx
        self.view_x1 += dx
        self.view_y0 += dy
        self.view_y1 += dy
        self._clamp_view()

    def _clamp_view(self) -> None:
        w = self.view_x1 - self.view_x0
        h = self.view_y1 - self.view_y0
        y_min, y_max = self._y_data_range()
        if self.view_x0 < 0:
            self.view_x0 = 0
            self.view_x1 = w
        if self.view_x1 > self.n_frames:
            self.view_x1 = float(self.n_frames)
            self.view_x0 = self.view_x1 - w
        if self.view_y0 < y_min:
            self.view_y0 = y_min
            self.view_y1 = y_min + h
        if self.view_y1 > y_max:
            self.view_y1 = y_max
            self.view_y0 = y_max - h
        self.view_x0 = max(0.0, self.view_x0)
        self.view_y0 = max(y_min, self.view_y0)

    def _zoom_at_screen(self, sx: int, sy: int, factor: float) -> None:
        dx, dy = self._screen_to_data(sx, sy)
        w = self.view_x1 - self.view_x0
        h = self.view_y1 - self.view_y0
        y_min, y_max = self._y_data_range()
        min_x = 4.0
        min_y = 0.01 if self.display_mode == "ts" else 4.0
        nw = w if self.zoom_lock == "freq" else max(min_x, w / factor)
        nh = h if self.zoom_lock == "time" else max(min_y, h / factor)
        # Clamp to full data range
        nw = min(nw, float(self.n_frames))
        nh = min(nh, y_max - y_min)
        rx, ry, rw, rh = self._spec_rect()
        fx = (sx - rx) / rw if rw > 0 else 0.5
        fy = 1.0 - ((sy - ry) / rh if rh > 0 else 0.5)
        self.view_x0 = dx - fx * nw
        self.view_x1 = self.view_x0 + nw
        self.view_y0 = dy - fy * nh
        self.view_y1 = self.view_y0 + nh
        self._clamp_view()

    def _zoom_to_rect(self, p0: tuple[int, int], p1: tuple[int, int]) -> None:
        d0 = self._screen_to_data(*p0)
        d1 = self._screen_to_data(*p1)
        x0, x1 = sorted([d0[0], d1[0]])
        y0, y1 = sorted([d0[1], d1[1]])
        min_y = 0.01 if self.display_mode == "ts" else 4.0
        if x1 - x0 < 4 or y1 - y0 < min_y:
            return
        if self.zoom_lock != "freq":
            self.view_x0, self.view_x1 = x0, x1
        if self.zoom_lock != "time":
            self.view_y0, self.view_y1 = y0, y1
        self._clamp_view()

    def _seek_to_screen_pos(self, pos: tuple[int, int]) -> None:
        dx, _ = self._screen_to_data(*pos)
        t = self._frame_to_time(dx)
        t = max(self.t_start, min(t, self.t_end))
        if self.player:
            self.player.seek(t)

    # ---- OpenGL -----------------------------------------------------------

    def _init_gl(self) -> None:
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.08, 0.08, 0.08, 1.0)

        vert = gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER)
        frag = gl_shaders.compileShader(FRAG_SRC, GL_FRAGMENT_SHADER)
        self._program = gl_shaders.compileProgram(vert, frag)

        # Invalidate textures (GL context may have been recreated)
        self._front_tex_ids = {}
        self._back_tex_ids = {}
        self._back_buffers = {}
        self._scan_key = None
        self._scan_complete = True

    # ---- Texture management -----------------------------------------------

    def _upload_texture_float(self, data: np.ndarray) -> int:
        """Upload an (H, W, 4) float32 RGBA array as a GL_RGBA32F texture."""
        h, w = data.shape[:2]
        raw = np.ascontiguousarray(data).tobytes()
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                     GL_RGBA, GL_FLOAT, raw)
        return tex_id



    def _upload_pygame_surface(self, surf: pygame.Surface) -> int:
        w, h = surf.get_size()
        raw = pygame.image.tobytes(surf, "RGBA", True)
        if self._gui_tex:
            glDeleteTextures([self._gui_tex])
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        self._gui_tex = tex_id
        return tex_id

    # ---- Uniform helpers --------------------------------------------------

    def _set_field_uniforms(self, tex_ids: dict[str, int]) -> None:
        """Bind all data textures and set per-field + output uniforms."""
        loc = lambda n: glGetUniformLocation(self._program, n)

        for tex_name, unit in self._tex_units.items():
            if tex_name not in tex_ids:
                continue
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, tex_ids[tex_name])
            glUniform1i(loc(f"uTex{unit}"), unit)

        gd = self.global_defaults
        field_idx = 0
        for i, fs in enumerate(self.field_sources):
            cfg = self.field_configs[i]
            if cfg.target < 0:
                continue
            if fs.tex_name not in self._tex_units:
                continue
            if field_idx >= 16:
                break
            tex_unit = self._tex_units[fs.tex_name]
            eff_norm = cfg.norm_mode if not cfg.use_global else gd.norm_mode
            eff_gamma = cfg.gamma if not cfg.use_global else gd.gamma
            eff_scale = cfg.scale if not cfg.use_global else gd.scale

            glUniform4i(loc(f"uFieldDesc[{field_idx}]"),
                        tex_unit, fs.tex_channel, cfg.target, eff_norm)
            glUniform4f(loc(f"uFieldParams[{field_idx}]"),
                        eff_gamma, eff_scale, 0.0, 0.0)
            field_idx += 1

        glUniform1i(loc("uFieldCount"), field_idx)

        glUniform4f(loc("uOutGamma"),
                    gd.out_gamma, gd.out_gamma, gd.out_gamma, gd.out_gamma)
        glUniform4f(loc("uOutScale"),
                    gd.out_scale, gd.out_scale, gd.out_scale, gd.out_scale)

    # ---- Rendering --------------------------------------------------------

    def _render(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        glViewport(0, 0, self.win_w, self.win_h)

        sx, sy, sw, sh = self._spec_rect()

        # --- Subplot tiling ---
        active = self._active_subplots()
        if active:
            cells = _tile_rects(sx, sy, sw, sh, len(active),
                                self.tiling_strategy)
            for subplot, cell in zip(active, cells):
                self._render_subplot_content(subplot, *cell)
            self._render_cell_borders(cells)
            self._render_subplot_labels(active, cells)
        else:
            if self.display_mode == "tf":
                self._render_tf(sx, sy, sw, sh)
            elif self.display_mode == "ts":
                self._render_ts(sx, sy, sw, sh)

        # --- Axes: grid lines + glyph-atlas labels (live per-frame) ---
        self._render_axes(sx, sy, sw, sh)

        # --- Playback cursor (file audio) ---
        if self.player:
            cur_t = self.player.get_position()
            cur_frame = self._time_to_frame(cur_t)
            if self.view_x0 <= cur_frame <= self.view_x1:
                frac = ((cur_frame - self.view_x0)
                        / (self.view_x1 - self.view_x0))
                cx = sx + frac * sw
                cx_ndc = 2.0 * cx / self.win_w - 1.0
                sy_ndc = 1.0 - 2.0 * sy / self.win_h
                sy_ndc_bot = 1.0 - 2.0 * (sy + sh) / self.win_h

                glLineWidth(2.0)
                glColor4f(1.0, 1.0, 0.0, 0.8)
                glBegin(GL_LINES)
                glVertex2f(cx_ndc, sy_ndc_bot)
                glVertex2f(cx_ndc, sy_ndc)
                glEnd()

        # --- Synth playback cursor (cyan) ---
        if self.synth and self.synth.playing:
            synth_t = self.synth.get_time()
            synth_frame = self._time_to_frame(synth_t)
            if self.view_x0 <= synth_frame <= self.view_x1:
                frac = ((synth_frame - self.view_x0)
                        / (self.view_x1 - self.view_x0))
                cx = sx + frac * sw
                cx_ndc = 2.0 * cx / self.win_w - 1.0
                sy_ndc = 1.0 - 2.0 * sy / self.win_h
                sy_ndc_bot = 1.0 - 2.0 * (sy + sh) / self.win_h

                glLineWidth(2.0)
                glColor4f(0.0, 1.0, 1.0, 0.9)
                glBegin(GL_LINES)
                glVertex2f(cx_ndc, sy_ndc_bot)
                glVertex2f(cx_ndc, sy_ndc)
                glEnd()

        # --- Rectangle selection overlay ---
        if self._rect_start and self._rect_end:
            r0, r1 = self._rect_start, self._rect_end
            rx0 = 2.0 * min(r0[0], r1[0]) / self.win_w - 1.0
            rx1 = 2.0 * max(r0[0], r1[0]) / self.win_w - 1.0
            ry0 = 1.0 - 2.0 * max(r0[1], r1[1]) / self.win_h
            ry1 = 1.0 - 2.0 * min(r0[1], r1[1]) / self.win_h

            glColor4f(0.3, 0.6, 1.0, 0.25)
            glBegin(GL_QUADS)
            glVertex2f(rx0, ry0); glVertex2f(rx1, ry0)
            glVertex2f(rx1, ry1); glVertex2f(rx0, ry1)
            glEnd()
            glLineWidth(1.0)
            glColor4f(0.4, 0.7, 1.0, 0.8)
            glBegin(GL_LINE_LOOP)
            glVertex2f(rx0, ry0); glVertex2f(rx1, ry0)
            glVertex2f(rx1, ry1); glVertex2f(rx0, ry1)
            glEnd()

        # --- GUI panels (dock) ---
        if self.dock:
            for panel in self.dock.panels_visible():
                panel_surf = panel.render()
                if not panel_surf:
                    continue
                dd_surf = panel.render_dropdown_overlay()
                if dd_surf and hasattr(panel, "_active_dropdown") and panel._active_dropdown:
                    dr = panel._dropdown_rect
                    panel_surf.blit(dd_surf, (dr.x, dr.y))

                tex = self._upload_pygame_surface(panel_surf)
                pw, ph = panel_surf.get_size()
                pr = panel.panel_rect
                px0 = 2.0 * pr.x / self.win_w - 1.0
                px1 = 2.0 * (pr.x + pw) / self.win_w - 1.0
                py1 = 1.0 - 2.0 * pr.y / self.win_h
                py0 = py1 - 2.0 * ph / self.win_h

                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex)
                glColor4f(1, 1, 1, 1)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(px0, py0)
                glTexCoord2f(1, 0); glVertex2f(px1, py0)
                glTexCoord2f(1, 1); glVertex2f(px1, py1)
                glTexCoord2f(0, 1); glVertex2f(px0, py1)
                glEnd()
                glDisable(GL_TEXTURE_2D)

        # --- Menu bar ---
        if self.menu_bar and self.menu_bar.visible:
            bar_surf = self.menu_bar.render(self.win_w)
            if bar_surf:
                tex = self._upload_pygame_surface(bar_surf)
                bw, bh = bar_surf.get_size()
                # top of screen in GL coords
                bx0 = -1.0
                bx1 = 1.0
                by1 = 1.0
                by0 = 1.0 - 2.0 * bh / self.win_h

                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex)
                glColor4f(1, 1, 1, 1)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 1); glVertex2f(bx0, by1)
                glTexCoord2f(1, 1); glVertex2f(bx1, by1)
                glTexCoord2f(1, 0); glVertex2f(bx1, by0)
                glTexCoord2f(0, 0); glVertex2f(bx0, by0)
                glEnd()
                glDisable(GL_TEXTURE_2D)

            dd_surf = self.menu_bar.render_dropdown()
            if dd_surf:
                dr = self.menu_bar.dropdown_screen_rect
                tex = self._upload_pygame_surface(dd_surf)
                dw, dh = dd_surf.get_size()
                dx0 = -1.0 + 2.0 * dr.x / self.win_w
                dx1 = dx0 + 2.0 * dw / self.win_w
                dy1 = 1.0 - 2.0 * dr.y / self.win_h
                dy0 = dy1 - 2.0 * dh / self.win_h

                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex)
                glColor4f(1, 1, 1, 1)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 1); glVertex2f(dx0, dy1)
                glTexCoord2f(1, 1); glVertex2f(dx1, dy1)
                glTexCoord2f(1, 0); glVertex2f(dx1, dy0)
                glTexCoord2f(0, 0); glVertex2f(dx0, dy0)
                glEnd()
                glDisable(GL_TEXTURE_2D)

        # --- HUD ---
        self._render_hud()

    # ---- TF / TS display sub-renderers ------------------------------------

    def _render_tf(self, sx: int, sy: int, sw: int, sh: int) -> None:
        """Draw the time-frequency spectrogram (double-buffered CQT data)."""
        if not (self._program and (self._front_tex_ids or self._back_tex_ids)):
            return
        glUseProgram(self._program)

        u0, u1 = 0.0, 1.0
        v0, v1 = 0.0, 1.0

        nx0 = 2.0 * sx / self.win_w - 1.0
        ny0 = 1.0 - 2.0 * (sy + sh) / self.win_h
        nx1 = 2.0 * (sx + sw) / self.win_w - 1.0
        ny1 = 1.0 - 2.0 * sy / self.win_h

        def _draw_spec_quad() -> None:
            glBegin(GL_QUADS)
            glTexCoord2f(u0, v0); glVertex2f(nx0, ny0)
            glTexCoord2f(u1, v0); glVertex2f(nx1, ny0)
            glTexCoord2f(u1, v1); glVertex2f(nx1, ny1)
            glTexCoord2f(u0, v1); glVertex2f(nx0, ny1)
            glEnd()

        if self._front_tex_ids:
            self._set_field_uniforms(self._front_tex_ids)
            _draw_spec_quad()

        if self._back_tex_ids and not self._scan_complete:
            scan_frac = self._scan_col / max(1, self._scan_total)
            scan_px = int(scan_frac * sw)
            if scan_px > 0:
                glEnable(GL_SCISSOR_TEST)
                glScissor(sx, self.win_h - sy - sh,
                          max(1, scan_px), sh)
                self._set_field_uniforms(self._back_tex_ids)
                _draw_spec_quad()
                glDisable(GL_SCISSOR_TEST)

        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)

    def _render_ts(self, sx: int, sy: int, sw: int, sh: int) -> None:
        """Draw the time-signal waveform in the spec-rect area."""
        if not self.player or sw <= 0 or sh <= 0 or not self._loaded:
            return

        audio = self.player.data       # int16 stereo (N, 2)
        sr = self.player.sr
        n_samples = len(audio)

        # Visible time range from viewport
        t0 = self._frame_to_time(self.view_x0)
        t1 = self._frame_to_time(self.view_x1)
        s0 = max(0, int(t0 * sr))
        s1 = min(n_samples, int(t1 * sr))
        if s1 <= s0:
            return

        # Mix to mono, normalise to [-1, 1]
        chunk = audio[s0:s1]
        mono = chunk.mean(axis=1).astype(np.float32) / 32768.0

        ww, wh = float(self.win_w), float(self.win_h)
        amp_lo = self.view_y0    # e.g. -1.0
        amp_hi = self.view_y1    # e.g. +1.0
        amp_range = amp_hi - amp_lo
        if amp_range <= 0:
            return

        # Downsample: min/max envelope per pixel column
        n = len(mono)
        cols = min(sw, n)
        indices = np.linspace(0, n, cols + 1, dtype=np.int64)
        env_min = np.empty(cols, dtype=np.float32)
        env_max = np.empty(cols, dtype=np.float32)
        for c in range(cols):
            seg = mono[indices[c]:max(indices[c] + 1, indices[c + 1])]
            env_min[c] = seg.min()
            env_max[c] = seg.max()

        # Map amplitude through view_y0..view_y1
        def amp_to_screen_y(a: np.ndarray) -> np.ndarray:
            frac = (a - amp_lo) / amp_range
            return sy + sh * (1.0 - frac)

        # Pre-compute NDC coordinates
        x_screen = sx + np.arange(cols) * (sw / cols)
        x_ndc = 2.0 * x_screen / ww - 1.0
        y_min_screen = amp_to_screen_y(env_min)
        y_max_screen = amp_to_screen_y(env_max)
        y_min_ndc = 1.0 - 2.0 * y_min_screen / wh
        y_max_ndc = 1.0 - 2.0 * y_max_screen / wh

        # Draw filled waveform envelope
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(0.25, 0.65, 0.35, 0.85)
        glBegin(GL_QUAD_STRIP)
        for i in range(cols):
            glVertex2f(float(x_ndc[i]), float(y_min_ndc[i]))
            glVertex2f(float(x_ndc[i]), float(y_max_ndc[i]))
        glEnd()

        # Draw centre line (zero amplitude) if visible
        if amp_lo < 0.0 < amp_hi:
            zero_frac = (0.0 - amp_lo) / amp_range
            zero_sy = sy + sh * (1.0 - zero_frac)
            nx0 = 2.0 * sx / ww - 1.0
            nx1 = 2.0 * (sx + sw) / ww - 1.0
            cy = 1.0 - 2.0 * zero_sy / wh
            glLineWidth(1.0)
            glColor4f(0.5, 0.5, 0.5, 0.5)
            glBegin(GL_LINES)
            glVertex2f(nx0, cy)
            glVertex2f(nx1, cy)
            glEnd()

        glDisable(GL_BLEND)

    # ---- Live axis rendering (grid lines + glyph quads) -------------------

    def _render_axes(self, sx: int, sy: int, sw: int, sh: int) -> None:
        """Draw grid lines across the spectrogram and labels in the margins."""
        atlas = self._glyph_atlas
        if not atlas.tex_id:
            return

        ww, wh = self.win_w, self.win_h
        view_w = self.view_x1 - self.view_x0
        view_h = self.view_y1 - self.view_y0
        if view_w <= 0 or view_h <= 0 or sw <= 0 or sh <= 0:
            return

        tick_color = (0.39, 0.39, 0.39, 0.6)
        grid_color = (0.25, 0.25, 0.25, 0.25)

        # Helper: data-space bin index -> screen y
        def bin_to_sy(b: float) -> float:
            frac = (b - self.view_y0) / view_h
            return sy + sh * (1.0 - frac)

        # Helper: data-space frame index -> screen x
        def frame_to_sx(f: float) -> float:
            frac = (f - self.view_x0) / view_w
            return sx + sw * frac

        def px_to_ndc_x(px: float) -> float:
            return 2.0 * px / ww - 1.0

        def px_to_ndc_y(py: float) -> float:
            return 1.0 - 2.0 * py / wh

        # ---- Frequency axis (horizontal grid lines, labels left + right) --
        if self.show_freq_axis and self.display_mode == "tf":
            # Decide which semitone rows are visible and not too dense
            visible_idxs: list[int] = []
            for i in np.flatnonzero(self._semitone_mask):
                if self.view_y0 <= i <= self.view_y1:
                    visible_idxs.append(int(i))

            # Thin out when too dense: skip labels closer than 14 px
            min_gap = 14
            shown_idxs: list[int] = []
            last_screen_y = -999.0
            for i in visible_idxs:
                scr_y = bin_to_sy(float(i))
                if abs(scr_y - last_screen_y) >= min_gap:
                    shown_idxs.append(i)
                    last_screen_y = scr_y

            # Grid lines
            glBegin(GL_LINES)
            for i in shown_idxs:
                scr_y = bin_to_sy(float(i))
                ny = px_to_ndc_y(scr_y)
                nx0 = px_to_ndc_x(float(sx))
                nx1 = px_to_ndc_x(float(sx + sw))
                glColor4f(*grid_color)
                glVertex2f(nx0, ny)
                glVertex2f(nx1, ny)
                # Tick marks in margins
                glColor4f(*tick_color)
                # left tick
                glVertex2f(px_to_ndc_x(float(sx) - 4), ny)
                glVertex2f(px_to_ndc_x(float(sx)), ny)
                # right tick
                glVertex2f(px_to_ndc_x(float(sx + sw)), ny)
                glVertex2f(px_to_ndc_x(float(sx + sw) + 4), ny)
            glEnd()

            # Labels via glyph atlas
            glEnable(GL_TEXTURE_2D)
            glUseProgram(0)
            for i in shown_idxs:
                name = self._note_names[i] if i < len(self._note_names) else ""
                if not name:
                    continue
                freq = float(self._freqs[i])
                label = f"{name}  {freq:.1f} Hz"
                scr_y = bin_to_sy(float(i))
                # Left margin: right-aligned
                atlas.draw_string(label, float(sx) - 8, scr_y, ww, wh,
                                  anchor_x=1.0, anchor_y=0.5)
                # Right margin: left-aligned
                atlas.draw_string(label, float(sx + sw) + 8, scr_y, ww, wh,
                                  anchor_x=0.0, anchor_y=0.5)
            glDisable(GL_TEXTURE_2D)

        # ---- Amplitude axis (horizontal grid, labels left + right) — TS ---
        if self.show_amp_axis and self.display_mode == "ts":
            amp_lo = self.view_y0
            amp_hi = self.view_y1
            amp_span = amp_hi - amp_lo
            if amp_span > 0:
                amp_step = _nice_step(amp_span, sh, min_spacing=40)
                a_first = math.ceil(amp_lo / amp_step) * amp_step

                def amp_to_screen(a: float) -> float:
                    frac = (a - amp_lo) / amp_span
                    return sy + sh * (1.0 - frac)

                # Collect ticks
                amp_ticks: list[float] = []
                a = a_first
                while a <= amp_hi + amp_step * 0.01:
                    amp_ticks.append(a)
                    a += amp_step

                # Grid lines + ticks
                glBegin(GL_LINES)
                for a in amp_ticks:
                    scr_y = amp_to_screen(a)
                    ny = px_to_ndc_y(scr_y)
                    nx0 = px_to_ndc_x(float(sx))
                    nx1 = px_to_ndc_x(float(sx + sw))
                    glColor4f(*grid_color)
                    glVertex2f(nx0, ny)
                    glVertex2f(nx1, ny)
                    glColor4f(*tick_color)
                    glVertex2f(px_to_ndc_x(float(sx) - 4), ny)
                    glVertex2f(px_to_ndc_x(float(sx)), ny)
                    glVertex2f(px_to_ndc_x(float(sx + sw)), ny)
                    glVertex2f(px_to_ndc_x(float(sx + sw) + 4), ny)
                glEnd()

                # Labels
                glEnable(GL_TEXTURE_2D)
                glUseProgram(0)
                for a in amp_ticks:
                    scr_y = amp_to_screen(a)
                    if abs(a) < 1e-9:
                        label = "0"
                    else:
                        db = 20.0 * math.log10(max(abs(a), 1e-12))
                        label = f"{a:+.2f}  {db:.0f} dB"
                    atlas.draw_string(label, float(sx) - 8, scr_y, ww, wh,
                                      anchor_x=1.0, anchor_y=0.5)
                    atlas.draw_string(label, float(sx + sw) + 8, scr_y, ww, wh,
                                      anchor_x=0.0, anchor_y=0.5)
                glDisable(GL_TEXTURE_2D)

        # ---- Time axis (vertical grid lines, labels top + bottom) ---------
        if self.show_time_axis:
            t0_view = self.t_start + (self.view_x0 / self.n_frames) * self.t_dur
            t1_view = self.t_start + (self.view_x1 / self.n_frames) * self.t_dur
            dur_view = t1_view - t0_view
            if dur_view > 0:
                tick_step = _nice_step(dur_view, sw)
                t_first = math.ceil(t0_view / tick_step) * tick_step

                # Determine decimal places from step size
                if tick_step >= 1.0:
                    fmt = "{:.0f}"
                elif tick_step >= 0.1:
                    fmt = "{:.1f}"
                elif tick_step >= 0.01:
                    fmt = "{:.2f}"
                else:
                    fmt = "{:.3f}"

                # Collect ticks
                ticks: list[tuple[float, str]] = []
                t = t_first
                while t <= t1_view + tick_step * 0.01:
                    frame = (t - self.t_start) / self.t_dur * self.n_frames
                    ticks.append((frame, fmt.format(t)))
                    t += tick_step

                # Grid lines
                glBegin(GL_LINES)
                for frame, _ in ticks:
                    scr_x = frame_to_sx(frame)
                    nx = px_to_ndc_x(scr_x)
                    ny_top = px_to_ndc_y(float(sy))
                    ny_bot = px_to_ndc_y(float(sy + sh))
                    glColor4f(*grid_color)
                    glVertex2f(nx, ny_bot)
                    glVertex2f(nx, ny_top)
                    # Tick marks
                    glColor4f(*tick_color)
                    # top tick
                    glVertex2f(nx, px_to_ndc_y(float(sy)))
                    glVertex2f(nx, px_to_ndc_y(float(sy) - 4))
                    # bottom tick
                    glVertex2f(nx, px_to_ndc_y(float(sy + sh)))
                    glVertex2f(nx, px_to_ndc_y(float(sy + sh) + 4))
                glEnd()

                # Labels via glyph atlas
                glEnable(GL_TEXTURE_2D)
                glUseProgram(0)
                for frame, label in ticks:
                    scr_x = frame_to_sx(frame)
                    # Top margin: centered above
                    atlas.draw_string(label, scr_x, float(sy) - 4, ww, wh,
                                      anchor_x=0.5, anchor_y=1.0)
                    # Bottom margin: centered below
                    atlas.draw_string(label, scr_x, float(sy + sh) + 4, ww, wh,
                                      anchor_x=0.5, anchor_y=0.0)
                glDisable(GL_TEXTURE_2D)

    def _render_hud(self) -> None:
        if not self.player:
            return
        try:
            font = pygame.font.SysFont("consolas", 13)
        except Exception:
            return

        pos = self.player.get_position()
        dur = self.player.duration
        state = "\u25b6" if self.player.playing else "\u23f8"

        _TGT_CH = {0: "R", 1: "G", 2: "B", 3: "A",
                   4: "W", 5: "C", 6: "M", 7: "Y"}
        actives = []
        for i, fs in enumerate(self.field_sources):
            cfg = self.field_configs[i]
            if cfg.target >= 0:
                ch = _TGT_CH.get(cfg.target, "?")
                actives.append(f"{fs.display_name}\u2192{ch}")
        mapping = ", ".join(actives) if actives else "no fields"
        lock_txt = ""
        if self.zoom_lock == "time":
            lock_txt = "  | Zoom: time"
        elif self.zoom_lock == "freq":
            lock_txt = "  | Zoom: freq"
        synth_txt = ""
        if self.synth and self.synth.playing:
            synth_txt = f"  | Synth {self.synth.get_position():.1f}s"
        txt = f"{state} {pos:.1f}s / {dur:.1f}s  |  {mapping}{lock_txt}{synth_txt}"
        hint = "X:lock time  Y:lock freq  S:synth view"
        surf = font.render(txt, True, (220, 220, 200))
        hint_surf = font.render(hint, True, (140, 140, 140))

        tw, th = surf.get_size()
        hw, hh = hint_surf.get_size()
        total_w = max(tw, hw) + 8
        total_h = th + hh + 6
        bg = pygame.Surface((total_w, total_h), pygame.SRCALPHA)
        bg.fill((20, 20, 20, 180))
        bg.blit(surf, (4, 2))
        bg.blit(hint_surf, (4, th + 4))

        tex = self._upload_pygame_surface(bg)
        bw, bh = bg.get_size()
        bx0 = -1.0
        bx1 = -1.0 + 2.0 * bw / self.win_w
        by0 = -1.0
        by1 = -1.0 + 2.0 * bh / self.win_h

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, tex)
        glColor4f(1, 1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(bx0, by0)
        glTexCoord2f(1, 0); glVertex2f(bx1, by0)
        glTexCoord2f(1, 1); glVertex2f(bx1, by1)
        glTexCoord2f(0, 1); glVertex2f(bx0, by1)
        glEnd()
        glDisable(GL_TEXTURE_2D)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive spectrogram viewer with per-field control.",
    )
    parser.add_argument("analysis_dir", nargs="?", default=None,
                        help="Path to the analysis output directory "
                             "(must contain cqt_data.npz). "
                             "If omitted, starts with an empty view.")
    parser.add_argument("audio", nargs="?", default=None,
                        help="Path to the source audio file (.wav). "
                             "Optional when composite_audio.wav exists in the "
                             "analysis directory or wav_path is stored in cqt_data.npz.")
    return parser.parse_args()


def _resolve_audio(analysis_dir: str, audio_arg: str | None) -> str:
    """Determine which audio file to use for playback."""
    composite_wav = os.path.join(analysis_dir, "composite_audio.wav")
    if os.path.isfile(composite_wav):
        return composite_wav
    if audio_arg is not None:
        return audio_arg
    npz_path = os.path.join(analysis_dir, "cqt_data.npz")
    if os.path.isfile(npz_path):
        d = np.load(npz_path, allow_pickle=True)
        if "wav_path" in d:
            p = str(d["wav_path"])
            d.close()
            if os.path.isfile(p):
                return p
        else:
            d.close()
    raise SystemExit(
        "No audio file found. Provide one as the second argument, or run "
        "analysis with --composite to generate composite_audio.wav."
    )


def main() -> None:
    args = parse_args()
    if args.analysis_dir is not None:
        audio = _resolve_audio(args.analysis_dir, args.audio)
        viewer = SpectrogramViewer(args.analysis_dir, audio)
    else:
        viewer = SpectrogramViewer()
    viewer.run()


if __name__ == "__main__":
    main()
