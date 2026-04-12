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
import io
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

from plot_widget import PlotWidget, PlotSeries, PlotMarker, HeatmapBar

# Preferred audio backend: soundfile (libsndfile) handles WAV, FLAC, OGG, AIFF, etc.
# Falls back to scipy.io.wavfile for WAV-only if soundfile is unavailable.
from scipy.io import wavfile
try:
    import soundfile as _sf
    _HAS_SOUNDFILE = True
except ImportError:
    _HAS_SOUNDFILE = False
    _wavfile_fallback = wavfile

_AUDIO_EXTS = frozenset((
    ".wav", ".flac", ".ogg", ".oga", ".opus", ".aiff", ".aif",
    ".mp3", ".w64", ".rf64", ".mat", ".pvf", ".htk", ".sds",
    ".raw", ".voc", ".sd2", ".xi", ".wve", ".mpc",
))


def _load_audio(path: str) -> tuple[int, np.ndarray]:
    """Load any supported audio file → (sample_rate, data).

    *data* is returned in its native dtype (int16, int32, float32, float64, …).
    Multi-channel files keep all channels: shape ``(n_samples, n_channels)``.
    Mono files are 1-D.
    """
    if _HAS_SOUNDFILE:
        data, sr = _sf.read(path, dtype=None, always_2d=False)
        # soundfile returns (samples, channels) and normalises int→float by default.
        # We request dtype=None which returns the file's native type for float
        # formats; for integer formats soundfile always decodes to float64.
        # Re-read as the native subtype when integer to preserve the raw dtype.
        info = _sf.info(path)
        subtype = info.subtype  # e.g. 'PCM_16', 'PCM_24', 'FLOAT', …
        if subtype.startswith("PCM_"):
            bits = int(subtype.split("_")[1])
            if bits <= 16:
                data, sr = _sf.read(path, dtype="int16", always_2d=False)
            elif bits <= 32:
                data, sr = _sf.read(path, dtype="int32", always_2d=False)
            else:
                data, sr = _sf.read(path, dtype="float64", always_2d=False)
        return int(sr), data
    # Fallback: scipy WAV-only
    sr, data = _wavfile_fallback.read(path)
    return int(sr), data


def _load_audio_float(path: str) -> tuple[int, np.ndarray]:
    """Load audio file → (sr, float-normalised-mono).  Always float64, 1-D."""
    sr, data = _load_audio(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float64) / max(abs(info.min), info.max)
    elif data.dtype != np.float64:
        data = data.astype(np.float64)
    return sr, data


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

OVERLAY_VERT_SRC = """
#version 330 compatibility
out vec2 vUV;
void main() {
    gl_Position = gl_Vertex;
    vUV = gl_MultiTexCoord0.xy;
}
"""

OVERLAY_FRAG_SRC = """
#version 330 compatibility
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = texture(uTex, vUV);
}
"""

VERT_SRC = """
#version 330 compatibility
out vec2 vUV;
void main() {
    gl_Position = gl_Vertex;
    vUV = gl_MultiTexCoord0.xy;
}
"""

# ---------------------------------------------------------------------------
# Pass 1 — Field gather: sample data textures, apply norm + gamma*scale,
#           aggregate per-channel via mono-mix.  NO clipping.
#           MRT output 0: vec4(ch0, ch1, ch2, ch3) — 4 data channel values
#           MRT output 1: vec4(alpha, 0, 0, 0)     — display alpha
# ---------------------------------------------------------------------------

PASS1_FRAG_SRC = """
#version 330 compatibility
#define MAX_FIELDS 16

in vec2 vUV;
layout(location = 0) out vec4 outChannels;   // 4 data channels (unbounded)
layout(location = 1) out vec4 outAlpha;      // alpha in .r

uniform sampler2D uTex0, uTex1, uTex2, uTex3;
uniform sampler2D uTex4, uTex5, uTex6, uTex7;

uniform int   uFieldCount;
uniform ivec4 uFieldDesc[MAX_FIELDS];   // (tex_unit, tex_chan, target, norm_mode)
uniform vec4  uFieldParams[MAX_FIELDS]; // (gamma, scale, 0, 0)

uniform int   uMonoMixMode;            // 0=avg 1=first 2=last 3=mid 4=side

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
        // dB: 10*log10 then remap [-60,0] -> [0,1] — NO clamp
        float db = 10.0 * log(max(v, 1e-12)) / log(10.0);
        return (db + 60.0) / 60.0;
    }
    // mode 0 (linear) and mode 2 (rank_order — pre-ranked on CPU)
    // NO clamp — preserve full dynamic range
    return v;
}

float mixChannel(float first, float last, float sum, float count) {
    if (count <= 0.0) return 0.0;
    if (count <= 1.0) return first;
    if (uMonoMixMode == 1) return first;
    if (uMonoMixMode == 2) return last;
    if (uMonoMixMode == 3) return (first + last) * 0.5;
    if (uMonoMixMode == 4) return abs(first - last) * 0.5;
    return sum / count;
}

void main() {
    vec4 accum   = vec4(0.0);
    vec4 counts  = vec4(0.0);
    vec4 first_v = vec4(0.0);
    vec4 last_v  = vec4(0.0);
    float alpha_accum = 0.0;
    float alpha_count = 0.0;
    float alpha_first = 0.0;
    float alpha_last  = 0.0;

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
        // pow requires non-negative base — clamp base only, not result
        float val    = pow(max(normed, 0.0), gamma) * scale;

        if (target <= 3) {
            if (target == 0) { if (counts.r == 0.0) first_v.r = val; last_v.r = val; accum.r += val; counts.r += 1.0; }
            if (target == 1) { if (counts.g == 0.0) first_v.g = val; last_v.g = val; accum.g += val; counts.g += 1.0; }
            if (target == 2) { if (counts.b == 0.0) first_v.b = val; last_v.b = val; accum.b += val; counts.b += 1.0; }
            if (target == 3) { if (counts.a == 0.0) first_v.a = val; last_v.a = val; accum.a += val; counts.a += 1.0; }
        } else if (target == 4) {
            if (alpha_count == 0.0) alpha_first = val;
            alpha_last = val;
            alpha_accum += val;
            alpha_count += 1.0;
        }
    }

    vec4 mixed;
    mixed.r = mixChannel(first_v.r, last_v.r, accum.r, counts.r);
    mixed.g = mixChannel(first_v.g, last_v.g, accum.g, counts.g);
    mixed.b = mixChannel(first_v.b, last_v.b, accum.b, counts.b);
    mixed.a = mixChannel(first_v.a, last_v.a, accum.a, counts.a);

    float alpha_val = mixChannel(alpha_first, alpha_last, alpha_accum, alpha_count);
    if (alpha_count == 0.0) alpha_val = 1.0;

    // Write all 4 data channels + alpha via MRT — NO clipping
    outChannels = mixed;
    outAlpha = vec4(alpha_val, 0.0, 0.0, 0.0);
}
"""

# ---------------------------------------------------------------------------
# Pass 2 — HSL hue blend ONLY.  Read 4 data channels from pass-1 MRT,
#           project each through its HSL hue/sat, blend additively or
#           subtractively into unbounded RGB.  NO gamma, NO scale, NO clip.
#           Output: RGBA32F (raw blended RGB, possibly outside [0,1])
# ---------------------------------------------------------------------------

PASS2_FRAG_SRC = """
#version 330 compatibility
#define N_CHAN 4

in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uPass1Channels;      // pass-1 MRT 0: 4 data channel values
uniform sampler2D uPass1Alpha;         // pass-1 MRT 1: alpha in .r

uniform vec4  uChannelHue;             // hue per data channel (0-1)
uniform vec4  uChannelSat;             // saturation per data channel (0-1)
uniform int   uBlendMode;             // 0=additive  1=subtractive

vec3 hsl2rgb(float h, float s, float l) {
    float c = (1.0 - abs(2.0 * l - 1.0)) * s;
    float hp = h * 6.0;
    float x = c * (1.0 - abs(mod(hp, 2.0) - 1.0));
    vec3 rgb;
    if      (hp < 1.0) rgb = vec3(c, x, 0.0);
    else if (hp < 2.0) rgb = vec3(x, c, 0.0);
    else if (hp < 3.0) rgb = vec3(0.0, c, x);
    else if (hp < 4.0) rgb = vec3(0.0, x, c);
    else if (hp < 5.0) rgb = vec3(x, 0.0, c);
    else               rgb = vec3(c, 0.0, x);
    float m = l - c * 0.5;
    return rgb + m;
}

void main() {
    vec4 ch = texture(uPass1Channels, vUV);   // 4 data channels
    float alpha_val = texture(uPass1Alpha, vUV).r;

    float vals[N_CHAN] = float[](ch.r, ch.g, ch.b, ch.a);
    float hues[N_CHAN] = float[](uChannelHue.x, uChannelHue.y,
                                 uChannelHue.z, uChannelHue.w);
    float sats[N_CHAN] = float[](uChannelSat.x, uChannelSat.y,
                                 uChannelSat.z, uChannelSat.w);

    vec3 rgb = vec3(0.0);

    if (uBlendMode == 0) {
        // Additive: HSL light sources projected onto black
        for (int i = 0; i < N_CHAN; i++) {
            float v = vals[i];
            if (v <= 0.0) continue;
            rgb += hsl2rgb(hues[i], sats[i], v * 0.5);
        }
    } else {
        // Subtractive: gel plates filtering white light
        rgb = vec3(1.0);
        bool any_active = false;
        for (int i = 0; i < N_CHAN; i++) {
            float v = vals[i];
            if (v <= 0.0) continue;
            any_active = true;
            vec3 ink = hsl2rgb(hues[i], sats[i], v * 0.5);
            rgb *= (vec3(1.0) - ink);
        }
        if (any_active) rgb = vec3(1.0) - rgb;
        else rgb = vec3(0.0);
    }

    // Raw blended RGB — NO gamma, NO scale, NO clamp
    fragColor = vec4(rgb, alpha_val);
}
"""

# ---------------------------------------------------------------------------
# Reduce — GPU parallel min/max reduction.
# Iteratively downsamples the pass-2 FBO (mip-chain style) to find the
# actual extent of the blended RGB data.  Dual-output: min in RT0, max in
# RT1.  Each step reads a 2×2 neighbourhood and takes component-wise
# min / max.  Final 1×1 texels are read back to set the HDR input range.
# ---------------------------------------------------------------------------

REDUCE_FRAG_SRC = """
#version 330 compatibility

in vec2 vUV;
layout(location = 0) out vec4 outMin;
layout(location = 1) out vec4 outMax;

uniform sampler2D uMinTex;
uniform sampler2D uMaxTex;
uniform vec2 uTexelSize;               // 1/width, 1/height of source

void main() {
    // Sample 2×2 block from source
    vec2 o = uTexelSize * 0.5;
    vec4 a_min = texture(uMinTex, vUV + vec2(-o.x, -o.y));
    vec4 b_min = texture(uMinTex, vUV + vec2( o.x, -o.y));
    vec4 c_min = texture(uMinTex, vUV + vec2(-o.x,  o.y));
    vec4 d_min = texture(uMinTex, vUV + vec2( o.x,  o.y));

    vec4 a_max = texture(uMaxTex, vUV + vec2(-o.x, -o.y));
    vec4 b_max = texture(uMaxTex, vUV + vec2( o.x, -o.y));
    vec4 c_max = texture(uMaxTex, vUV + vec2(-o.x,  o.y));
    vec4 d_max = texture(uMaxTex, vUV + vec2( o.x,  o.y));

    outMin = min(min(a_min, b_min), min(c_min, d_min));
    outMax = max(max(a_max, b_max), max(c_max, d_max));
}
"""

# ---------------------------------------------------------------------------
# Reduce — seed pass: copy pass-2 output into both min and max RT.
# ---------------------------------------------------------------------------

REDUCE_SEED_FRAG_SRC = """
#version 330 compatibility

in vec2 vUV;
layout(location = 0) out vec4 outMin;
layout(location = 1) out vec4 outMax;

uniform sampler2D uSrcTex;

void main() {
    vec4 c = texture(uSrcTex, vUV);
    outMin = c;
    outMax = c;
}
"""

# ---------------------------------------------------------------------------
# Pass 3 — HDR normalize + out_gamma/out_scale + final display clamp.
#           Input range is auto-detected from GPU reduction.
#           This is the ONLY place clipping occurs.
# ---------------------------------------------------------------------------

PASS3_FRAG_SRC = """
#version 330 compatibility

in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uPass2Tex;           // raw blended RGB from pass 2

uniform int   uHdrNormalizer;          // 0=none 1=clamp 2=reinhard 3=sigmoid 4=linear
uniform vec2  uHdrInputRange;         // (min, max) auto-detected from data
uniform vec2  uHdrOutputRange;        // (lo, hi) user-controlled output range
uniform vec4  uOutGamma;              // display gamma per component
uniform vec4  uOutScale;              // display scale per component

void main() {
    vec4 data = texture(uPass2Tex, vUV);
    vec3 rgb = data.rgb;
    float alpha_val = data.a;

    // HDR normalize: map [auto_min, auto_max] → [out_lo, out_hi]
    if (uHdrNormalizer > 0) {
        float span_in = uHdrInputRange.y - uHdrInputRange.x;
        if (abs(span_in) < 1e-12) span_in = 1.0;
        float span_out = uHdrOutputRange.y - uHdrOutputRange.x;
        for (int c = 0; c < 3; c++) {
            float v = (c == 0) ? rgb.r : ((c == 1) ? rgb.g : rgb.b);
            float t = (v - uHdrInputRange.x) / span_in;
            if (uHdrNormalizer == 1) {       // Clamp
                t = clamp(t, 0.0, 1.0);
            } else if (uHdrNormalizer == 2) { // Reinhard
                t = max(t, 0.0);
                t = t / (1.0 + t);
            } else if (uHdrNormalizer == 3) { // Sigmoid
                t = 1.0 / (1.0 + exp(-6.0 * (t - 0.5)));
            } else if (uHdrNormalizer == 4) { // Linear Compress
                t = clamp(t, 0.0, 1.0);
            }
            float mapped = uHdrOutputRange.x + t * span_out;
            if (c == 0) rgb.r = mapped;
            else if (c == 1) rgb.g = mapped;
            else rgb.b = mapped;
        }
    }

    // Display output gamma/scale — after HDR normalization
    rgb.r = pow(max(rgb.r, 0.0), uOutGamma.x) * uOutScale.x;
    rgb.g = pow(max(rgb.g, 0.0), uOutGamma.y) * uOutScale.y;
    rgb.b = pow(max(rgb.b, 0.0), uOutGamma.z) * uOutScale.z;
    alpha_val = pow(max(alpha_val, 0.0), uOutGamma.w) * uOutScale.w;

    // THE ONLY CLAMP IN THE ENTIRE PIPELINE
    fragColor = clamp(vec4(rgb, alpha_val), 0.0, 1.0);
}
"""

# Legacy single-pass shader kept as concatenation for _apply_curve compat
FRAG_SRC = PASS1_FRAG_SRC



# ---------------------------------------------------------------------------
# Audio player
# ---------------------------------------------------------------------------

class AudioPlayer:
    def __init__(self, path: str) -> None:
        self.path = path
        sr, data = _load_audio(path)
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
        self._play_offset: float = 0.0
        self._music_buf: io.BytesIO | None = None

    @staticmethod
    def _write_wav_header(buf: io.BytesIO, sr: int, n_bytes: int) -> None:
        """Write a minimal 44-byte WAV header for stereo 16-bit PCM."""
        import struct
        buf.write(b'RIFF')
        buf.write(struct.pack('<I', 36 + n_bytes))
        buf.write(b'WAVEfmt ')
        buf.write(struct.pack('<IHHIIHH', 16, 1, 2, sr, sr * 4, 4, 16))
        buf.write(b'data')
        buf.write(struct.pack('<I', n_bytes))

    def _make_music_buffer(self, start_sample: int) -> io.BytesIO:
        """Build an in-memory WAV from *start_sample* onward."""
        raw = self.data[start_sample:].tobytes()
        buf = io.BytesIO()
        self._write_wav_header(buf, int(self.sr), len(raw))
        buf.write(raw)
        buf.seek(0)
        return buf

    def play(self, start_s: float | None = None) -> None:
        if start_s is not None:
            self.position = max(0.0, min(start_s, self.duration))
        self.stop()
        start_sample = int(self.position * self.sr)
        if start_sample >= len(self.data):
            return
        self._play_offset = self.position
        self._music_buf = self._make_music_buffer(start_sample)
        pygame.mixer.music.load(self._music_buf)
        pygame.mixer.music.play()
        self.playing = True

    def stop(self) -> None:
        if self.playing:
            self.position = self.get_position()
            pygame.mixer.music.stop()
            self.playing = False
        self._music_buf = None

    def toggle(self) -> None:
        if self.playing:
            self.stop()
        else:
            self.play()

    def get_position(self) -> float:
        if self.playing:
            ms = pygame.mixer.music.get_pos()
            if ms < 0 or not pygame.mixer.music.get_busy():
                self.playing = False
                return self._play_offset
            pos = self._play_offset + ms / 1000.0
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
TARGET_LABELS = ["none", "Ch1", "Ch2", "Ch3", "Ch4", "Alpha"]
TARGET_VALUES = [-1, 0, 1, 2, 3, 4]
BLEND_MODES = ["Additive", "Subtractive"]

# HDR normalizer modes for pre-clip post-blend compression
HDR_NORMALIZERS = ["None", "Clamp", "Reinhard", "Sigmoid", "Linear Compress"]

# Default channel hues (evenly spaced on the colour wheel, 0-1 range)
DEFAULT_CHANNEL_HUES = [0.0, 0.25, 0.5, 0.75]      # red, yellow-green, cyan, violet
DEFAULT_CHANNEL_SATS = [1.0, 1.0, 1.0, 1.0]

# Virtual texture channel layouts — same channel packing as bass_plot.py
_TEXTURE_LAYOUTS: dict[str, dict[str, str | None]] = {
    "cqt_left_complex":  {"R": "real_left",  "G": "imag_left",  "B": None,              "A": None},
    "cqt_right_complex": {"R": "real_right", "G": "imag_right", "B": None,              "A": None},
    "cqt_left_polar":    {"R": "mag_left",   "G": "phase_left", "B": None,              "A": None},
    "cqt_right_polar":   {"R": "mag_right",  "G": "phase_right","B": None,              "A": None},
    "cqt_mag_stereo":    {"R": "mag_left",   "G": "mag_right",  "B": "mag_similarity",  "A": None},
    "cqt_phase_stereo":  {"R": "phase_left", "G": "phase_right","B": "phase_similarity","A": None},
    "onset":             {"R": "onset",       "G": None,         "B": None,              "A": None},
    "wavelet_coeffs":    {"R": "wvlt_approx","G": "wvlt_detail", "B": None,              "A": None},
}


@dataclass
class FieldSource:
    name: str
    display_name: str
    tex_name: str
    tex_channel: int  # 0=R 1=G 2=B 3=A
    category: str = "cqt"  # "cqt" | "fb" | "wvlt"
    available: bool = True  # False → greyed out in tree


# Category display order and labels for the data tree
_FIELD_CATEGORIES = [
    ("cqt", "CQT"),
    ("fb", "Filterbank"),
    ("wvlt", "Wavelet"),
]

# Map field names to categories
_FIELD_CATEGORY_MAP: dict[str, str] = {
    "real_left": "cqt", "imag_left": "cqt",
    "real_right": "cqt", "imag_right": "cqt",
    "mag_left": "cqt", "phase_left": "cqt",
    "mag_right": "cqt", "phase_right": "cqt",
    "mag_similarity": "cqt", "phase_similarity": "cqt",
    "onset": "cqt",
    "fb_mag_left": "fb", "fb_phase_left": "fb",
    "fb_mag_right": "fb", "fb_phase_right": "fb",
    "wvlt_approx": "wvlt", "wvlt_detail": "wvlt",
}


@dataclass
class FieldConfig:
    target: int = -1       # data channel 0-3, alpha=4, -1=disabled
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
    channel_hues: list[float] | None = None   # per-channel hue (0-1), len 4
    channel_sats: list[float] | None = None   # per-channel saturation (0-1), len 4
    blend_mode: int = 0                        # 0=additive, 1=subtractive
    # HDR dynamic range controls
    hdr_output_lo: float = 0.0    # desired output floor
    hdr_output_hi: float = 1.0    # desired output ceiling
    hdr_normalizer: int = 0       # index into HDR_NORMALIZERS

    def __post_init__(self) -> None:
        if self.channel_hues is None:
            self.channel_hues = list(DEFAULT_CHANNEL_HUES)
        if self.channel_sats is None:
            self.channel_sats = list(DEFAULT_CHANNEL_SATS)


# ---------------------------------------------------------------------------
# Overlay display mode — common grid normalization + layer compositing
# ---------------------------------------------------------------------------

OVERLAY_AXIS_ROLES = ["time", "frequency", "amplitude", "scale"]
OVERLAY_AXIS_LABELS = {
    "time": "Time",
    "frequency": "Frequency",
    "amplitude": "Amplitude",
    "scale": "Scale",
}

OVERLAY_SOURCES = ["cqt", "fb", "wavelet"]
OVERLAY_SOURCE_LABELS = {"cqt": "CQT", "fb": "Filterbank", "wavelet": "Wavelet"}


@dataclass
class OverlayLayer:
    """One data source layer in the overlay composite."""
    source: str = "cqt"        # "cqt" | "fb" | "wavelet"
    enabled: bool = True
    opacity: float = 1.0       # 0..1
    x_role: str = "time"       # which data dimension maps to screen X
    y_role: str = "frequency"  # which data dimension maps to screen Y
    channel: int = 0           # data channel used (0=Ch1 .. 3=Ch4)

    @property
    def transposed(self) -> bool:
        """True when axes are swapped from the native orientation."""
        return self.x_role == "frequency" and self.y_role == "time"


@dataclass
class OverlayConfig:
    """Global settings for the overlay display mode."""
    layers: list[OverlayLayer] | None = None
    common_x: str = "time"        # master X axis interpretation
    common_y: str = "frequency"   # master Y axis interpretation

    def __post_init__(self) -> None:
        if self.layers is None:
            self.layers = [
                OverlayLayer("cqt", True, 1.0, "time", "frequency", 0),
                OverlayLayer("fb", False, 0.7, "time", "frequency", 0),
                OverlayLayer("wavelet", False, 0.5, "time", "scale", 0),
            ]


def _overlay_resample_to_grid(
    data: np.ndarray,
    src_x: np.ndarray,
    src_y: np.ndarray,
    dst_x: np.ndarray,
    dst_y: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    """Resample a 2-D (rows=Y, cols=X) array onto a common grid.

    *src_x*, *src_y* are the native coordinate arrays of *data*.
    *dst_x*, *dst_y* define the target grid.  Nearest-neighbour lookup
    is used so the native dtype and precision are preserved.

    If *transpose* is True the data's axes are swapped before resampling
    (rows become X, cols become Y).
    """
    if transpose:
        data = data.T
        src_x, src_y = src_y, src_x

    out_h = len(dst_y)
    out_w = len(dst_x)
    if out_h == 0 or out_w == 0 or data.size == 0:
        return np.zeros((out_h, out_w), dtype=data.dtype)

    h_src, w_src = data.shape[:2]

    # Map destination coordinates into source index space via
    # nearest-neighbour lookup in the sorted source axes.
    xi = np.clip(np.searchsorted(src_x, dst_x) - 1, 0, w_src - 1)
    yi = np.clip(np.searchsorted(src_y, dst_y) - 1, 0, h_src - 1)

    return data[np.ix_(yi, xi)]


def _hsl2rgb_vec(h: float, s: float, l: float) -> tuple[float, float, float]:
    """CPU replica of the GLSL hsl2rgb used in the spectrogram shader."""
    c = (1.0 - abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - abs((hp % 2.0) - 1.0))
    if hp < 1.0:
        rgb = (c, x, 0.0)
    elif hp < 2.0:
        rgb = (x, c, 0.0)
    elif hp < 3.0:
        rgb = (0.0, c, x)
    elif hp < 4.0:
        rgb = (0.0, x, c)
    elif hp < 5.0:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    m = l - c * 0.5
    return (rgb[0] + m, rgb[1] + m, rgb[2] + m)


def _broadcast_hsl2rgb_additive(
    out: np.ndarray, v: np.ndarray, h: float, s: float,
) -> None:
    """Add per-pixel HSL(h, s, v*0.5) → RGB into *out* (H×W×3).

    Replicates the shader's ``rgb += hsl2rgb(hue, sat, v * 0.5);``
    loop exactly.  *v* is (H, W) float32 in [0, 1].
    """
    # l = v * 0.5  →  c = (1 - |2l - 1|) * s = (1 - |v - 1|) * s = v * s
    # (since v in [0,1], 2l-1 = v-1 ∈ [-1,0], abs → 1-v, 1-abs → v)
    c = v * s
    hp = h * 6.0
    x = c * (1.0 - abs((hp % 2.0) - 1.0))
    m = v * 0.5 - c * 0.5   # l - c/2

    if hp < 1.0:
        r, g, b = c, x, np.float32(0.0)
    elif hp < 2.0:
        r, g, b = x, c, np.float32(0.0)
    elif hp < 3.0:
        r, g, b = np.float32(0.0), c, x
    elif hp < 4.0:
        r, g, b = np.float32(0.0), x, c
    elif hp < 5.0:
        r, g, b = x, np.float32(0.0), c
    else:
        r, g, b = c, np.float32(0.0), x

    out[:, :, 0] += r + m
    out[:, :, 1] += g + m
    out[:, :, 2] += b + m


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
# Grid style options for axis grid lines
# ---------------------------------------------------------------------------

_GRID_STYLES = ["Subtle", "Bold", "None"]
_GRID_STYLE_COLORS = {
    "Subtle": ((0.25, 0.25, 0.25, 0.25), (0.39, 0.39, 0.39, 0.6)),
    "Bold":   ((0.0, 0.0, 0.0, 0.8),     (0.0, 0.0, 0.0, 1.0)),
    "None":   (None, None),
}


# ---------------------------------------------------------------------------
# Filter bank decomposition — arbitrary crossover with perfect reconstruction
# ---------------------------------------------------------------------------

_FILTER_TYPES = ["Linkwitz-Riley 4", "Butterworth 4", "Butterworth 8"]
_CQT_ALGORITHMS = ["librosa", "nsgt"]
_FB_CONFIG_MODES = ["Manual", "Auto (log)"]


def hybrid_crossover_freq(bpo: int, hop: int, sr: int) -> float:
    """Return the frequency below which filterbank Hilbert envelopes
    provide tighter time resolution than the CQT at the given settings.

    CQT bin bandwidth at frequency *f* is ``f * (2^(1/bpo) - 1)``.
    The CQT hop adequately samples the envelope when
    ``hop <= sr / (2 * bandwidth)``, i.e. above::

        f_cross = sr / (2 * hop * (2^(1/bpo) - 1))
    """
    ratio = 2.0 ** (1.0 / bpo) - 1.0
    if ratio <= 0:
        return float(sr / 2)
    return sr / (2.0 * hop * ratio)

_RESAMPLE_ENGINES = ["scipy", "soxr"]
_RESAMPLE_INTERPS = ["sinc", "linear", "nearest", "cubic"]
_RESAMPLE_TAPS = ["8", "16", "32", "64", "128", "256"]
_RESAMPLE_PRECISIONS = ["16-bit", "24-bit", "32-bit float", "64-bit float"]

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


def _snap_to_note_up(freq: float) -> float:
    """Snap *freq* UP to the nearest semitone >= freq."""
    if freq <= 0:
        return freq
    midi = 69 + 12 * math.log2(freq / 440.0)
    return _midi_to_freq(math.ceil(midi - 1e-9))


def _snap_to_note_down(freq: float) -> float:
    """Snap *freq* DOWN to the nearest semitone <= freq."""
    if freq <= 0:
        return freq
    midi = 69 + 12 * math.log2(freq / 440.0)
    return _midi_to_freq(math.floor(midi + 1e-9))


def _freq_fmt(val: float) -> str:
    """Return a format string with appropriate decimal precision for *val* Hz."""
    av = abs(val)
    if av >= 100:
        return "{:.0f}"
    if av >= 10:
        return "{:.1f}"
    if av >= 1:
        return "{:.2f}"
    if av >= 0.01:
        return "{:.4f}"
    return "{:.6f}"


def _snap_bpo(val: float) -> int:
    """Snap a raw BPO value to the nearest entry in _BPO_SNAPS."""
    best = _BPO_SNAPS[0]
    best_dist = abs(val - best)
    for s in _BPO_SNAPS[1:]:
        d = abs(val - s)
        if d < best_dist:
            best, best_dist = s, d
    return best


_FB_LABEL_MODES = ["Range", "Mid+BW"]


@dataclass
class BandDef:
    """One band in a crossover filter bank."""
    fmin: float    # 0.0 means from DC
    fmax: float    # >= Nyquist means to Nyquist
    label: str = ""
    enabled: bool = True

    @property
    def centre(self) -> float:
        """Geometric centre frequency of the band."""
        lo = max(self.fmin, 1e-3)
        return math.sqrt(lo * self.fmax)

    def auto_label(self, sr: int) -> str:
        nyq = sr / 2.0
        lo = "DC" if self.fmin <= 0.0 else f"{self.fmin:.0f}"
        hi = "Nyq" if self.fmax >= nyq else f"{self.fmax:.0f}"
        return f"{lo}-{hi} Hz"

    def mid_label(self, sr: int) -> str:
        """Label showing middle frequency and bandwidth."""
        nyq = sr / 2.0
        fc = self.centre
        bw = min(self.fmax, nyq) - max(self.fmin, 0.0)
        note = _freq_to_note_name(fc)
        if fc >= 1000:
            mid_str = f"{fc / 1000:.2f}k"
        else:
            mid_str = f"{fc:.1f}"
        if bw >= 1000:
            bw_str = f"{bw / 1000:.1f}k"
        else:
            bw_str = f"{bw:.0f}"
        return f"{mid_str}Hz {note} bw{bw_str}"


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

    @staticmethod
    def bands_from_log_spacing(
        bands_per_octave: int,
        fmin: float,
        fmax: float,
        sr: int,
        ceiling: float | None = None,
    ) -> list[BandDef]:
        """Generate log-spaced crossover bands analogous to CQT bins.

        *bands_per_octave* bands are placed per octave between *fmin*
        and *fmax*.  Each crossover sits at the geometric midpoint
        between adjacent band centres, so the resulting bands tile the
        frequency axis without gaps or overlaps.

        If *ceiling* is given, the last band's upper edge is clamped to
        *ceiling* instead of extending to Nyquist.
        """
        nyq = sr / 2.0
        fmin = max(fmin, 1.0)
        fmax = min(fmax, nyq)
        top = ceiling if ceiling is not None else nyq
        top = min(top, nyq)
        # When ceiling is set, also limit which centres are generated
        centre_limit = min(fmax, top) if ceiling is not None else fmax
        if centre_limit <= fmin or bands_per_octave < 1:
            return [BandDef(fmin=0.0, fmax=top,
                            label=BandDef(0.0, top).auto_label(sr))]
        # Band centres snapped to the nearest equal-temperament semitone
        # so that every band is musically meaningful regardless of BPO.
        # Start from the semitone nearest to fmin and step by
        # 12/bpo semitones (so bpo=12 → 1 semitone, bpo=6 → 2, etc.).
        step_semitones = 12.0 / bands_per_octave
        # Nearest semitone index to fmin (relative to A4=440)
        midi_start = round(69 + 12 * math.log2(fmin / 440.0))
        centres: list[float] = []
        k = 0
        while True:
            midi = midi_start + k * step_semitones
            fc = 440.0 * (2.0 ** ((midi - 69) / 12.0))
            if fc > centre_limit:
                break
            if fc >= fmin * 0.99:  # allow tiny undershoot from rounding
                centres.append(fc)
            k += 1
        if not centres:
            centres = [_snap_to_note(fmin)]
        # Crossovers at geometric midpoints between consecutive centres
        crossovers: list[float] = []
        for i in range(len(centres) - 1):
            crossovers.append(math.sqrt(centres[i] * centres[i + 1]))
        # Build bands: DC→first_xo, xo_i→xo_{i+1}, …, last_xo→ceiling/Nyq
        bands: list[BandDef] = []
        prev = 0.0
        for xf in crossovers:
            bands.append(BandDef(fmin=prev, fmax=xf))
            prev = xf
        bands.append(BandDef(fmin=prev, fmax=top))
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

    def compute_envelopes(self, hop: int | list[int] | None = None,
                          hilbert_progress: Any = None,
                          ) -> tuple[list[np.ndarray], list[np.ndarray],
                                     list[int]]:
        """Compute per-band magnitude and phase envelopes via analytic signal.

        *hop* can be a single int (shared), a per-band list, or ``None``
        to auto-compute from each band's bandwidth.

        Returns ``(mags, phases, hops)`` where each element is a list of
        1-D ``float32`` arrays (one per band) and *hops* records the hop
        used for each band.

        When PyTorch + CUDA are available the Hilbert transforms and
        frame-wise peak extraction run as batched GPU operations.
        Falls back to serial CPU (scipy) otherwise.
        """
        if not self.subbands:
            return [], [], []
        n_bands = len(self.subbands)
        n_samples = len(self.subbands[0])

        # Resolve per-band hops
        if hop is None:
            hops = [self.band_hop(b, self.sr) for b in self.bands]
        elif isinstance(hop, int):
            hops = [hop] * n_bands
        else:
            hops = list(hop)

        # --- Try GPU path ---------------------------------------------------
        try:
            import torch
            if torch.cuda.is_available():
                return self._compute_envelopes_gpu(
                    torch, n_samples, hops,
                    hilbert_progress=hilbert_progress)
        except ImportError:
            pass

        # --- CPU fallback ----------------------------------------------------
        return self._compute_envelopes_cpu(
            n_samples, hops, hilbert_progress=hilbert_progress)

    def _compute_envelopes_gpu(self, torch: Any, n_samples: int,
                               hops: list[int],
                               hilbert_progress: Any = None,
                               ) -> tuple[list[np.ndarray], list[np.ndarray],
                                          list[int]]:
        """Batched GPU Hilbert transform with VRAM-aware chunking.

        Preserves the native precision of the subbands.  Extracts frame
        peaks per-chunk so neither VRAM nor system RAM need full
        (n_bands, n_samples) arrays.
        """
        n_bands = len(self.subbands)
        device = torch.device("cuda")

        # Detect native precision from the first subband
        native_dtype = np.result_type(self.subbands[0])
        if native_dtype == np.float64:
            real_dt = np.float64
            cplx_dt = np.complex128
            real_bytes = 8
            cplx_bytes = 16
        else:
            real_dt = np.float32
            cplx_dt = np.complex64
            real_bytes = 4
            cplx_bytes = 8

        # --- VRAM budget --------------------------------------------------
        # Exact peak trace for B bands of N samples (R=real, C=complex bytes):
        #
        # Resident always: h_filt = N*C
        #
        # Step  Tensors on device              Total VRAM
        # ----  ----------------------------   ----------------------------
        #  1    sig.to(device)                  N*C + B*N*R
        #  2    Xf = fft(sig)  [sig+Xf live]   N*C + B*N*R + B*N*C   ← candidate
        #  3    del sig                         N*C + B*N*C
        #  4    Xf.mul_(h_filt) [in-place]      N*C + B*N*C
        #  5    analytic = ifft(Xf) [Xf+ana]    N*C + 2*B*N*C         ← candidate
        #  6    del Xf                          N*C + B*N*C
        #  7    .abs()/.angle() + .float()       N*C + B*N*C + B*N*4   ← candidate
        #
        # Peak = max(step2, step5, step7)
        #   step2 = N*C + B*N*(R + C)
        #   step5 = N*C + 2*B*N*C
        #   step7 = N*C + B*N*(C + 4)
        #
        # For f64: R=8,C=16 → step2= B*N*24, step5= B*N*32 → peak=step5
        # For f32: R=4,C=8  → step2= B*N*12, step5= B*N*16 → peak=step5
        # In both cases step5 dominates.
        #
        # peak_total = N*C + 2*B*N*C
        # Solve: B = (usable - N*C) / (2*N*C)
        try:
            free_mem = torch.cuda.mem_get_info()[0]
        except Exception:
            free_mem = 2 * 1024**3
        usable = int(free_mem * 0.5)                  # use at most 50% of free VRAM
        h_filt_cost = n_samples * cplx_bytes          # N*C  (h_filt)
        per_band_peak = 2 * n_samples * cplx_bytes    # 2*N*C per band at step5
        max_bands = max(1, int((usable - h_filt_cost) // per_band_peak))

        print(f"  VRAM budget: {free_mem / 1024**2:.0f} MB free, "
              f"{(h_filt_cost + per_band_peak * max_bands) / 1024**2:.0f} MB for "
              f"{max_bands}/{n_bands} bands/chunk")

        # --- One-sided Hilbert filter (native complex, uploaded once) -----
        h_filter_np = np.zeros(n_samples, dtype=cplx_dt)
        h_filter_np[0] = 1.0
        if n_samples % 2 == 0:
            h_filter_np[1:n_samples // 2] = 2.0
            h_filter_np[n_samples // 2] = 1.0
        else:
            h_filter_np[1:(n_samples + 1) // 2] = 2.0
        h_filt = torch.from_numpy(h_filter_np).to(device)

        mags_out: list[np.ndarray] = [None] * n_bands   # type: ignore[list-item]
        phases_out: list[np.ndarray] = [None] * n_bands  # type: ignore[list-item]

        chunk_start = 0
        while chunk_start < n_bands:
            chunk_end = min(chunk_start + max_bands, n_bands)

            while True:
                chunk_size = chunk_end - chunk_start
                try:
                    stacked = np.stack([
                        np.asarray(self.subbands[i], dtype=real_dt)
                        for i in range(chunk_start, chunk_end)
                    ])
                    sig = torch.from_numpy(stacked).to(device)
                    del stacked

                    Xf = torch.fft.fft(sig, dim=-1)
                    del sig
                    Xf.mul_(h_filt)
                    analytic = torch.fft.ifft(Xf, dim=-1)
                    del Xf
                    torch.cuda.empty_cache()

                    env_chunk = analytic.abs().float().cpu().numpy()
                    phi_chunk = analytic.angle().float().cpu().numpy()
                    del analytic
                    torch.cuda.empty_cache()
                    break  # success

                except torch.cuda.OutOfMemoryError:
                    # Clean up whatever partial allocs exist
                    for name in ('stacked', 'sig', 'Xf', 'analytic',
                                 'env_chunk', 'phi_chunk'):
                        if name in locals():
                            del locals()[name]
                    torch.cuda.empty_cache()

                    max_bands = max(1, chunk_size // 2)
                    chunk_end = min(chunk_start + max_bands, n_bands)
                    print(f"  OOM — reducing to {max_bands} bands/chunk")
                    if chunk_size <= 1:
                        raise  # truly cannot fit even 1 band

            # Frame peak extraction per band (CPU, immediate)
            for j, i in enumerate(range(chunk_start, chunk_end)):
                h = hops[i]
                n_frames = int(math.ceil(n_samples / h))
                env = env_chunk[j]
                phi = phi_chunk[j]
                m = np.empty(n_frames, dtype=np.float32)
                p = np.empty(n_frames, dtype=np.float32)
                for f in range(n_frames):
                    s0 = f * h
                    s1 = min(s0 + h, n_samples)
                    if s1 <= s0:
                        break
                    seg = env[s0:s1]
                    idx = int(np.argmax(seg))
                    m[f] = seg[idx]
                    p[f] = phi[s0 + idx]
                mags_out[i] = m
                phases_out[i] = p
            del env_chunk, phi_chunk

            print(f"  Hilbert envelopes: {chunk_end}/{n_bands} bands on GPU")
            if hilbert_progress:
                hilbert_progress(chunk_end, n_bands)

            chunk_start = chunk_end

        del h_filt
        torch.cuda.empty_cache()
        return mags_out, phases_out, hops

    def _compute_envelopes_cpu(self, n_samples: int,
                               hops: list[int],
                               hilbert_progress: Any = None,
                               ) -> tuple[list[np.ndarray], list[np.ndarray],
                                          list[int]]:
        """Serial CPU Hilbert envelope computation (scipy fallback)."""
        from scipy.signal import hilbert

        n_bands = len(self.subbands)
        mags: list[np.ndarray] = []
        phases: list[np.ndarray] = []
        for i, sub in enumerate(self.subbands):
            h = hops[i]
            n_frames = int(math.ceil(n_samples / h))
            analytic = hilbert(np.asarray(sub, dtype=np.float64))
            env = np.abs(analytic).astype(np.float32)
            phi = np.angle(analytic).astype(np.float32)
            m = np.empty(n_frames, dtype=np.float32)
            p = np.empty(n_frames, dtype=np.float32)
            for f in range(n_frames):
                s0 = f * h
                s1 = min(s0 + h, n_samples)
                if s1 <= s0:
                    break
                seg = env[s0:s1]
                idx = np.argmax(seg)
                m[f] = seg[idx]
                p[f] = phi[s0 + idx]
            mags.append(m)
            phases.append(p)
            if hilbert_progress and (i % 10 == 0 or i == n_bands - 1):
                hilbert_progress(i + 1, n_bands)
        return mags, phases, hops

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

    @staticmethod
    def band_hop(band: "BandDef", sr: int) -> int:
        """Nyquist-limited hop for one band based on its bandwidth."""
        if sr <= 0:
            return 1
        nyq = sr / 2.0
        bw = min(band.fmax, nyq) - max(band.fmin, 0.0)
        if bw <= 0:
            return 1
        return max(1, int(sr / (2.0 * bw)))

    def _filter_one_band(self, signal: np.ndarray, band: "BandDef",
                         nyq: float) -> np.ndarray:
        """Filter *signal* for a single band and return the subband array."""
        from scipy.signal import butter, sosfilt

        if not band.enabled:
            return np.zeros_like(signal)
        flo, fhi = band.fmin, band.fmax
        if flo <= 0.0 and fhi >= nyq:
            return signal.copy()

        if self.filter_type.startswith("Linkwitz-Riley"):
            order = int(self.filter_type.split()[-1])
            half = order // 2
            if flo <= 0.0:
                wn = min(fhi / nyq, 0.9999)
                sos = butter(half, wn, btype="low", output="sos")
                return sosfilt(sos, sosfilt(sos, signal)).astype(np.float32)
            if fhi >= nyq:
                wn = max(flo / nyq, 0.0001)
                sos = butter(half, wn, btype="high", output="sos")
                return sosfilt(sos, sosfilt(sos, signal)).astype(np.float32)
            wn_lo = max(flo / nyq, 0.0001)
            wn_hi = min(fhi / nyq, 0.9999)
            sos_lo = butter(half, wn_lo, btype="high", output="sos")
            sos_hi = butter(half, wn_hi, btype="low", output="sos")
            return sosfilt(sos_hi, sosfilt(sos_hi,
                        sosfilt(sos_lo, sosfilt(sos_lo, signal)))).astype(np.float32)
        else:
            order = int(self.filter_type.split()[-1])
            if flo <= 0.0:
                wn = min(fhi / nyq, 0.9999)
                sos = butter(order, wn, btype="low", output="sos")
            elif fhi >= nyq:
                wn = max(flo / nyq, 0.0001)
                sos = butter(order, wn, btype="high", output="sos")
            else:
                wn = [max(flo / nyq, 0.0001), min(fhi / nyq, 0.9999)]
                sos = butter(order, wn, btype="band", output="sos")
            return sosfilt(sos, signal).astype(np.float32)

    def compute_and_save(self, signal: np.ndarray, bands: list["BandDef"],
                         analysis_dir: str, sr: int,
                         batch_size: int = 32,
                         wav_path: str | None = None,
                         progress_cb: Any = None) -> None:
        """Filter, save WAVs, and bake envelopes in batches.

        Filtering uses the existing IIR sosfilt (preserving exact LR
        phase alignment) but runs bands in parallel threads since
        sosfilt releases the GIL.  Hilbert envelope computation is
        dispatched to GPU when available via ``compute_envelopes``.

        After completion ``self.bands`` is set but ``self.subbands`` is
        empty (the WAV files on disk are the authoritative copies).
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        self._source_signal = signal.copy()
        self.bands = list(bands)
        self.subbands = []
        nyq = self.sr / 2.0
        n_bands = len(bands)
        n_samples = len(signal)

        fb_dir = os.path.join(analysis_dir, "filterbank")
        os.makedirs(fb_dir, exist_ok=True)

        meta_bands: list[dict] = [None] * n_bands  # type: ignore[list-item]
        all_subs: list[np.ndarray | None] = [None] * n_bands
        t0 = time.monotonic()

        # --- Parallel IIR filtering + WAV write ---------------------------
        # sosfilt releases the GIL so threads give true parallelism here.
        n_workers = min(os.cpu_count() or 4, 16)

        def _filter_and_write(i: int) -> dict:
            sub = self._filter_one_band(signal, bands[i], nyq)
            all_subs[i] = sub
            fname = f"band_{i:02d}.wav"
            fpath = os.path.join(fb_dir, fname)
            peak = float(np.abs(sub).max())
            if peak > 0:
                scaled = (sub / peak * 32767).astype(np.int16)
            else:
                scaled = np.zeros(len(sub), dtype=np.int16)
            wavfile.write(fpath, sr, scaled)
            return {
                "index": i,
                "fmin": bands[i].fmin,
                "fmax": bands[i].fmax,
                "label": bands[i].label,
                "file": fname,
                "peak_amplitude": peak,
            }

        done_count = 0
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_filter_and_write, i): i
                       for i in range(n_bands)}
            for fut in as_completed(futures):
                i = futures[fut]
                meta_bands[i] = fut.result()
                done_count += 1
                frac = done_count / n_bands * 0.5  # filtering = 0..50%
                if done_count % 20 == 0 or done_count == n_bands:
                    elapsed = time.monotonic() - t0
                    eta = (elapsed / done_count) * (n_bands - done_count)
                    print(f"\r  Filtering bands: {done_count}/{n_bands} "
                          f"({done_count * 100 // n_bands}%) — "
                          f"{elapsed:.1f}s elapsed, ~{eta:.0f}s remaining",
                          end="", flush=True)
                if progress_cb:
                    progress_cb(frac, f"Filtering {done_count}/{n_bands}")
        print()

        # --- Compute Hilbert envelopes (GPU when available) ---------------
        # Temporarily set self.subbands so compute_envelopes can use them,
        # then clear to free memory.
        if progress_cb:
            progress_cb(0.5, "Hilbert envelopes")
        print("  Computing Hilbert envelopes ...", end="", flush=True)
        t1 = time.monotonic()
        self.subbands = [s for s in all_subs]  # type: ignore[misc]
        del all_subs

        def _hilbert_progress(chunk_done: int, chunk_total: int) -> None:
            # Hilbert = 50..90% of total
            frac = 0.5 + 0.4 * (chunk_done / max(chunk_total, 1))
            if progress_cb:
                progress_cb(frac, f"Hilbert {chunk_done}/{chunk_total}")

        mags, phases, hops = self.compute_envelopes(
            hilbert_progress=_hilbert_progress)
        self.subbands = []  # free the subband memory
        print(f" done ({time.monotonic() - t1:.1f}s)")
        if progress_cb:
            progress_cb(0.9, "Saving")

        # --- Write metadata JSON ------------------------------------------
        import json as _json
        meta: dict = {
            "filter_type": self.filter_type,
            "sr": sr,
            "n_bands": n_bands,
            "bands": meta_bands,
        }
        if wav_path:
            meta["wav_path"] = os.path.abspath(wav_path)
        with open(os.path.join(fb_dir, "filterbank_meta.json"), "w") as f:
            _json.dump(meta, f, indent=2)

        # --- Write envelopes NPZ -----------------------------------------
        self.save_envelopes(analysis_dir, mags, phases, hops)
        print(f"  Filterbank complete: {n_bands} bands in "
              f"{time.monotonic() - t0:.1f}s")
        if progress_cb:
            progress_cb(1.0, "FB done")

    def save(self, analysis_dir: str, sr: int) -> None:
        """Save subbands as WAV files + metadata, and bake Hilbert
        envelopes into *fb_envelopes.npz* with per-band hops derived
        from each band's bandwidth.

        If subbands are already in memory (from a prior ``compute()``),
        this saves them directly.  For large decompositions use
        ``compute_and_save()`` which batches to control peak memory.
        """
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

        # Bake Hilbert envelopes while subbands are still in memory
        mags, phases, hops = self.compute_envelopes()
        self.save_envelopes(analysis_dir, mags, phases, hops)

    def save_envelopes(self, analysis_dir: str,
                       mags: list[np.ndarray],
                       phases: list[np.ndarray],
                       hops: list[int]) -> None:
        """Persist per-band envelopes as *fb_envelopes.npz*."""
        fb_dir = os.path.join(analysis_dir, "filterbank")
        os.makedirs(fb_dir, exist_ok=True)
        data: dict[str, np.ndarray] = {
            "n_bands": np.int64(len(mags)),
            "hops": np.array(hops, dtype=np.int64),
        }
        for i, (m, p) in enumerate(zip(mags, phases)):
            data[f"mag_{i}"] = m
            data[f"phase_{i}"] = p
        np.savez_compressed(
            os.path.join(fb_dir, "fb_envelopes.npz"), **data)

    @staticmethod
    def load_envelopes(analysis_dir: str
                       ) -> tuple[list[np.ndarray], list[np.ndarray],
                                  list[int]] | None:
        """Load cached per-band envelopes.  Returns
        ``(mags, phases, hops)`` or *None* if no cache exists."""
        path = os.path.join(analysis_dir, "filterbank",
                            "fb_envelopes.npz")
        if not os.path.isfile(path):
            return None
        d = np.load(path)
        if "n_bands" not in d:
            # Legacy single-hop format — ignore, will recompute
            return None
        n = int(d["n_bands"])
        hops = d["hops"].tolist()
        mags = [d[f"mag_{i}"] for i in range(n)]
        phases = [d[f"phase_{i}"] for i in range(n)]
        return mags, phases, hops

    @staticmethod
    def load_meta(analysis_dir: str
                  ) -> tuple[dict, list[dict]] | None:
        """Load only the filterbank metadata (no WAV data).

        Returns ``(top_meta, bands_list)`` or *None*.
        """
        import json
        fb_dir = os.path.join(analysis_dir, "filterbank")
        meta_path = os.path.join(fb_dir, "filterbank_meta.json")
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path) as f:
            meta = json.load(f)
        return meta, meta["bands"]

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
        n_bands = len(meta["bands"])
        t0 = time.monotonic()
        for idx, bm in enumerate(meta["bands"]):
            fb.bands.append(BandDef(
                fmin=bm["fmin"], fmax=bm["fmax"], label=bm.get("label", "")))
            fpath = os.path.join(fb_dir, bm["file"])
            if os.path.isfile(fpath):
                sr_wav, data = _load_audio(fpath)
                if np.issubdtype(data.dtype, np.integer):
                    sub = data.astype(np.float32) / max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max) * bm.get("peak_amplitude", 1.0)
                else:
                    sub = data.astype(np.float32) * bm.get("peak_amplitude", 1.0)
                fb.subbands.append(sub)
            else:
                fb.subbands.append(np.zeros(0, dtype=np.float32))
            # Progress every 20 bands or on the last one
            if idx % 20 == 0 or idx == n_bands - 1:
                elapsed = time.monotonic() - t0
                pct = (idx + 1) / n_bands * 100
                eta = (elapsed / (idx + 1)) * (n_bands - idx - 1) if idx > 0 else 0.0
                print(f"\r  Loading filterbank WAVs: {idx + 1}/{n_bands} "
                      f"({pct:.0f}%) — {elapsed:.1f}s elapsed, "
                      f"~{eta:.0f}s remaining", end="", flush=True)
        print()  # newline after progress
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


def _build_field_sources(has_onset: bool,
                         has_fb: bool = False,
                         has_wvlt: bool = False) -> list[FieldSource]:
    """Build the list of data fields from the virtual texture layouts.

    Includes placeholder entries for filterbank and wavelet sources
    so they appear in the tree (greyed out when unavailable).

    Audio is always stored as L+R internally; mono-derived fields use
    the shader's *uMonoMixMode* uniform to mix L and R on the GPU.
    """
    seen: set[str] = set()
    sources: list[FieldSource] = []
    for tex_name, layout in _TEXTURE_LAYOUTS.items():
        if tex_name == "onset" and not has_onset:
            # Still add as unavailable so it shows in tree
            sources.append(FieldSource(
                name="onset",
                display_name=FIELD_DISPLAY.get("onset", "onset"),
                tex_name=tex_name,
                tex_channel=0,
                category="cqt",
                available=False,
            ))
            seen.add("onset")
            continue
        if tex_name == "wavelet_coeffs":
            # Wavelet fields handled separately below with has_wvlt flag
            continue
        for ch_idx, ch_key in enumerate("RGBA"):
            field_name = layout.get(ch_key)
            if field_name is None or field_name in seen:
                continue
            seen.add(field_name)
            cat = _FIELD_CATEGORY_MAP.get(field_name, "cqt")
            sources.append(FieldSource(
                name=field_name,
                display_name=FIELD_DISPLAY.get(field_name, field_name),
                tex_name=tex_name,
                tex_channel=ch_idx,
                category=cat,
                available=True,
            ))

    # Filterbank L/R fields.
    # tex_names use dedicated "fb_*_polar" keys so they do NOT alias
    # the CQT textures — _compute_fb_envelopes uploads under these keys
    # and _render_fb_tf binds them through _set_field_uniforms.
    fb_fields = [
        ("fb_mag_left",    "FB Mag L",   "fb_left_polar",  0),
        ("fb_phase_left",  "FB Phase L", "fb_left_polar",  1),
        ("fb_mag_right",   "FB Mag R",   "fb_right_polar", 0),
        ("fb_phase_right", "FB Phase R", "fb_right_polar", 1),
    ]
    for fname, dname, tname, ch in fb_fields:
        if fname not in seen:
            seen.add(fname)
            sources.append(FieldSource(
                name=fname,
                display_name=dname,
                tex_name=tname,
                tex_channel=ch,
                category="fb",
                available=has_fb,
            ))

    # Wavelet approximation/detail
    wvlt_fields = [
        ("wvlt_approx", "Wvlt Approx", "wavelet_coeffs", 0),
        ("wvlt_detail", "Wvlt Detail", "wavelet_coeffs", 1),
    ]
    for fname, dname, tname, ch in wvlt_fields:
        if fname not in seen:
            seen.add(fname)
            sources.append(FieldSource(
                name=fname,
                display_name=dname,
                tex_name=tname,
                tex_channel=ch,
                category="wvlt",
                available=has_wvlt,
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
                    opened_submenu = False
                    for j, ir in enumerate(col.item_rects):
                        if ir.collidepoint(mx, my):
                            item = col.items[j]
                            if not item.is_separator:
                                col.hover_idx = j
                                # Auto-open submenu on hover
                                if item.has_submenu and item.is_enabled:
                                    self._open_submenu(j, col)
                                    opened_submenu = True
                            break
                    # If mouse is in a parent column and did NOT just
                    # open a submenu, collapse any deeper columns
                    if not opened_submenu and depth < len(self._columns) - 1:
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
                                self._open_submenu(j, col)
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
                            self._open_submenu(col.hover_idx, col)
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

    def _open_submenu(self, parent_idx: int, col: '_MenuColumn | None' = None) -> None:
        if col is None:
            col = self._columns[-1]
        item = col.items[parent_idx]
        if not item.children:
            return
        # Find the depth of the parent column
        try:
            col_depth = self._columns.index(col)
        except ValueError:
            return
        # Avoid re-opening the same submenu
        if (col_depth + 1 < len(self._columns)
                and self._columns[col_depth + 1].title == item.label):
            return
        # Pop any columns deeper than the parent
        while len(self._columns) > col_depth + 1:
            self._columns.pop()
        ir = col.item_rects[parent_idx]
        sub = self._build_column(item.children, ir.right, ir.y, item.label)
        self._columns.append(sub)


# ---------------------------------------------------------------------------
# Scrollable item list — reusable scrolling row list with selection
# ---------------------------------------------------------------------------

def _hue_to_rgb(h: float, s: float = 1.0) -> tuple[int, int, int]:
    """Convert hue (0-1) and saturation to an RGB tuple for UI display."""
    import colorsys
    r, g, b = colorsys.hls_to_rgb(h, 0.5, s)
    return (int(r * 255), int(g * 255), int(b * 255))


_TGT_COLORS: dict[str, tuple[int, int, int]] = {
    "Ch1": _hue_to_rgb(DEFAULT_CHANNEL_HUES[0]),
    "Ch2": _hue_to_rgb(DEFAULT_CHANNEL_HUES[1]),
    "Ch3": _hue_to_rgb(DEFAULT_CHANNEL_HUES[2]),
    "Ch4": _hue_to_rgb(DEFAULT_CHANNEL_HUES[3]),
    "Alpha": (200, 200, 200),
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

    def scroll_to_selected(self) -> None:
        """Adjust scroll offset so the selected item is visible."""
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
# Tree item list — collapsible category tree for data sources
# ---------------------------------------------------------------------------

@dataclass
class TreeCategory:
    """One collapsible category in a :class:`TreeItemList`."""
    key: str
    label: str
    expanded: bool = True


class TreeItemList:
    """Scrollable tree with collapsible category headers.

    Items are grouped by category. Categories can be expanded/collapsed.
    Unavailable items appear greyed out and cannot be selected.
    """

    ROW_H = 22
    CAT_H = 24
    MORE_H = 18
    PAD = 6
    INDENT = 16  # extra indent for items under a category

    def __init__(self, title: str, max_visible: int = 12) -> None:
        self.title = title
        self.categories: list[TreeCategory] = []
        self.items: list[ListItem] = []
        # Map item key → category key
        self.item_categories: dict[str, str] = {}
        # Map item key → available flag
        self.item_available: dict[str, bool] = {}
        self.selected_idx: int = 0
        self.scroll_offset: int = 0
        self.max_visible = max_visible
        self._row_rects: list[pygame.Rect] = []
        self._cat_rects: dict[str, pygame.Rect] = {}
        self._flat_indices: list[int] = []  # maps visible row → items index
        self._more_up_rect: pygame.Rect | None = None
        self._more_down_rect: pygame.Rect | None = None
        self._rendered_h: int = 0

    def _build_flat_list(self) -> list[tuple[str, int | None]]:
        """Build a flat list of (type, data) tuples for rendering.

        Returns list of ("cat", cat_idx) and ("item", item_idx).
        """
        flat: list[tuple[str, int | None]] = []
        cat_items: dict[str, list[int]] = {c.key: [] for c in self.categories}
        for i, item in enumerate(self.items):
            cat_key = self.item_categories.get(item.key, "")
            if cat_key in cat_items:
                cat_items[cat_key].append(i)

        for ci, cat in enumerate(self.categories):
            items_in_cat = cat_items.get(cat.key, [])
            if not items_in_cat:
                continue
            flat.append(("cat", ci))
            if cat.expanded:
                for ii in items_in_cat:
                    flat.append(("item", ii))
        return flat

    @property
    def needs_scroll(self) -> bool:
        return len(self._build_flat_list()) > self.max_visible

    def clamp(self) -> None:
        n = len(self.items)
        if n == 0:
            self.selected_idx = 0
            self.scroll_offset = 0
            return
        self.selected_idx = max(0, min(self.selected_idx, n - 1))
        flat = self._build_flat_list()
        max_off = max(0, len(flat) - self.max_visible)
        self.scroll_offset = max(0, min(self.scroll_offset, max_off))

    def scroll_to_selected(self) -> None:
        flat = self._build_flat_list()
        # Find the flat index of the selected item
        flat_idx = -1
        for fi, (ftype, fdata) in enumerate(flat):
            if ftype == "item" and fdata == self.selected_idx:
                flat_idx = fi
                break
        if flat_idx < 0:
            return
        if flat_idx < self.scroll_offset:
            self.scroll_offset = flat_idx
        if flat_idx >= self.scroll_offset + self.max_visible:
            self.scroll_offset = flat_idx - self.max_visible + 1

    def render(self, surf: pygame.Surface, font: pygame.font.Font,
               x: int, y: int, w: int) -> int:
        self._row_rects = []
        self._cat_rects = {}
        self._flat_indices = []
        self._more_up_rect = None
        self._more_down_rect = None
        pad = self.PAD
        start_y = y

        # Title
        ttl = font.render(f"\u2500\u2500 {self.title} \u2500\u2500",
                           True, (180, 180, 180))
        surf.blit(ttl, (pad, y + 2))
        y += self.ROW_H + 2

        flat = self._build_flat_list()
        if not flat:
            dim = font.render("(no sources)", True, (100, 100, 100))
            surf.blit(dim, (pad + 4, y + 3))
            y += self.ROW_H + 2
            self._rendered_h = y - start_y
            return self._rendered_h

        self.clamp()
        vis_start = self.scroll_offset
        vis_end = min(len(flat), vis_start + self.max_visible)

        # Scroll-up
        if vis_start > 0:
            btn = pygame.Rect(pad, y, w - 2 * pad, self.MORE_H)
            pygame.draw.rect(surf, (40, 40, 55), btn)
            pygame.draw.rect(surf, (60, 60, 80), btn, 1)
            arr = font.render(f"\u25b2 more ({vis_start})", True, (140, 140, 170))
            surf.blit(arr, (w // 2 - arr.get_width() // 2, y + 1))
            self._more_up_rect = btn
            y += self.MORE_H + 2

        for fi in range(vis_start, vis_end):
            ftype, fdata = flat[fi]
            if ftype == "cat":
                cat = self.categories[fdata]
                rect = pygame.Rect(pad, y, w - 2 * pad, self.CAT_H)
                pygame.draw.rect(surf, (38, 38, 48), rect)
                pygame.draw.rect(surf, (70, 70, 90), rect, 1)
                arrow = "\u25bc" if cat.expanded else "\u25b6"
                ct = font.render(f"{arrow} {cat.label}", True,
                                 (200, 200, 220))
                surf.blit(ct, (pad + 4, y + 4))
                self._cat_rects[cat.key] = rect
                y += self.CAT_H + 1
            elif ftype == "item":
                item = self.items[fdata]
                avail = self.item_available.get(item.key, True)
                rect = pygame.Rect(pad + self.INDENT, y,
                                   w - 2 * pad - self.INDENT, self.ROW_H)
                if fdata == self.selected_idx and avail:
                    pygame.draw.rect(surf, (50, 50, 70), rect)
                pygame.draw.rect(surf, (55, 55, 68) if avail
                                 else (40, 40, 45), rect, 1)
                tc = (200, 200, 200) if avail else (80, 80, 90)
                txt = font.render(item.display_name, True, tc)
                surf.blit(txt, (pad + self.INDENT + 4, y + 3))
                if avail:
                    color = _TGT_COLORS.get(item.tag_color_key, (80, 80, 80))
                    arrow = font.render(f"\u2192 {item.tag}", True, color)
                else:
                    arrow = font.render("\u2014", True, (60, 60, 70))
                surf.blit(arrow, (w - 60, y + 3))
                self._row_rects.append(rect)
                self._flat_indices.append(fdata)
                y += self.ROW_H + 2

        # Scroll-down
        remaining = len(flat) - vis_end
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
        # Category headers — toggle expand/collapse
        for cat_key, rect in self._cat_rects.items():
            if rect.collidepoint(lx, ly):
                for cat in self.categories:
                    if cat.key == cat_key:
                        cat.expanded = not cat.expanded
                        break
                return -1  # consumed, no selection change

        # More buttons
        if self._more_up_rect and self._more_up_rect.collidepoint(lx, ly):
            self.scroll_offset = max(0, self.scroll_offset - self.max_visible)
            return -1
        if self._more_down_rect and self._more_down_rect.collidepoint(lx, ly):
            flat = self._build_flat_list()
            max_off = max(0, len(flat) - self.max_visible)
            self.scroll_offset = min(max_off,
                                     self.scroll_offset + self.max_visible)
            return -1

        # Item rows
        for vi, rect in enumerate(self._row_rects):
            if rect.collidepoint(lx, ly):
                idx = self._flat_indices[vi]
                avail = self.item_available.get(
                    self.items[idx].key, True)
                if not avail:
                    return -1  # unavailable — consume but don't select
                self.selected_idx = idx
                return idx
        return None

    def handle_scroll(self, direction: int) -> bool:
        flat = self._build_flat_list()
        if len(flat) <= self.max_visible:
            return False
        max_off = max(0, len(flat) - self.max_visible)
        self.scroll_offset = max(0, min(self.scroll_offset + direction,
                                        max_off))
        return True


# ---------------------------------------------------------------------------
# Synthesis product — one output from the synthesizer
# ---------------------------------------------------------------------------

PHASE_MODES = ["original", "griffin_lim", "zero_phase", "freq_carrier"]
PHASE_LABELS = {
    "original":    "Original Phase",
    "griffin_lim":  "Griffin-Lim",
    "zero_phase":  "Zero Phase",
    "freq_carrier": "Freq Carrier",
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
        self._panel_scroll_y: int = 0  # vertical scroll offset for panel content
        self._content_h: int = 0       # total height of last rendered content

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

    def _clamp_panel_scroll(self) -> None:
        """Keep panel scroll within bounds."""
        pr = self.panel_rect
        max_scroll = max(0, self._content_h - pr.h)
        self._panel_scroll_y = max(0, min(self._panel_scroll_y, max_scroll))

    def _apply_panel_scroll(self, full_surf: pygame.Surface) -> pygame.Surface:
        """Clip *full_surf* to the available panel height, applying scroll."""
        pr = self.panel_rect
        self._content_h = full_surf.get_height()
        self._clamp_panel_scroll()
        avail_h = pr.h
        if self._content_h <= avail_h:
            return full_surf
        clipped = pygame.Surface((full_surf.get_width(), avail_h),
                                 pygame.SRCALPHA)
        clipped.blit(full_surf, (0, -self._panel_scroll_y))
        # Draw a thin scrollbar track on the right edge
        w = full_surf.get_width()
        bar_w = 4
        track_h = avail_h
        thumb_frac = avail_h / self._content_h
        thumb_h = max(16, int(track_h * thumb_frac))
        max_scroll = self._content_h - avail_h
        thumb_y = int((track_h - thumb_h) * self._panel_scroll_y
                      / max(max_scroll, 1))
        pygame.draw.rect(clipped, (60, 60, 70),
                         pygame.Rect(w - bar_w, 0, bar_w, track_h))
        pygame.draw.rect(clipped, (120, 120, 150),
                         pygame.Rect(w - bar_w, thumb_y, bar_w, thumb_h))
        return clipped

    def _handle_panel_wheel(self, event: pygame.event.Event) -> bool:
        """Scroll panel content on mouse wheel. Call from handle_event."""
        if event.type != MOUSEWHEEL:
            return False
        mx, my = pygame.mouse.get_pos()
        pr = self.panel_rect
        if not pr.collidepoint(mx, my):
            return False
        self._panel_scroll_y -= event.y * 28
        self._clamp_panel_scroll()
        return True

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


def _apply_curve_unclipped(xs: np.ndarray, norm_mode: int,
                           gamma: float, scale: float) -> np.ndarray:
    """Like _apply_curve but without final [0,1] clamp — preserves HDR range.

    The normalization step still maps the domain, but the gamma*scale
    output is left unbounded so the curves plot can show where values
    exceed the normal [0,1] display range.
    """
    if norm_mode == 1:  # dB
        safe = np.maximum(xs, 1e-12)
        db = 10.0 * np.log10(safe)
        normed = (db + 60.0) / 60.0   # no clamp
    else:
        normed = xs  # no clamp
    return (np.power(np.maximum(normed, 0.0), gamma) * scale).astype(np.float32)


def _hdr_normalize(ys: np.ndarray, gd: "GlobalDefaults",
                   auto_lo: float = 0.0, auto_hi: float = 1.0) -> np.ndarray:
    """Apply the HDR normalizer to pre-clip post-blend values.

    Maps from the auto-detected input range [auto_lo, auto_hi]
    into [hdr_output_lo, hdr_output_hi] using the selected tone-mapping
    curve.  Operates on raw unclipped values.
    """
    mode = gd.hdr_normalizer
    lo_in = auto_lo
    hi_in = auto_hi
    lo_out = gd.hdr_output_lo
    hi_out = gd.hdr_output_hi
    span_in = hi_in - lo_in if abs(hi_in - lo_in) > 1e-12 else 1.0
    span_out = hi_out - lo_out

    if mode == 0:  # None — raw passthrough
        return ys

    # Remap into [0, 1] input space
    t = (ys - lo_in) / span_in

    if mode == 1:  # Clamp
        t = np.clip(t, 0.0, 1.0)
    elif mode == 2:  # Reinhard
        t = np.maximum(t, 0.0)
        t = t / (1.0 + t)
    elif mode == 3:  # Sigmoid
        t = 1.0 / (1.0 + np.exp(-6.0 * (t - 0.5)))
    elif mode == 4:  # Linear Compress
        t = np.clip(t, 0.0, 1.0)

    # Map to output range
    return (lo_out + t * span_out).astype(np.float32)


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
    """Panel with a tree of data fields grouped by source category,
    per-field controls, global defaults, and output controls.

    Data sources are displayed in a collapsible tree with categories
    (CQT, Filterbank, Wavelet).  Unavailable sources appear greyed out.
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
        self._prev_selected_idx: int = -1
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        self._dragging: str | None = None
        self._item_map: dict[str, tuple[pygame.Rect, Any]] = {}

        # Data-field tree list (top section) — grouped by category
        self.data_list = TreeItemList("Data Sources", max_visible=14)
        self._sync_data_list()

        # Plot widget (bottom section — shows normalised data curves)
        self.plot = PlotWidget()
        self.plot.title = "Data Curves"
        self.plot.y_min = None  # auto-range to show full HDR extent
        self.plot.y_max = None
        self.plot.clip_lo = 0.0
        self.plot.clip_hi = 1.0
        self._plot_h: int = 140  # pixel height of the plot box

    def _sync_data_list(self) -> None:
        """Rebuild data_list items and categories from fields/configs."""
        # Build categories from _FIELD_CATEGORIES
        cats: list[TreeCategory] = []
        seen_cats: set[str] = set()
        for fs in self.fields:
            if fs.category not in seen_cats:
                seen_cats.add(fs.category)
        for cat_key, cat_label in _FIELD_CATEGORIES:
            if cat_key in seen_cats:
                # Preserve existing expanded state if category exists
                existing = None
                for c in self.data_list.categories:
                    if c.key == cat_key:
                        existing = c
                        break
                if existing:
                    cats.append(existing)
                else:
                    cats.append(TreeCategory(key=cat_key, label=cat_label))
        self.data_list.categories = cats

        items: list[ListItem] = []
        item_categories: dict[str, str] = {}
        item_available: dict[str, bool] = {}
        for i, fs in enumerate(self.fields):
            cfg = self.configs[i]
            tgt_text = TARGET_LABELS[TARGET_VALUES.index(cfg.target)] if fs.available else "—"
            items.append(ListItem(
                key=fs.name,
                display_name=fs.display_name,
                tag=tgt_text,
                tag_color_key=tgt_text if fs.available else "none",
            ))
            item_categories[fs.name] = fs.category
            item_available[fs.name] = fs.available
        self.data_list.items = items
        self.data_list.item_categories = item_categories
        self.data_list.item_available = item_available
        self.data_list.selected_idx = self.selected_idx

    _CURVE_XS = np.linspace(0.0, 1.0, 256, dtype=np.float32)

    def _rebuild_plot_series(self) -> None:
        """Recompute transfer-curve series from current field configs.

        Uses unclipped curve values so the plot shows exactly where data
        exceeds [0,1].  If an HDR normalizer is active, a second
        (normalized) series is overlaid.  Clips show as red segments.
        """
        xs = self._CURVE_XS
        gd = self.global_defaults
        target_seen: dict[int, int] = {}  # target → count seen so far
        series: list[PlotSeries] = []

        xs_hdr = xs

        for i, (fs, cfg) in enumerate(zip(self.fields, self.configs)):
            if cfg.target < 0:
                continue
            norm_mode = cfg.norm_mode if not cfg.use_global else gd.norm_mode
            gamma = cfg.gamma if not cfg.use_global else gd.gamma
            scale = cfg.scale if not cfg.use_global else gd.scale

            # Unclipped curve — shows true output before any clamping
            ys_raw = _apply_curve_unclipped(xs_hdr, norm_mode, gamma, scale)

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
                data_x=xs_hdr,
                data_y=ys_raw,
                clip_color=(200, 50, 50),  # red for clipped segments
            ))

            # If HDR normalizer is active, add the normalized curve
            if gd.hdr_normalizer > 0:
                ys_norm = _hdr_normalize(ys_raw, gd,
                                         float(ys_raw.min()),
                                         float(ys_raw.max()))
                series.append(PlotSeries(
                    key=f"{fs.name}_hdr",
                    label=f"{fs.display_name} HDR",
                    color=_jitter_color(
                        (min(255, color[0] + 40),
                         min(255, color[1] + 40),
                         min(255, color[2] + 40)), 0),
                    line=True,
                    dots=False,
                    data_x=xs_hdr,
                    data_y=ys_norm,
                ))

        # Update clip boundary lines based on output range
        self.plot.clip_lo = gd.hdr_output_lo
        self.plot.clip_hi = gd.hdr_output_hi
        self.plot.series = series

    # ---- Rendering --------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._item_map = {}

        # Keep list selections in sync — scroll only when selection changes
        sel_changed = (self.selected_idx != self._prev_selected_idx)
        self._prev_selected_idx = self.selected_idx
        self.data_list.selected_idx = self.selected_idx
        self._sync_data_list()
        if sel_changed:
            self.data_list.scroll_to_selected()

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

        # Blend mode
        upper_rows.append(("dropdown", ("blend_mode",
                                         BLEND_MODES[gd.blend_mode], BLEND_MODES)))

        # Channel HSL controls
        upper_rows.append(("label", "\u2500\u2500 Channel Hues \u2500\u2500"))
        for ci in range(4):
            upper_rows.append(("label", f"Ch{ci + 1} Hue / Sat"))
            upper_rows.append(("slider", (f"ch{ci}_hue", gd.channel_hues[ci], 0.0, 1.0, "{:.2f}")))
            upper_rows.append(("slider", (f"ch{ci}_sat", gd.channel_sats[ci], 0.0, 1.0, "{:.2f}")))

        # HDR Dynamic Range controls
        upper_rows.append(("label", ""))
        upper_rows.append(("label", "\u2500\u2500 HDR / Dynamic Range \u2500\u2500"))
        upper_rows.append(("dropdown", ("hdr_norm",
                                         HDR_NORMALIZERS[gd.hdr_normalizer],
                                         HDR_NORMALIZERS)))
        upper_rows.append(("label", "Output Range"))
        upper_rows.append(("slider", ("hdr_out_lo", gd.hdr_output_lo, -1.0, 1.0, "{:.2f}")))
        upper_rows.append(("slider", ("hdr_out_hi", gd.hdr_output_hi, 0.0, 3.0, "{:.2f}")))

        # --- Pre-compute total height ---
        # Data list — estimate from flat list produced by tree
        flat = self.data_list._build_flat_list()
        vis_rows = min(len(flat), self.data_list.max_visible)
        data_list_h = self.data_list.ROW_H + 2  # title
        if flat:
            for fi in range(self.data_list.scroll_offset,
                            self.data_list.scroll_offset + vis_rows):
                if fi >= len(flat):
                    break
                ftype, _ = flat[fi]
                if ftype == "cat":
                    data_list_h += self.data_list.CAT_H + 1
                else:
                    data_list_h += self.ROW_H + 2
            if self.data_list.scroll_offset > 0:
                data_list_h += self.data_list.MORE_H + 2
            vis_end = self.data_list.scroll_offset + self.data_list.max_visible
            if vis_end < len(flat):
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
                          "g_norm": "Nrm:",
                          "blend_mode": "Blend:",
                          "hdr_norm": "HDR:"}.get(key, key[:4])
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
            ly += self._panel_scroll_y

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
            for key in ("f_target", "f_norm", "g_norm", "blend_mode",
                        "hdr_norm"):
                if key in self._item_map:
                    rect, opts = self._item_map[key]
                    if rect.collidepoint(lx, ly):
                        self._open_dropdown(key, opts, rect, pr)
                        return True

            # Slider +/- buttons and track drag — collect all slider keys
            _slider_keys = ["f_gamma", "f_scale",
                            "g_gamma", "g_scale",
                            "o_gamma", "o_scale"]
            for ci in range(4):
                _slider_keys.append(f"ch{ci}_hue")
                _slider_keys.append(f"ch{ci}_sat")
            _slider_keys.extend(["hdr_out_lo", "hdr_out_hi"])
            for key in _slider_keys:
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
            for key in _slider_keys:
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
            return self._handle_panel_wheel(event)

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
        elif key == "blend_mode":
            self.global_defaults.blend_mode = BLEND_MODES.index(value)
        elif key == "hdr_norm":
            self.global_defaults.hdr_normalizer = HDR_NORMALIZERS.index(value)

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
        elif key.startswith("ch") and key.endswith("_hue"):
            ci = int(key[2])
            gd.channel_hues[ci] = val
        elif key.startswith("ch") and key.endswith("_sat"):
            ci = int(key[2])
            gd.channel_sats[ci] = val
        elif key == "hdr_out_lo":
            gd.hdr_output_lo = val
        elif key == "hdr_out_hi":
            gd.hdr_output_hi = val

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
            "hdr_out_lo": gd.hdr_output_lo, "hdr_out_hi": gd.hdr_output_hi,
        }
        for ci in range(4):
            current[f"ch{ci}_hue"] = gd.channel_hues[ci]
            current[f"ch{ci}_sat"] = gd.channel_sats[ci]
        current = current.get(key, 0.0)
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

# Backing data channel → synthesis role routing
ROUTE_SOURCES = ["Ch1", "Ch2", "Ch3", "Ch4", "Alpha"]
ROUTE_SOURCE_TARGETS = {"Ch1": 0, "Ch2": 1, "Ch3": 2, "Ch4": 3, "Alpha": 4}
ROUTE_ROLES = ["magnitude", "phase", "mask"]
ROUTE_ROLE_LABELS = {"magnitude": "Mag", "phase": "Phase", "mask": "Mask"}
ROUTE_AUDIO_CHANNELS = ["left", "right"]
ROUTE_AUDIO_CH_LABELS = {"left": "L", "right": "R"}
MONO_MIX_MODES = ["Average", "L Only", "R Only", "Mid", "Side"]


@dataclass
class ChannelRouting:
    """Maps backing data channels (Ch1-Ch4, Alpha) to synthesis roles.

    Audio is always stored as L+R internally.  Mono signals are
    derived from L+R via *mono_mix_mode* (index into MONO_MIX_MODES).
    """
    left_mag: str = "Ch1"
    left_phase: str = "Ch2"
    left_mask: str = "Alpha"
    right_mag: str = "Ch3"
    right_phase: str = "Ch4"
    right_mask: str = "Alpha"
    mono_mix_mode: int = 0  # index into MONO_MIX_MODES

    def get(self, audio_ch: str, role: str) -> str:
        """Return the backing data channel key for given audio_ch and role."""
        return getattr(self, f"{audio_ch}_{role}", "Ch1")

    def set(self, audio_ch: str, role: str, source: str) -> None:
        setattr(self, f"{audio_ch}_{role}", source)

    def resolve_field(self, source: str,
                      field_sources: list,
                      field_configs: list) -> str | None:
        """Return the raw_arrays field name assigned to a backing channel.

        Scans *field_configs* for the first field whose *target* matches
        the numeric target of *source* (e.g. Ch1→0).  Returns the
        corresponding :class:`FieldSource` name, or *None*.
        """
        tgt = ROUTE_SOURCE_TARGETS.get(source)
        if tgt is None:
            return None
        for fs, cfg in zip(field_sources, field_configs):
            if cfg.target == tgt and fs.available:
                return fs.name
        return None


class SynthesisPanel(Panel):
    """Async synthesis job queue panel.

    Holds a list of :class:`SynthJob` entries.  Jobs are executed on a
    background thread; completed jobs can be played back instantly.
    """

    ROW_H = 22
    BTN_W = 40
    PAD = 6
    _SYNTH_MODE_KEYS = ["icqt", "fb", "hybrid", "fb_tf", "wavelet", "transplant", "hybrid_cwt"]
    _SYNTH_MODE_OPTIONS = [
        "iCQT (full CQT)",
        "Filterbank (full FB)",
        "Hybrid (FB low / iCQT high)",
        "FB from TF (viewport)",
        "Wavelet reconstruction",
        "TF Transplant (viewport)",
        "Hybrid CWT (CWT low / iCQT high)",
    ]

    def __init__(self, *, side: str = "right") -> None:
        super().__init__(title="Synthesis", side=side)
        self.jobs: list[SynthJob] = []
        self.job_list = ScrollableItemList("Synthesis Jobs", max_visible=10)
        self._next_id: int = 1
        self._worker_thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.synth_mode_idx: int = 0  # index into _SYNTH_MODE_KEYS
        self.channel_routing = ChannelRouting()
        self.use_adjusted: bool = False   # apply viewer norm/gamma/scale to synth data
        # Callbacks set by the viewer
        self.on_play: Any = None          # (SynthJob) -> None
        self.on_stop: Any = None          # () -> None
        self.on_synth_icqt: Any = None    # () -> None
        self.on_synth_fb: Any = None      # () -> None
        self.on_synth_hybrid: Any = None  # () -> None
        self.on_synth_fb_tf: Any = None   # () -> None
        self.on_synth_wavelet: Any = None # () -> None
        self.on_synth_transplant: Any = None  # () -> None
        self.on_synth_hybrid_cwt: Any = None  # () -> None
        self._playing_job_id: int | None = None
        # Per-row button rects (surface-local)
        self._play_rects: dict[int, pygame.Rect] = {}
        self._del_rects: dict[int, pygame.Rect] = {}
        # Controls state — bound to viewer attrs via get/set callbacks
        self._get_phase_mode: Any = lambda: "original"
        self._set_phase_mode: Any = lambda v: None
        self._get_normalize: Any = lambda: True
        self._set_normalize: Any = lambda v: None
        self._get_fb_n_bands: Any = lambda: 16
        self._set_fb_n_bands: Any = lambda v: None
        self._has_audio: Any = lambda: False
        self._has_synth: Any = lambda: False
        self._has_wavelet: Any = lambda: False
        # Transplant destination frequency (Hz)
        self.transplant_dst_f0: float = 200.0
        self.transplant_dst_f1: float = 800.0
        # Dropdown UI state
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        # Button rects for controls
        self._ctrl_rects: dict[str, pygame.Rect] = {}


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

    @staticmethod
    def _apply_field_adjustment(arr: np.ndarray,
                                cfg: "FieldConfig",
                                gd: "GlobalDefaults | None") -> np.ndarray:
        """Apply norm → gamma → scale to *arr* on CPU, mirroring the shader.

        Uses per-field config; falls back to *gd* when ``cfg.use_global``
        is set.  Returns a new array — never mutates the input.
        """
        norm_mode = gd.norm_mode if (cfg.use_global and gd) else cfg.norm_mode
        gamma = gd.gamma if (cfg.use_global and gd) else cfg.gamma
        scale = gd.scale if (cfg.use_global and gd) else cfg.scale
        return _apply_curve(arr, norm_mode, gamma, scale)

    @staticmethod
    def _resolve_routing(p: dict) -> dict[str, np.ndarray]:
        """Build a *routed_arrays* dict from channel routing + field mappings.

        Returns a dict with keys like ``"magnitude_left"``,
        ``"phase_left"``, ``"mask_left"``, etc., mapping to the
        native-dtype raw arrays from the backing data channels the user
        selected in the routing panel.

        When ``p["use_adjusted"]`` is truthy, applies the per-field
        norm/gamma/scale pipeline (mirroring the GPU shader) to each
        array before returning it.

        Falls back to the default CQT fields when no routing is provided.

        The optional ``p["field_resolver"]`` callable ``(str) -> ndarray | None``
        is used to look up *any* field name — including derived fields
        (``mag_left``, ``mag_right``, …) and filterbank grids
        (``fb_mag_left``, …) that are not directly stored in
        ``p["raw_arrays"]``.
        """
        raw: dict[str, np.ndarray] = p["raw_arrays"]
        resolver = p.get("field_resolver")  # callable(str) -> ndarray|None
        routing: ChannelRouting | None = p.get("routing")
        field_sources: list | None = p.get("field_sources")
        field_configs: list | None = p.get("field_configs")
        use_adjusted: bool = bool(p.get("use_adjusted", False))
        global_defaults: GlobalDefaults | None = p.get("global_defaults")

        def _lookup(fname: str) -> np.ndarray | None:
            """Try the resolver first, then fall back to raw dict."""
            if resolver is not None:
                arr = resolver(fname)
                if arr is not None:
                    return arr
            return raw.get(fname)

        result: dict[str, np.ndarray] = {}

        if routing is None or field_sources is None or field_configs is None:
            # Legacy fallback — use the hardcoded CQT complex fields
            for side in ("left", "right"):
                mag = _lookup(f"mag_{side}")
                if mag is None:
                    mag = _lookup(f"real_{side}")
                pha = _lookup(f"phase_{side}")
                if pha is None:
                    pha = _lookup(f"imag_{side}")
                result[f"magnitude_{side}"] = mag
                result[f"phase_{side}"] = pha
            return result

        for audio_ch in ROUTE_AUDIO_CHANNELS:
            for role in ROUTE_ROLES:
                src_key = routing.get(audio_ch, role)
                fname = routing.resolve_field(
                    src_key, field_sources, field_configs)
                if fname is not None:
                    arr = _lookup(fname)
                    if arr is not None:
                        if use_adjusted:
                            tgt = ROUTE_SOURCE_TARGETS.get(src_key)
                            for fs, fc in zip(field_sources, field_configs):
                                if fc.target == tgt and fs.available and fs.name == fname:
                                    arr = SynthesisPanel._apply_field_adjustment(
                                        arr, fc, global_defaults)
                                    break
                        result[f"{role}_{audio_ch}"] = arr
                    else:
                        result[f"{role}_{audio_ch}"] = None  # type: ignore[assignment]
                else:
                    result[f"{role}_{audio_ch}"] = None  # type: ignore[assignment]
        return result

    def _execute(self, job: SynthJob) -> None:
        p = job.params
        synth: ViewportSynthPlayer = p["synth"]
        if job.source_type == "cqt":
            routed = self._resolve_routing(p)
            result = synth.synthesise_to_array(
                p["raw_arrays"], p["freqs"], p["times"],
                p["view_x0"], p["view_x1"],
                p["view_y0"], p["view_y1"],
                p["n_frames"], p["n_bins"],
                p["t_start"], p["t_dur"], p["sr"],
                p["is_stereo"], p.get("normalize", True),
                phase_mode=p.get("phase_mode", "original"),
                routed_arrays=routed)
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
        elif job.source_type == "fb_audio":
            result = synth.synthesise_filterbank_from_audio(
                p["audio_path"], p["sr"],
                p["t0"], p["t1"],
                p["f_lo"], p["f_hi"],
                n_bands=p.get("n_bands", 16),
                is_stereo=p.get("is_stereo", True),
                normalize=p.get("normalize", True))
            if result is None:
                raise ValueError("Empty viewport or no audio")
            stereo, dur, vt0, vt1 = result
            job.result_stereo = stereo
            job.duration = dur
            job.view_t0 = vt0
            job.view_t1 = vt1
        elif job.source_type == "fb_tf":
            self._execute_fb_tf(job, synth, p)
        elif job.source_type == "wavelet":
            self._execute_wavelet(job, p)
        elif job.source_type == "hybrid":
            self._execute_hybrid(job, synth, p)
        elif job.source_type == "hybrid_cwt":
            self._execute_hybrid_cwt(job, synth, p)
        elif job.source_type == "transplant":
            self._execute_transplant(job, p)

    def _execute_fb_tf(self, job: SynthJob,
                       synth: "ViewportSynthPlayer",
                       p: dict) -> None:
        """FB-from-TF synthesis: interpret TF viewport data as band
        envelopes and reconstruct via additive synthesis.

        Each frequency row in the viewport is treated as one band.
        The routed magnitude data supplies per-band amplitude envelopes.

        Phase behaviour follows ``phase_mode`` (same options as iCQT):

        * ``"zero_phase"`` — pure envelope summation, no carrier or
          phase modulation.  This is the default.
        * ``"original"`` — use routed phase data directly as
          instantaneous phase (can be CQT phase, filterbank phase,
          or any data channel the user routes to the phase slot).
        * ``"griffin_lim"`` — derive phase via Griffin-Lim iteration
          over a per-band STFT round-trip.
        """
        sr: int = p["sr"]
        freqs: np.ndarray = p["freqs"]
        times: np.ndarray = p["times"]
        x0, x1 = int(p["view_x0"]), int(p["view_x1"])
        y0, y1 = int(p["view_y0"]), int(p["view_y1"])
        is_stereo: bool = p.get("is_stereo", True)
        normalize: bool = p.get("normalize", True)
        phase_mode: str = p.get("phase_mode", "zero_phase")

        routed = self._resolve_routing(p)

        n_frames = x1 - x0
        n_bins = y1 - y0
        if n_frames <= 0 or n_bins <= 0:
            raise ValueError("Empty viewport")

        view_freqs = freqs[y0:y1]
        view_times = times[x0:x1]
        dur = float(view_times[-1] - view_times[0]) if len(view_times) > 1 else 0.0
        vt0 = float(view_times[0])
        vt1 = float(view_times[-1])
        n_samples = int(round(dur * sr)) if dur > 0 else n_frames * synth.hop_length
        if n_samples <= 0:
            raise ValueError("Zero-length output")

        def _channel_from_routed(routed_d: dict, ch: str) -> np.ndarray:
            mag_arr = routed_d.get(f"magnitude_{ch}")
            pha_arr = routed_d.get(f"phase_{ch}")
            # Build envelope grid  (n_bins, n_frames)
            if mag_arr is not None:
                env = np.abs(np.asarray(mag_arr[y0:y1, x0:x1],
                                        dtype=np.float64))
            else:
                env = np.ones((n_bins, n_frames), dtype=np.float64)
            if pha_arr is not None and phase_mode == "original":
                pha = np.asarray(pha_arr[y0:y1, x0:x1], dtype=np.float64)
            else:
                pha = None

            output = np.zeros(n_samples, dtype=np.float64)
            frame_t = np.linspace(0.0, 1.0, n_frames)
            sample_t = np.linspace(0.0, 1.0, n_samples)
            t = np.arange(n_samples, dtype=np.float64) / sr

            if phase_mode == "zero_phase":
                # Pure envelope summation — no carrier, no phase
                for bi in range(n_bins):
                    amp = np.interp(sample_t, frame_t, env[bi])
                    output += amp
            elif phase_mode == "freq_carrier":
                # Free-running cosine carrier at each band centre freq
                for bi in range(n_bins):
                    fc = float(view_freqs[bi])
                    amp = np.interp(sample_t, frame_t, env[bi])
                    output += amp * np.cos(2.0 * np.pi * fc * t)
            elif phase_mode == "original" and pha is not None:
                # Use routed phase data as instantaneous phase
                for bi in range(n_bins):
                    amp = np.interp(sample_t, frame_t, env[bi])
                    inst_phase = np.interp(sample_t, frame_t, pha[bi])
                    output += amp * np.cos(inst_phase)
            elif phase_mode == "griffin_lim":
                # Griffin-Lim: iteratively estimate phase from magnitude.
                # Build a complex spectrogram-like matrix from the band
                # envelopes, run Griffin-Lim round-trips through per-bin
                # iFFT/FFT, then sum the resulting band signals.
                n_gl_iters = 20
                rng = np.random.default_rng(42)
                # Random initial phase per (bin, frame)
                est_phase = 2.0 * np.pi * rng.random((n_bins, n_frames))
                hop = max(1, n_samples // n_frames)
                for _it in range(n_gl_iters):
                    # Synthesise with current phase estimate
                    sig = np.zeros(n_samples, dtype=np.float64)
                    for bi in range(n_bins):
                        fc = float(view_freqs[bi])
                        amp = np.interp(sample_t, frame_t, env[bi])
                        ph = np.interp(sample_t, frame_t, est_phase[bi])
                        sig += amp * np.cos(2.0 * np.pi * fc * t + ph)
                    # Re-analyse: extract instantaneous phase at each
                    # band centre via windowed DFT
                    for bi in range(n_bins):
                        fc = float(view_freqs[bi])
                        for fi in range(n_frames):
                            centre = int(fi * hop)
                            half_w = hop
                            s0 = max(0, centre - half_w)
                            s1 = min(n_samples, centre + half_w)
                            seg = sig[s0:s1]
                            if len(seg) == 0:
                                continue
                            tt = np.arange(len(seg), dtype=np.float64) / sr
                            ref = np.exp(-2j * np.pi * fc * tt)
                            est_phase[bi, fi] = np.angle(np.sum(seg * ref))
                # Final synthesis with converged phases
                for bi in range(n_bins):
                    fc = float(view_freqs[bi])
                    amp = np.interp(sample_t, frame_t, env[bi])
                    ph = np.interp(sample_t, frame_t, est_phase[bi])
                    output += amp * np.cos(2.0 * np.pi * fc * t + ph)
            else:
                # Fallback: if "original" selected but no phase data
                # routed, degrade gracefully to zero-phase
                for bi in range(n_bins):
                    amp = np.interp(sample_t, frame_t, env[bi])
                    output += amp
            return output

        sig_l = _channel_from_routed(routed, "left")
        if is_stereo:
            sig_r = _channel_from_routed(routed, "right")
        else:
            sig_r = sig_l

        # Fade edges
        fade = min(256, n_samples // 4)
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade)
            sig_l[:fade] *= ramp
            sig_l[-fade:] *= ramp[::-1]
            sig_r[:fade] *= ramp
            sig_r[-fade:] *= ramp[::-1]

        if normalize:
            peak = max(np.abs(sig_l).max(), np.abs(sig_r).max(), 1e-12)
            sig_l = sig_l / peak * 0.8
            sig_r = sig_r / peak * 0.8

        i16_l = (sig_l * 32767).clip(-32768, 32767).astype(np.int16)
        i16_r = (sig_r * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([i16_l, i16_r])
        job.result_stereo = stereo
        job.duration = dur
        job.view_t0 = vt0
        job.view_t1 = vt1

    @staticmethod
    def _execute_wavelet(job: SynthJob, p: dict) -> None:
        """Wavelet reconstruction: inverse DWT or iCWT of viewport coefficients.

        Auto-detects DWT vs CWT from ``wv_meta["type"]``.
        """
        wv_meta: dict = p.get("wv_meta", {})
        sr: int = p["sr"]
        t0: float = p["t0"]
        t1: float = p["t1"]
        normalize: bool = p.get("normalize", True)
        wv_type = wv_meta.get("type", "dwt")

        if wv_type == "cwt":
            # ── CWT path: use torch_cqt_new.icwt ──
            import torch
            from torch_cqt_new import icwt as _torch_icwt

            W_complex: np.ndarray = p["wv_W"]
            wv_freqs: np.ndarray = p["wv_freqs"]
            if W_complex is None or wv_freqs is None:
                raise ValueError("No CWT scalogram data")

            hop = int(wv_meta.get("hop_length", 1))
            n_scales, n_frames = W_complex.shape
            # Determine total duration from meta or data
            total_dur = n_frames * hop / sr if sr > 0 else 1.0
            frac0 = max(0.0, t0 / total_dur) if total_dur > 0 else 0.0
            frac1 = min(1.0, t1 / total_dur) if total_dur > 0 else 1.0
            f0 = int(frac0 * n_frames)
            f1 = max(f0 + 1, int(frac1 * n_frames))
            f1 = min(f1, n_frames)
            W_slice = W_complex[:, f0:f1]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            W_t = torch.as_tensor(W_slice, device=device)
            freqs_t = torch.as_tensor(wv_freqs, device=device)

            wavelet_name = wv_meta.get("wavelet", "morlet")
            wavelet_kw: dict = {}
            if wavelet_name == "morlet":
                wavelet_kw["sigma"] = wv_meta.get("sigma", 6.0)
            eps = wv_meta.get("epsilon", None)
            if eps is not None and eps <= 0:
                eps = None

            out_len = (f1 - f0) * hop
            sig_t = _torch_icwt(
                W_t, freqs_t, sr,
                hop_length=hop,
                wavelet=wavelet_name,
                wavelet_kw=wavelet_kw,
                device=device,
                epsilon=eps,
                length=out_len,
            )
            sig = sig_t.cpu().numpy().astype(np.float64)
        else:
            # ── DWT path: pywt.waverec ──
            import pywt

            wv_coeffs: list[np.ndarray] = p["wv_coeffs"]
            wavelet_name: str = wv_meta.get("wavelet", "db4")

            if not wv_coeffs:
                raise ValueError("No wavelet coefficients")

            finest = wv_coeffs[-1]
            total_samples = len(finest) * 2
            dur_total = total_samples / sr if sr > 0 else 1.0

            frac0 = max(0.0, t0 / dur_total) if dur_total > 0 else 0.0
            frac1 = min(1.0, t1 / dur_total) if dur_total > 0 else 1.0

            windowed: list[np.ndarray] = []
            for coeff in wv_coeffs:
                n = len(coeff)
                s0 = int(frac0 * n)
                s1 = max(s0 + 1, int(frac1 * n))
                s1 = min(s1, n)
                windowed.append(coeff[s0:s1].copy())

            target_len = len(windowed[0])
            aligned: list[np.ndarray] = [windowed[0]]
            for i in range(1, len(windowed)):
                expected = target_len
                c = windowed[i]
                if len(c) < expected:
                    c = np.pad(c, (0, expected - len(c)))
                elif len(c) > expected:
                    c = c[:expected]
                aligned.append(c)
                target_len = expected * 2

            sig = pywt.waverec(aligned, wavelet_name)
            sig = np.asarray(sig, dtype=np.float64)

        # Fade edges
        n_samples = len(sig)
        fade = min(256, n_samples // 4)
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade)
            sig[:fade] *= ramp
            sig[-fade:] *= ramp[::-1]

        if normalize:
            peak = max(np.abs(sig).max(), 1e-12)
            sig = sig / peak * 0.8

        dur = float(t1 - t0)
        i16 = (sig * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([i16, i16])
        job.result_stereo = stereo
        job.duration = dur
        job.view_t0 = t0
        job.view_t1 = t1

    def _execute_hybrid(self, job: SynthJob, synth: "ViewportSynthPlayer",
                        p: dict) -> None:
        """Hybrid synthesis: FB below crossover + iCQT above, with
        complementary Linkwitz-Riley filtering at the boundary.
        Uses crossover.lr_decompose for perfect reconstruction.
        """
        import torch
        from crossover import lr_decompose, lr_reconstruct

        sr = p["sr"]
        cross_hz = float(p["cross_hz"])
        is_stereo: bool = p.get("is_stereo", True)
        normalize: bool = p.get("normalize", True)

        # --- iCQT portion (full viewport, will be high-passed) ---
        routed = self._resolve_routing(p)
        icqt_result = synth.synthesise_to_array(
            p["raw_arrays"], p["freqs"], p["times"],
            p["view_x0"], p["view_x1"],
            p["view_y0"], p["view_y1"],
            p["n_frames"], p["n_bins"],
            p["t_start"], p["t_dur"], sr,
            is_stereo, False,
            phase_mode=p.get("phase_mode", "original"),
            routed_arrays=routed)
        if icqt_result is None:
            raise ValueError("Empty viewport")
        icqt_stereo_i16, dur, vt0, vt1 = icqt_result
        icqt_f = icqt_stereo_i16.astype(np.float64) / 32767.0

        # --- FB portion (from audio, will be low-passed) ---
        fb_result = synth.synthesise_filterbank_from_audio(
            p["audio_path"], sr, vt0, vt1,
            float(p["f_lo"]), cross_hz,
            n_bands=p.get("n_bands", 16),
            is_stereo=is_stereo,
            normalize=False)
        if fb_result is None:
            job.result_stereo = icqt_stereo_i16
            job.duration = dur
            job.view_t0 = vt0
            job.view_t1 = vt1
            return

        fb_stereo_i16, _, _, _ = fb_result
        fb_f = fb_stereo_i16.astype(np.float64) / 32767.0

        # Match lengths
        n = min(len(icqt_f), len(fb_f))
        icqt_f = icqt_f[:n]
        fb_f = fb_f[:n]

        # --- Crossover filtering via crossover.py (perfect reconstruction) ---
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for ch in range(icqt_f.shape[1]):
            # Split iCQT into [low, high] bands at crossover
            icqt_bands = lr_decompose(
                torch.as_tensor(icqt_f[:, ch], device=device, dtype=torch.float64),
                [cross_hz], sr, order=8)
            # Keep only the high band from iCQT
            icqt_high = icqt_bands[..., 1, :].cpu().numpy()

            # Split FB into [low, high] bands at crossover
            fb_bands = lr_decompose(
                torch.as_tensor(fb_f[:, ch], device=device, dtype=torch.float64),
                [cross_hz], sr, order=8)
            # Keep only the low band from FB
            fb_low = fb_bands[..., 0, :].cpu().numpy()

            icqt_f[:n, ch] = icqt_high[:n]
            fb_f[:n, ch] = fb_low[:n]

        mixed = icqt_f + fb_f

        if normalize:
            peak = max(np.abs(mixed).max(), 1e-12)
            mixed = mixed / peak * 0.8
        else:
            mixed = np.clip(mixed, -1.0, 1.0)
        stereo = (mixed * 32767).clip(-32768, 32767).astype(np.int16)
        job.result_stereo = stereo
        job.duration = dur
        job.view_t0 = vt0
        job.view_t1 = vt1

    def _execute_hybrid_cwt(self, job: SynthJob,
                            synth: "ViewportSynthPlayer",
                            p: dict) -> None:
        """Hybrid CWT synthesis: CWT below crossover + iCQT above,
        with complementary Linkwitz-Riley filtering at the boundary.
        Uses crossover.lr_decompose for perfect reconstruction.
        """
        import torch
        from crossover import lr_decompose
        from torch_cqt_new import icwt as _torch_icwt

        sr = p["sr"]
        cross_hz = float(p["cross_hz"])
        is_stereo: bool = p.get("is_stereo", True)
        normalize: bool = p.get("normalize", True)

        # --- iCQT portion (full viewport, will be high-passed) ---
        routed = self._resolve_routing(p)
        icqt_result = synth.synthesise_to_array(
            p["raw_arrays"], p["freqs"], p["times"],
            p["view_x0"], p["view_x1"],
            p["view_y0"], p["view_y1"],
            p["n_frames"], p["n_bins"],
            p["t_start"], p["t_dur"], sr,
            is_stereo, False,
            phase_mode=p.get("phase_mode", "original"),
            routed_arrays=routed)
        if icqt_result is None:
            raise ValueError("Empty viewport")
        icqt_stereo_i16, dur, vt0, vt1 = icqt_result
        icqt_f = icqt_stereo_i16.astype(np.float64) / 32767.0

        # --- CWT portion (from stored scalogram, will be low-passed) ---
        W_complex: np.ndarray | None = p.get("wv_W")
        wv_freqs: np.ndarray | None = p.get("wv_freqs")
        wv_meta: dict = p.get("wv_meta", {})
        if W_complex is None or wv_freqs is None:
            job.result_stereo = icqt_stereo_i16
            job.duration = dur
            job.view_t0 = vt0
            job.view_t1 = vt1
            return

        # Reconstruct time-domain from CWT scalogram
        hop = int(wv_meta.get("hop_length", 1))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        W_t = torch.as_tensor(W_complex, device=device)
        freqs_t = torch.as_tensor(wv_freqs, device=device)
        wavelet_name = wv_meta.get("wavelet", "morlet")
        wavelet_kw: dict = {}
        if wavelet_name == "morlet":
            wavelet_kw["sigma"] = wv_meta.get("sigma", 6.0)
        eps = wv_meta.get("epsilon", None)
        if eps is not None and eps <= 0:
            eps = None

        # Window the CWT scalogram to the viewport time range
        n_scales, n_frames_cwt = W_complex.shape
        total_dur_cwt = n_frames_cwt * hop / sr if sr > 0 else 1.0
        frac0 = max(0.0, vt0 / total_dur_cwt) if total_dur_cwt > 0 else 0.0
        frac1 = min(1.0, vt1 / total_dur_cwt) if total_dur_cwt > 0 else 1.0
        f0 = int(frac0 * n_frames_cwt)
        f1 = max(f0 + 1, int(frac1 * n_frames_cwt))
        f1 = min(f1, n_frames_cwt)
        W_slice = W_t[:, f0:f1]

        target_len = len(icqt_f)
        cwt_sig = _torch_icwt(
            W_slice, freqs_t, sr,
            hop_length=hop,
            wavelet=wavelet_name,
            wavelet_kw=wavelet_kw,
            device=device,
            epsilon=eps,
            length=target_len,
        ).cpu().numpy().astype(np.float64)

        # Make mono CWT into stereo shape
        cwt_f = np.column_stack([cwt_sig, cwt_sig])

        # Match lengths
        n = min(len(icqt_f), len(cwt_f))
        icqt_f = icqt_f[:n]
        cwt_f = cwt_f[:n]

        # --- Crossover filtering via crossover.py (perfect reconstruction) ---
        for ch in range(icqt_f.shape[1]):
            icqt_bands = lr_decompose(
                torch.as_tensor(icqt_f[:, ch], device=device, dtype=torch.float64),
                [cross_hz], sr, order=8)
            icqt_f[:n, ch] = icqt_bands[..., 1, :].cpu().numpy()[:n]

            cwt_bands = lr_decompose(
                torch.as_tensor(cwt_f[:, ch], device=device, dtype=torch.float64),
                [cross_hz], sr, order=8)
            cwt_f[:n, ch] = cwt_bands[..., 0, :].cpu().numpy()[:n]

        mixed = icqt_f + cwt_f
        if normalize:
            peak = max(np.abs(mixed).max(), 1e-12)
            mixed = mixed / peak * 0.8
        else:
            mixed = np.clip(mixed, -1.0, 1.0)
        stereo = (mixed * 32767).clip(-32768, 32767).astype(np.int16)
        job.result_stereo = stereo
        job.duration = dur
        job.view_t0 = vt0
        job.view_t1 = vt1

    def _execute_transplant(self, job: SynthJob, p: dict) -> None:
        """TF transplant synthesis: move a viewport TF region to a new
        frequency range using the auto-dispatched STFT/CWT engine.

        The source region is the current viewport rectangle.  The
        destination frequency range is specified as ``dst_f0``/``dst_f1``
        in the job params.  Time span is preserved.
        """
        import torch
        from tf_surgery import (
            TFRegion, affine_translate, transplant, working_range,
        )

        sr = p["sr"]
        freqs = p["freqs"]
        duration = p["duration"]
        normalize = p.get("normalize", True)

        # Source region from viewport
        src_t0 = p["src_t0"]
        src_t1 = p["src_t1"]
        src_f0 = p["src_f0"]
        src_f1 = p["src_f1"]

        # Destination frequency range
        dst_f0 = p["dst_f0"]
        dst_f1 = p["dst_f1"]

        # Clamp to working range
        wr = working_range(sr, duration)
        src_f0 = max(src_f0, wr.f_min)
        src_f1 = min(src_f1, wr.f_max)
        dst_f0 = max(dst_f0, wr.f_min)
        dst_f1 = min(dst_f1, wr.f_max)

        if src_f1 <= src_f0 or dst_f1 <= dst_f0:
            raise ValueError("Degenerate frequency range after clamping "
                             f"to working range [{wr.f_min:.2f}, {wr.f_max:.1f}] Hz")

        src = TFRegion(t0=src_t0, t1=src_t1, f0=src_f0, f1=src_f1)

        # Build affine: pure frequency translation in normalised coords
        # df_norm = (dst_centre - src_centre) / src_span → normalised shift
        src_fc = (src_f0 + src_f1) / 2
        dst_fc = (dst_f0 + dst_f1) / 2
        src_span = src_f1 - src_f0
        df_norm = (dst_fc - src_fc) / src_span if src_span > 0 else 0.0
        affine = affine_translate(dt=0.0, df=df_norm)

        # Load audio and convert to torch
        audio_path = p.get("audio_path")
        if audio_path:
            _sr, data = _load_audio_float(audio_path)
            sig = torch.from_numpy(data)
        else:
            raise ValueError("Transplant requires an audio file path")

        result = transplant(sig, sr, src, affine)

        out = result.signal.numpy()
        if normalize:
            peak = max(np.abs(out).max(), 1e-12)
            out = out / peak * 0.8
        else:
            out = np.clip(out, -1.0, 1.0)

        i16 = (out * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([i16, i16])

        job.result_stereo = stereo
        job.duration = duration
        job.view_t0 = src_t0
        job.view_t1 = src_t1
        job.transplant_result = result  # stash for uncertainty display

    def remove_job(self, job_id: int) -> None:
        with self._lock:
            self.jobs = [j for j in self.jobs if j.job_id != job_id]

    # -- list sync ---------------------------------------------------------

    def _sync_list(self) -> None:
        items: list[ListItem] = []
        with self._lock:
            for j in self.jobs:
                tag = j.status
                color_key = {"done": "Ch2", "running": "Ch3",
                             "error": "Ch1", "queued": "none"}.get(j.status, "none")
                items.append(ListItem(
                    key=str(j.job_id),
                    display_name=j.label,
                    tag=tag,
                    tag_color_key=color_key,
                ))
        self.job_list.items = items

    # -- render ------------------------------------------------------------

    def _render_dropdown_btn(self, surf: pygame.Surface, font: Any,
                             y: int, w: int, key: str, prefix: str,
                             options: list[str], sel_idx: int) -> int:
        lbl = font.render(prefix, True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 2))
        btn_x = 80
        btn_w = w - btn_x - self.PAD
        btn_rect = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
        bg = (80, 80, 120) if self._active_dropdown == key else (60, 60, 80)
        pygame.draw.rect(surf, bg, btn_rect)
        pygame.draw.rect(surf, (100, 100, 120), btn_rect, 1)
        txt = font.render(options[sel_idx] if 0 <= sel_idx < len(options)
                          else "?", True, (220, 220, 220))
        surf.blit(txt, (btn_x + 4, y + 3))
        self._ctrl_rects[key] = btn_rect
        return y + self.ROW_H + 2

    def _render_toggle_btn(self, surf: pygame.Surface, font: Any,
                           y: int, w: int, key: str, label: str,
                           state: bool) -> int:
        btn_rect = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
        bg = (50, 100, 60) if state else (60, 60, 80)
        pygame.draw.rect(surf, bg, btn_rect)
        pygame.draw.rect(surf, (100, 100, 120), btn_rect, 1)
        mark = "\u2713 " if state else "  "
        txt = font.render(mark + label, True, (220, 220, 220))
        surf.blit(txt, (self.PAD + 4, y + 3))
        self._ctrl_rects[key] = btn_rect
        return y + self.ROW_H + 2

    def _render_action_btn(self, surf: pygame.Surface, font: Any,
                           x: int, y: int, bw: int, key: str,
                           label: str, enabled: bool,
                           color: tuple[int, int, int] = (60, 80, 160)) -> int:
        btn_rect = pygame.Rect(x, y, bw, self.ROW_H)
        bg = color if enabled else (45, 45, 55)
        pygame.draw.rect(surf, bg, btn_rect)
        pygame.draw.rect(surf, (100, 100, 120), btn_rect, 1)
        tc = (220, 220, 220) if enabled else (90, 90, 100)
        txt = font.render(label, True, tc)
        surf.blit(txt, (x + 4, y + 3))
        self._ctrl_rects[key] = btn_rect
        return x + bw + 4

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._sync_list()
        self._play_rects.clear()
        self._del_rects.clear()
        self._ctrl_rects.clear()

        # --- Controls section ---
        cur_phase = self._get_phase_mode()
        phase_idx = (self._PHASE_KEYS.index(cur_phase)
                     if cur_phase in self._PHASE_KEYS else 0)
        cur_bands = self._get_fb_n_bands()
        bands_idx = (self._FB_BAND_VALUES.index(cur_bands)
                     if cur_bands in self._FB_BAND_VALUES else 2)

        est_h = 800
        surf = pygame.Surface((w, est_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))
        y = self.PAD

        # Section: iCQT phase mode
        y = self._render_dropdown_btn(surf, font, y, w, "phase_mode",
                                      "Phase:", self._PHASE_OPTIONS, phase_idx)

        # Section: Normalize toggle
        y = self._render_toggle_btn(surf, font, y, w, "normalize",
                                    "Normalize Loudness",
                                    self._get_normalize())

        # Section: Use viewer adjustments toggle
        y = self._render_toggle_btn(surf, font, y, w, "use_adjusted",
                                    "Apply Viewer Adjustments",
                                    self.use_adjusted)

        # Section: Synth mode dropdown
        y = self._render_dropdown_btn(surf, font, y, w, "synth_mode",
                                      "Synth:", self._SYNTH_MODE_OPTIONS,
                                      self.synth_mode_idx)

        # Section: Transplant destination (only when transplant mode is active)
        cur_smode = self._SYNTH_MODE_KEYS[self.synth_mode_idx]
        if cur_smode == "transplant":
            hdr = font.render("── Transplant Dst ──", True, (140, 180, 220))
            surf.blit(hdr, (self.PAD, y + 2))
            y += self.ROW_H

            # Dst F0
            f0_lbl = font.render(f"Lo: {self.transplant_dst_f0:.1f} Hz",
                                 True, (200, 200, 200))
            surf.blit(f0_lbl, (self.PAD, y + 2))
            minus_rect = pygame.Rect(w - self.PAD - 2 * self.BTN_W - 4, y,
                                     self.BTN_W, self.ROW_H)
            plus_rect = pygame.Rect(w - self.PAD - self.BTN_W, y,
                                    self.BTN_W, self.ROW_H)
            for r, txt in [(minus_rect, "−"), (plus_rect, "+")]:
                pygame.draw.rect(surf, (60, 60, 80), r)
                pygame.draw.rect(surf, (100, 100, 120), r, 1)
                t = font.render(txt, True, (220, 220, 220))
                surf.blit(t, (r.x + r.w // 2 - t.get_width() // 2,
                              r.y + 3))
            self._ctrl_rects["txp_f0_minus"] = minus_rect
            self._ctrl_rects["txp_f0_plus"] = plus_rect
            y += self.ROW_H + 2

            # Dst F1
            f1_lbl = font.render(f"Hi: {self.transplant_dst_f1:.1f} Hz",
                                 True, (200, 200, 200))
            surf.blit(f1_lbl, (self.PAD, y + 2))
            minus_rect = pygame.Rect(w - self.PAD - 2 * self.BTN_W - 4, y,
                                     self.BTN_W, self.ROW_H)
            plus_rect = pygame.Rect(w - self.PAD - self.BTN_W, y,
                                    self.BTN_W, self.ROW_H)
            for r, txt in [(minus_rect, "−"), (plus_rect, "+")]:
                pygame.draw.rect(surf, (60, 60, 80), r)
                pygame.draw.rect(surf, (100, 100, 120), r, 1)
                t = font.render(txt, True, (220, 220, 220))
                surf.blit(t, (r.x + r.w // 2 - t.get_width() // 2,
                              r.y + 3))
            self._ctrl_rects["txp_f1_minus"] = minus_rect
            self._ctrl_rects["txp_f1_plus"] = plus_rect
            y += self.ROW_H + 2

        # Section: Synthesize button
        has_synth = self._has_synth()
        has_audio = self._has_audio()
        has_wavelet = self._has_wavelet()
        cur_smode = self._SYNTH_MODE_KEYS[self.synth_mode_idx]
        can_synth = has_synth and (
            cur_smode in ("icqt", "fb_tf")
            or (cur_smode in ("fb", "hybrid", "transplant") and has_audio)
            or (cur_smode == "wavelet" and has_wavelet)
            or (cur_smode == "hybrid_cwt" and has_wavelet and has_audio)
        )
        btn_w = w - 2 * self.PAD
        self._render_action_btn(surf, font, self.PAD, y, btn_w,
                                "synth_go", "Synthesize", can_synth,
                                color=(50, 80, 140))
        y += self.ROW_H + 4

        # Section: Filterbank bands
        y = self._render_dropdown_btn(surf, font, y, w, "fb_n_bands",
                                      "FB Bands:", self._FB_BAND_OPTIONS, bands_idx)

        # Section: Channel Routing
        pygame.draw.line(surf, (60, 60, 70), (self.PAD, y),
                         (w - self.PAD, y))
        y += 2
        hdr = font.render("\u2500\u2500 Channel Routing \u2500\u2500",
                          True, (180, 180, 180))
        surf.blit(hdr, (self.PAD, y + 2))
        y += self.ROW_H + 2

        routing = self.channel_routing
        n_src = len(ROUTE_SOURCES)
        lbl_w = 60  # width for role label
        avail_w = w - self.PAD - lbl_w - self.PAD
        col_w = max(36, avail_w // n_src)  # auto-size for 5 buttons
        for audio_ch in ROUTE_AUDIO_CHANNELS:
            ch_label = ROUTE_AUDIO_CH_LABELS[audio_ch]
            ch_hdr = font.render(f"{ch_label}:", True, (140, 180, 220))
            surf.blit(ch_hdr, (self.PAD, y + 2))
            y += self.ROW_H
            for role in ROUTE_ROLES:
                role_lbl = ROUTE_ROLE_LABELS[role]
                rl = font.render(role_lbl, True, (140, 140, 150))
                surf.blit(rl, (self.PAD + 8, y + 3))
                bx = self.PAD + lbl_w
                cur_src = routing.get(audio_ch, role)
                for src in ROUTE_SOURCES:
                    btn_key = f"rt_{audio_ch}_{role}_{src}"
                    active = (cur_src == src)
                    # Tint button with the channel's hue colour
                    tint = _TGT_COLORS.get(src, (80, 80, 80))
                    if active:
                        bg = tint
                    else:
                        bg = (tint[0] // 4, tint[1] // 4, tint[2] // 4)
                    br = pygame.Rect(bx, y, col_w - 2, self.ROW_H)
                    pygame.draw.rect(surf, bg, br)
                    border = tint if not active else (220, 220, 220)
                    pygame.draw.rect(surf, border, br, 1)
                    tc = (255, 255, 255) if active else tint
                    st = font.render(src, True, tc)
                    surf.blit(st, (bx + col_w // 2 - st.get_width() // 2,
                                   y + 3))
                    self._ctrl_rects[btn_key] = br
                    bx += col_w
                y += self.ROW_H + 2

        # Mono mix mode selector
        y = self._render_dropdown_btn(surf, font, y, w, "mono_mix",
                                      "Mono Mix:", MONO_MIX_MODES,
                                      routing.mono_mix_mode)

        # Separator
        pygame.draw.line(surf, (60, 60, 70), (self.PAD, y),
                         (w - self.PAD, y))
        y += 4

        ctrl_h = y  # height consumed by the controls section

        # Job list rendering
        y_after_list = self.job_list.render(surf, font, 0, ctrl_h, w)
        y_cur = ctrl_h + y_after_list

        # Action buttons for selected job
        sel_job = self._selected_job()
        if sel_job is not None:
            pad = self.PAD
            bx = pad
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
            if sel_job.duration > 0:
                dur_txt = f"{sel_job.duration:.2f}s"
                dtl = font.render(dur_txt, True, (160, 160, 160))
                surf.blit(dtl, (bx, y_cur + 3))
                bx += dtl.get_width() + 8
            if sel_job.status == "error":
                etl = font.render(sel_job.error_msg[:30], True, (255, 80, 80))
                surf.blit(etl, (bx, y_cur + 3))
            del_w = 26
            del_x = w - pad - del_w
            dr = pygame.Rect(del_x, y_cur, del_w, self.ROW_H)
            pygame.draw.rect(surf, (120, 40, 40), dr)
            pygame.draw.rect(surf, (200, 200, 200), dr, 1)
            xl = font.render("X", True, (255, 255, 255))
            surf.blit(xl, (del_x + 7, y_cur + 3))
            self._del_rects[sel_job.job_id] = dr
            y_cur += self.ROW_H + 4

        # Crop the oversized surface to actual content
        final_h = max(y_cur + self.PAD, ctrl_h + 30)
        final_surf = pygame.Surface((w, final_h), pygame.SRCALPHA)
        final_surf.blit(surf, (0, 0))
        return final_surf

    def _selected_job(self) -> SynthJob | None:
        idx = self.job_list.selected_idx
        with self._lock:
            if 0 <= idx < len(self.jobs):
                return self.jobs[idx]
        return None

    # -- events ------------------------------------------------------------

    _PHASE_KEYS = ["original", "griffin_lim", "zero_phase", "freq_carrier"]
    _PHASE_OPTIONS = ["Original Phase", "Griffin-Lim", "Zero Phase", "Freq Carrier"]
    _FB_BAND_VALUES = [4, 8, 16, 32, 64]
    _FB_BAND_OPTIONS = ["4", "8", "16", "32", "64"]

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        pr = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Dropdown overlay gets first priority (screen coords)
            if self._active_dropdown:
                for i, ir in enumerate(self._dropdown_item_rects):
                    if ir.collidepoint(mx, my):
                        self._select_dropdown(i)
                        return True
                self._active_dropdown = None
                return True

            if not pr.collidepoint(mx, my):
                return False
            lx, ly = mx - pr.x, my - pr.y
            ly += self._panel_scroll_y

            # Controls
            for key, rect in self._ctrl_rects.items():
                if rect.collidepoint(lx, ly):
                    if key == "phase_mode":
                        self._open_dropdown("phase_mode",
                                            self._PHASE_OPTIONS,
                                            pygame.Rect(rect.x,
                                                        rect.y + rect.h,
                                                        rect.w, 0))
                    elif key == "normalize":
                        self._set_normalize(not self._get_normalize())
                    elif key == "use_adjusted":
                        self.use_adjusted = not self.use_adjusted
                    elif key == "synth_mode":
                        self._open_dropdown("synth_mode",
                                            self._SYNTH_MODE_OPTIONS,
                                            pygame.Rect(rect.x,
                                                        rect.y + rect.h,
                                                        rect.w, 0))
                    elif key == "synth_go":
                        smode = self._SYNTH_MODE_KEYS[self.synth_mode_idx]
                        if smode == "icqt" and self._has_synth() and self.on_synth_icqt:
                            self.on_synth_icqt()
                        elif smode == "fb" and self._has_synth() and self._has_audio() and self.on_synth_fb:
                            self.on_synth_fb()
                        elif smode == "hybrid" and self._has_synth() and self._has_audio() and self.on_synth_hybrid:
                            self.on_synth_hybrid()
                        elif smode == "fb_tf" and self._has_synth() and self.on_synth_fb_tf:
                            self.on_synth_fb_tf()
                        elif smode == "wavelet" and self._has_wavelet() and self.on_synth_wavelet:
                            self.on_synth_wavelet()
                        elif smode == "transplant" and self._has_synth() and self._has_audio() and self.on_synth_transplant:
                            self.on_synth_transplant()
                        elif smode == "hybrid_cwt" and self._has_synth() and self._has_wavelet() and self.on_synth_hybrid_cwt:
                            self.on_synth_hybrid_cwt()
                    elif key == "fb_n_bands":
                        self._open_dropdown("fb_n_bands",
                                            self._FB_BAND_OPTIONS,
                                            pygame.Rect(rect.x,
                                                        rect.y + rect.h,
                                                        rect.w, 0))
                    elif key.startswith("txp_f"):
                        # Transplant destination freq ±
                        step = 10.0
                        if key == "txp_f0_minus":
                            self.transplant_dst_f0 = max(1.0, self.transplant_dst_f0 - step)
                        elif key == "txp_f0_plus":
                            self.transplant_dst_f0 = min(self.transplant_dst_f1 - 1.0,
                                                         self.transplant_dst_f0 + step)
                        elif key == "txp_f1_minus":
                            self.transplant_dst_f1 = max(self.transplant_dst_f0 + 1.0,
                                                         self.transplant_dst_f1 - step)
                        elif key == "txp_f1_plus":
                            self.transplant_dst_f1 += step
                    elif key == "mono_mix":
                        self._open_dropdown("mono_mix",
                                            MONO_MIX_MODES,
                                            pygame.Rect(rect.x,
                                                        rect.y + rect.h,
                                                        rect.w, 0))
                    elif key.startswith("rt_"):
                        # rt_{audio_ch}_{role}_{source}
                        parts = key.split("_", 3)
                        if len(parts) == 4:
                            _, audio_ch, role, source = parts
                            self.channel_routing.set(audio_ch, role, source)

                    return True

            # Job action buttons
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
            return self._handle_panel_wheel(event)

        return False

    def _open_dropdown(self, key: str, options: list[str],
                       anchor: pygame.Rect) -> None:
        self._active_dropdown = key
        self._dropdown_opts = options
        self._dropdown_rect = anchor

    def _select_dropdown(self, idx: int) -> None:
        key = self._active_dropdown
        self._active_dropdown = None
        if key == "phase_mode" and 0 <= idx < len(self._PHASE_KEYS):
            self._set_phase_mode(self._PHASE_KEYS[idx])
        elif key == "synth_mode" and 0 <= idx < len(self._SYNTH_MODE_KEYS):
            self.synth_mode_idx = idx
        elif key == "fb_n_bands" and 0 <= idx < len(self._FB_BAND_VALUES):
            self._set_fb_n_bands(self._FB_BAND_VALUES[idx])
        elif key == "mono_mix" and 0 <= idx < len(MONO_MIX_MODES):
            self.channel_routing.mono_mix_mode = idx


    def render_dropdown_overlay(self) -> pygame.Surface | None:
        if not self._active_dropdown:
            return None
        opts = self._dropdown_opts
        if not opts:
            return None
        self._ensure_font()
        font = self.font
        pr = self.panel_rect
        item_h = self.ROW_H
        ow = self._dropdown_rect.w
        oh = len(opts) * item_h
        overlay = pygame.Surface((ow, oh), pygame.SRCALPHA)
        overlay.fill((40, 40, 55, 240))
        self._dropdown_item_rects = []
        for i, label in enumerate(opts):
            iy = i * item_h
            # Screen-space rect for hit-testing in handle_event
            ir_screen = pygame.Rect(pr.x + self._dropdown_rect.x,
                                    pr.y + self._dropdown_rect.y + iy
                                    - self._panel_scroll_y,
                                    ow, item_h)
            self._dropdown_item_rects.append(ir_screen)
            pygame.draw.rect(overlay, (60, 60, 80), (0, iy, ow, item_h), 1)
            txt = font.render(label, True, (220, 220, 220))
            overlay.blit(txt, (4, iy + 3))
        return overlay

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
# Image input panel — load images and convert to TF / TSC / FB
# ---------------------------------------------------------------------------

_IMG_CONVERT_MODES = ["TF", "TSC", "FB"]
_IMG_CONVERT_LABELS = {
    "TF": "Time-Frequency",
    "TSC": "Time-Scale Coefficients",
    "FB": "Filterbank (TF)",
}
_IMG_INTERP_MODES = ["bilinear", "bicubic", "nearest", "lanczos"]
_IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class ImageInputPanel(Panel):
    """Panel for loading image files, converting them to TF representations.

    Images can be converted to:
    * TF — mapped directly to time-frequency magnitude/phase arrays
    * TSC — treated as time-scale coefficient arrays
    * FB — mapped to filterbank envelope arrays

    Configurable: frequency range, time range, interpolation, target
    dimensions (bins × frames).
    """

    PANEL_W = 280
    ROW_H = 22
    PAD = 6

    def __init__(self, *, side: str = "left",
                 image_dir: str = "") -> None:
        super().__init__(title="Image Input", side=side)
        self.image_dir: str = image_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "input_images")
        os.makedirs(self.image_dir, exist_ok=True)

        # Image file list
        self.image_list = ScrollableItemList("Image Files", max_visible=8)
        self._image_paths: list[str] = []
        self._refresh_images()

        # Conversion settings
        self.convert_mode_idx: int = 0   # index into _IMG_CONVERT_MODES
        self.interp_idx: int = 0         # index into _IMG_INTERP_MODES
        self.target_bins: int = 1200     # frequency bins
        self.target_frames: int = 1000   # time frames
        self.freq_min: float = 16.35     # Hz (C0)
        self.freq_max: float = 19912.13  # Hz (~D#9)
        self.time_start: float = 0.0     # seconds
        self.time_duration: float = 10.0 # seconds

        # Result data (populated after conversion)
        self._converted_data: np.ndarray | None = None
        self._converted_mode: str | None = None

        # UI state
        self._ctrl_rects: dict[str, pygame.Rect] = {}
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        self._dragging: str | None = None
        self._item_map: dict[str, tuple[pygame.Rect, Any]] = {}
        self._status_msg: str = ""

        # Callback: (data, mode, freq_range, time_range) -> None
        self.on_converted: Any = None

    # -- scanning -----------------------------------------------------------

    def _refresh_images(self) -> None:
        self._image_paths = []
        items: list[ListItem] = []
        if os.path.isdir(self.image_dir):
            for fn in sorted(os.listdir(self.image_dir)):
                ext = os.path.splitext(fn)[1].lower()
                if ext in _IMG_EXTENSIONS:
                    self._image_paths.append(
                        os.path.join(self.image_dir, fn))
                    items.append(ListItem(
                        key=fn, display_name=fn,
                        tag=ext, tag_color_key="none"))
        self.image_list.items = items

    # -- conversion ---------------------------------------------------------

    def _load_image_as_array(self, path: str) -> np.ndarray:
        """Load image file as float array, shape (H, W) or (H, W, C)."""
        from PIL import Image
        img = Image.open(path)
        arr = np.asarray(img)
        return arr

    def _interpolate_2d(self, data: np.ndarray,
                        out_h: int, out_w: int) -> np.ndarray:
        """Resize 2D array to (out_h, out_w) using selected interpolation."""
        from scipy.ndimage import zoom
        mode = _IMG_INTERP_MODES[self.interp_idx]
        order = {"nearest": 0, "bilinear": 1, "bicubic": 3,
                 "lanczos": 5}.get(mode, 1)
        h, w = data.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((out_h, out_w), dtype=data.dtype)
        factor_h = out_h / h
        factor_w = out_w / w
        if data.ndim == 2:
            return zoom(data, (factor_h, factor_w), order=order)
        # Multi-channel: zoom spatial dims only
        return zoom(data, (factor_h, factor_w, 1), order=order)

    def convert_selected(self) -> None:
        """Convert the currently selected image to the chosen format."""
        idx = self.image_list.selected_idx
        if idx < 0 or idx >= len(self._image_paths):
            self._status_msg = "No image selected"
            return
        path = self._image_paths[idx]
        mode = _IMG_CONVERT_MODES[self.convert_mode_idx]
        try:
            raw = self._load_image_as_array(path)
        except Exception as exc:
            self._status_msg = f"Load error: {exc}"
            return

        # Convert to grayscale intensity if multi-channel
        if raw.ndim == 3:
            if raw.shape[2] == 4:
                # RGBA — use alpha-premultiplied luminance
                rgb = raw[:, :, :3].astype(np.float64)
                alpha = raw[:, :, 3:4].astype(np.float64) / 255.0
                gray = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1]
                        + 0.0722 * rgb[:, :, 2]) * alpha[:, :, 0]
            else:
                gray = (0.2126 * raw[:, :, 0].astype(np.float64)
                        + 0.7152 * raw[:, :, 1].astype(np.float64)
                        + 0.0722 * raw[:, :, 2].astype(np.float64))
        else:
            gray = raw.astype(np.float64)

        # Normalise to 0–1
        vmin, vmax = gray.min(), gray.max()
        if vmax > vmin:
            gray = (gray - vmin) / (vmax - vmin)
        else:
            gray = np.zeros_like(gray)

        # Resize to target dimensions: (bins, frames)
        resized = self._interpolate_2d(
            gray, self.target_bins, self.target_frames)

        self._converted_data = resized
        self._converted_mode = mode
        self._status_msg = (f"Converted: {resized.shape[0]}×{resized.shape[1]}"
                            f" ({mode})")

        if self.on_converted:
            freq_range = (self.freq_min, self.freq_max)
            time_range = (self.time_start,
                          self.time_start + self.time_duration)
            self.on_converted(resized, mode, freq_range, time_range)

    # -- render -------------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._ctrl_rects.clear()
        self._item_map.clear()
        self._refresh_images()

        est_h = 700
        surf = pygame.Surface((w, est_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))
        y = self.PAD

        # Section: Image file list
        data_h = self.image_list.render(surf, font, 0, y, w)
        y += data_h

        # Separator
        pygame.draw.line(surf, (60, 60, 70), (self.PAD, y),
                         (w - self.PAD, y))
        y += 4

        # Section: Convert mode dropdown
        lbl = font.render("Output:", True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 2))
        btn_x = 80
        btn_w = w - btn_x - self.PAD
        btn_rect = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
        bg = (80, 80, 120) if self._active_dropdown == "conv_mode" \
            else (60, 60, 80)
        pygame.draw.rect(surf, bg, btn_rect)
        pygame.draw.rect(surf, (100, 100, 120), btn_rect, 1)
        ct = font.render(
            _IMG_CONVERT_LABELS[_IMG_CONVERT_MODES[self.convert_mode_idx]],
            True, (220, 220, 220))
        surf.blit(ct, (btn_x + 4, y + 3))
        self._ctrl_rects["conv_mode"] = btn_rect
        y += self.ROW_H + 2

        # Section: Interpolation dropdown
        lbl2 = font.render("Interp:", True, (160, 160, 160))
        surf.blit(lbl2, (self.PAD, y + 2))
        btn_rect2 = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
        bg2 = (80, 80, 120) if self._active_dropdown == "interp" \
            else (60, 60, 80)
        pygame.draw.rect(surf, bg2, btn_rect2)
        pygame.draw.rect(surf, (100, 100, 120), btn_rect2, 1)
        it = font.render(_IMG_INTERP_MODES[self.interp_idx],
                         True, (220, 220, 220))
        surf.blit(it, (btn_x + 4, y + 3))
        self._ctrl_rects["interp"] = btn_rect2
        y += self.ROW_H + 2

        # Section: Sliders for bins, frames, freq, time
        slider_defs = [
            ("bins", "Bins:", self.target_bins, 64, 4800),
            ("frames", "Frames:", self.target_frames, 100, 10000),
            ("freq_min", "Freq Lo:", self.freq_min, 1.0, 20000.0),
            ("freq_max", "Freq Hi:", self.freq_max, 10.0, 22050.0),
            ("time_dur", "Duration:", self.time_duration, 0.1, 300.0),
        ]
        for key, label, val, vmin, vmax in slider_defs:
            sl = font.render(label, True, (140, 140, 150))
            surf.blit(sl, (self.PAD, y + 3))
            # Value
            if isinstance(val, int):
                vt = font.render(str(val), True, (200, 200, 200))
            else:
                vt = font.render(f"{val:.1f}", True, (200, 200, 200))
            surf.blit(vt, (self.PAD + 72, y + 3))

            # +/- buttons
            minus_r = pygame.Rect(w - self.PAD - 44, y, 20, self.ROW_H)
            plus_r = pygame.Rect(w - self.PAD - 22, y, 20, self.ROW_H)
            pygame.draw.rect(surf, (50, 50, 65), minus_r)
            pygame.draw.rect(surf, (80, 80, 100), minus_r, 1)
            m_lbl = font.render("\u2212", True, (180, 180, 180))
            surf.blit(m_lbl, (minus_r.x + 5, y + 3))
            pygame.draw.rect(surf, (50, 50, 65), plus_r)
            pygame.draw.rect(surf, (80, 80, 100), plus_r, 1)
            p_lbl = font.render("+", True, (180, 180, 180))
            surf.blit(p_lbl, (plus_r.x + 5, y + 3))
            self._ctrl_rects[f"{key}_minus"] = minus_r
            self._ctrl_rects[f"{key}_plus"] = plus_r
            y += self.ROW_H + 2

        # Convert button
        y += 2
        conv_btn = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
        pygame.draw.rect(surf, (50, 80, 140), conv_btn)
        pygame.draw.rect(surf, (100, 100, 120), conv_btn, 1)
        cl = font.render("Convert Image", True, (220, 220, 220))
        surf.blit(cl, (w // 2 - cl.get_width() // 2, y + 3))
        self._ctrl_rects["convert_go"] = conv_btn
        y += self.ROW_H + 4

        # Status line
        if self._status_msg:
            st = font.render(self._status_msg[:40], True, (160, 160, 160))
            surf.blit(st, (self.PAD, y + 2))
            y += self.ROW_H + 2

        # Crop
        final_h = max(y + self.PAD, 100)
        final_surf = pygame.Surface((w, final_h), pygame.SRCALPHA)
        final_surf.blit(surf, (0, 0))
        return final_surf

    # -- events -------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        pr = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Dropdown overlay first
            if self._active_dropdown:
                for i, ir in enumerate(self._dropdown_item_rects):
                    if ir.collidepoint(mx, my):
                        self._select_dropdown(i)
                        return True
                self._active_dropdown = None
                return True

            if not pr.collidepoint(mx, my):
                return False
            lx, ly = mx - pr.x, my - pr.y
            ly += self._panel_scroll_y

            # Image list click
            hit = self.image_list.handle_click(lx, ly)
            if hit is not None:
                return True

            # Controls
            for key, rect in self._ctrl_rects.items():
                if rect.collidepoint(lx, ly):
                    if key == "conv_mode":
                        self._open_dropdown(
                            "conv_mode",
                            [_IMG_CONVERT_LABELS[m]
                             for m in _IMG_CONVERT_MODES],
                            rect)
                    elif key == "interp":
                        self._open_dropdown("interp",
                                            _IMG_INTERP_MODES, rect)
                    elif key == "convert_go":
                        self.convert_selected()
                    elif key.endswith("_minus") or key.endswith("_plus"):
                        self._nudge_slider(key)
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
            return self._handle_panel_wheel(event)

        return False

    def _open_dropdown(self, key: str, options: list[str],
                       anchor: pygame.Rect) -> None:
        self._active_dropdown = key
        self._dropdown_opts = options
        self._dropdown_rect = anchor

    def _select_dropdown(self, idx: int) -> None:
        key = self._active_dropdown
        self._active_dropdown = None
        if key == "conv_mode" and 0 <= idx < len(_IMG_CONVERT_MODES):
            self.convert_mode_idx = idx
        elif key == "interp" and 0 <= idx < len(_IMG_INTERP_MODES):
            self.interp_idx = idx

    def _nudge_slider(self, btn_key: str) -> None:
        """Handle +/- button clicks for value sliders."""
        base = btn_key.rsplit("_", 1)[0]
        direction = 1 if btn_key.endswith("_plus") else -1
        if base == "bins":
            step = 100 * direction
            self.target_bins = max(64, min(4800,
                                           self.target_bins + step))
        elif base == "frames":
            step = 100 * direction
            self.target_frames = max(100, min(10000,
                                              self.target_frames + step))
        elif base == "freq_min":
            step = 10.0 * direction
            self.freq_min = max(1.0, min(self.freq_max - 10,
                                         self.freq_min + step))
        elif base == "freq_max":
            step = 100.0 * direction
            self.freq_max = max(self.freq_min + 10,
                                min(22050.0, self.freq_max + step))
        elif base == "time_dur":
            step = 1.0 * direction
            self.time_duration = max(0.1, min(300.0,
                                              self.time_duration + step))

    def render_dropdown_overlay(self) -> pygame.Surface | None:
        if not self._active_dropdown:
            return None
        opts = self._dropdown_opts
        if not opts:
            return None
        self._ensure_font()
        font = self.font
        pr = self.panel_rect
        item_h = self.ROW_H
        ow = self._dropdown_rect.w
        oh = len(opts) * item_h
        overlay = pygame.Surface((ow, oh), pygame.SRCALPHA)
        overlay.fill((40, 40, 55, 240))
        self._dropdown_item_rects = []
        for i, label in enumerate(opts):
            iy = i * item_h
            rect = pygame.Rect(
                pr.x + self._dropdown_rect.x,
                pr.y + self._dropdown_rect.y + self._dropdown_rect.h + iy
                - self._panel_scroll_y,
                ow, item_h)
            self._dropdown_item_rects.append(rect)
            txt = font.render(label, True, (220, 220, 220))
            overlay.blit(txt, (4, iy + 3))
            if i < len(opts) - 1:
                pygame.draw.line(overlay, (60, 60, 70),
                                 (0, iy + item_h - 1),
                                 (ow, iy + item_h - 1))
        pygame.draw.rect(overlay, (100, 100, 130),
                         pygame.Rect(0, 0, ow, oh), 1)
        return overlay


# ---------------------------------------------------------------------------
# Source panel — WAV browser, analysis settings, filter bank, folder mgmt
# ---------------------------------------------------------------------------

_CHANNEL_MODES = ["Stereo", "L", "R", "Mono Mix"]

# Wavelet families — complete orthogonal dictionaries first
_WAVELET_FAMILIES = [
    "Daubechies", "Coiflets", "Symlets", "Biorthogonal",
    "Reverse Biorth.", "Discrete Meyer",
]
_WAVELET_FAMILY_PREFIX = {
    "Daubechies": "db", "Coiflets": "coif", "Symlets": "sym",
    "Biorthogonal": "bior", "Reverse Biorth.": "rbio",
    "Discrete Meyer": "dmey",
}
_WAVELET_ORDERS: dict[str, list[str]] = {
    "Daubechies":      [str(i) for i in range(1, 21)],
    "Coiflets":        [str(i) for i in range(1, 18)],
    "Symlets":         [str(i) for i in range(2, 21)],
    "Biorthogonal":    ["1.1","1.3","1.5","2.2","2.4","2.6","2.8",
                        "3.1","3.3","3.5","3.7","3.9",
                        "4.4","5.5","6.8"],
    "Reverse Biorth.": ["1.1","1.3","1.5","2.2","2.4","2.6","2.8",
                        "3.1","3.3","3.5","3.7","3.9",
                        "4.4","5.5","6.8"],
    "Discrete Meyer":  ["—"],
}
_WAVELET_EXTENSIONS = ["symmetric", "zero", "constant", "periodic",
                       "smooth", "antisymmetric", "reflect"]
_WAVELET_MODES = ["DWT (pywt)", "CWT (torch)"]
_CWT_WAVELET_TYPES = ["morlet", "morse", "ricker"]


# ---------------------------------------------------------------------------
# Signal Generator panel — tone / silence / recording WAV builder
# ---------------------------------------------------------------------------

_SIG_WAVE_SHAPES = ["Sine", "Triangle", "Sawtooth", "Square", "Pulse",
                    "White Noise", "Pink Noise"]
_SIG_GLIDE_MODES = ["None", "Linear", "Exponential", "Logarithmic"]
_SIG_BIT_DEPTHS = ["16-bit int", "24-bit int", "32-bit float"]
_SIG_CHANNELS = ["Mono", "Stereo"]
_SIG_GEN_MODES = ["Tone Generator", "Silent Track", "Record Input"]


def _generate_wave_cycle(shape: str, phase: np.ndarray,
                         pulse_width: float = 0.5) -> np.ndarray:
    """Return one waveform cycle for *phase* array (0…2π repeating)."""
    t = phase / (2.0 * np.pi)
    t = t - np.floor(t)
    if shape == "Sine":
        return np.sin(phase)
    elif shape == "Triangle":
        return 2.0 * np.abs(2.0 * t - 1.0) - 1.0
    elif shape == "Sawtooth":
        return 2.0 * t - 1.0
    elif shape == "Square":
        return np.where(t < 0.5, 1.0, -1.0)
    elif shape == "Pulse":
        return np.where(t < pulse_width, 1.0, -1.0)
    elif shape == "White Noise":
        return np.random.default_rng().uniform(-1.0, 1.0, size=phase.shape)
    elif shape == "Pink Noise":
        # Approximate pink noise via spectral shaping
        n = len(phase)
        white = np.random.default_rng().standard_normal(n)
        fft = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0)
        freqs[0] = 1.0  # avoid div-by-zero
        fft /= np.sqrt(freqs)
        pink = np.fft.irfft(fft, n=n)
        peak = np.abs(pink).max()
        if peak > 0:
            pink /= peak
        return pink
    return np.sin(phase)


def _apply_harmonics(fundamental_phase: np.ndarray, shape: str,
                     harmonics: list[tuple[int, float, float]],
                     pulse_width: float = 0.5) -> np.ndarray:
    """Sum fundamental + overtones.

    *harmonics* is a list of ``(harmonic_number, amplitude, phase_offset)``
    tuples.  Harmonic 1 = fundamental.
    """
    out = np.zeros_like(fundamental_phase, dtype=np.float64)
    for h_num, h_amp, h_phase in harmonics:
        if h_amp == 0.0:
            continue
        ph = fundamental_phase * h_num + h_phase
        out += h_amp * _generate_wave_cycle(shape, ph, pulse_width)
    return out


def _build_freq_envelope(n_samples: int, sr: int,
                         f_start: float, f_end: float,
                         glide: str) -> np.ndarray:
    """Return instantaneous-frequency envelope in Hz, length *n_samples*."""
    if glide == "None" or abs(f_end - f_start) < 1e-6:
        return np.full(n_samples, f_start, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n_samples, endpoint=False, dtype=np.float64)
    if glide == "Linear":
        return f_start + (f_end - f_start) * t
    elif glide == "Exponential":
        safe_start = max(f_start, 1e-3)
        safe_end = max(f_end, 1e-3)
        return safe_start * (safe_end / safe_start) ** t
    elif glide == "Logarithmic":
        safe_start = max(f_start, 1e-3)
        safe_end = max(f_end, 1e-3)
        log_s = np.log(safe_start)
        log_e = np.log(safe_end)
        return np.exp(log_s + (log_e - log_s) * t)
    return np.full(n_samples, f_start, dtype=np.float64)


def _freq_to_phase(freq_env: np.ndarray, sr: int) -> np.ndarray:
    """Integrate instantaneous frequency into cumulative phase."""
    return np.cumsum(2.0 * np.pi * freq_env / sr)


def _generate_signal(sr: int, duration: float, n_channels: int,
                     shape: str, f_start: float, f_end: float,
                     glide: str, n_voices: int, detune_cents: float,
                     harmonics: list[tuple[int, float, float]],
                     pulse_width: float,
                     fade_ms: float) -> np.ndarray:
    """Build the full multi-voice signal.  Returns float64 array shape (N,) or (N,2)."""
    n_samples = int(sr * duration)
    if n_samples < 1:
        n_samples = 1
    out = np.zeros((n_samples, n_channels), dtype=np.float64)

    for vi in range(n_voices):
        # Detune: spread voices symmetrically in cents
        if n_voices == 1:
            cent_offset = 0.0
        else:
            cent_offset = detune_cents * (vi / (n_voices - 1) - 0.5)
        ratio = 2.0 ** (cent_offset / 1200.0)
        freq_env = _build_freq_envelope(n_samples, sr,
                                        f_start * ratio, f_end * ratio, glide)
        phase = _freq_to_phase(freq_env, sr)
        sig = _apply_harmonics(phase, shape, harmonics, pulse_width)
        for ch in range(n_channels):
            out[:, ch] += sig

    # Normalize
    peak = np.abs(out).max()
    if peak > 0:
        out /= peak

    # Fade in/out
    if fade_ms > 0:
        fade_samples = min(int(sr * fade_ms / 1000.0), n_samples // 2)
        if fade_samples > 0:
            fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float64)
            fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float64)
            for ch in range(n_channels):
                out[:fade_samples, ch] *= fade_in
                out[-fade_samples:, ch] *= fade_out

    if n_channels == 1:
        return out[:, 0]
    return out


class SignalGeneratorPanel(Panel):
    """Panel for generating WAV files: tone synthesis, silence, or recording.

    Sections:
      1. Generation mode selector (Tone / Silence / Record)
      2. Wave shape + harmonic configuration
      3. Frequency / glide path
      4. Voices & detune
      5. Output format (SR, bit depth, channels, duration)
      6. Generate / Record button + status
    """

    PANEL_W = 280
    ROW_H = 22
    PAD = 6

    def __init__(self, *, side: str = "left",
                 output_dir: str = "") -> None:
        super().__init__(title="Signal Generator", side=side)
        self.output_dir: str = output_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "input")
        os.makedirs(self.output_dir, exist_ok=True)

        # Generation mode
        self.gen_mode_idx: int = 0  # index into _SIG_GEN_MODES

        # Wave shape
        self.wave_shape_idx: int = 0  # index into _SIG_WAVE_SHAPES
        self.pulse_width: float = 0.5  # only for Pulse shape

        # Frequency path
        self.freq_start: float = 440.0
        self.freq_end: float = 440.0
        self.glide_mode_idx: int = 0  # index into _SIG_GLIDE_MODES

        # Voices
        self.n_voices: int = 1
        self.detune_cents: float = 10.0

        # Harmonics: list of (harmonic_number, amplitude, phase_offset)
        self.harmonics: list[list[float]] = [[1, 1.0, 0.0]]
        self._harmonics_expanded: bool = False

        # Output format
        self.sample_rate: int = 44100
        self.bit_depth_idx: int = 0  # index into _SIG_BIT_DEPTHS
        self.channels_idx: int = 0   # index into _SIG_CHANNELS
        self.duration: float = 5.0
        self.fade_ms: float = 10.0

        # Recording
        self._recording: bool = False
        self._record_thread: threading.Thread | None = None
        self._recorded_data: np.ndarray | None = None

        # UI state
        self._ctrl_rects: dict[str, pygame.Rect] = {}
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        self._status_msg: str = ""

        # File list for generated outputs
        self.output_list = ScrollableItemList("Generated Files", max_visible=4)
        self._output_paths: list[str] = []
        self._refresh_outputs()

        # Callback: called with (wav_path,) after generation
        self.on_generated: Any = None

    # -- file scanning ------------------------------------------------------

    def _refresh_outputs(self) -> None:
        self._output_paths = []
        items: list[ListItem] = []
        if os.path.isdir(self.output_dir):
            for fn in sorted(os.listdir(self.output_dir)):
                if fn.lower().endswith(".wav") and fn.startswith("gen_"):
                    path = os.path.join(self.output_dir, fn)
                    self._output_paths.append(path)
                    items.append(ListItem(
                        key=fn, display_name=fn,
                        tag="wav", tag_color_key="none"))
        self.output_list.items = items

    # -- generation ---------------------------------------------------------

    def _output_filename(self, prefix: str) -> str:
        """Build a unique output filename with timestamp."""
        ts = time.strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.output_dir, f"gen_{prefix}_{ts}.wav")

    def _write_wav(self, path: str, data: np.ndarray, sr: int) -> None:
        """Write audio data to WAV, respecting the selected bit depth."""
        depth = _SIG_BIT_DEPTHS[self.bit_depth_idx]
        if depth == "32-bit float":
            wavfile.write(path, sr, data.astype(np.float32))
        elif depth == "24-bit int":
            peak = np.abs(data).max()
            if peak > 0:
                scaled = (data / peak * 8388607).astype(np.int32)
            else:
                scaled = np.zeros(data.shape, dtype=np.int32)
            # scipy wavfile doesn't natively support 24-bit, write as 32-bit int
            wavfile.write(path, sr, scaled)
        else:  # 16-bit int
            peak = np.abs(data).max()
            if peak > 0:
                scaled = (data / peak * 32767).astype(np.int16)
            else:
                scaled = np.zeros(data.shape, dtype=np.int16)
            wavfile.write(path, sr, scaled)

    def generate_tone(self) -> None:
        """Generate a tone signal and write to WAV."""
        sr = self.sample_rate
        dur = self.duration
        n_ch = 2 if self.channels_idx == 1 else 1
        shape = _SIG_WAVE_SHAPES[self.wave_shape_idx]
        harmonics = [(int(h[0]), float(h[1]), float(h[2]))
                     for h in self.harmonics]
        if not harmonics:
            harmonics = [(1, 1.0, 0.0)]
        data = _generate_signal(
            sr, dur, n_ch, shape,
            self.freq_start, self.freq_end,
            _SIG_GLIDE_MODES[self.glide_mode_idx],
            self.n_voices, self.detune_cents,
            harmonics, self.pulse_width, self.fade_ms)
        path = self._output_filename("tone")
        self._write_wav(path, data, sr)
        self._status_msg = f"Wrote {os.path.basename(path)}"
        self._refresh_outputs()
        if self.on_generated:
            self.on_generated(path)

    def generate_silence(self) -> None:
        """Generate a silent WAV file."""
        sr = self.sample_rate
        dur = self.duration
        n_ch = 2 if self.channels_idx == 1 else 1
        n_samples = int(sr * dur)
        if n_ch == 1:
            data = np.zeros(n_samples, dtype=np.float64)
        else:
            data = np.zeros((n_samples, 2), dtype=np.float64)
        path = self._output_filename("silence")
        self._write_wav(path, data, sr)
        self._status_msg = f"Wrote {os.path.basename(path)}"
        self._refresh_outputs()
        if self.on_generated:
            self.on_generated(path)

    def start_recording(self) -> None:
        """Start recording from loopback/system input in a background thread."""
        if self._recording:
            return
        self._recording = True
        self._status_msg = "Recording..."
        self._record_thread = threading.Thread(
            target=self._record_worker, daemon=True)
        self._record_thread.start()

    def stop_recording(self) -> None:
        """Signal the recording thread to stop."""
        self._recording = False

    def _record_worker(self) -> None:
        """Background thread: capture audio from default input device."""
        try:
            import sounddevice as sd
        except ImportError:
            self._status_msg = "pip install sounddevice"
            self._recording = False
            return
        sr = self.sample_rate
        n_ch = 2 if self.channels_idx == 1 else 1
        dur = self.duration
        n_frames = int(sr * dur)
        try:
            data = sd.rec(n_frames, samplerate=sr, channels=n_ch,
                          dtype='float64', blocking=True)
        except Exception as exc:
            self._status_msg = f"Record error: {exc}"
            self._recording = False
            return
        self._recorded_data = data
        path = self._output_filename("rec")
        self._write_wav(path, data, sr)
        self._status_msg = f"Recorded {os.path.basename(path)}"
        self._recording = False
        self._refresh_outputs()
        if self.on_generated:
            self.on_generated(path)

    def _do_generate(self) -> None:
        """Dispatch generation based on current mode."""
        mode = _SIG_GEN_MODES[self.gen_mode_idx]
        if mode == "Tone Generator":
            self.generate_tone()
        elif mode == "Silent Track":
            self.generate_silence()
        elif mode == "Record Input":
            if self._recording:
                self.stop_recording()
            else:
                self.start_recording()

    # -- render -------------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._ctrl_rects.clear()
        mode = _SIG_GEN_MODES[self.gen_mode_idx]

        est_h = 1200
        surf = pygame.Surface((w, est_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))
        y = self.PAD

        # -- Section: Generation mode dropdown --
        lbl = font.render("Mode:", True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 2))
        btn_x = 60
        btn_w = w - btn_x - self.PAD
        r = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
        bg = (80, 80, 120) if self._active_dropdown == "gen_mode" else (60, 60, 80)
        pygame.draw.rect(surf, bg, r)
        pygame.draw.rect(surf, (100, 100, 120), r, 1)
        ct = font.render(mode, True, (220, 220, 220))
        surf.blit(ct, (btn_x + 4, y + 3))
        self._ctrl_rects["gen_mode"] = r
        y += self.ROW_H + 4

        # Separator
        pygame.draw.line(surf, (60, 60, 70), (self.PAD, y), (w - self.PAD, y))
        y += 4

        # -- Section: Output format (always shown) --
        sec = font.render("─ Output Format ─", True, (180, 180, 180))
        surf.blit(sec, (self.PAD, y + 2))
        y += self.ROW_H

        # Sample rate
        y = self._render_slider(surf, font, y, "sr", "SR:",
                                self.sample_rate, 8000, 192000)
        # Duration
        y = self._render_slider(surf, font, y, "dur", "Duration:",
                                self.duration, 0.1, 600.0)
        # Bit depth dropdown
        y = self._render_dropdown_row(surf, font, y, "bit_depth", "Depth:",
                                      _SIG_BIT_DEPTHS[self.bit_depth_idx])
        # Channels dropdown
        y = self._render_dropdown_row(surf, font, y, "channels", "Channels:",
                                      _SIG_CHANNELS[self.channels_idx])
        # Fade
        y = self._render_slider(surf, font, y, "fade", "Fade ms:",
                                self.fade_ms, 0.0, 500.0)

        y += 2
        pygame.draw.line(surf, (60, 60, 70), (self.PAD, y), (w - self.PAD, y))
        y += 4

        if mode == "Tone Generator":
            y = self._render_tone_section(surf, font, y)
        elif mode == "Silent Track":
            note = font.render("Generates empty audio", True, (130, 130, 140))
            surf.blit(note, (self.PAD, y + 2))
            y += self.ROW_H + 2
        elif mode == "Record Input":
            note = font.render("Records from system input", True, (130, 130, 140))
            surf.blit(note, (self.PAD, y + 2))
            y += self.ROW_H
            note2 = font.render("(requires sounddevice)", True, (100, 100, 110))
            surf.blit(note2, (self.PAD, y + 2))
            y += self.ROW_H + 2

        # -- Generate / Record button --
        y += 2
        if mode == "Record Input" and self._recording:
            btn_label = "■ Stop Recording"
            btn_color = (140, 50, 50)
        elif mode == "Record Input":
            btn_label = "● Record"
            btn_color = (140, 50, 50)
        else:
            btn_label = "Generate"
            btn_color = (50, 80, 140)
        go_btn = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H + 4)
        pygame.draw.rect(surf, btn_color, go_btn)
        pygame.draw.rect(surf, (100, 100, 120), go_btn, 1)
        gl = font.render(btn_label, True, (220, 220, 220))
        surf.blit(gl, (w // 2 - gl.get_width() // 2, y + 5))
        self._ctrl_rects["go"] = go_btn
        y += self.ROW_H + 8

        # Status
        if self._status_msg:
            st = font.render(self._status_msg[:42], True, (160, 160, 160))
            surf.blit(st, (self.PAD, y + 2))
            y += self.ROW_H + 2

        # Separator
        pygame.draw.line(surf, (60, 60, 70), (self.PAD, y), (w - self.PAD, y))
        y += 4

        # -- Generated file list --
        self._refresh_outputs()
        data_h = self.output_list.render(surf, font, 0, y, w)
        y += data_h

        # Crop to actual content
        final_h = max(y + self.PAD, 100)
        final_surf = pygame.Surface((w, final_h), pygame.SRCALPHA)
        final_surf.blit(surf, (0, 0))
        return self._apply_panel_scroll(final_surf)

    def _render_tone_section(self, surf: pygame.Surface,
                             font: pygame.font.Font,
                             y: int) -> int:
        """Render tone-specific controls (shape, freq, voices, harmonics)."""
        w = self.PANEL_W

        # -- Wave shape --
        sec = font.render("─ Wave Shape ─", True, (180, 180, 180))
        surf.blit(sec, (self.PAD, y + 2))
        y += self.ROW_H
        y = self._render_dropdown_row(surf, font, y, "wave_shape", "Shape:",
                                      _SIG_WAVE_SHAPES[self.wave_shape_idx])
        if _SIG_WAVE_SHAPES[self.wave_shape_idx] == "Pulse":
            y = self._render_slider(surf, font, y, "pulse_w", "Pulse W:",
                                    self.pulse_width, 0.01, 0.99)

        # -- Frequency path --
        y += 2
        sec2 = font.render("─ Frequency Path ─", True, (180, 180, 180))
        surf.blit(sec2, (self.PAD, y + 2))
        y += self.ROW_H
        _nyq_tg = self.sample_rate / 2.0
        y = self._render_slider(surf, font, y, "f_start", "F Start:",
                                self.freq_start, 1.0, _nyq_tg)
        y = self._render_slider(surf, font, y, "f_end", "F End:",
                                self.freq_end, 1.0, _nyq_tg)
        y = self._render_dropdown_row(surf, font, y, "glide", "Glide:",
                                      _SIG_GLIDE_MODES[self.glide_mode_idx])

        # -- Voices --
        y += 2
        sec3 = font.render("─ Voices ─", True, (180, 180, 180))
        surf.blit(sec3, (self.PAD, y + 2))
        y += self.ROW_H
        y = self._render_slider(surf, font, y, "voices", "Count:",
                                self.n_voices, 1, 64)
        if self.n_voices > 1:
            y = self._render_slider(surf, font, y, "detune", "Detune ¢:",
                                    self.detune_cents, 0.0, 200.0)

        # -- Harmonics --
        y += 2
        toggle_label = "▼ Harmonics" if self._harmonics_expanded else "▶ Harmonics"
        ht = font.render(toggle_label, True, (180, 180, 180))
        toggle_r = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
        surf.blit(ht, (self.PAD, y + 2))
        self._ctrl_rects["harmonics_toggle"] = toggle_r
        y += self.ROW_H

        if self._harmonics_expanded:
            for hi, h in enumerate(self.harmonics):
                h_num, h_amp, h_ph = h
                row_lbl = font.render(
                    f"H{int(h_num)}: A={h_amp:.2f} φ={h_ph:.1f}",
                    True, (170, 170, 180))
                surf.blit(row_lbl, (self.PAD + 4, y + 3))

                # Remove button
                rm_r = pygame.Rect(w - self.PAD - 22, y, 20, self.ROW_H)
                pygame.draw.rect(surf, (100, 40, 40), rm_r)
                pygame.draw.rect(surf, (80, 80, 100), rm_r, 1)
                rm_l = font.render("×", True, (200, 180, 180))
                surf.blit(rm_l, (rm_r.x + 5, y + 3))
                self._ctrl_rects[f"h_rm_{hi}"] = rm_r

                # Amplitude +/- buttons
                a_minus = pygame.Rect(w - self.PAD - 90, y, 20, self.ROW_H)
                a_plus = pygame.Rect(w - self.PAD - 68, y, 20, self.ROW_H)
                pygame.draw.rect(surf, (50, 50, 65), a_minus)
                pygame.draw.rect(surf, (80, 80, 100), a_minus, 1)
                surf.blit(font.render("−", True, (180, 180, 180)),
                          (a_minus.x + 5, y + 3))
                pygame.draw.rect(surf, (50, 50, 65), a_plus)
                pygame.draw.rect(surf, (80, 80, 100), a_plus, 1)
                surf.blit(font.render("+", True, (180, 180, 180)),
                          (a_plus.x + 5, y + 3))
                self._ctrl_rects[f"h_amp_minus_{hi}"] = a_minus
                self._ctrl_rects[f"h_amp_plus_{hi}"] = a_plus

                y += self.ROW_H + 2

            # Phase +/- for each harmonic
            for hi, h in enumerate(self.harmonics):
                ph_lbl = font.render(
                    f"  H{int(h[0])} phase:", True, (130, 130, 140))
                surf.blit(ph_lbl, (self.PAD, y + 3))
                ph_minus = pygame.Rect(w - self.PAD - 44, y, 20, self.ROW_H)
                ph_plus = pygame.Rect(w - self.PAD - 22, y, 20, self.ROW_H)
                pygame.draw.rect(surf, (50, 50, 65), ph_minus)
                pygame.draw.rect(surf, (80, 80, 100), ph_minus, 1)
                surf.blit(font.render("−", True, (180, 180, 180)),
                          (ph_minus.x + 5, y + 3))
                pygame.draw.rect(surf, (50, 50, 65), ph_plus)
                pygame.draw.rect(surf, (80, 80, 100), ph_plus, 1)
                surf.blit(font.render("+", True, (180, 180, 180)),
                          (ph_plus.x + 5, y + 3))
                self._ctrl_rects[f"h_ph_minus_{hi}"] = ph_minus
                self._ctrl_rects[f"h_ph_plus_{hi}"] = ph_plus
                y += self.ROW_H + 2

            # Add harmonic button
            add_r = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
            pygame.draw.rect(surf, (50, 65, 50), add_r)
            pygame.draw.rect(surf, (80, 100, 80), add_r, 1)
            al = font.render("+ Add Harmonic", True, (180, 220, 180))
            surf.blit(al, (self.PAD + 4, y + 3))
            self._ctrl_rects["h_add"] = add_r
            y += self.ROW_H + 2

        return y

    # -- render helpers -----------------------------------------------------

    def _render_slider(self, surf: pygame.Surface, font: pygame.font.Font,
                       y: int, key: str, label: str,
                       val: float | int, vmin: float, vmax: float) -> int:
        """Render a label + value + ± buttons row. Returns new y."""
        w = self.PANEL_W
        sl = font.render(label, True, (140, 140, 150))
        surf.blit(sl, (self.PAD, y + 3))
        if isinstance(val, int):
            vt = font.render(str(val), True, (200, 200, 200))
        else:
            if val >= 100:
                vt = font.render(f"{val:.1f}", True, (200, 200, 200))
            else:
                vt = font.render(f"{val:.2f}", True, (200, 200, 200))
        surf.blit(vt, (self.PAD + 78, y + 3))
        minus_r = pygame.Rect(w - self.PAD - 44, y, 20, self.ROW_H)
        plus_r = pygame.Rect(w - self.PAD - 22, y, 20, self.ROW_H)
        pygame.draw.rect(surf, (50, 50, 65), minus_r)
        pygame.draw.rect(surf, (80, 80, 100), minus_r, 1)
        surf.blit(font.render("−", True, (180, 180, 180)),
                  (minus_r.x + 5, y + 3))
        pygame.draw.rect(surf, (50, 50, 65), plus_r)
        pygame.draw.rect(surf, (80, 80, 100), plus_r, 1)
        surf.blit(font.render("+", True, (180, 180, 180)),
                  (plus_r.x + 5, y + 3))
        self._ctrl_rects[f"{key}_minus"] = minus_r
        self._ctrl_rects[f"{key}_plus"] = plus_r
        return y + self.ROW_H + 2

    def _render_dropdown_row(self, surf: pygame.Surface,
                             font: pygame.font.Font,
                             y: int, key: str, label: str,
                             current: str) -> int:
        """Render a label + dropdown button row. Returns new y."""
        w = self.PANEL_W
        lbl = font.render(label, True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 2))
        btn_x = 78
        btn_w = w - btn_x - self.PAD
        r = pygame.Rect(btn_x, y, btn_w, self.ROW_H)
        bg = (80, 80, 120) if self._active_dropdown == key else (60, 60, 80)
        pygame.draw.rect(surf, bg, r)
        pygame.draw.rect(surf, (100, 100, 120), r, 1)
        ct = font.render(current, True, (220, 220, 220))
        surf.blit(ct, (btn_x + 4, y + 3))
        self._ctrl_rects[key] = r
        return y + self.ROW_H + 2

    # -- events -------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        pr = self.panel_rect

        if event.type == MOUSEBUTTONDOWN and event.button == 1:
            mx, my = event.pos

            # Dropdown overlay first
            if self._active_dropdown:
                for i, ir in enumerate(self._dropdown_item_rects):
                    if ir.collidepoint(mx, my):
                        self._select_dropdown(i)
                        return True
                self._active_dropdown = None
                return True

            if not pr.collidepoint(mx, my):
                return False
            lx, ly = mx - pr.x, my - pr.y
            ly += self._panel_scroll_y

            # Output list click
            hit = self.output_list.handle_click(lx, ly)
            if hit is not None:
                return True

            # Controls
            for key, rect in self._ctrl_rects.items():
                if rect.collidepoint(lx, ly):
                    self._handle_ctrl_click(key)
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
            return self._handle_panel_wheel(event)

        return False

    def _handle_ctrl_click(self, key: str) -> None:
        """Route a control click to the right handler."""
        if key == "gen_mode":
            self._open_dropdown("gen_mode", _SIG_GEN_MODES,
                                self._ctrl_rects[key])
        elif key == "wave_shape":
            self._open_dropdown("wave_shape", _SIG_WAVE_SHAPES,
                                self._ctrl_rects[key])
        elif key == "glide":
            self._open_dropdown("glide", _SIG_GLIDE_MODES,
                                self._ctrl_rects[key])
        elif key == "bit_depth":
            self._open_dropdown("bit_depth", _SIG_BIT_DEPTHS,
                                self._ctrl_rects[key])
        elif key == "channels":
            self._open_dropdown("channels", _SIG_CHANNELS,
                                self._ctrl_rects[key])
        elif key == "go":
            self._do_generate()
        elif key == "harmonics_toggle":
            self._harmonics_expanded = not self._harmonics_expanded
        elif key == "h_add":
            next_h = max(int(h[0]) for h in self.harmonics) + 1 \
                if self.harmonics else 1
            self.harmonics.append([float(next_h), 0.5, 0.0])
        elif key.startswith("h_rm_"):
            idx = int(key[5:])
            if 0 <= idx < len(self.harmonics) and len(self.harmonics) > 1:
                self.harmonics.pop(idx)
        elif key.endswith("_minus") or key.endswith("_plus"):
            self._nudge(key)

    def _open_dropdown(self, key: str, options: list[str],
                       anchor: pygame.Rect) -> None:
        self._active_dropdown = key
        self._dropdown_opts = options
        self._dropdown_rect = anchor

    def _select_dropdown(self, idx: int) -> None:
        key = self._active_dropdown
        self._active_dropdown = None
        if key == "gen_mode" and 0 <= idx < len(_SIG_GEN_MODES):
            self.gen_mode_idx = idx
        elif key == "wave_shape" and 0 <= idx < len(_SIG_WAVE_SHAPES):
            self.wave_shape_idx = idx
        elif key == "glide" and 0 <= idx < len(_SIG_GLIDE_MODES):
            self.glide_mode_idx = idx
        elif key == "bit_depth" and 0 <= idx < len(_SIG_BIT_DEPTHS):
            self.bit_depth_idx = idx
        elif key == "channels" and 0 <= idx < len(_SIG_CHANNELS):
            self.channels_idx = idx

    def _nudge(self, btn_key: str) -> None:
        """Handle +/- button clicks."""
        direction = 1 if btn_key.endswith("_plus") else -1
        base = btn_key.rsplit("_", 1)[0]
        # Strip trailing _minus/_plus that was already consumed
        if base.endswith("_minus") or base.endswith("_plus"):
            base = base.rsplit("_", 1)[0]

        if base == "sr":
            steps = [8000, 11025, 16000, 22050, 32000, 44100,
                     48000, 88200, 96000, 176400, 192000]
            try:
                ci = steps.index(self.sample_rate)
            except ValueError:
                ci = 5  # default to 44100
            ci = max(0, min(len(steps) - 1, ci + direction))
            self.sample_rate = steps[ci]
            _nyq_tg = self.sample_rate / 2.0
            self.freq_start = min(self.freq_start, _nyq_tg)
            self.freq_end = min(self.freq_end, _nyq_tg)
        elif base == "dur":
            step = 1.0 if self.duration >= 10 else 0.5
            self.duration = max(0.1, min(600.0, self.duration + step * direction))
        elif base == "fade":
            self.fade_ms = max(0.0, min(500.0, self.fade_ms + 5.0 * direction))
        elif base == "f_start":
            _nyq_tg = self.sample_rate / 2.0
            self.freq_start = max(1.0, min(_nyq_tg,
                                           self.freq_start * (2 ** (direction / 12.0))))
        elif base == "f_end":
            _nyq_tg = self.sample_rate / 2.0
            self.freq_end = max(1.0, min(_nyq_tg,
                                         self.freq_end * (2 ** (direction / 12.0))))
        elif base == "voices":
            self.n_voices = max(1, min(64, self.n_voices + direction))
        elif base == "detune":
            self.detune_cents = max(0.0, min(200.0,
                                             self.detune_cents + 2.0 * direction))
        elif base == "pulse_w":
            self.pulse_width = max(0.01, min(0.99,
                                             self.pulse_width + 0.05 * direction))
        elif base.startswith("h_amp"):
            hi = int(base.split("_")[-1])
            if 0 <= hi < len(self.harmonics):
                self.harmonics[hi][1] = max(0.0, min(2.0,
                    self.harmonics[hi][1] + 0.05 * direction))
        elif base.startswith("h_ph"):
            hi = int(base.split("_")[-1])
            if 0 <= hi < len(self.harmonics):
                self.harmonics[hi][2] = (self.harmonics[hi][2]
                                         + 0.1 * direction) % (2.0 * np.pi)

    def render_dropdown_overlay(self) -> pygame.Surface | None:
        if not self._active_dropdown:
            return None
        opts = self._dropdown_opts
        if not opts:
            return None
        self._ensure_font()
        font = self.font
        pr = self.panel_rect
        item_h = self.ROW_H
        ow = self._dropdown_rect.w
        oh = len(opts) * item_h
        overlay = pygame.Surface((ow, oh), pygame.SRCALPHA)
        overlay.fill((40, 40, 55, 240))
        self._dropdown_item_rects = []
        for i, label in enumerate(opts):
            iy = i * item_h
            rect = pygame.Rect(
                pr.x + self._dropdown_rect.x,
                pr.y + self._dropdown_rect.y + self._dropdown_rect.h + iy
                - self._panel_scroll_y,
                ow, item_h)
            self._dropdown_item_rects.append(rect)
            txt = font.render(label, True, (220, 220, 220))
            overlay.blit(txt, (4, iy + 3))
            if i < len(opts) - 1:
                pygame.draw.line(overlay, (60, 60, 70),
                                 (0, iy + item_h - 1),
                                 (ow, iy + item_h - 1))
        pygame.draw.rect(overlay, (100, 100, 130),
                         pygame.Rect(0, 0, ow, oh), 1)
        return overlay


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
        self._update_wav_duration()

        # Channel mode
        self.channel_mode_idx: int = 0

        # Source include flags (non-exclusive checkboxes)
        self.include_cqt: bool = True
        self.include_fb: bool = False
        self.fb_hybrid: bool = False  # hybrid = FB low / CQT high
        self.fb_q_norm: bool = False  # divide by sqrt(bw) for spectral density
        self.include_wavelet: bool = False

        # Wavelet settings
        self.wavelet_family_idx: int = 0   # index into _WAVELET_FAMILIES
        self.wavelet_order_idx: int = 0    # index into family-specific orders
        self.wavelet_depth_pct: float = 100.0  # 0-100% of max available depth
        self.wavelet_depth_abs: int = 5        # absolute level 0-50
        self.wavelet_depth_mode: int = 0       # 0=percentage, 1=absolute
        self.wavelet_ext_idx: int = 0      # index into _WAVELET_EXTENSIONS
        self.wavelet_mode_idx: int = 0     # 0=DWT, 1=CWT
        self.cwt_wavelet_idx: int = 0      # index into _CWT_WAVELET_TYPES
        self.cwt_scales_per_octave: int = 12
        self.cwt_sigma: float = 6.0
        self.cwt_epsilon: float = 0.01
        self.cwt_fmin: float = 0.1      # Hz, lower bound for CWT analysis
        self.cwt_fmax: float = 0.0      # Hz, upper bound; 0 = sr/4.0

        # CQT settings
        self.cqt_algorithm_idx: int = 0  # index into _CQT_ALGORITHMS
        self.hop_length: int = 512
        self.bins_per_octave: int = 1200
        self.cqt_filter_scale: float = 1.0  # librosa CQT filter quality factor
        self.cqt_fmin: float = 16.35  # C0
        self.cqt_fmax: float = 19912.13  # D#9 (snapped)
        self._cqt_expanded: bool = True
        self._loss_map = PlotWidget()
        self._loss_map.title = ""
        self._loss_map.grid_lines = 0

        # Filter bank settings
        self._fb_expanded: bool = True
        self.fb_config_mode_idx: int = 0  # index into _FB_CONFIG_MODES
        self.fb_crossovers: list[float] = [
            _snap_to_note(200.0),   # ~G3
            _snap_to_note(800.0),   # ~G#5
            _snap_to_note(4000.0),  # ~B7
        ]
        self.fb_filter_type_idx: int = 0  # index into _FILTER_TYPES
        self.fb_label_mode_idx: int = 1  # 0=Range, 1=Mid+BW
        self.fb_hop: int = 512  # hop length for FB envelope computation
        # Auto (log) settings — mirrors CQT-style controls
        self.fb_bpo: int = 12        # bands per octave
        self.fb_fmin: float = 16.35  # C0
        self.fb_fmax: float = 19912.13  # D#9
        self._fb_decomp: FilterBankDecomposition | None = None

        # Resampling preferences
        self._rs_expanded: bool = True
        self.resample_engine_idx: int = 0  # index into _RESAMPLE_ENGINES
        self.resample_interp_idx: int = 0  # index into _RESAMPLE_INTERPS
        self.resample_taps_idx: int = 3    # index into _RESAMPLE_TAPS (64)
        self.resample_precision_idx: int = 2  # index into _RESAMPLE_PRECISIONS

        # Time region (partial analysis)
        self._region_expanded: bool = True
        self.region_start: float = 0.0    # seconds
        self.region_end: float = 0.0      # seconds (0 = full file)
        self._wav_duration: float = 0.0   # total duration of selected WAV
        self._wav_sr: int = 0                 # sample rate of selected WAV
        self._wav_n_samples: int = 0          # sample count of selected WAV

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
        self._analysis_progress: float = 0.0   # 0.0 – 1.0
        self._analysis_phase: str = ""         # short label for bar
        self._fb_editing_idx: int = -1  # which crossover is being edited
        # Band list — scrollable fixed-height list for band preview
        self._band_list = ScrollableItemList("Filter Bands", max_visible=8)
        self._band_list_bands: list[BandDef] = []  # current preview bands

        # Solver state
        self._solving: bool = False
        self._solver_thread: threading.Thread | None = None
        self._solver_progress: str = ""
        self._solver_result: Any = None  # SolverResult or None
        self._solver_error: str | None = None
        self._pre_solve_losses: tuple[float, float, float] | None = None
        # Schedule-mode params: set of (engine, param_name)
        # A param in _schedule_params is solved per-octave instead of as a
        # single global value.  It must NOT also be in _param_locks.
        self._schedule_params: set[tuple[str, str]] = set()
        # Solver-discovered schedules (ScheduleVec per param), populated after solve
        self._solver_schedules: dict[str, dict[str, Any]] = {}
        # Parameter locks: {engine: {param_name: True}}
        self._param_locks: dict[str, set[str]] = {
            "cqt": set(), "fb": set(), "cwt": set()}

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
                if os.path.splitext(fn)[1].lower() in _AUDIO_EXTS:
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
        self._update_wav_duration()
        self._refresh_folders()

    def _update_wav_duration(self) -> None:
        """Read the audio header for the selected file and update region limits."""
        idx = self.wav_list.selected_idx
        if idx < 0 or idx >= len(self._wav_paths):
            self._wav_duration = 0.0
            return
        try:
            if _HAS_SOUNDFILE:
                info = _sf.info(self._wav_paths[idx])
                sr_file = info.samplerate
                n_samples = info.frames
            else:
                sr_file, raw = _load_audio(self._wav_paths[idx])
                n_samples = raw.shape[0] if raw.ndim >= 1 else len(raw)
            self._wav_duration = n_samples / sr_file
            self._wav_sr = sr_file
            self._wav_n_samples = n_samples
        except Exception:
            self._wav_duration = 0.0
            self._wav_sr = 0
            self._wav_n_samples = 0
        # Set region to full file on new selection
        if self._wav_duration > 0:
            self.region_start = 0.0
            self.region_end = round(self._wav_duration, 2)
        self._auto_freq_range()

    def _effective_duration(self) -> float:
        """Effective analysis duration in seconds (respects region selection)."""
        if self._wav_duration <= 0:
            return 0.0
        if self.region_end > 0:
            return max(0.01, self.region_end - self.region_start)
        return self._wav_duration

    def _freq_limits(self) -> tuple[float, float]:
        """Return (abs_min, abs_max) frequency limits for the current file."""
        sr = self._wav_sr if self._wav_sr > 0 else 44100
        nyq = sr / 2.0
        dur = self._effective_duration()
        lo = 1.0 / dur if dur > 0 else 1.0
        return (lo, nyq)

    def _auto_freq_range(self) -> None:
        """Auto-set fb and cqt fmin/fmax to semitone-snapped limits."""
        lo_raw, nyq = self._freq_limits()
        # fmax: snap down to semitone at or below Nyquist
        fmax = _snap_to_note_down(nyq)
        if fmax <= 0:
            fmax = _snap_to_note(nyq)
        # fmin: snap up to semitone at or above floor
        fmin = _snap_to_note_up(lo_raw)
        if fmin >= fmax:
            fmin = _snap_to_note_down(fmax / 2.0)
        self.fb_fmax = round(fmax, 6)
        self.fb_fmin = round(fmin, 6)
        self.cqt_fmax = round(fmax, 6)
        self.cqt_fmin = round(fmin, 6)

    def _clamp_freq_range(self) -> None:
        """Clamp all freq sliders to valid range after region change."""
        lo, hi = self._freq_limits()
        if self.fb_fmin < lo:
            self.fb_fmin = round(_snap_to_note_up(lo), 6)
        if self.fb_fmax > hi:
            self.fb_fmax = round(_snap_to_note_down(hi), 6)
        if self.cqt_fmin < lo:
            self.cqt_fmin = round(_snap_to_note_up(lo), 6)
        if self.cqt_fmax > hi:
            self.cqt_fmax = round(_snap_to_note_down(hi), 6)

    def _clamp_hybrid_cqt_fmin(self) -> None:
        """In hybrid mode, raise cqt_fmin to the crossover frequency."""
        if not (self.include_fb and self.fb_hybrid):
            return
        sr = self._wav_sr if self._wav_sr > 0 else 44100
        xf = hybrid_crossover_freq(self.bins_per_octave, self.hop_length, sr)
        if self.cqt_fmin < xf:
            self.cqt_fmin = round(_snap_to_note_up(xf), 6)

    def _current_wavelet_name(self) -> str:
        """Resolve the pywt wavelet name from current UI settings."""
        fam = _WAVELET_FAMILIES[self.wavelet_family_idx]
        if fam == "Discrete Meyer":
            return "dmey"
        prefix = _WAVELET_FAMILY_PREFIX.get(fam, "db")
        orders = _WAVELET_ORDERS.get(fam, ["\u2014"])
        safe_idx = min(self.wavelet_order_idx, len(orders) - 1)
        return prefix + orders[safe_idx]

    def _wavelet_max_level(self) -> int:
        """Max DWT decomposition level for current signal & wavelet."""
        n = self._wav_n_samples
        if n <= 0:
            return 20  # fallback when no file selected
        # Account for region selection
        if self._wav_sr > 0:
            if self.region_start > 0 or self.region_end > 0:
                s0 = int(self.region_start * self._wav_sr) if self.region_start > 0 else 0
                s1 = int(self.region_end * self._wav_sr) if self.region_end > 0 else n
                n = max(1, min(n, s1) - max(0, s0))
        try:
            import pywt
            return max(1, pywt.dwt_max_level(n, self._current_wavelet_name()))
        except Exception:
            # Fallback: conservative estimate
            if n > 1:
                return max(1, int(math.floor(math.log2(n))))
            return 1

    def _wavelet_level_from_pct(self) -> int:
        """Compute the actual decomposition level from current depth settings."""
        max_lev = self._wavelet_max_level()
        if self.wavelet_depth_mode == 1:
            return max(1, min(self.wavelet_depth_abs, max_lev))
        return max(1, int(round(self.wavelet_depth_pct / 100.0 * max_lev)))

    # ---- Solver -----------------------------------------------------------

    def _solver_limit(self, engine: str, param: str,
                      default_hi: float) -> float:
        """Return the upper bound for a param, expanded by solver if available."""
        if self._solver_result is not None:
            bounds = self._solver_result.bounds_used.get(engine, {})
            if param in bounds:
                _, hi = bounds[param]
                return max(default_hi, hi)
            # Also check if solver pushed the value beyond the old limit
            val = self._solver_result.params.get(engine, {}).get(param)
            if val is not None and isinstance(val, (int, float)):
                return max(default_hi, float(val) * 1.2)
        return default_hi

    def _run_solver(self) -> None:
        """Launch the fidelity solver in a background thread."""
        if self._solving:
            return
        self._solving = True
        self._solver_error = None
        self._solver_progress = "Starting"
        self._solver_result = None
        # Capture current manual losses for before/after comparison
        try:
            _bars = [b for b in self._loss_map.bars if b.rgb.shape[0] > 0]
            _pr = max((float(b.rgb[:, 0].max()) for b in _bars), default=0.0)
            _pg = max((float(b.rgb[:, 1].max()) for b in _bars), default=0.0)
            _pb = max((float(b.rgb[:, 2].max()) for b in _bars), default=0.0)
            self._pre_solve_losses = (_pr, _pg, _pb)
        except Exception:
            self._pre_solve_losses = None

        sr = self._wav_sr if self._wav_sr > 0 else 44100
        engines: list[str] = []
        if self.include_cqt:
            engines.append("cqt")
        if self.include_fb:
            engines.append("fb")
        if self.include_wavelet:
            engines.append("cwt")
        if not engines:
            engines = ["cqt"]

        # Build locks from current values of locked params
        locks: dict[str, dict[str, Any]] = {}
        _cwt_fmax_eff = self.cwt_fmax if self.cwt_fmax > 0 else sr / 4.0
        lock_map = {
            "cqt": {
                "algorithm": _CQT_ALGORITHMS[self.cqt_algorithm_idx],
                "bins_per_octave": self.bins_per_octave,
                "hop_length": self.hop_length,
                "fmin": self.cqt_fmin,
                "fmax": self.cqt_fmax,
                "filter_scale": self.cqt_filter_scale,
            },
            "fb": {
                "bands_per_octave": self.fb_bpo,
                "hop_length": self.fb_hop,
                "fmin": self.fb_fmin,
                "fmax": self.fb_fmax,
                "filter_type": (_FILTER_TYPES[self.fb_filter_type_idx]
                                if self.fb_filter_type_idx < len(_FILTER_TYPES)
                                else "Linkwitz-Riley 4"),
            },
            "cwt": {
                "scales_per_octave": self.cwt_scales_per_octave,
                "hop_length": 1,
                "fmin": self.cwt_fmin,
                "fmax": _cwt_fmax_eff,
                "sigma": self.cwt_sigma,
                "wavelet": _CWT_WAVELET_TYPES[
                    min(self.cwt_wavelet_idx, len(_CWT_WAVELET_TYPES) - 1)],
            },
        }
        for eng in engines:
            for pname in self._param_locks.get(eng, set()):
                if pname in lock_map.get(eng, {}):
                    if eng not in locks:
                        locks[eng] = {}
                    locks[eng][pname] = lock_map[eng][pname]

        # Always lock the CQT algorithm — solver shouldn't flip it
        if "cqt" in engines:
            locks.setdefault("cqt", {})["algorithm"] = \
                _CQT_ALGORITHMS[self.cqt_algorithm_idx]

        def _cb(xk, convergence=0):
            self._solver_progress = f"conv={convergence:.4f}"

        _schedule_params_snapshot = set(self._schedule_params)

        def _worker() -> None:
            try:
                from fidelity_solver import solve
                result = solve(
                    sr=sr,
                    engines=engines,
                    locks=locks,
                    schedule_params=_schedule_params_snapshot,
                    max_iter=150,
                    seed=42,
                    callback=_cb,
                )
                self._solver_result = result
                self._apply_solver_result(result)
            except Exception as exc:
                self._solver_error = str(exc)
            finally:
                self._solving = False

        self._solver_thread = threading.Thread(target=_worker, daemon=True)
        self._solver_thread.start()

    def _apply_solver_result(self, result: Any) -> None:
        """Push solver-found values back into panel controls."""
        cqt_vals = result.params.get("cqt", {})
        fb_vals = result.params.get("fb", {})
        cwt_vals = result.params.get("cwt", {})

        # CQT
        if "algorithm" not in result.locked.get("cqt", {}):
            if "algorithm" in cqt_vals:
                algo = str(cqt_vals["algorithm"])
                if algo in _CQT_ALGORITHMS:
                    self.cqt_algorithm_idx = _CQT_ALGORITHMS.index(algo)
        if "bins_per_octave" not in result.locked.get("cqt", {}):
            if "bins_per_octave" in cqt_vals:
                self.bins_per_octave = _snap_bpo(cqt_vals["bins_per_octave"])
        if "hop_length" not in result.locked.get("cqt", {}):
            if "hop_length" in cqt_vals:
                raw = max(16, min(8192, int(cqt_vals["hop_length"])))
                self.hop_length = 2 ** round(math.log2(raw))
        if "fmin" not in result.locked.get("cqt", {}):
            if "fmin" in cqt_vals:
                self.cqt_fmin = round(_snap_to_note(
                    max(0.5, float(cqt_vals["fmin"]))), 6)
        if "fmax" not in result.locked.get("cqt", {}):
            if "fmax" in cqt_vals:
                self.cqt_fmax = round(_snap_to_note(
                    max(20.0, float(cqt_vals["fmax"]))), 6)
        if "filter_scale" not in result.locked.get("cqt", {}):
            if "filter_scale" in cqt_vals:
                self.cqt_filter_scale = max(0.1, min(8.0, float(cqt_vals["filter_scale"])))

        # FB
        if "bands_per_octave" not in result.locked.get("fb", {}):
            if "bands_per_octave" in fb_vals:
                self.fb_bpo = max(1, min(96, int(fb_vals["bands_per_octave"])))
        if "hop_length" not in result.locked.get("fb", {}):
            if "hop_length" in fb_vals:
                self.fb_hop = max(16, min(8192, int(fb_vals["hop_length"])))
        if "fmin" not in result.locked.get("fb", {}):
            if "fmin" in fb_vals:
                self.fb_fmin = round(_snap_to_note(
                    max(0.5, float(fb_vals["fmin"]))), 6)
        if "fmax" not in result.locked.get("fb", {}):
            if "fmax" in fb_vals:
                self.fb_fmax = round(_snap_to_note(
                    max(20.0, float(fb_vals["fmax"]))), 6)
        if "filter_type" not in result.locked.get("fb", {}):
            if "filter_type" in fb_vals:
                ft = str(fb_vals["filter_type"])
                if ft in _FILTER_TYPES:
                    self.fb_filter_type_idx = _FILTER_TYPES.index(ft)

        # CWT
        if "scales_per_octave" not in result.locked.get("cwt", {}):
            if "scales_per_octave" in cwt_vals:
                self.cwt_scales_per_octave = max(
                    1, min(240, int(cwt_vals["scales_per_octave"])))
        if "sigma" not in result.locked.get("cwt", {}):
            if "sigma" in cwt_vals:
                self.cwt_sigma = max(1.0, min(60.0, float(cwt_vals["sigma"])))
        if "epsilon" not in result.locked.get("cwt", {}):
            if "epsilon" in cwt_vals:
                self.cwt_epsilon = max(0.0, min(1.0, float(cwt_vals["epsilon"])))
        if "fmin" not in result.locked.get("cwt", {}):
            if "fmin" in cwt_vals:
                self.cwt_fmin = round(max(0.001, float(cwt_vals["fmin"])), 6)
        if "fmax" not in result.locked.get("cwt", {}):
            if "fmax" in cwt_vals:
                self.cwt_fmax = round(max(0.001, float(cwt_vals["fmax"])), 6)

        # Store solver schedules so analysis can use bpo_func / hop_func
        try:
            from fidelity_solver import ScheduleVec as _SV
            self._solver_schedules: dict[str, dict[str, Any]] = {}
            for _eng, _sched_dict in (result.schedules or {}).items():
                self._solver_schedules[_eng] = dict(_sched_dict)
        except Exception:
            pass

    # ---- Analysis ---------------------------------------------------------

    def _run_analysis(self) -> None:
        idx = self.wav_list.selected_idx
        if idx < 0 or idx >= len(self._wav_paths):
            return
        wav_path = self._wav_paths[idx]
        self._analyzing = True
        self._analysis_error = None
        self._analysis_progress = 0.0
        self._analysis_phase = "Starting"
        do_cqt = self.include_cqt
        do_fb = self.include_fb
        do_hybrid = self.include_fb and self.fb_hybrid
        do_wavelet = self.include_wavelet

        # Count enabled phases for proportional progress
        _phases = []
        if do_cqt or do_hybrid:
            _phases.append("cqt")
        if do_fb or do_hybrid:
            _phases.append("fb")
        if do_wavelet:
            _phases.append("wv")
        n_phases = max(len(_phases), 1)

        def _set_progress(frac: float, phase: str) -> None:
            self._analysis_progress = max(0.0, min(1.0, frac))
            self._analysis_phase = phase

        def _worker() -> None:
            try:
                base_name = os.path.splitext(os.path.basename(wav_path))[0]
                outdir = os.path.join(self.output_root,
                                      f"{base_name}_analysis")
                os.makedirs(outdir, exist_ok=True)

                # Time region for partial analysis
                r_start = self.region_start
                r_end = self.region_end
                has_region = (r_start > 0 or
                              (r_end > 0 and r_end < self._wav_duration))

                # Pre-read audio if needed for FB / Hybrid / Wavelet
                sr_wav: int | None = None
                raw_wav: np.ndarray | None = None
                if do_fb or do_hybrid or do_wavelet:
                    sr_wav, raw_wav = _load_audio(wav_path)

                # Compute hybrid crossover once (used by both halves)
                hybrid_xf: float | None = None
                if do_hybrid and sr_wav is not None:
                    hybrid_xf = hybrid_crossover_freq(
                        self.bins_per_octave, self.hop_length, sr_wav)

                # CQT analysis
                if do_cqt or do_hybrid:
                    phase_idx = _phases.index("cqt")
                    _set_progress(phase_idx / n_phases, "CQT analysis")
                    import subprocess, sys
                    cqt_fmin = self.cqt_fmin
                    if do_hybrid and hybrid_xf is not None:
                        # CQT only above crossover
                        cqt_fmin = max(self.cqt_fmin, hybrid_xf)
                    taps = int(_RESAMPLE_TAPS[self.resample_taps_idx])
                    cmd = [
                        sys.executable, "-m", "bass_analysis",
                        wav_path,
                        "--hop-length", str(self.hop_length),
                        "--bins-per-octave", str(self.bins_per_octave),
                        "--cqt-fmin", str(cqt_fmin),
                        "--cqt-fmax", str(self.cqt_fmax),
                        "--resample-taps", str(taps),
                        "--outdir", outdir,
                    ]
                    if has_region:
                        if r_start > 0:
                            cmd += ["--start-time", str(r_start)]
                        if r_end > 0:
                            cmd += ["--end-time", str(r_end)]
                    subprocess.check_call(cmd,
                                          cwd=os.path.dirname(
                                              os.path.abspath(__file__)))
                    _set_progress((phase_idx + 1) / n_phases, "CQT done")

                # Filter bank decomposition
                if do_fb or do_hybrid:
                    phase_idx = _phases.index("fb")
                    phase_base = phase_idx / n_phases
                    phase_span = 1.0 / n_phases

                    def _fb_progress(frac: float, label: str) -> None:
                        _set_progress(phase_base + frac * phase_span, label)

                    def _to_float32(arr: np.ndarray) -> np.ndarray:
                        if np.issubdtype(arr.dtype, np.integer):
                            return arr.astype(np.float32) / max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)
                        return arr.astype(np.float32)

                    if raw_wav.ndim == 2:
                        mono = _to_float32(raw_wav.mean(axis=1))
                    else:
                        mono = _to_float32(raw_wav)
                    is_wav_stereo = raw_wav.ndim == 2
                    if is_wav_stereo:
                        left_ch = _to_float32(raw_wav[:, 0])
                        right_ch = _to_float32(raw_wav[:, 1])
                    # Slice to time region if set
                    if has_region and sr_wav is not None:
                        s0 = int(r_start * sr_wav) if r_start > 0 else 0
                        s1 = int(r_end * sr_wav) if r_end > 0 else len(mono)
                        s0 = max(0, min(len(mono), s0))
                        s1 = max(s0, min(len(mono), s1))
                        mono = mono[s0:s1]
                        if is_wav_stereo:
                            left_ch = left_ch[s0:s1]
                            right_ch = right_ch[s0:s1]
                    ftype = _FILTER_TYPES[self.fb_filter_type_idx]
                    fb = FilterBankDecomposition(sr_wav, ftype)
                    fb_auto = _FB_CONFIG_MODES[
                        self.fb_config_mode_idx] != "Manual"
                    # In Hybrid mode, FB covers only below crossover
                    fb_ceil = hybrid_xf if (do_hybrid
                                            and hybrid_xf is not None) else None
                    if fb_auto:
                        bands = FilterBankDecomposition.bands_from_log_spacing(
                            self.fb_bpo, self.fb_fmin, self.fb_fmax,
                            sr_wav, ceiling=fb_ceil)
                    else:
                        bands = FilterBankDecomposition.bands_from_crossovers(
                            self.fb_crossovers, sr_wav)
                    fb.compute_and_save(mono, bands, outdir, sr_wav,
                                        wav_path=wav_path,
                                        progress_cb=_fb_progress)
                    self._fb_decomp = fb
                    if self.on_fb_computed:
                        self.on_fb_computed(fb)

                    # Always produce L+R envelopes (mono input → L=R)
                    fb_dir = os.path.join(outdir, "filterbank")
                    if is_wav_stereo:
                        ch_pairs = [("left", left_ch),
                                    ("right", right_ch)]
                    else:
                        ch_pairs = [("left", mono)]
                    for ch_label, ch_sig in ch_pairs:
                        fb_ch = FilterBankDecomposition(sr_wav, ftype)
                        fb_ch.compute(ch_sig, bands)
                        mags_ch, phases_ch, hops_ch = (
                            fb_ch.compute_envelopes())
                        data_ch: dict[str, np.ndarray] = {
                            "n_bands": np.int64(len(mags_ch)),
                            "hops": np.array(hops_ch, dtype=np.int64),
                        }
                        for ii, (m, p) in enumerate(
                                zip(mags_ch, phases_ch)):
                            data_ch[f"mag_{ii}"] = m
                            data_ch[f"phase_{ii}"] = p
                        np.savez_compressed(
                            os.path.join(fb_dir,
                                         f"fb_envelopes_{ch_label}.npz"),
                            **data_ch)
                    if not is_wav_stereo:
                        import shutil
                        shutil.copy2(
                            os.path.join(fb_dir, "fb_envelopes_left.npz"),
                            os.path.join(fb_dir, "fb_envelopes_right.npz"))

                # Wavelet decomposition
                if do_wavelet:
                    phase_idx = _phases.index("wv")
                    _set_progress(phase_idx / n_phases, "Wavelet decomp")
                    if raw_wav is None:
                        sr_wav, raw_wav = _load_audio(wav_path)
                    if raw_wav.ndim == 2:
                        wv_signal = raw_wav.mean(axis=1)
                    else:
                        wv_signal = raw_wav.copy()
                    if np.issubdtype(wv_signal.dtype, np.integer):
                        wv_signal = wv_signal.astype(np.float32) / max(abs(np.iinfo(raw_wav.dtype).min), np.iinfo(raw_wav.dtype).max)
                    else:
                        wv_signal = wv_signal.astype(np.float32)
                    if has_region and sr_wav is not None:
                        s0 = int(r_start * sr_wav) if r_start > 0 else 0
                        s1 = int(r_end * sr_wav) if r_end > 0 else len(wv_signal)
                        s0 = max(0, min(len(wv_signal), s0))
                        s1 = max(s0, min(len(wv_signal), s1))
                        wv_signal = wv_signal[s0:s1]

                    wv_dir = os.path.join(outdir, "wavelet")
                    os.makedirs(wv_dir, exist_ok=True)
                    import json as _json

                    if self.wavelet_mode_idx == 1:
                        # ── CWT path (torch_cqt_new) ──
                        import torch
                        from torch_cqt_new import cwt as _torch_cwt

                        cwt_wtype = _CWT_WAVELET_TYPES[self.cwt_wavelet_idx]
                        cwt_kw: dict = {}
                        if cwt_wtype == "morlet":
                            cwt_kw["sigma"] = self.cwt_sigma
                        device = torch.device(
                            "cuda" if torch.cuda.is_available() else "cpu")
                        y_t = torch.from_numpy(wv_signal).to(device)

                        cwt_eps = self.cwt_epsilon if self.cwt_epsilon > 0 else None
                        # Frequency range: subsonic up to Nyquist/4
                        cwt_fmax = float(sr_wav) / 4.0
                        cwt_fmin = max(0.1, 1.0 / (len(wv_signal) / sr_wav))

                        W, freqs_t = _torch_cwt(
                            y_t, sr_wav,
                            fmin=cwt_fmin,
                            fmax=cwt_fmax,
                            scales_per_octave=self.cwt_scales_per_octave,
                            hop_length=max(1, self.hop_length),
                            wavelet=cwt_wtype,
                            wavelet_kw=cwt_kw,
                            device=device,
                            epsilon=cwt_eps,
                        )
                        W_np = W.cpu().numpy()
                        freqs_np = freqs_t.cpu().numpy()
                        np.savez_compressed(
                            os.path.join(wv_dir, "wavelet_data.npz"),
                            W_real=W_np.real,
                            W_imag=W_np.imag,
                            freqs=freqs_np,
                        )
                        wv_meta = {
                            "type": "cwt",
                            "wavelet": cwt_wtype,
                            "sigma": self.cwt_sigma,
                            "epsilon": self.cwt_epsilon,
                            "sr": sr_wav,
                            "hop_length": self.hop_length,
                            "scales_per_octave": self.cwt_scales_per_octave,
                            "n_scales": int(W_np.shape[0]),
                            "fmin": float(cwt_fmin),
                            "fmax": float(cwt_fmax),
                        }
                        if wav_path:
                            wv_meta["wav_path"] = os.path.abspath(wav_path)
                        with open(os.path.join(wv_dir, "wavelet_meta.json"),
                                  "w") as wf:
                            _json.dump(wv_meta, wf, indent=2)
                        del W, freqs_t, W_np, y_t
                    else:
                        # ── DWT path (pywt) ──
                        import pywt
                        fam = _WAVELET_FAMILIES[self.wavelet_family_idx]
                        prefix = _WAVELET_FAMILY_PREFIX.get(fam, "db")
                        orders = _WAVELET_ORDERS.get(fam, ["\u2014"])
                        if fam == "Discrete Meyer":
                            wv_name = "dmey"
                        else:
                            safe_idx = min(self.wavelet_order_idx,
                                           len(orders) - 1)
                            wv_name = prefix + orders[safe_idx]
                        ext_mode = _WAVELET_EXTENSIONS[self.wavelet_ext_idx]
                        wv_max = pywt.dwt_max_level(len(wv_signal), wv_name)
                        if self.wavelet_depth_mode == 1:
                            wv_level = max(1, min(self.wavelet_depth_abs, wv_max))
                        else:
                            wv_level = max(1, int(round(
                                self.wavelet_depth_pct / 100.0 * wv_max)))
                        coeffs = pywt.wavedec(wv_signal, wv_name,
                                              mode=ext_mode,
                                              level=wv_level)
                        wv_data: dict[str, np.ndarray] = {
                            "n_levels": np.int64(len(coeffs) - 1),
                            "approx": coeffs[0].astype(np.float32),
                        }
                        for li, detail in enumerate(coeffs[1:], 1):
                            wv_data[f"detail_{li}"] = detail.astype(np.float32)
                        np.savez_compressed(
                            os.path.join(wv_dir, "wavelet_data.npz"),
                            **wv_data)
                        wv_meta = {
                            "type": "dwt",
                            "wavelet": wv_name, "level": wv_level,
                            "extension": ext_mode, "sr": sr_wav,
                            "n_levels": len(coeffs) - 1,
                            "depth_pct": self.wavelet_depth_pct,
                            "max_level": wv_max,
                        }
                        if wav_path:
                            wv_meta["wav_path"] = os.path.abspath(wav_path)
                        with open(os.path.join(wv_dir, "wavelet_meta.json"),
                                  "w") as wf:
                            _json.dump(wv_meta, wf, indent=2)

                _set_progress(1.0, "Complete")
                self._refresh_folders()
            except Exception as exc:
                self._analysis_error = str(exc)
            finally:
                self._analyzing = False
                self._analysis_progress = 0.0
                self._analysis_phase = ""

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

    def _render_checkbox(self, surf: pygame.Surface, font: Any,
                         x: int, y: int, max_w: int, key: str,
                         label: str, checked: bool) -> int:
        """Draw a checkbox and return the x coordinate after it."""
        box_sz = 14
        box = pygame.Rect(x + 2, y + 4, box_sz, box_sz)
        pygame.draw.rect(surf, (80, 80, 100), box, 1)
        if checked:
            pygame.draw.line(surf, (120, 200, 120),
                             (box.x + 2, box.y + 7),
                             (box.x + 5, box.y + 11), 2)
            pygame.draw.line(surf, (120, 200, 120),
                             (box.x + 5, box.y + 11),
                             (box.x + 12, box.y + 2), 2)
        txt = font.render(label, True, (200, 200, 200))
        surf.blit(txt, (x + box_sz + 6, y + 3))
        total_w = box_sz + 6 + txt.get_width() + 8
        click_rect = pygame.Rect(x, y, total_w, self.ROW_H)
        self._item_map[key] = (click_rect, None)
        return x + total_w

    # ---- Main render ------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._item_map = {}
        self._refresh_folders()

        # Pre-compute total height — band list now scrolls internally
        est_h = 1800
        surf = pygame.Surface((w, est_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))
        y = self.PAD

        # === WAV file list ===
        y += self.wav_list.render(surf, font, 0, y, w)

        # === Channel mode ===
        y = self._render_dropdown_btn(surf, font, y, w, "chan_mode",
                                      "Chan:", _CHANNEL_MODES,
                                      self.channel_mode_idx)

        # === Source include checkboxes ===
        lbl = font.render("Include:", True, (160, 160, 160))
        surf.blit(lbl, (self.PAD, y + 3))
        cx = self.PAD + lbl.get_width() + 6
        cx = self._render_checkbox(surf, font, cx, y, w, "inc_cqt",
                                   "CQT", self.include_cqt)
        cx = self._render_checkbox(surf, font, cx, y, w, "inc_fb",
                                   "FB", self.include_fb)
        cx = self._render_checkbox(surf, font, cx, y, w, "inc_wavelet",
                                   "WV", self.include_wavelet)
        y += self.ROW_H + 2
        # Hybrid checkbox (only relevant when FB is included)
        if self.include_fb:
            cx2 = self.PAD + 10
            cx2 = self._render_checkbox(surf, font, cx2, y, w,
                                        "fb_hybrid", "Hybrid (FB low / CQT high)",
                                        self.fb_hybrid)
            y += self.ROW_H + 2
            cx3 = self.PAD + 10
            cx3 = self._render_checkbox(surf, font, cx3, y, w,
                                        "fb_q_norm", "Q-width normalization",
                                        self.fb_q_norm)
            y += self.ROW_H + 2

        # Derive effective mode string for legacy sections
        show_cqt = self.include_cqt
        show_fb = self.include_fb
        is_hybrid = self.include_fb and self.fb_hybrid
        sr_preview = self._wav_sr if self._wav_sr > 0 else 44100

        # === CQT settings (collapsible) ===
        if show_cqt or is_hybrid:
            y = self._render_section_header(surf, font, y, w,
                                            "CQT Settings", "cqt_hdr",
                                            self._cqt_expanded)
            if self._cqt_expanded:
                y = self._render_dropdown_btn(surf, font, y, w,
                                              "cqt_algo", "Algorithm:",
                                              _CQT_ALGORITHMS,
                                              self.cqt_algorithm_idx)
                _is_nsgt = _CQT_ALGORITHMS[self.cqt_algorithm_idx] == "nsgt"
                if not _is_nsgt:
                    y = self._render_slider(surf, font, y, w, "hop", "Hop:",
                                            float(self.hop_length), 64.0, 2048.0,
                                            log=True)
                y = self._render_slider(surf, font, y, w, "bpo", "BPO:",
                                        float(self.bins_per_octave), 12.0, 1200.0,
                                        log=True)
                if not _is_nsgt:
                    _fs_hi = max(4.0, self.cqt_filter_scale,
                                 self._solver_limit("cqt", "filter_scale", 4.0))
                    y = self._render_slider(surf, font, y, w, "cqt_filter_scale",
                                            "FilterQ:", self.cqt_filter_scale,
                                            0.1, _fs_hi, "{:.2f}", log=False)
                _cqt_lo, _cqt_hi = self._freq_limits()
                if is_hybrid:
                    # In Hybrid the CQT fMin is auto-raised to crossover
                    xf_cqt = hybrid_crossover_freq(self.bins_per_octave,
                                                   self.hop_length, sr_preview)
                    eff_fmin = max(self.cqt_fmin, xf_cqt)
                    y = self._render_slider(surf, font, y, w, "fmin", "fMin:",
                                            eff_fmin, _cqt_lo, _cqt_hi,
                                            _freq_fmt(eff_fmin),
                                            note_label=True, log=True)
                else:
                    y = self._render_slider(surf, font, y, w, "fmin", "fMin:",
                                            self.cqt_fmin, _cqt_lo, _cqt_hi,
                                            _freq_fmt(self.cqt_fmin),
                                            note_label=True, log=True)
                y = self._render_slider(surf, font, y, w, "fmax", "fMax:",
                                        self.cqt_fmax, _cqt_lo, _cqt_hi,
                                        _freq_fmt(self.cqt_fmax),
                                        note_label=True, log=True)
        # === Hybrid crossover info ===
        if is_hybrid:
            xf = hybrid_crossover_freq(self.bins_per_octave,
                                       self.hop_length, sr_preview)
            xf_note = _freq_to_note_name(xf) if xf < (sr_preview / 2.0) else ""
            xf_label = f"Crossover: {xf:.1f} Hz"
            if xf_note:
                xf_label += f"  ({xf_note})"
            xt = font.render(xf_label, True, (180, 220, 140))
            surf.blit(xt, (self.PAD, y + 2))
            y += self.ROW_H + 2
            hint = font.render("FB below  \u2502  CQT above", True,
                               (120, 140, 120))
            surf.blit(hint, (self.PAD, y + 2))
            y += self.ROW_H + 2

        # === Filter bank settings (collapsible) ===
        if show_fb or is_hybrid:
            y = self._render_section_header(surf, font, y, w,
                                            "Filter Bank", "fb_hdr",
                                            self._fb_expanded)
            if self._fb_expanded:
                # Filter type
                y = self._render_dropdown_btn(surf, font, y, w,
                                              "fb_ftype", "Type:",
                                              _FILTER_TYPES,
                                              self.fb_filter_type_idx)

                # Config mode (Manual / Auto)
                y = self._render_dropdown_btn(surf, font, y, w,
                                              "fb_cfg", "Config:",
                                              _FB_CONFIG_MODES,
                                              self.fb_config_mode_idx)

                # Label mode (Range / Mid+BW)
                y = self._render_dropdown_btn(surf, font, y, w,
                                              "fb_lbl", "Labels:",
                                              _FB_LABEL_MODES,
                                              self.fb_label_mode_idx)

                fb_auto = _FB_CONFIG_MODES[self.fb_config_mode_idx] != "Manual"

                # FB controls — always visible (mirrors CQT controls)
                _fb_bpo_hi = max(48.0, float(self.fb_bpo), self._solver_limit("fb", "bands_per_octave", 48.0))
                y = self._render_slider(
                    surf, font, y, w, "fb_bpo", "BPO:",
                    float(self.fb_bpo), 1.0, _fb_bpo_hi, log=False)
                _fb_lo, _fb_hi = self._freq_limits()
                y = self._render_slider(
                    surf, font, y, w, "fb_fmin", "fMin:",
                    self.fb_fmin, _fb_lo, _fb_hi,
                    _freq_fmt(self.fb_fmin),
                    note_label=True, log=True)
                y = self._render_slider(
                    surf, font, y, w, "fb_fmax", "fMax:",
                    self.fb_fmax, _fb_lo, _fb_hi,
                    _freq_fmt(self.fb_fmax),
                    note_label=True, log=True)
                y = self._render_slider(
                    surf, font, y, w, "fb_hop", "Hop:",
                    float(self.fb_hop), 64.0, 2048.0, log=True)

                if not fb_auto:
                    # --- Manual crossover mode ---
                    lbl = font.render("Crossovers:", True, (160, 160, 160))
                    surf.blit(lbl, (self.PAD, y + 2))
                    y += self.ROW_H

                    _xo_lo, _xo_hi = self._freq_limits()
                    for i, xo in enumerate(self.fb_crossovers):
                        key = f"xo_{i}"
                        y = self._render_slider(surf, font, y, w, key,
                                                f" X{i + 1}:",
                                                xo, _xo_lo, _xo_hi,
                                                _freq_fmt(xo),
                                                note_label=True, log=True)

                    # Add / Remove crossover buttons
                    btn_w2 = (w - 3 * self.PAD) // 2
                    add_r = pygame.Rect(self.PAD, y, btn_w2, self.ROW_H)
                    rem_r = pygame.Rect(self.PAD * 2 + btn_w2, y,
                                        btn_w2, self.ROW_H)
                    pygame.draw.rect(surf, (40, 70, 40), add_r)
                    pygame.draw.rect(surf, (70, 100, 70), add_r, 1)
                    at = font.render("+ Add XO", True, (160, 230, 160))
                    surf.blit(at, (add_r.x + add_r.w // 2
                                   - at.get_width() // 2, y + 3))
                    self._item_map["fb_add_xo"] = (add_r, None)

                    pygame.draw.rect(surf, (70, 40, 40), rem_r)
                    pygame.draw.rect(surf, (100, 70, 70), rem_r, 1)
                    rt = font.render("- Remove", True, (230, 160, 160))
                    surf.blit(rt, (rem_r.x + rem_r.w // 2
                                   - rt.get_width() // 2, y + 3))
                    self._item_map["fb_rem_xo"] = (rem_r, None)
                    y += self.ROW_H + 2

                    bands = FilterBankDecomposition.bands_from_crossovers(
                        self.fb_crossovers, sr_preview)
                else:
                    # --- Auto (log-spaced) band computation ---
                    fb_preview_ceil: float | None = None
                    if is_hybrid:
                        fb_preview_ceil = hybrid_crossover_freq(
                            self.bins_per_octave, self.hop_length,
                            sr_preview)
                    bands = FilterBankDecomposition.bands_from_log_spacing(
                        self.fb_bpo, self.fb_fmin, self.fb_fmax,
                        sr_preview, ceiling=fb_preview_ceil)

                # --- Resulting bands preview (fixed-height scrollable) ---
                self._band_list_bands = bands
                band_items: list[ListItem] = []
                for bi, band in enumerate(bands):
                    has_fb = (self._fb_decomp is not None and
                              bi < len(self._fb_decomp.subbands))
                    tag = "\u2197" if has_fb else ""
                    band_items.append(ListItem(
                        key=f"band_{bi}",
                        display_name=(band.mid_label(sr_preview)
                                      if _FB_LABEL_MODES[self.fb_label_mode_idx] == "Mid+BW"
                                      else band.auto_label(sr_preview)),
                        tag=tag,
                        tag_color_key="Ch3" if has_fb else "none",
                    ))
                self._band_list.items = band_items
                y += self._band_list.render(surf, font, 0, y, w)

                # Graduate selected band button
                sel_bi = self._band_list.selected_idx
                has_sel_fb = (self._fb_decomp is not None and
                              0 <= sel_bi < len(self._fb_decomp.subbands))
                grad_r = pygame.Rect(self.PAD, y, 80, self.ROW_H)
                gc = (50, 70, 90) if has_sel_fb else (40, 40, 45)
                pygame.draw.rect(surf, gc, grad_r)
                pygame.draw.rect(surf, (80, 100, 120), grad_r, 1)
                gt = font.render("\u2197 WAV", True,
                                 (160, 200, 230) if has_sel_fb
                                 else (80, 80, 90))
                surf.blit(gt, (grad_r.x + 4, y + 2))
                self._item_map["grad_selected"] = (grad_r, sel_bi)
                y += self.ROW_H + 2

                # Reconstruction error (if computed)
                if self._fb_decomp and self._fb_decomp.subbands:
                    err = self._fb_decomp.reconstruction_error()
                    err_txt = f"Recon err: {err:.6f} RMS"
                    ec = (120, 230, 120) if err < 0.01 else (230, 180, 120)
                    et = font.render(err_txt, True, ec)
                    surf.blit(et, (self.PAD, y + 2))
                    y += self.ROW_H

        # === Wavelet decomposition (collapsible) ===
        if self.include_wavelet:
            y = self._render_section_header(surf, font, y, w,
                                            "Wavelet Decomposition",
                                            "wv_hdr", True)
            # Mode: DWT or CWT
            y = self._render_dropdown_btn(
                surf, font, y, w, "wv_mode", "Mode:",
                _WAVELET_MODES, self.wavelet_mode_idx)

            if self.wavelet_mode_idx == 1:
                # ── CWT settings ──
                y = self._render_dropdown_btn(
                    surf, font, y, w, "cwt_wavelet", "Wavelet:",
                    _CWT_WAVELET_TYPES, self.cwt_wavelet_idx)
                _cwt_spo_hi = max(120.0, float(self.cwt_scales_per_octave), self._solver_limit("cwt", "scales_per_octave", 120.0))
                y = self._render_slider(
                    surf, font, y, w, "cwt_spo", "Scales/oct:",
                    float(self.cwt_scales_per_octave), 1.0, _cwt_spo_hi,
                    "{:.0f}", log=False)
                if _CWT_WAVELET_TYPES[self.cwt_wavelet_idx] == "morlet":
                    _sig_hi = max(30.0, self.cwt_sigma, self._solver_limit("cwt", "sigma", 30.0))
                    y = self._render_slider(
                        surf, font, y, w, "cwt_sigma", "σ:",
                        self.cwt_sigma, 1.0, _sig_hi, "{:.1f}", log=False)
                y = self._render_slider(
                    surf, font, y, w, "cwt_epsilon", "ε:",
                    self.cwt_epsilon, 0.0, 1.0, "{:.3f}", log=False)
                _cwt_flim_lo, _cwt_flim_hi = self._freq_limits()
                _cwt_fmax_eff = self.cwt_fmax if self.cwt_fmax > 0 else sr_preview / 4.0
                y = self._render_slider(
                    surf, font, y, w, "cwt_fmin", "fMin:",
                    self.cwt_fmin, max(0.001, _cwt_flim_lo), _cwt_flim_hi,
                    _freq_fmt(self.cwt_fmin), note_label=True, log=True)
                y = self._render_slider(
                    surf, font, y, w, "cwt_fmax", "fMax:",
                    _cwt_fmax_eff, max(0.001, _cwt_flim_lo), _cwt_flim_hi,
                    _freq_fmt(_cwt_fmax_eff), note_label=True, log=True)
                wn_txt = font.render(
                    f"CWT: {_CWT_WAVELET_TYPES[self.cwt_wavelet_idx]}  "
                    f"σ={self.cwt_sigma:.1f}  ε={self.cwt_epsilon:.3f}",
                    True, (180, 200, 140))
                surf.blit(wn_txt, (self.PAD, y + 2))
                y += self.ROW_H + 2
            else:
                # ── DWT settings ──
                fam = _WAVELET_FAMILIES[self.wavelet_family_idx]
                y = self._render_dropdown_btn(
                    surf, font, y, w, "wv_family", "Family:",
                    _WAVELET_FAMILIES, self.wavelet_family_idx)

                orders = _WAVELET_ORDERS.get(fam, ["\u2014"])
                if orders != ["\u2014"]:
                    safe_idx = min(self.wavelet_order_idx, len(orders) - 1)
                    y = self._render_dropdown_btn(
                        surf, font, y, w, "wv_order", "Order:",
                        orders, safe_idx)

                _depth_modes = ["% of max", "Absolute"]
                y = self._render_dropdown_btn(
                    surf, font, y, w, "wv_depth_mode", "Depth:",
                    _depth_modes, self.wavelet_depth_mode)
                max_lev = self._wavelet_max_level()
                actual_lev = self._wavelet_level_from_pct()
                if self.wavelet_depth_mode == 0:
                    y = self._render_slider(
                        surf, font, y, w, "wv_depth_pct", "",
                        self.wavelet_depth_pct, 1.0, 100.0, "{:.0f}%",
                        log=False)
                else:
                    y = self._render_slider(
                        surf, font, y, w, "wv_depth_abs", "",
                        float(self.wavelet_depth_abs), 1.0, 50.0, "{:.0f}",
                        log=False)
                depth_txt = font.render(
                    f"  \u2192 level {actual_lev} / {max_lev}", True,
                    (140, 160, 180))
                surf.blit(depth_txt, (self.PAD, y + 1))
                y += self.ROW_H

                y = self._render_dropdown_btn(
                    surf, font, y, w, "wv_ext", "Ext:",
                    _WAVELET_EXTENSIONS, self.wavelet_ext_idx)

                prefix = _WAVELET_FAMILY_PREFIX.get(fam, "db")
                if fam == "Discrete Meyer":
                    wv_name = "dmey"
                else:
                    safe_idx = min(self.wavelet_order_idx,
                                   len(orders) - 1)
                    wv_name = prefix + orders[safe_idx]
                wn_txt = font.render(f"Wavelet: {wv_name}", True,
                                     (180, 200, 140))
                surf.blit(wn_txt, (self.PAD, y + 2))
                y += self.ROW_H + 2

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

        # === Time region (partial analysis) ===
        y += 2
        y = self._render_section_header(surf, font, y, w,
                                        "Time Region", "region_hdr",
                                        self._region_expanded)
        if self._region_expanded:
            dur = self._wav_duration
            if dur > 0:
                y = self._render_slider(surf, font, y, w, "region_start",
                                        "Start:",
                                        self.region_start, 0.0, dur,
                                        "{:.2f}", log=False)
                y = self._render_slider(surf, font, y, w, "region_end",
                                        "End:",
                                        self.region_end, 0.0, dur,
                                        "{:.2f}", log=False)
                # Info label
                if self.region_start > 0 or (self.region_end > 0
                                             and self.region_end < dur):
                    rs = self.region_start
                    re = self.region_end if self.region_end > 0 else dur
                    span = re - rs
                    rl = font.render(
                        f"Region: {rs:.2f}s \u2013 {re:.2f}s ({span:.2f}s)",
                        True, (180, 200, 140))
                    surf.blit(rl, (self.PAD, y + 2))
                    y += self.ROW_H
                else:
                    fl = font.render(
                        f"Full file: {dur:.2f}s", True, (140, 140, 140))
                    surf.blit(fl, (self.PAD, y + 2))
                    y += self.ROW_H
            else:
                nl = font.render("Select a WAV to set region",
                                 True, (100, 100, 100))
                surf.blit(nl, (self.PAD, y + 2))
                y += self.ROW_H

        # === Signal Loss Map (shared across all active sources) ===
        any_source = show_cqt or show_fb or self.include_wavelet
        if any_source:
            try:
                from torch_cqt_new import (fidelity_curve as _fidelity_curve,
                                           fidelity_curve_fb as _fidelity_fb,
                                           fidelity_curve_cwt as _fidelity_cwt)
                from torch_nsgt import fidelity_curve_nsgt as _fidelity_nsgt
                self._loss_map.bars.clear()

                if show_cqt:
                    _algo = _CQT_ALGORITHMS[self.cqt_algorithm_idx]
                    if _algo == "nsgt":
                        fc = _fidelity_nsgt(
                            sr_preview,
                            fmin=self.cqt_fmin,
                            fmax=self.cqt_fmax,
                            bins_per_octave=self.bins_per_octave)
                        _pr = fc.get("painless")
                        _cond = fc.get("condition_number", 0.0)
                        if _pr is not None and _pr.is_painless:
                            _cqt_label = f"NSGT \u2713 cond={_cond:.0f}"
                        elif _pr is not None:
                            _cqt_label = f"NSGT \u26a0 NOT PAINLESS"
                        else:
                            _cqt_label = "NSGT"
                    else:
                        fc = _fidelity_curve(
                            sr_preview, self.hop_length,
                            self.cqt_fmin, self.cqt_fmax,
                            self.bins_per_octave)
                        _cqt_label = "CQT"
                    f_cqt = fc["freqs"]
                    if len(f_cqt) > 1:
                        log_f = np.log10(np.maximum(f_cqt, 1e-6)).astype(np.float32)
                        self._loss_map.add_bar(HeatmapBar(
                            key="cqt", label=_cqt_label,
                            rgb=fc["loss_rgb"], x=log_f))

                if show_fb:
                    ft = _FILTER_TYPES[self.fb_filter_type_idx] \
                        if self.fb_filter_type_idx < len(_FILTER_TYPES) \
                        else "Linkwitz-Riley 4"
                    fb = _fidelity_fb(
                        sr_preview,
                        bands_per_octave=self.fb_bpo,
                        fmin=self.fb_fmin, fmax=self.fb_fmax,
                        hop_length=self.fb_hop,
                        filter_type=ft)
                    f_fb = fb["freqs"]
                    if len(f_fb) > 1:
                        log_fb = np.log10(np.maximum(f_fb, 1e-6)).astype(np.float32)
                        self._loss_map.add_bar(HeatmapBar(
                            key="fb", label="FB",
                            rgb=fb["loss_rgb"], x=log_fb))

                if self.include_wavelet:
                    wv_type = "morlet"
                    if self.wavelet_mode_idx == 1:
                        wv_type = _CWT_WAVELET_TYPES[
                            min(self.cwt_wavelet_idx,
                                len(_CWT_WAVELET_TYPES) - 1)]
                    _cwt_fmax_lm = (self.cwt_fmax if self.cwt_fmax > 0
                                    else sr_preview / 4.0)
                    wv = _fidelity_cwt(
                        sr_preview,
                        fmin=self.cwt_fmin, fmax=_cwt_fmax_lm,
                        scales_per_octave=self.cwt_scales_per_octave,
                        hop_length=1,
                        wavelet=wv_type,
                        sigma=self.cwt_sigma)
                    f_wv = wv["freqs"]
                    if len(f_wv) > 1:
                        log_wv = np.log10(np.maximum(f_wv, 1e-6)).astype(np.float32)
                        self._loss_map.add_bar(HeatmapBar(
                            key="wv", label="WV",
                            rgb=wv["loss_rgb"], x=log_wv))

                # Frequency markers — subsonic through audible
                _marker_freqs = [
                    # Geologic / volcanic
                    (0.01, "Geo"),     # Earth hum, seismic
                    (0.1, "μSeis"),    # Ocean microseism
                    (1.0, "1Hz"),      # Volcanic tremor
                    # Whale song
                    (15.0, "Fin♪"),    # Fin whale fundamental
                    (20.0, "Blue♪"),   # Blue whale call
                    # Musical
                    (32.7, "C1"), (65.4, "C2"), (130.8, "C3"),
                    (261.6, "C4"), (440.0, "A4"), (1000.0, "1k"),
                    (4000.0, "4k"), (10000.0, "10k"),
                ]
                all_x = [b.x for b in self._loss_map.bars if b.x.size]
                if all_x:
                    lo = min(float(a.min()) for a in all_x)
                    hi = max(float(a.max()) for a in all_x)
                    markers: list[PlotMarker] = []
                    for mf, ml in _marker_freqs:
                        lmf = math.log10(mf)
                        if lo <= lmf <= hi:
                            markers.append(PlotMarker(
                                x_value=lmf, label=ml,
                                color=(200, 200, 200)))
                    self._loss_map.markers = markers

                # Title: worst overall
                worst_r = max(
                    (float(b.rgb[:, 0].max()) for b in self._loss_map.bars
                     if b.rgb.shape[0] > 0), default=0.0)
                worst_g = max(
                    (float(b.rgb[:, 1].max()) for b in self._loss_map.bars
                     if b.rgb.shape[0] > 0), default=0.0)
                if worst_r > 0.05:
                    self._loss_map.title = (
                        f"Loss map  \u26a0 catastrophic {worst_r:.0%}")
                elif worst_g > 0.1:
                    self._loss_map.title = (
                        f"Loss map  degraded {worst_g:.0%}")
                else:
                    self._loss_map.title = "Loss map  \u2713 clean"
            except Exception:
                pass

            n_bars = len([b for b in self._loss_map.bars if b.rgb.shape[0] > 0])
            plot_h = max(40, 16 + n_bars * 22)
            y += self._loss_map.render(surf, 0, y, w, plot_h, font)
            y += 2

        # === Solver locks & button ===
        any_source = show_cqt or show_fb or self.include_wavelet
        if any_source:
            # Lock checkboxes — show lockable params per active engine
            lbl = font.render("Lock:", True, (160, 140, 100))
            surf.blit(lbl, (self.PAD, y + 3))
            cx = self.PAD + lbl.get_width() + 4

            _lock_items: list[tuple[str, str, str]] = []  # (key, label, eng)
            _is_nsgt_lock = _CQT_ALGORITHMS[self.cqt_algorithm_idx] == "nsgt"
            if show_cqt:
                _lock_items += [
                    ("lock_cqt_bpo", "BPO", "cqt"),
                    ("lock_cqt_fmin", "fMin", "cqt"),
                    ("lock_cqt_fmax", "fMax", "cqt"),
                ]
                if not _is_nsgt_lock:
                    _lock_items += [
                        ("lock_cqt_hop", "Hop", "cqt"),
                        ("lock_cqt_fs", "Q", "cqt"),
                    ]
            if show_fb:
                _lock_items += [
                    ("lock_fb_bpo", "fbBPO", "fb"),
                    ("lock_fb_hop", "fbHop", "fb"),
                    ("lock_fb_fmin", "fbFmin", "fb"),
                    ("lock_fb_fmax", "fbFmax", "fb"),
                ]
            if self.include_wavelet and self.wavelet_mode_idx == 1:
                _lock_items += [
                    ("lock_cwt_spo", "SPO", "cwt"),
                    ("lock_cwt_sigma", "\u03c3", "cwt"),
                    ("lock_cwt_fmin", "cFmin", "cwt"),
                    ("lock_cwt_fmax", "cFmax", "cwt"),
                ]

            _lock_param_map = {
                "lock_cqt_bpo": ("cqt", "bins_per_octave"),
                "lock_cqt_hop": ("cqt", "hop_length"),
                "lock_cqt_fmin": ("cqt", "fmin"),
                "lock_cqt_fmax": ("cqt", "fmax"),
                "lock_cqt_fs": ("cqt", "filter_scale"),
                "lock_fb_bpo": ("fb", "bands_per_octave"),
                "lock_fb_hop": ("fb", "hop_length"),
                "lock_fb_fmin": ("fb", "fmin"),
                "lock_fb_fmax": ("fb", "fmax"),
                "lock_cwt_spo": ("cwt", "scales_per_octave"),
                "lock_cwt_sigma": ("cwt", "sigma"),
                "lock_cwt_fmin": ("cwt", "fmin"),
                "lock_cwt_fmax": ("cwt", "fmax"),
            }

            for lk_key, lk_label, lk_eng in _lock_items:
                eng_name, param_name = _lock_param_map[lk_key]
                is_locked = param_name in self._param_locks.get(eng_name, set())
                # Wrap to next row if needed
                needed = len(lk_label) * 7 + 18
                if cx + needed > w - self.PAD:
                    y += self.ROW_H
                    cx = self.PAD + 4
                cx = self._render_checkbox(surf, font, cx, y, w, lk_key,
                                           lk_label, is_locked)
            y += self.ROW_H + 2

            # Solver scope hint — what is free to optimize
            _short_names = {
                "bins_per_octave": "BPO", "hop_length": "hop",
                "fmin": "fMin", "fmax": "fMax", "filter_scale": "Q",
                "bands_per_octave": "BPO", "scales_per_octave": "SPO",
                "sigma": "σ", "epsilon": "ε",
            }
            _free_parts: list[str] = []
            _hint_engines: list[tuple[str, list[str]]] = []
            if show_cqt:
                _cqt_all = (["hop_length", "filter_scale"]
                            if not _is_nsgt_lock else [])
                _cqt_all += ["bins_per_octave", "fmin", "fmax"]
                _hint_engines.append(("cqt", _cqt_all))
            if show_fb:
                _hint_engines.append(
                    ("fb", ["bands_per_octave", "hop_length", "fmin", "fmax"]))
            if self.include_wavelet and self.wavelet_mode_idx == 1:
                _hint_engines.append(
                    ("cwt", ["scales_per_octave", "sigma", "fmin", "fmax"]))
            for _eng, _all_params in _hint_engines:
                _locked_set = self._param_locks.get(_eng, set())
                _free = [p for p in _all_params if p not in _locked_set]
                if _free:
                    _free_parts.append(
                        f"{_eng}: "
                        f"{' '.join(_short_names.get(p, p) for p in _free)}")
            if _free_parts:
                hint_txt = font.render(
                    "Free: " + "  |  ".join(_free_parts),
                    True, (110, 130, 110))
                surf.blit(hint_txt, (self.PAD, y + 2))
                y += self.ROW_H + 2

            # Schedule-mode checkboxes — per-octave vs global for free params
            # Only schedulable, non-locked free params are shown
            _sched_label = font.render("Schedule:", True, (120, 150, 110))
            surf.blit(_sched_label, (self.PAD, y + 3))
            _scx = self.PAD + _sched_label.get_width() + 4

            # Build list of schedulable free params across active engines
            _sched_items: list[tuple[str, str, str]] = []  # (cb_key, label, eng)
            _sched_candidates = {
                "cqt": ([] if _is_nsgt_lock else
                        [("sched_cqt_bpo", "BPO", "cqt"),
                         ("sched_cqt_hop", "Hop", "cqt"),
                         ("sched_cqt_fs", "Q", "cqt")]),
                "fb":  [("sched_fb_bpo", "fBPO", "fb"),
                        ("sched_fb_hop", "fHop", "fb")],
                "cwt": [("sched_cwt_spo", "SPO", "cwt"),
                        ("sched_cwt_sig", "\u03c3", "cwt")],
            }
            _sched_param_map = {
                "sched_cqt_bpo": ("cqt", "bins_per_octave"),
                "sched_cqt_hop": ("cqt", "hop_length"),
                "sched_cqt_fs":  ("cqt", "filter_scale"),
                "sched_fb_bpo":  ("fb", "bands_per_octave"),
                "sched_fb_hop":  ("fb", "hop_length"),
                "sched_cwt_spo": ("cwt", "scales_per_octave"),
                "sched_cwt_sig": ("cwt", "sigma"),
            }
            _eng_active = (
                (["cqt"] if show_cqt else []) +
                (["fb"] if show_fb else []) +
                (["cwt"] if self.include_wavelet and self.wavelet_mode_idx == 1
                 else [])
            )
            for _eng in _eng_active:
                for item in _sched_candidates.get(_eng, []):
                    cb_key, lbl, eng = item
                    eng_name, param_name = _sched_param_map[cb_key]
                    # Skip if locked
                    if param_name in self._param_locks.get(eng_name, set()):
                        continue
                    _sched_items.append(item)

            for cb_key, lbl, eng in _sched_items:
                eng_name, param_name = _sched_param_map[cb_key]
                is_sched = (eng_name, param_name) in self._schedule_params
                needed = len(lbl) * 7 + 18
                if _scx + needed > w - self.PAD:
                    y += self.ROW_H
                    _scx = self.PAD + 4
                _scx = self._render_checkbox(surf, font, _scx, y, w, cb_key,
                                             lbl, is_sched)
            y += self.ROW_H + 2

            # Solve button
            solve_rect = pygame.Rect(self.PAD, y,
                                     w - 2 * self.PAD, self.ROW_H)
            if self._solving:
                pygame.draw.rect(surf, (30, 30, 55), solve_rect)
                pygame.draw.rect(surf, (80, 80, 160), solve_rect, 1)
                st = font.render(
                    f"Solving\u2026 {self._solver_progress}",
                    True, (160, 160, 255))
            else:
                pygame.draw.rect(surf, (30, 40, 55), solve_rect)
                pygame.draw.rect(surf, (60, 90, 130), solve_rect, 1)
                st = font.render("Solve for best fidelity",
                                 True, (140, 200, 255))
            surf.blit(st,
                      (solve_rect.x + solve_rect.w // 2 - st.get_width() // 2,
                       y + 3))
            self._item_map["solve_btn"] = (solve_rect, None)
            y += self.ROW_H + 1

            # Show solver result summary if available
            if self._solver_result is not None:
                sr_res = self._solver_result
                # Before row
                if self._pre_solve_losses is not None:
                    pr, pg, pb = self._pre_solve_losses
                    pre_txt = (f"R={pr:.0%} G={pg:.0%} B={pb:.0%}")
                    pre_col = (200, 100, 100) if pr > 0.05 else \
                               (200, 200, 100) if pg > 0.1 else \
                               (120, 180, 120)
                    bl = font.render(f"Before: {pre_txt}", True, pre_col)
                    surf.blit(bl, (self.PAD, y + 2))
                    y += self.ROW_H
                # After row
                r_txt = (f"R={sr_res.loss_red:.0%} "
                         f"G={sr_res.loss_green:.0%} "
                         f"B={sr_res.loss_blue:.0%}")
                col = (200, 100, 100) if sr_res.loss_red > 0.05 else \
                      (200, 200, 100) if sr_res.loss_green > 0.1 else \
                      (100, 200, 100)
                rl = font.render(f"Solved: {r_txt}", True, col)
                surf.blit(rl, (self.PAD, y + 2))
                y += self.ROW_H
            elif self._solver_error:
                el = font.render(f"Solver error: {self._solver_error[:50]}",
                                 True, (255, 100, 100))
                surf.blit(el, (self.PAD, y + 2))
                y += self.ROW_H

        # === Analyze button ===
        btn_rect = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
        if self._analyzing:
            # Background: dark amber
            pygame.draw.rect(surf, (50, 40, 15), btn_rect)
            # Progress fill: growing amber highlight
            prog = max(0.0, min(1.0, self._analysis_progress))
            fill_w = int(btn_rect.w * prog)
            if fill_w > 0:
                fill_rect = pygame.Rect(btn_rect.x, btn_rect.y,
                                        fill_w, btn_rect.h)
                pygame.draw.rect(surf, (180, 140, 30), fill_rect)
            # Border: amber
            pygame.draw.rect(surf, (200, 160, 40), btn_rect, 1)
            # Label: phase text + percentage
            pct = int(prog * 100)
            phase = self._analysis_phase or "Analyzing"
            label = f"{phase} {pct}%"
            txt = font.render(label, True, (255, 230, 150))
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
            ly += self._panel_scroll_y

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
                self._update_wav_duration()
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
            if "wv_hdr" in self._item_map:
                rect, _ = self._item_map["wv_hdr"]
                if rect.collidepoint(lx, ly):
                    return True  # always expanded when include_wavelet
            if "rs_hdr" in self._item_map:
                rect, _ = self._item_map["rs_hdr"]
                if rect.collidepoint(lx, ly):
                    self._rs_expanded = not self._rs_expanded
                    return True
            if "region_hdr" in self._item_map:
                rect, _ = self._item_map["region_hdr"]
                if rect.collidepoint(lx, ly):
                    self._region_expanded = not self._region_expanded
                    return True

            # Source include checkboxes
            for ck_key, attr in [("inc_cqt", "include_cqt"),
                                 ("inc_fb", "include_fb"),
                                 ("inc_wavelet", "include_wavelet"),
                                 ("fb_hybrid", "fb_hybrid"),
                                 ("fb_q_norm", "fb_q_norm")]:
                if ck_key in self._item_map:
                    rect, _ = self._item_map[ck_key]
                    if rect.collidepoint(lx, ly):
                        setattr(self, attr, not getattr(self, attr))
                        return True

            # Lock checkboxes (solver invariants)
            _lock_param_map = {
                "lock_cqt_bpo": ("cqt", "bins_per_octave"),
                "lock_cqt_hop": ("cqt", "hop_length"),
                "lock_cqt_fmin": ("cqt", "fmin"),
                "lock_cqt_fmax": ("cqt", "fmax"),
                "lock_cqt_fs": ("cqt", "filter_scale"),
                "lock_fb_bpo": ("fb", "bands_per_octave"),
                "lock_fb_hop": ("fb", "hop_length"),
                "lock_fb_fmin": ("fb", "fmin"),
                "lock_fb_fmax": ("fb", "fmax"),
                "lock_cwt_spo": ("cwt", "scales_per_octave"),
                "lock_cwt_sigma": ("cwt", "sigma"),
                "lock_cwt_fmin": ("cwt", "fmin"),
                "lock_cwt_fmax": ("cwt", "fmax"),
            }
            for lk_key, (lk_eng, lk_param) in _lock_param_map.items():
                if lk_key in self._item_map:
                    rect, _ = self._item_map[lk_key]
                    if rect.collidepoint(lx, ly):
                        s = self._param_locks.setdefault(lk_eng, set())
                        if lk_param in s:
                            s.discard(lk_param)
                        else:
                            s.add(lk_param)
                        return True

            # Schedule-mode checkboxes (per-octave plan for free params)
            _sched_param_map_ev = {
                "sched_cqt_bpo": ("cqt", "bins_per_octave"),
                "sched_cqt_hop": ("cqt", "hop_length"),
                "sched_cqt_fs":  ("cqt", "filter_scale"),
                "sched_fb_bpo":  ("fb", "bands_per_octave"),
                "sched_fb_hop":  ("fb", "hop_length"),
                "sched_cwt_spo": ("cwt", "scales_per_octave"),
                "sched_cwt_sig": ("cwt", "sigma"),
            }
            for sk_key, (sk_eng, sk_param) in _sched_param_map_ev.items():
                if sk_key in self._item_map:
                    rect, _ = self._item_map[sk_key]
                    if rect.collidepoint(lx, ly):
                        key_pair = (sk_eng, sk_param)
                        if key_pair in self._schedule_params:
                            self._schedule_params.discard(key_pair)
                        else:
                            # Can't be both locked and scheduled
                            self._param_locks.get(sk_eng, set()).discard(sk_param)
                            self._schedule_params.add(key_pair)
                        return True

            # Solve button
            if "solve_btn" in self._item_map:
                rect, _ = self._item_map["solve_btn"]
                if rect.collidepoint(lx, ly) and not self._solving:
                    self._run_solver()
                    return True

            # Band list click
            if self._band_list.handle_click(lx, ly) is not None:
                return True

            # Dropdowns
            for key in ("chan_mode", "cqt_algo",
                        "fb_ftype", "fb_cfg", "fb_lbl",
                        "wv_mode", "cwt_wavelet",
                        "wv_family", "wv_order", "wv_ext",
                        "wv_depth_mode",
                        "rs_engine", "rs_interp", "rs_taps", "rs_prec"):
                if key in self._item_map:
                    rect, opts = self._item_map[key]
                    if rect.collidepoint(lx, ly):
                        self._open_dropdown(key, opts, rect, pr)
                        return True

            # Slider +/- buttons and track drag
            slider_keys = ["hop", "bpo", "fmin", "fmax",
                           "cqt_filter_scale",
                           "fb_bpo", "fb_fmin", "fb_fmax", "fb_hop",
                           "wv_depth_pct", "wv_depth_abs",
                           "cwt_spo", "cwt_sigma", "cwt_epsilon",
                           "cwt_fmin", "cwt_fmax",
                           "region_start", "region_end"]
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

            # Graduate selected band button
            if "grad_selected" in self._item_map:
                rect, bi = self._item_map["grad_selected"]
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
            return self._handle_panel_wheel(event)

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
        elif key == "cqt_algo":
            self.cqt_algorithm_idx = _CQT_ALGORITHMS.index(value)
        elif key == "wv_mode":
            self.wavelet_mode_idx = _WAVELET_MODES.index(value)
        elif key == "cwt_wavelet":
            self.cwt_wavelet_idx = _CWT_WAVELET_TYPES.index(value)
        elif key == "wv_family":
            self.wavelet_family_idx = _WAVELET_FAMILIES.index(value)
            self.wavelet_order_idx = 0
        elif key == "wv_order":
            fam = _WAVELET_FAMILIES[self.wavelet_family_idx]
            orders = _WAVELET_ORDERS[fam]
            self.wavelet_order_idx = orders.index(value)
        elif key == "wv_ext":
            self.wavelet_ext_idx = _WAVELET_EXTENSIONS.index(value)
        elif key == "wv_depth_mode":
            self.wavelet_depth_mode = ["% of max", "Absolute"].index(value)
        elif key == "fb_ftype":
            self.fb_filter_type_idx = _FILTER_TYPES.index(value)
        elif key == "fb_cfg":
            self.fb_config_mode_idx = _FB_CONFIG_MODES.index(value)
        elif key == "fb_lbl":
            self.fb_label_mode_idx = _FB_LABEL_MODES.index(value)
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
            self._clamp_hybrid_cqt_fmin()
        elif key == "bpo":
            self.bins_per_octave = _snap_bpo(val)
            self._clamp_hybrid_cqt_fmin()
        elif key == "fmin":
            lo, hi = self._freq_limits()
            self.cqt_fmin = round(_snap_to_note(max(lo, min(hi, val))), 6)
        elif key == "fmax":
            lo, hi = self._freq_limits()
            self.cqt_fmax = round(_snap_to_note(max(lo, min(hi, val))), 6)
        elif key == "fb_bpo":
            _hi = max(48, int(self._solver_limit("fb", "bands_per_octave", 48)))
            self.fb_bpo = max(1, min(_hi, int(round(val))))
        elif key == "fb_fmin":
            lo, hi = self._freq_limits()
            self.fb_fmin = round(_snap_to_note(max(lo, min(hi, val))), 6)
        elif key == "fb_fmax":
            lo, hi = self._freq_limits()
            self.fb_fmax = round(_snap_to_note(max(lo, min(hi, val))), 6)
        elif key == "fb_hop":
            raw = max(64, min(2048, int(val)))
            self.fb_hop = 2 ** round(math.log2(raw))
        elif key == "cqt_filter_scale":
            _fs_hi = max(4.0, self._solver_limit("cqt", "filter_scale", 4.0))
            self.cqt_filter_scale = max(0.1, min(_fs_hi, float(val)))
        elif key == "cwt_fmin":
            lo, hi = self._freq_limits()
            self.cwt_fmin = round(max(0.001, min(hi, val)), 6)
        elif key == "cwt_fmax":
            lo, hi = self._freq_limits()
            sr = self._wav_sr if self._wav_sr > 0 else 44100
            self.cwt_fmax = round(max(0.001, min(sr / 2.0, val)), 6)
        elif key == "region_start":
            dur = self._wav_duration or 1.0
            self.region_start = round(max(0.0, min(dur, val)), 2)
            if self.region_end > 0 and self.region_start > self.region_end:
                self.region_end = self.region_start
            self._auto_freq_range()
        elif key == "region_end":
            dur = self._wav_duration or 1.0
            self.region_end = round(max(0.0, min(dur, val)), 2)
            if self.region_end > 0 and self.region_end < self.region_start:
                self.region_start = self.region_end
            self._auto_freq_range()
        elif key == "wv_depth_pct":
            self.wavelet_depth_pct = max(1.0, min(100.0, float(val)))
        elif key == "wv_depth_abs":
            self.wavelet_depth_abs = max(1, min(50, int(round(val))))
        elif key == "cwt_spo":
            _hi = max(120, int(self._solver_limit("cwt", "scales_per_octave", 120)))
            self.cwt_scales_per_octave = max(1, min(_hi, int(round(val))))
        elif key == "cwt_sigma":
            _hi = max(30.0, self._solver_limit("cwt", "sigma", 30.0))
            self.cwt_sigma = max(1.0, min(_hi, float(val)))
        elif key == "cwt_epsilon":
            self.cwt_epsilon = max(0.0, min(1.0, float(val)))
        elif key.startswith("xo_"):
            idx = int(key[3:])
            if 0 <= idx < len(self.fb_crossovers):
                lo, hi = self._freq_limits()
                self.fb_crossovers[idx] = round(
                    _snap_to_note(max(lo, min(hi, val))), 6)
                self.fb_crossovers.sort()

    def _nudge_slider(self, key: str, direction: int) -> None:
        """Move a slider by one logical step. direction: +1 or -1."""
        if key == "hop":
            exp = round(math.log2(self.hop_length)) + direction
            self.hop_length = max(64, min(2048, 2 ** exp))
            self._clamp_hybrid_cqt_fmin()
        elif key == "bpo":
            idx = _BPO_SNAPS.index(self.bins_per_octave) if \
                self.bins_per_octave in _BPO_SNAPS else 0
            idx = max(0, min(len(_BPO_SNAPS) - 1, idx + direction))
            self.bins_per_octave = _BPO_SNAPS[idx]
            self._clamp_hybrid_cqt_fmin()
        elif key == "fmin":
            lo, hi = self._freq_limits()
            midi = 69 + 12 * math.log2(self.cqt_fmin / 440.0)
            midi = round(midi) + direction
            self.cqt_fmin = round(max(lo, min(hi,
                                  _midi_to_freq(midi))), 6)
        elif key == "fmax":
            lo, hi = self._freq_limits()
            midi = 69 + 12 * math.log2(self.cqt_fmax / 440.0)
            midi = round(midi) + direction
            self.cqt_fmax = round(max(lo, min(hi,
                                  _midi_to_freq(midi))), 6)
        elif key == "fb_bpo":
            _hi = max(48, int(self._solver_limit("fb", "bands_per_octave", 48)))
            self.fb_bpo = max(1, min(_hi, self.fb_bpo + direction))
        elif key == "fb_fmin":
            lo, hi = self._freq_limits()
            midi = 69 + 12 * math.log2(self.fb_fmin / 440.0)
            midi = round(midi) + direction
            self.fb_fmin = round(max(lo, min(hi,
                                 _midi_to_freq(midi))), 6)
        elif key == "fb_fmax":
            lo, hi = self._freq_limits()
            midi = 69 + 12 * math.log2(self.fb_fmax / 440.0)
            midi = round(midi) + direction
            self.fb_fmax = round(max(lo, min(hi,
                                 _midi_to_freq(midi))), 6)
        elif key == "region_start":
            step = 0.1
            dur = self._wav_duration or 1.0
            self.region_start = round(
                max(0.0, min(dur, self.region_start + step * direction)), 2)
            if self.region_end > 0 and self.region_start > self.region_end:
                self.region_end = self.region_start
            self._auto_freq_range()
        elif key == "region_end":
            step = 0.1
            dur = self._wav_duration or 1.0
            self.region_end = round(
                max(0.0, min(dur, self.region_end + step * direction)), 2)
            if self.region_end > 0 and self.region_end < self.region_start:
                self.region_start = self.region_end
            self._auto_freq_range()
        elif key == "wv_depth_pct":
            self.wavelet_depth_pct = max(1.0, min(100.0,
                self.wavelet_depth_pct + 5.0 * direction))
        elif key == "wv_depth_abs":
            self.wavelet_depth_abs = max(1, min(50,
                self.wavelet_depth_abs + direction))
        elif key == "cwt_spo":
            _hi = max(120, int(self._solver_limit("cwt", "scales_per_octave", 120)))
            self.cwt_scales_per_octave = max(1, min(_hi,
                self.cwt_scales_per_octave + direction))
        elif key == "cwt_sigma":
            _hi = max(30.0, self._solver_limit("cwt", "sigma", 30.0))
            self.cwt_sigma = max(1.0, min(_hi,
                self.cwt_sigma + 0.5 * direction))
        elif key == "cwt_epsilon":
            self.cwt_epsilon = max(0.0, min(1.0,
                self.cwt_epsilon + 0.005 * direction))
        elif key == "fb_hop":
            exp = round(math.log2(self.fb_hop)) + direction
            self.fb_hop = max(64, min(2048, 2 ** exp))
        elif key == "cqt_filter_scale":
            _fs_hi = max(4.0, self._solver_limit("cqt", "filter_scale", 4.0))
            self.cqt_filter_scale = max(0.1, min(_fs_hi,
                round(self.cqt_filter_scale + 0.1 * direction, 2)))
        elif key == "cwt_fmin":
            midi = 69 + 12 * math.log2(max(self.cwt_fmin, 0.001) / 440.0)
            midi = round(midi) + direction
            lo, hi = self._freq_limits()
            self.cwt_fmin = round(max(0.001, min(hi, _midi_to_freq(midi))), 6)
        elif key == "cwt_fmax":
            sr = self._wav_sr if self._wav_sr > 0 else 44100
            eff = self.cwt_fmax if self.cwt_fmax > 0 else sr / 4.0
            midi = 69 + 12 * math.log2(max(eff, 0.001) / 440.0)
            midi = round(midi) + direction
            lo, hi = self._freq_limits()
            self.cwt_fmax = round(max(0.001, min(sr / 2.0, _midi_to_freq(midi))), 6)
        elif key.startswith("xo_"):
            idx = int(key[3:])
            if 0 <= idx < len(self.fb_crossovers):
                lo, hi = self._freq_limits()
                midi = 69 + 12 * math.log2(
                    self.fb_crossovers[idx] / 440.0)
                midi = round(midi) + direction
                self.fb_crossovers[idx] = round(max(lo, min(hi,
                    _midi_to_freq(midi))), 6)
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
                 bins_per_octave: int,
                 resample_taps: int = 64,
                 precision: str = "32-bit float") -> None:
        self.sr = sr
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.resample_taps = resample_taps
        self.precision = precision
        if precision == "64-bit float":
            self._np_real = np.float64
            self._np_complex = np.complex128
        else:
            self._np_real = np.float32
            self._np_complex = np.complex64
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
    def _compute_ola_norms(self, view_freqs: np.ndarray,
                           y0: int, n_total_bins: int,
                           n_view_frames: int) -> dict[int, np.ndarray]:
        """Compute per-octave OLA normalization arrays from ALL given bins.

        The returned dict maps ``octave_index -> norm_array``.  Pass it as
        ``ola_norms`` to :meth:`_inverse_cqt_channel` (or the GPU variant)
        so that two complementary TF partitions share the same normalizer
        and their time-domain outputs sum exactly to the full iCQT result.
        """
        bpo = self.bins_per_octave
        hop = self.hop_length
        sr = self.sr
        Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
        n_octaves = int(math.ceil(n_total_bins / bpo))
        norms: dict[int, np.ndarray] = {}

        for octave in range(n_octaves):
            gbin_start = max(0, n_total_bins - (octave + 1) * bpo)
            gbin_end = n_total_bins - octave * bpo
            vis_start = max(gbin_start, y0)
            vis_end = min(gbin_end, y0 + len(view_freqs))
            if vis_end <= vis_start:
                continue

            local_start = vis_start - y0
            local_end = vis_end - y0
            repeat = 1 << octave
            oct_sr = sr >> octave
            oct_freqs = np.asarray(view_freqs[local_start:local_end],
                                   dtype=np.float64)

            n_oct_frames = len(range(0, n_view_frames, repeat))
            train_len = (n_oct_frames - 1) * hop + 1
            norm_cache: dict[int, np.ndarray] = {}
            oct_out_len = 0
            oct_norm: np.ndarray | None = None

            for fk in oct_freqs:
                klen = int(math.ceil(Q * oct_sr / fk))
                if klen < 1:
                    continue
                conv_len = train_len + klen - 1
                if oct_norm is None:
                    oct_out_len = conv_len
                    oct_norm = np.zeros(oct_out_len, dtype=np.float64)
                elif conv_len > oct_out_len:
                    oct_norm = np.pad(oct_norm, (0, conv_len - oct_out_len))
                    oct_out_len = conv_len

                if klen not in norm_cache:
                    t = np.arange(klen, dtype=np.float64)
                    win = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / klen)
                    g = win / klen  # L1-normalised — must match _inverse_cqt_channel
                    win_sq = g * g
                    norm_train = np.zeros(train_len, dtype=np.float64)
                    norm_train[::hop] = 1.0
                    norm_cache[klen] = self._fftconvolve(norm_train, win_sq)

                nc = norm_cache[klen]
                trim = min(len(nc), oct_out_len)
                oct_norm[:trim] += nc[:trim]

            if oct_norm is not None:
                norms[octave] = oct_norm

        return norms

    # ------------------------------------------------------------------
    # Unified inverse CQT — single torch path for CPU and CUDA
    # ------------------------------------------------------------------
    def _inverse_cqt_channel(self, cqt_slice: np.ndarray,
                             view_freqs: np.ndarray,
                             y0: int, n_total_bins: int,
                             n_view_frames: int,
                             *, ola_norms: dict[int, np.ndarray] | None = None
                             ) -> np.ndarray:
        """Reconstruct one channel via torch_cqt_new.icqt.

        Delegates to the production-tested ``torch_cqt_new.icqt`` engine.
        The STFT back-projection is linear, so complementary TF partitions
        sum exactly without shared normalisers — *ola_norms* is accepted
        for API compatibility but ignored.
        """
        from torch_cqt_new import icqt as _torch_icqt
        import torch

        device = torch.device("cuda" if self._has_cuda else "cpu")
        _torch_real = torch.float64 if self._np_real == np.float64 else torch.float32
        _torch_complex = torch.complex128 if self._np_complex == np.complex128 else torch.complex64

        fmin = float(view_freqs[0])
        out_len = n_view_frames * self.hop_length

        C = torch.from_numpy(np.asarray(cqt_slice, dtype=self._np_complex)).to(
            device=device, dtype=_torch_complex)

        resample_kw = {
            "lowpass_filter_width": self.resample_taps,
            "resampling_method": "sinc_interp_kaiser",
            "beta": 14.769656459379492,
        }

        y = _torch_icqt(
            C, self.sr,
            hop_length=self.hop_length,
            fmin=fmin,
            bins_per_octave=self.bins_per_octave,
            length=out_len,
            device=device,
            dtype=_torch_real,
            resample_kw=resample_kw,
        )

        result = y.cpu().numpy()
        del C, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

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
        _icqt = self._inverse_cqt_channel
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
    # Forward CQT (analysis) — padded, AA-decimated, FFT-convolution
    # matching bass_analysis.compute_cqt conventions exactly.
    # ------------------------------------------------------------------
    def _forward_cqt_channel(self, signal: np.ndarray,
                             view_freqs: np.ndarray,
                             y0: int, n_total_bins: int,
                             n_view_frames: int,
                             *, zero_front: bool = False) -> np.ndarray:
        """Compute CQT coefficients via torch_cqt_new.cqt.

        Delegates to the production-tested ``torch_cqt_new.cqt`` engine.
        Returns complex array of shape ``(n_view_bins, n_view_frames)``.
        Precision is controlled by ``self.precision``.
        """
        from torch_cqt_new import cqt as _torch_cqt
        import torch

        device = torch.device("cuda" if self._has_cuda else "cpu")
        _torch_real = torch.float64 if self._np_real == np.float64 else torch.float32

        n_view_bins = len(view_freqs)
        fmin = float(view_freqs[0])

        y = torch.from_numpy(np.asarray(signal, dtype=self._np_real)).to(
            device=device, dtype=_torch_real)

        resample_kw = {
            "lowpass_filter_width": self.resample_taps,
            "resampling_method": "sinc_interp_kaiser",
            "beta": 14.769656459379492,
        }

        C, freqs = _torch_cqt(
            y, self.sr,
            hop_length=self.hop_length,
            fmin=fmin,
            n_bins=n_view_bins,
            bins_per_octave=self.bins_per_octave,
            pad_mode="constant" if zero_front else "reflect",
            device=device,
            dtype=_torch_real,
            resample_kw=resample_kw,
        )

        result = C.cpu().numpy().astype(self._np_complex)

        # Trim or pad frames to match expected n_view_frames
        actual_frames = result.shape[-1]
        if actual_frames > n_view_frames:
            result = result[..., :n_view_frames]
        elif actual_frames < n_view_frames:
            pad_amt = n_view_frames - actual_frames
            result = np.pad(result, [(0, 0), (0, pad_amt)], mode='edge')

        del C, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return result

    # ------------------------------------------------------------------
    def _apply_phase_mode(self, cqt: np.ndarray, mode: str,
                          view_freqs: np.ndarray, y0: int,
                          n_total_bins: int) -> np.ndarray:
        """Transform CQT coefficients according to the phase mode.

        *cqt* has shape ``(n_bins, n_frames)`` complex.
        Returns a new array (never mutates the input).
        """
        if mode == "zero_phase":
            return np.abs(cqt).astype(self._np_complex)
        if mode == "freq_carrier":
            mag = np.abs(cqt)
            n_bins, n_frames = cqt.shape
            hop = self.hop_length
            sr = self.sr
            phase = np.zeros_like(mag)
            for bi in range(n_bins):
                fc = float(view_freqs[bi])
                for fi in range(n_frames):
                    t_sec = fi * hop / sr
                    phase[bi, fi] = 2.0 * np.pi * fc * t_sec
            return mag * np.exp(1j * phase)
        if mode == "griffin_lim":
            mag = np.abs(cqt)
            n_bins, n_frames = cqt.shape
            # Initialize with random phase
            rng = np.random.default_rng(42)
            phase = np.exp(2j * np.pi * rng.random(cqt.shape))
            estimate = mag * phase

            _icqt = self._inverse_cqt_channel
            n_iters = 20
            for it in range(n_iters):
                # Inverse: coefficients → time domain
                sig = _icqt(estimate, view_freqs, y0,
                            n_total_bins, n_frames)
                # Forward: time domain → coefficients
                reanalysis = self._forward_cqt_channel(
                    sig, view_freqs, y0, n_total_bins, n_frames)
                # Project: keep original magnitudes, take new phases
                phase = np.exp(1j * np.angle(reanalysis))
                estimate = mag * phase
            return estimate
        # "original" — pass through
        return cqt.copy()

    def _ola_pad_frames(self, y0: int, y1: int,
                        freqs: np.ndarray, n_bins: int,
                        sr: int) -> int:
        """Return the number of extra CQT frames each side of the viewport
        needs so that every contributing octave has enough overlap-add
        context for the reconstruction kernels.
        """
        bpo = self.bins_per_octave
        hop = self.hop_length
        Q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
        n_octaves = int(math.ceil(n_bins / bpo))
        pad = 0
        for octave in range(n_octaves):
            gbin_start = max(0, n_bins - (octave + 1) * bpo)
            gbin_end = n_bins - octave * bpo
            vis_start = max(gbin_start, y0)
            vis_end = min(gbin_end, y1)
            if vis_end <= vis_start:
                continue
            f_low = max(float(freqs[vis_start]), 1.0)
            oct_sr = sr >> octave
            klen_max = int(math.ceil(Q * oct_sr / f_low))
            repeat = 1 << octave
            # Enough decimated-frame context × repeat → full-rate frames
            oct_pad = int(math.ceil(klen_max / hop)) * repeat
            pad = max(pad, oct_pad)
        return pad

    # -- Routing-aware CQT helpers ----------------------------------------

    def _build_cqt_from_routing(
        self,
        raw_arrays: dict[str, np.ndarray],
        routed: dict[str, np.ndarray | None] | None,
        audio_ch: str,
        y0: int, y1: int, ex0: int, ex1: int,
        phase_mode: str,
        view_freqs: np.ndarray,
        n_bins: int,
    ) -> np.ndarray:
        """Build a complex CQT slice for *audio_ch* using routed fields.

        When *routed* is ``None`` or the relevant fields are missing, falls
        back to the default ``real_<ch> + 1j * imag_<ch>`` path.

        The magnitude slot is treated as magnitude, the phase slot as
        phase — always ``m * exp(1j * p)``.  The user controls what
        data lands in each slot via the routing patch panel.
        """
        mag_arr = routed.get(f"magnitude_{audio_ch}") if routed else None
        pha_arr = routed.get(f"phase_{audio_ch}") if routed else None

        if mag_arr is not None and pha_arr is not None:
            m = np.asarray(mag_arr[y0:y1, ex0:ex1], dtype=np.float64)
            p = np.asarray(pha_arr[y0:y1, ex0:ex1], dtype=np.float64)
            cqt = m * np.exp(1j * p)
        else:
            # Default fallback — hardcoded CQT complex fields
            real = np.asarray(
                raw_arrays[f"real_{audio_ch}"][y0:y1, ex0:ex1],
                dtype=np.float64)
            imag = np.asarray(
                raw_arrays[f"imag_{audio_ch}"][y0:y1, ex0:ex1],
                dtype=np.float64)
            cqt = real + 1j * imag

        return self._apply_phase_mode(cqt, phase_mode,
                                       view_freqs, y0, n_bins)

    @staticmethod
    def _extract_routing_mask(
        raw_arrays: dict[str, np.ndarray],
        routed: dict[str, np.ndarray | None] | None,
        audio_ch: str,
        y0: int, y1: int, ex0: int, ex1: int,
    ) -> np.ndarray | None:
        """Return the mask array from routing, or None if unset."""
        if routed is None:
            return None
        mask_arr = routed.get(f"mask_{audio_ch}")
        if mask_arr is None:
            return None
        return np.asarray(mask_arr[y0:y1, ex0:ex1], dtype=np.float64)

    def synthesise_to_array(self, raw_arrays: dict[str, np.ndarray],
                            freqs: np.ndarray, times: np.ndarray,
                            view_x0: float, view_x1: float,
                            view_y0: float, view_y1: float,
                            n_frames: int, n_bins: int,
                            t_start: float, t_dur: float,
                            sr: int, is_stereo: bool,
                            normalize: bool = True,
                            phase_mode: str = "original",
                            tf_mask: np.ndarray | None = None,
                            ola_norms: dict[int, np.ndarray] | None = None,
                            routed_arrays: dict[str, np.ndarray | None] | None = None,
                            ) -> tuple[np.ndarray, float, float, float] | None:
        """Run iCQT and return ``(stereo_i16, duration, view_t0, view_t1)``.

        Returns *None* if the viewport is empty.  The stereo array has shape
        ``(N, 2)`` dtype ``int16``, ready for ``pygame.sndarray.make_sound``.

        Parameters
        ----------
        tf_mask : array (n_view_bins, n_extended_frames), optional
            Multiplicative mask in [0, 1] applied to the CQT coefficients
            before inversion.  When used with *ola_norms* computed from the
            unmasked bin set, two complementary masks (M and 1-M) will
            produce time-domain outputs that sum exactly to the unmasked
            iCQT result.
        ola_norms : dict, optional
            Pre-computed per-octave OLA normalization from
            :meth:`_compute_ola_norms`.  When provided the iCQT uses these
            instead of computing its own, ensuring consistent normalisation
            across partitioned synthesis calls.
        routed_arrays : dict, optional
            Mapping from ``"magnitude_left"``, ``"phase_left"``,
            ``"mask_left"``, etc. to raw numpy arrays (native dtype,
            unscaled) resolved from the channel routing panel.  When
            provided, the synthesiser reads data from these arrays
            instead of the hardcoded CQT fields.
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

        # --- Extend frame range so every octave has full OLA context ---
        pad_frames = self._ola_pad_frames(y0, y1, freqs, n_bins, sr)
        ex0 = max(0, x0 - pad_frames)
        ex1 = min(n_frames, x1 + pad_frames)
        trim_start = (x0 - ex0) * self.hop_length
        n_vf = ex1 - ex0

        # Build complex CQT slices (extended range)
        cqt_l = self._build_cqt_from_routing(
            raw_arrays, routed_arrays, "left",
            y0, y1, ex0, ex1, phase_mode, view_freqs, n_bins)

        # Apply routing mask if available
        mask_l = self._extract_routing_mask(
            raw_arrays, routed_arrays, "left", y0, y1, ex0, ex1)
        if mask_l is not None:
            cqt_l = cqt_l * mask_l

        # Apply TF mask if provided
        if tf_mask is not None:
            cqt_l = cqt_l * tf_mask

        print(f"Inverse CQT: {y1 - y0} bins × {x1 - x0} frames "
              f"(padded to {n_vf}), {dur:.2f}s  phase={phase_mode} ...")
        t0 = time.time()

        _icqt = self._inverse_cqt_channel
        print(f"  using {'CUDA' if self._has_cuda else 'CPU'} (torch)")
        sig_l_full = _icqt(cqt_l, view_freqs, y0, n_bins, n_vf,
                           ola_norms=ola_norms)

        if is_stereo:
            cqt_r = self._build_cqt_from_routing(
                raw_arrays, routed_arrays, "right",
                y0, y1, ex0, ex1, phase_mode, view_freqs, n_bins)
            mask_r = self._extract_routing_mask(
                raw_arrays, routed_arrays, "right", y0, y1, ex0, ex1)
            if mask_r is not None:
                cqt_r = cqt_r * mask_r
            if tf_mask is not None:
                cqt_r = cqt_r * tf_mask
            sig_r_full = _icqt(cqt_r, view_freqs, y0, n_bins, n_vf,
                               ola_norms=ola_norms)
        else:
            sig_r_full = sig_l_full

        elapsed = time.time() - t0
        print(f"  done in {elapsed:.2f}s")

        # Extract the viewport interior where OLA is fully converged
        n_view_samples = (x1 - x0) * self.hop_length
        sig_l = sig_l_full[trim_start:trim_start + n_view_samples]
        sig_r = sig_r_full[trim_start:trim_start + n_view_samples]

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

    def synthesise_complementary_pair(
            self, raw_arrays: dict[str, np.ndarray],
            freqs: np.ndarray, times: np.ndarray,
            view_x0: float, view_x1: float,
            view_y0: float, view_y1: float,
            n_frames: int, n_bins: int,
            t_start: float, t_dur: float,
            sr: int, is_stereo: bool,
            tf_mask: np.ndarray,
            phase_mode: str = "original",
    ) -> tuple[tuple[np.ndarray, np.ndarray], float, float, float] | None:
        """Synthesise two complementary partitions that sum to the full iCQT.

        *tf_mask* is a float array with shape ``(y1-y0, ex1-ex0)``
        (viewport bins × extended frames) with values in [0, 1].
        Partition A uses *tf_mask*, partition B uses ``1 - tf_mask``.

        Both partitions share the same OLA normalisation (computed once
        from all viewport bins), so ``A + B == full_icqt`` exactly in
        float64.  The returned float64 arrays are NOT normalised to int16
        so the caller can sum or post-process them freely.

        Returns ``((sig_A, sig_B), duration, vt0, vt1)`` where each
        signal is a 2-D float64 array of shape ``(n_samples, 2)``
        (stereo), or *None* if the viewport is empty.
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

        pad_frames = self._ola_pad_frames(y0, y1, freqs, n_bins, sr)
        ex0 = max(0, x0 - pad_frames)
        ex1 = min(n_frames, x1 + pad_frames)
        trim_start = (x0 - ex0) * self.hop_length
        n_vf = ex1 - ex0

        # Shared OLA normalization from ALL viewport bins
        ola_norms = self._compute_ola_norms(view_freqs, y0, n_bins, n_vf)

        # Build complex CQT (extended range)
        real_l = np.asarray(raw_arrays["real_left"][y0:y1, ex0:ex1],
                            dtype=np.float64)
        imag_l = np.asarray(raw_arrays["imag_left"][y0:y1, ex0:ex1],
                            dtype=np.float64)
        cqt_l = self._apply_phase_mode(real_l + 1j * imag_l, phase_mode,
                                         view_freqs, y0, n_bins)

        _icqt = self._inverse_cqt_channel

        # Ensure mask matches the extended frame range
        mask_ext = tf_mask
        if mask_ext.shape != cqt_l.shape:
            raise ValueError(
                f"tf_mask shape {mask_ext.shape} does not match CQT slice "
                f"shape {cqt_l.shape}  (bins={y1-y0}, ext_frames={n_vf})")

        # Partition A
        sig_la = _icqt(cqt_l * mask_ext, view_freqs, y0, n_bins, n_vf,
                        ola_norms=ola_norms)
        # Partition B (complement)
        sig_lb = _icqt(cqt_l * (1.0 - mask_ext), view_freqs, y0, n_bins,
                        n_vf, ola_norms=ola_norms)

        if is_stereo:
            real_r = np.asarray(raw_arrays["real_right"][y0:y1, ex0:ex1],
                                dtype=np.float64)
            imag_r = np.asarray(raw_arrays["imag_right"][y0:y1, ex0:ex1],
                                dtype=np.float64)
            cqt_r = self._apply_phase_mode(real_r + 1j * imag_r, phase_mode,
                                             view_freqs, y0, n_bins)
            sig_ra = _icqt(cqt_r * mask_ext, view_freqs, y0, n_bins, n_vf,
                            ola_norms=ola_norms)
            sig_rb = _icqt(cqt_r * (1.0 - mask_ext), view_freqs, y0,
                            n_bins, n_vf, ola_norms=ola_norms)
        else:
            sig_ra, sig_rb = sig_la, sig_lb

        # Extract interior / trim
        n_view_samples = (x1 - x0) * self.hop_length
        n_samples = int(dur * sr)

        def _trim(sig: np.ndarray) -> np.ndarray:
            s = sig[trim_start:trim_start + n_view_samples][:n_samples]
            if len(s) < n_samples:
                s = np.pad(s, (0, n_samples - len(s)))
            return s

        out_a = np.column_stack([_trim(sig_la), _trim(sig_ra)])
        out_b = np.column_stack([_trim(sig_lb), _trim(sig_rb)])

        return (out_a, out_b), dur, vt0, vt1

    # ------------------------------------------------------------------
    # Filterbank-from-audio synthesis
    # ------------------------------------------------------------------
    def synthesise_filterbank_from_audio(
            self, audio_path: str, sr: int,
            t0: float, t1: float,
            f_lo: float, f_hi: float,
            n_bands: int = 16,
            is_stereo: bool = True,
            normalize: bool = True,
    ) -> tuple[np.ndarray, float, float, float] | None:
        """Load the original audio, extract a time window, decompose
        with *n_bands* log-spaced bandpass filters covering [f_lo, f_hi],
        and return ``(stereo_i16, duration, t0, t1)``.

        This bypasses CQT entirely — it goes back to the source audio and
        isolates the viewport's time/frequency region via a fresh filterbank.
        """
        dur = t1 - t0
        if dur <= 0 or f_hi <= f_lo:
            return None

        # Load audio
        sr_file, raw = _load_audio(audio_path)
        if np.issubdtype(raw.dtype, np.integer):
            max_abs = max(abs(np.iinfo(raw.dtype).min), np.iinfo(raw.dtype).max)
            raw = raw.astype(np.float64) / max_abs
        elif raw.dtype != np.float64:
            raw = raw.astype(np.float64)

        if raw.ndim == 2:
            left = raw[:, 0]
            right = raw[:, 1]
        else:
            left = raw
            right = raw

        # Extract time window (with small margin for filter settling)
        margin = min(0.05, dur * 0.1)  # 50ms or 10% of duration
        s0 = max(0, int((t0 - margin) * sr_file))
        s1 = min(len(left), int((t1 + margin) * sr_file))
        margin_pre = int((t0 - margin) * sr_file) - s0 if t0 - margin >= 0 else int(t0 * sr_file) - s0
        left_win = left[s0:s1].copy()
        right_win = right[s0:s1].copy()

        # Build log-spaced crossover frequencies
        f_lo_safe = max(f_lo, 1.0)
        f_hi_safe = min(f_hi, sr_file / 2.0 * 0.95)
        if f_hi_safe <= f_lo_safe or n_bands < 1:
            return None

        # Build n_bands bands spanning exactly [f_lo, f_hi].
        # geomspace gives n_bands+1 edge frequencies; adjacent pairs
        # form the bands.  Nothing outside this range is included.
        edges = np.geomspace(f_lo_safe, f_hi_safe, n_bands + 1)
        bands: list[BandDef] = []
        for i in range(n_bands):
            b = BandDef(fmin=float(edges[i]), fmax=float(edges[i + 1]))
            b.label = b.auto_label(sr_file)
            bands.append(b)

        print(f"Filterbank synthesis: {n_bands} bands, "
              f"{f_lo_safe:.0f}-{f_hi_safe:.0f} Hz, {dur:.2f}s ...")

        fb = FilterBankDecomposition(sr_file)
        fb.compute(left_win, bands)
        sig_l = fb.reconstruct()

        if is_stereo and left is not right:
            fb_r = FilterBankDecomposition(sr_file)
            fb_r.compute(right_win, bands)
            sig_r = fb_r.reconstruct()
        else:
            sig_r = sig_l

        # Trim out the margin
        actual_s0 = int(t0 * sr_file) - s0
        actual_s1 = int(t1 * sr_file) - s0
        actual_s0 = max(0, actual_s0)
        actual_s1 = min(len(sig_l), actual_s1)
        sig_l = sig_l[actual_s0:actual_s1].astype(np.float64)
        sig_r = sig_r[actual_s0:actual_s1].astype(np.float64)

        n_samples = len(sig_l)
        if n_samples == 0:
            return None

        # Fade edges
        fade_samples = min(int(0.005 * sr_file), n_samples // 4)
        if fade_samples > 1:
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            sig_l[:fade_samples] *= fade_in
            sig_l[-fade_samples:] *= fade_out
            sig_r[:fade_samples] *= fade_in
            sig_r[-fade_samples:] *= fade_out

        if normalize:
            peak = max(np.abs(sig_l).max(), np.abs(sig_r).max(), 1e-12)
            sig_l = sig_l / peak * 0.8
            sig_r = sig_r / peak * 0.8
        left_i16 = (sig_l * 32767).clip(-32768, 32767).astype(np.int16)
        right_i16 = (sig_r * 32767).clip(-32768, 32767).astype(np.int16)
        stereo = np.column_stack([left_i16, right_i16])
        return stereo, dur, t0, t1

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
                            normalize: bool = True,
                            phase_mode: str = "original") -> None:
        self.stop()
        result = self.synthesise_to_array(
            raw_arrays, freqs, times, view_x0, view_x1,
            view_y0, view_y1, n_frames, n_bins,
            t_start, t_dur, sr, is_stereo, normalize,
            phase_mode=phase_mode)
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
        _has_fb = (analysis_dir is not None and
                   os.path.isdir(os.path.join(analysis_dir, "filterbank")))
        _has_wvlt = (analysis_dir is not None and
                     os.path.isdir(os.path.join(analysis_dir, "wavelet")))
        self.field_sources = _build_field_sources(
            self._has_onset, has_fb=_has_fb, has_wvlt=_has_wvlt)
        self.field_configs = [FieldConfig() for _ in self.field_sources]
        self.global_defaults = GlobalDefaults()

        for i, fs in enumerate(self.field_sources):
            if fs.name == "mag_left":
                self.field_configs[i].target = 0   # Ch1
            elif fs.name == "phase_left":
                self.field_configs[i].target = 1   # Ch2
            elif fs.name == "mag_right":
                self.field_configs[i].target = 2   # Ch3
            elif fs.name == "phase_right":
                self.field_configs[i].target = 3   # Ch4

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
        self.grid_style_idx: int = 0   # 0=Subtle, 1=Bold, 2=None
        self.fb_label_mode_idx: int = 1  # 0=Range, 1=Mid+BW (default mid)
        self.display_mode: str = "tf"   # "tf" | "ts"
        self.tf_source: str = "cqt"     # "cqt" | "fb"
        self.axis_margin_left = 200
        self.axis_margin_right = 200
        self.axis_margin_top = 32
        self.axis_margin_bottom = 32
        self.axis_margin_amp = 60

        # Subplot tiling
        self.subplots_tf_cqt: list[SubPlot] = [SubPlot("spectrogram", "Spectrogram")]
        self.subplots_tf_fb: list[SubPlot] = [
            SubPlot("fb_magnitude", "FB Magnitude"),
            SubPlot("fb_phase", "FB Phase", visible=False),
        ]
        self.subplots_ts: list[SubPlot] = [SubPlot("waveform", "Waveform")]
        self.subplots_tsc: list[SubPlot] = [SubPlot("scalogram", "Scalogram")]
        self.subplots_overlay: list[SubPlot] = [SubPlot("overlay", "Overlay")]
        self.tiling_strategy: TilingStrategy = TilingStrategy.COLUMNS

        # Overlay composite config
        self.overlay_config = OverlayConfig()
        self._overlay_tex_id: int = 0
        self._overlay_dirty: bool = True
        self._overlay_gd_snap: tuple = ()  # snapshot of GD values at last bake

        # Wavelet time-scale cache
        self._wv_coeffs: list[np.ndarray] = []    # [approx, d1, d2, ...] (DWT)
        self._wv_meta: dict = {}
        self._wv_n_frames: int = 0                 # tex width
        self._wv_n_levels: int = 0                  # tex height (rows)
        self._wv_shader_tex_ids: dict[str, int] = {}
        self._wv_dirty: bool = True
        # CWT-specific cache (populated when meta["type"] == "cwt")
        self._wv_W: np.ndarray | None = None       # complex (n_scales, n_frames)
        self._wv_freqs: np.ndarray | None = None   # (n_scales,) Hz high-to-low

        if analysis_dir is not None:
            self._load_wavelet_data()

        # Filterbank-as-TF cache (per-band ragged arrays)
        self._fb_bands: list[BandDef] = []
        self._fb_mags: list[np.ndarray] = []   # per-band 1-D mag arrays
        self._fb_phases: list[np.ndarray] = []
        self._fb_hops: list[int] = []          # per-band hops
        self._fb_shader_tex_ids: dict[str, int] = {}
        self._fb_dirty: bool = True

        # TS filterbank band cache
        self._ts_fb: FilterBankDecomposition | None = None
        self._ts_fb_dir: str | None = None  # analysis_dir it was loaded from

        self._rect_start: tuple[int, int] | None = None
        self._rect_end: tuple[int, int] | None = None
        self._is_panning = False
        self._pan_last: tuple[int, int] | None = None

        # Region tool state
        self._tool_mode: str = "zoom"   # "zoom" or "region"
        self._region_data: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) data
        self._region_octave_shift: float = 0.0
        self._region_time_shift: float = 0.0
        self._region_panel_rect: pygame.Rect | None = None
        self._region_hitboxes: dict[str, pygame.Rect] = {}

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

        # Broadening state (Hybrid partial → full dataset)
        self._broadening: bool = False
        self._broadening_thread: threading.Thread | None = None
        self._broaden_pending_synth: str | None = None  # "cqt" or "fb_audio"

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

        print("Computing normalization ranges ...", end="", flush=True)
        t0_norm = time.monotonic()
        abs_maxes = []
        norm_keys = ("real_left", "real_right", "imag_left", "imag_right")
        for ki, key in enumerate(norm_keys):
            arr = self._raw_arrays[key]
            hi = float(arr.max())
            lo = float(arr.min())
            abs_maxes.append(max(abs(hi), abs(lo)))
            print(f"\r  Normalization scan: {ki + 1}/{len(norm_keys)} arrays",
                  end="", flush=True)
        ri_abs_max = max(abs_maxes)
        print(f"\r  Normalization scan: done ({time.monotonic() - t0_norm:.1f}s)")
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
            "wvlt_approx":      (0.0, 1.0),
            "wvlt_detail":      (0.0, 1.0),
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

    def _resolve_field_array(self, field_name: str) -> np.ndarray | None:
        """Return the full 2-D array for *field_name* at native dtype.

        Handles raw arrays stored directly in ``_raw_arrays`` as well as
        derived fields (``mag_left``, ``mag_right``, ``mag_similarity``,
        ``phase_similarity``) that are computed from the raw complex
        components.

        Returns ``None`` when the field cannot be resolved.
        """
        if field_name in self._raw_arrays:
            return self._raw_arrays[field_name]

        if field_name == "mag_left":
            r = self._raw_arrays["real_left"]
            i = self._raw_arrays["imag_left"]
            return np.sqrt(np.asarray(r, dtype=r.dtype) ** 2 +
                           np.asarray(i, dtype=i.dtype) ** 2)
        if field_name == "mag_right":
            r = self._raw_arrays["real_right"]
            i = self._raw_arrays["imag_right"]
            return np.sqrt(np.asarray(r, dtype=r.dtype) ** 2 +
                           np.asarray(i, dtype=i.dtype) ** 2)
        if field_name == "mag_similarity":
            ml = self._resolve_field_array("mag_left")
            mr = self._resolve_field_array("mag_right")
            if ml is None or mr is None:
                return None
            peak = np.maximum(ml, mr)
            return np.where(peak < EPS, 1.0,
                            np.minimum(ml, mr) / peak)
        if field_name == "phase_similarity":
            pl = self._raw_arrays.get("phase_left")
            pr = self._raw_arrays.get("phase_right")
            if pl is None or pr is None:
                return None
            return (1.0 + np.cos(
                np.asarray(pl, dtype=pl.dtype) -
                np.asarray(pr, dtype=pr.dtype))) / 2.0

        return None

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
        if self._mipmap is None:
            self._scan_complete = True
            return

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
        # For hybrid, convert unified Y to CQT bin indices
        cqt_off = self._hybrid_cross_bin() if self.tf_source == "hybrid" else 0.0
        cqt_skip = self._hybrid_cqt_skip_bins() if self.tf_source == "hybrid" else 0
        y0 = max(0, int(math.floor(self.view_y0 - cqt_off)) + cqt_skip)
        y1 = min(self.n_bins, int(math.ceil(self.view_y1 - cqt_off)) + cqt_skip)
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
        if self._scan_complete or self._scan_params is None or self._mipmap is None:
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
                        # Rank-order norm must be done CPU-side
                        fnorm = 0
                        for fi, fsi in enumerate(self.field_sources):
                            if fsi.name == field_name:
                                cfg = self.field_configs[fi]
                                fnorm = cfg.norm_mode if not cfg.use_global else self.global_defaults.norm_mode
                                break
                        if fnorm == 2:
                            flat = data.ravel()
                            n = flat.size
                            if n > 0:
                                order = flat.argsort().argsort()
                                data = (order.astype(np.float32) / max(n - 1, 1)).reshape(data.shape)
                            else:
                                data = np.zeros_like(data, dtype=np.float32)
                        channels.append(data.astype(np.float32))

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
                sw, sh, targets, self.tf_source)

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
        self.synth_panel.on_synth_icqt = self._toggle_synth
        self.synth_panel.on_synth_fb = self._submit_fb_audio_job
        self.synth_panel.on_synth_hybrid = self._submit_hybrid_synth_job
        self.synth_panel.on_synth_fb_tf = self._submit_fb_tf_job
        self.synth_panel.on_synth_wavelet = self._submit_wavelet_synth_job
        self.synth_panel.on_synth_transplant = self._submit_transplant_job
        self.synth_panel.on_synth_hybrid_cwt = self._submit_hybrid_cwt_synth_job
        self.synth_panel._get_phase_mode = lambda: self.phase_mode
        self.synth_panel._set_phase_mode = lambda v: setattr(self, "phase_mode", v)
        self.synth_panel._get_normalize = lambda: self.normalize_loudness
        self.synth_panel._set_normalize = lambda v: setattr(self, "normalize_loudness", v)
        self.synth_panel._get_fb_n_bands = lambda: self.fb_n_bands
        self.synth_panel._set_fb_n_bands = lambda v: setattr(self, "fb_n_bands", v)
        self.synth_panel._has_audio = lambda: self.audio_path is not None
        self.synth_panel._has_synth = lambda: self.synth is not None
        self.synth_panel._has_wavelet = lambda: bool(self._wv_coeffs) or self._wv_W is not None

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
        self.image_panel = ImageInputPanel(
            side="left",
            image_dir=os.path.join(_app_root, "input_images"))
        self.dock.register("images", self.image_panel)
        self.signal_panel = SignalGeneratorPanel(
            side="left", output_dir=input_dir)
        self.signal_panel.on_generated = lambda path: (
            self.file_panel._refresh_wavs() if self.file_panel else None)
        self.dock.register("signal_gen", self.signal_panel)
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
            taps_val = int(_RESAMPLE_TAPS[self.resample_taps_idx])
            prec_val = _RESAMPLE_PRECISIONS[self.resample_precision_idx]
            self.synth = ViewportSynthPlayer(sr_val, hop_val, bpo_val,
                                             taps_val, prec_val)
        self.phase_mode: str = "original"
        self.normalize_loudness: bool = True
        self.fb_n_bands: int = 16

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
                    self._gui_tex = 0

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

            # Check for async filterbank subband load completion
            self._check_ts_fb_ready()

            # Auto-scroll to keep cursor visible
            if self._loaded and self.player and self.player.playing:
                cur_frame = self._time_to_frame(self.player.get_position())
                vw = self.view_x1 - self.view_x0
                if cur_frame < self.view_x0 or cur_frame > self.view_x1:
                    self.view_x0 = cur_frame - vw * 0.25
                    self.view_x1 = self.view_x0 + vw
                    self._clamp_view()

            # Check for broadening completion → reload and submit synth
            if (self._broaden_pending_synth is not None
                    and not self._broadening
                    and self._broadening_thread is not None
                    and not self._broadening_thread.is_alive()):
                mode = self._broaden_pending_synth
                self._broaden_pending_synth = None
                self._broadening_thread = None
                # Reload the expanded dataset
                if self.analysis_dir:
                    self._load_analysis_folder(self.analysis_dir)
                # Re-submit the synth job now that data is complete
                if mode == "cqt":
                    self._submit_synth_job()
                elif mode == "fb_audio":
                    self._submit_fb_audio_job()

            # Progressive viewport texture scan (CQT / Hybrid TF)
            if self._loaded and self.display_mode == "tf" and self.tf_source in ("cqt", "hybrid"):
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
        def _subplot_toggles(src: list[SubPlot], mode: str) -> list[MenuItem]:
            items: list[MenuItem] = []
            for sp in src:
                def _action(s=sp, m=mode):
                    s.visible = not s.visible
                    if s.visible:
                        self._set_display_mode(m)
                items.append(MenuItem(
                    sp.label,
                    action=_action,
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
                MenuItem("Time-Scale",
                         action=lambda: self._set_display_mode("tsc"),
                         toggle_state=lambda: self.display_mode == "tsc",
                         radio_group="display_mode"),
                MenuItem("Time-Signal",
                         action=lambda: self._set_display_mode("ts"),
                         toggle_state=lambda: self.display_mode == "ts",
                         radio_group="display_mode"),
                MenuItem("Overlay",
                         action=lambda: self._set_display_mode("overlay"),
                         toggle_state=lambda: self.display_mode == "overlay",
                         radio_group="display_mode"),
            ]),
            MenuItem("TF Source", children=[
                MenuItem("CQT",
                         action=lambda: self._set_tf_source("cqt"),
                         toggle_state=lambda: self.tf_source == "cqt",
                         radio_group="tf_source"),
                MenuItem("Filterbank",
                         action=lambda: self._set_tf_source("fb"),
                         toggle_state=lambda: self.tf_source == "fb",
                         radio_group="tf_source"),
                MenuItem("Hybrid (FB low / CQT high)",
                         action=lambda: self._set_tf_source("hybrid"),
                         toggle_state=lambda: self.tf_source == "hybrid",
                         radio_group="tf_source"),
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
            MenuItem("CQT Subplots", children=_subplot_toggles(self.subplots_tf_cqt, "tf")),
            MenuItem("FB Subplots", children=_subplot_toggles(self.subplots_tf_fb, "tf")),
            MenuItem("TS Subplots", children=_subplot_toggles(self.subplots_ts, "ts")),
            MenuItem("Overlay Subplots", children=_subplot_toggles(self.subplots_overlay, "overlay")),
            MenuItem("-"),
            MenuItem("-"),
            MenuItem("Right-Drag Tool", children=[
                MenuItem("Zoom",
                         action=lambda: self._set_tool_mode("zoom"),
                         toggle_state=lambda: self._tool_mode == "zoom",
                         radio_group="tool_mode"),
                MenuItem("Region Select",
                         action=lambda: self._set_tool_mode("region"),
                         toggle_state=lambda: self._tool_mode == "region",
                         radio_group="tool_mode"),
            ]),
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
            MenuItem("Grid Lines", children=[
                MenuItem("Subtle",
                         action=lambda: setattr(self, "grid_style_idx", 0),
                         toggle_state=lambda: self.grid_style_idx == 0,
                         radio_group="grid_style"),
                MenuItem("Bold",
                         action=lambda: setattr(self, "grid_style_idx", 1),
                         toggle_state=lambda: self.grid_style_idx == 1,
                         radio_group="grid_style"),
                MenuItem("None",
                         action=lambda: setattr(self, "grid_style_idx", 2),
                         toggle_state=lambda: self.grid_style_idx == 2,
                         radio_group="grid_style"),
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

    def _rebuild_menus(self) -> None:
        """Tear down and rebuild all menus (e.g. after TS subplots change)."""
        self.menu_bar._menus.clear()
        self.menu_bar._title_rects.clear()
        self.menu_bar._close_all()
        self._build_menus()

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
        if mode == "ts":
            self._ensure_ts_subplots()
        elif mode == "tsc":
            if self._wv_dirty:
                if not self._wv_coeffs:
                    self._load_wavelet_data()
                self._compute_wavelet_scalogram()
        elif mode == "overlay":
            # Ensure all backing data is loaded
            if self._fb_dirty:
                self._compute_fb_envelopes()
            if self._wv_dirty:
                if not self._wv_coeffs:
                    self._load_wavelet_data()
                self._compute_wavelet_scalogram()
            self._overlay_dirty = True
        elif mode == "tf" and self.tf_source == "fb" and self._fb_dirty:
            self._compute_fb_envelopes()
        y_min, y_max = self._y_data_range()
        self.view_y0 = y_min
        self.view_y1 = y_max
        # Reset X viewport for wavelet (different frame count)
        if mode == "tsc" and self._wv_n_frames > 0:
            self.view_x0 = 0.0
            self.view_x1 = float(self._wv_n_frames)
        elif mode != "tsc" and self.n_frames > 0:
            self.view_x0 = 0.0
            self.view_x1 = float(self.n_frames)

    def _set_tool_mode(self, mode: str) -> None:
        self._tool_mode = mode
        if mode == "zoom":
            self._region_data = None
            self._region_panel_rect = None
            self._region_hitboxes = {}

    def _set_tf_source(self, source: str) -> None:
        """Switch TF content source (cqt / fb / hybrid) and reset Y viewport."""
        if source == self.tf_source:
            return
        self.tf_source = source
        if source in ("fb", "hybrid") and self._fb_dirty:
            self._compute_fb_envelopes()
        # Ensure we're in TF mode when switching source
        if self.display_mode != "tf":
            self.display_mode = "tf"
        y_min, y_max = self._y_data_range()
        self.view_y0 = y_min
        self.view_y1 = y_max

    def _ensure_ts_subplots(self) -> None:
        """Populate TS subplots from filterbank band WAVs on disk.

        The heavy WAV loading runs in a background thread so the UI
        stays responsive.  A loading message is printed until done.
        """
        if not self.analysis_dir:
            return
        # Avoid reloading if same folder
        if self._ts_fb is not None and self._ts_fb_dir == self.analysis_dir:
            return
        # Avoid launching a second background load
        if (getattr(self, '_ts_fb_loading', False)
                and getattr(self, '_ts_fb_loading_dir', None) == self.analysis_dir):
            return
        self._ts_fb_loading = True
        self._ts_fb_loading_dir = self.analysis_dir
        target_dir = self.analysis_dir

        def _bg_load() -> None:
            fb = FilterBankDecomposition.load(target_dir)
            # Store result; the main loop picks it up next frame
            self._ts_fb_pending = (target_dir, fb)
            self._ts_fb_loading = False

        t = threading.Thread(target=_bg_load, daemon=True)
        t.start()
        print("Loading filterbank subbands in background ...")

    def _check_ts_fb_ready(self) -> None:
        """Finalize TS subplot setup if the background load finished."""
        pending = getattr(self, '_ts_fb_pending', None)
        if pending is None:
            return
        target_dir, fb = pending
        self._ts_fb_pending = None
        if fb is None or not fb.subbands:
            return
        self._ts_fb = fb
        self._ts_fb_dir = target_dir
        # Build subplots: one per band + optionally full waveform
        new_ts: list[SubPlot] = []
        if self.player:
            new_ts.append(SubPlot("waveform", "Waveform"))
        for i, band in enumerate(fb.bands):
            label = band.label or f"Band {i}"
            new_ts.append(SubPlot(f"fb_band_{i}", label))
        self.subplots_ts = new_ts
        self._rebuild_menus()
        print("Filterbank subbands ready.")

    # ---- Subplot helpers --------------------------------------------------

    def _active_subplots(self) -> list[SubPlot]:
        """Return visible subplots for the current display mode."""
        if self.display_mode == "tf":
            if self.tf_source == "fb":
                src = self.subplots_tf_fb
            else:
                src = self.subplots_tf_cqt
        elif self.display_mode == "tsc":
            src = self.subplots_tsc
        elif self.display_mode == "overlay":
            src = self.subplots_overlay
        else:
            src = self.subplots_ts
        return [sp for sp in src if sp.visible]

    def _render_subplot_content(self, subplot: SubPlot,
                                sx: int, sy: int, sw: int, sh: int) -> None:
        """Dispatch rendering for a single subplot into its cell rect."""
        if self.display_mode == "tf":
            if self.tf_source == "cqt":
                if subplot.key == "spectrogram":
                    self._render_tf(sx, sy, sw, sh)
            elif self.tf_source == "hybrid":
                if subplot.key == "spectrogram":
                    self._render_hybrid_tf(sx, sy, sw, sh)
            else:
                if subplot.key == "fb_magnitude":
                    self._render_fb_tf(sx, sy, sw, sh, "mag")
                elif subplot.key == "fb_phase":
                    self._render_fb_tf(sx, sy, sw, sh, "phase")
        elif self.display_mode == "tsc":
            if subplot.key == "scalogram":
                self._render_tsc(sx, sy, sw, sh)
        elif self.display_mode == "overlay":
            if subplot.key == "overlay":
                self._render_overlay(sx, sy, sw, sh)
        elif self.display_mode == "ts":
            if subplot.key == "waveform":
                self._render_ts(sx, sy, sw, sh)
            elif subplot.key.startswith("fb_band_"):
                bi = int(subplot.key.split("_")[-1])
                self._render_ts_band(sx, sy, sw, sh, bi)

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
        import glob as _glob

        # Determine what data is available
        if version_hash:
            cqt_target = os.path.join(folder_path,
                                      f"cqt_data_{version_hash}.npz")
        else:
            cqt_target = os.path.join(folder_path, "cqt_data.npz")
        has_cqt = os.path.isfile(cqt_target)
        if not has_cqt:
            candidates = _glob.glob(
                os.path.join(folder_path, "cqt_data_*.npz"))
            if candidates:
                has_cqt = True
            elif os.path.isfile(os.path.join(folder_path, "cqt_data.npz")):
                has_cqt = True
        has_fb = os.path.isdir(os.path.join(folder_path, "filterbank"))
        has_wv = os.path.isfile(os.path.join(folder_path, "wavelet",
                                              "wavelet_data.npz"))

        if not has_cqt and not has_fb and not has_wv:
            print(f"No CQT, filterbank, or wavelet data in {folder_path}")
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

        if has_cqt:
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
        else:
            # No CQT: derive viewport from FB or wavelet data
            self._npz = None
            self._raw_arrays = {}
            self._has_onset = False
            self._settings_hash = ""
            self.is_stereo = False
            self._freqs = np.zeros(0, dtype=np.float32)
            self._semitone_mask = np.zeros(0, dtype=bool)
            self._note_names = []
            self._pad_influence = None
            self._field_ranges = {}
            self._mipmap = None

            # Derive n_bins/n_frames/t_dur from filterbank metadata + envelopes
            fb_dir = os.path.join(folder_path, "filterbank")
            fb_meta_path = os.path.join(fb_dir, "filterbank_meta.json")
            fb_env_path = os.path.join(fb_dir, "fb_envelopes.npz")
            fb_sr = 44100
            fb_n_bands = 0
            if os.path.isfile(fb_meta_path):
                import json as _json
                with open(fb_meta_path, "r") as f:
                    _fbm = _json.load(f)
                fb_sr = int(_fbm.get("sr", 44100))
                fb_n_bands = int(_fbm.get("n_bands", 0))

            fb_max_frames = 0
            fb_total_samples = 0
            if os.path.isfile(fb_env_path):
                _env = np.load(fb_env_path)
                _env_n = int(_env.get("n_bands", 0)) if "n_bands" in _env else 0
                _env_hops = _env["hops"].tolist() if "hops" in _env else []
                for _bi in range(_env_n):
                    _mk = f"mag_{_bi}"
                    if _mk in _env:
                        _blen = len(_env[_mk])
                        if _blen > fb_max_frames:
                            fb_max_frames = _blen
                        if _bi < len(_env_hops):
                            _samp = _blen * _env_hops[_bi]
                            if _samp > fb_total_samples:
                                fb_total_samples = _samp
                if fb_n_bands == 0:
                    fb_n_bands = _env_n

            self.n_bins = fb_n_bands
            self.n_frames = fb_max_frames
            if fb_total_samples > 0 and fb_sr > 0:
                self.t_dur = fb_total_samples / fb_sr
            else:
                self.t_dur = 0.0

            # Wavelet-only fallback: derive t_dur from wavelet metadata
            if self.t_dur == 0.0 and has_wv:
                wv_meta_path = os.path.join(folder_path, "wavelet",
                                            "wavelet_meta.json")
                wv_npz_path = os.path.join(folder_path, "wavelet",
                                           "wavelet_data.npz")
                wv_sr = fb_sr
                if os.path.isfile(wv_meta_path):
                    import json as _json
                    with open(wv_meta_path, "r") as f:
                        _wvm = _json.load(f)
                    wv_sr = int(_wvm.get("sr", 44100))
                if os.path.isfile(wv_npz_path):
                    _wnpz = np.load(wv_npz_path, allow_pickle=False)
                    _n_lev = int(_wnpz["n_levels"])
                    # Finest detail has len ≈ original_samples / 2
                    # Reconstruct: finest_len * 2 ≈ original samples
                    _finest_key = f"detail_{_n_lev}"
                    if _finest_key in _wnpz:
                        _total_samp = len(_wnpz[_finest_key]) * 2
                        self.t_dur = _total_samp / wv_sr if wv_sr > 0 else 0.0

            self.t_start = 0.0
            self.t_end = self.t_dur
            self.times = np.linspace(
                0.0, self.t_dur, self.n_frames,
                dtype=np.float32) if self.n_frames > 0 else np.zeros(
                    0, dtype=np.float32)

        # Reset viewport
        self.view_x0 = 0.0
        self.view_y0 = 0.0
        self.view_x1 = float(self.n_frames)
        self.view_y1 = float(self.n_bins)

        # Reset scan state
        self._front_tex_ids = {}
        self._back_tex_ids = {}
        self._back_buffers = {}
        self._fb_shader_tex_ids = {}
        self._fb_dirty = True
        self._wv_coeffs = []
        self._wv_meta = {}
        self._wv_n_frames = 0
        self._wv_n_levels = 0
        for old_id in self._wv_shader_tex_ids.values():
            glDeleteTextures([old_id])
        self._wv_shader_tex_ids = {}
        self._wv_dirty = True
        self._load_wavelet_data()
        self._scan_key = None
        self._scan_complete = True

        # Rebuild field sources
        _has_fb = os.path.isdir(
            os.path.join(self.analysis_dir, "filterbank")) if self.analysis_dir else False
        _has_wvlt = os.path.isdir(
            os.path.join(self.analysis_dir, "wavelet")) if self.analysis_dir else False
        self.field_sources = _build_field_sources(
            self._has_onset, has_fb=_has_fb, has_wvlt=_has_wvlt)
        self.field_configs = [FieldConfig() for _ in self.field_sources]
        for i, fs in enumerate(self.field_sources):
            if fs.name == "mag_left":
                self.field_configs[i].target = 0   # Ch1
            elif fs.name == "phase_left":
                self.field_configs[i].target = 1   # Ch2
            elif fs.name == "mag_right":
                self.field_configs[i].target = 2   # Ch3
            elif fs.name == "phase_right":
                self.field_configs[i].target = 3   # Ch4
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
        if self._npz is not None:
            sr_val = int(self._npz["sr"]) if "sr" in self._npz else 44100
            hop_val = int(self._npz["hop_length"]) if "hop_length" in self._npz else 512
            bpo_val = int(self._npz["bins_per_octave"]) if "bins_per_octave" in self._npz else 1200
        else:
            # No CQT: read sr from filterbank or wavelet metadata
            sr_val, hop_val, bpo_val = 44100, 512, 1200
            fb_meta_path = os.path.join(folder_path, "filterbank",
                                        "filterbank_meta.json")
            if os.path.isfile(fb_meta_path):
                import json as _json
                with open(fb_meta_path, "r") as f:
                    fb_meta = _json.load(f)
                sr_val = int(fb_meta.get("sr", 44100))
            else:
                wv_meta_path = os.path.join(folder_path, "wavelet",
                                            "wavelet_meta.json")
                if os.path.isfile(wv_meta_path):
                    import json as _json
                    with open(wv_meta_path, "r") as f:
                        wv_meta = _json.load(f)
                    sr_val = int(wv_meta.get("sr", 44100))
        taps_val = int(_RESAMPLE_TAPS[self.resample_taps_idx])
        prec_val = _RESAMPLE_PRECISIONS[self.resample_precision_idx]
        self.synth = ViewportSynthPlayer(sr_val, hop_val, bpo_val,
                                         taps_val, prec_val)

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
        self._fb_shader_tex_ids = {}
        self._fb_dirty = True
        self._wv_coeffs = []
        self._wv_meta = {}
        self._wv_n_frames = 0
        self._wv_n_levels = 0
        for old_id in self._wv_shader_tex_ids.values():
            glDeleteTextures([old_id])
        self._wv_shader_tex_ids = {}
        self._wv_dirty = True
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

    # ------------------------------------------------------------------
    # Hybrid broadening — fill missing CQT/FB and persist to disk
    # ------------------------------------------------------------------

    def _is_cqt_partial(self) -> bool:
        """True if the loaded CQT likely starts at the hybrid crossover
        and a filterbank directory also exists (indicating Hybrid analysis)."""
        if not self._loaded or self._npz is None or not self.analysis_dir:
            return False
        sr = int(self._npz["sr"]) if "sr" in self._npz else 44100
        bpo = int(self._npz["bins_per_octave"]) if "bins_per_octave" in self._npz else 1200
        hop = int(self._npz["hop_length"]) if "hop_length" in self._npz else 512
        xf = hybrid_crossover_freq(bpo, hop, sr)
        f0 = float(self._freqs[0])
        if f0 < xf * 0.8:
            return False
        fb_dir = os.path.join(self.analysis_dir, "filterbank")
        return os.path.isdir(fb_dir)

    def _is_fb_partial(self) -> bool:
        """True if the loaded FB only covers below the hybrid crossover
        and CQT data also exists (indicating Hybrid analysis)."""
        if not self.analysis_dir or self._npz is None:
            return False
        sr = int(self._npz["sr"]) if "sr" in self._npz else 44100
        bpo = int(self._npz["bins_per_octave"]) if "bins_per_octave" in self._npz else 1200
        hop = int(self._npz["hop_length"]) if "hop_length" in self._npz else 512
        xf = hybrid_crossover_freq(bpo, hop, sr)
        result = FilterBankDecomposition.load_meta(self.analysis_dir)
        if result is None:
            return False
        _meta, bands = result
        if not bands:
            return False
        top = max(b["fmax"] for b in bands)
        return top <= xf * 1.2

    def _broaden_cqt(self) -> None:
        """Re-run CQT analysis with the full frequency range, replacing
        the partial (Hybrid) NPZ with a complete one."""
        if not self.audio_path or not self.analysis_dir or self._npz is None:
            return
        import subprocess as _sp, sys as _sys
        bpo = int(self._npz["bins_per_octave"]) if "bins_per_octave" in self._npz else 1200
        hop = int(self._npz["hop_length"]) if "hop_length" in self._npz else 512
        fmin = self.file_panel.cqt_fmin if self.file_panel else 16.35
        fmax = self.file_panel.cqt_fmax if self.file_panel else 19912.13
        cmd = [
            _sys.executable, "-m", "bass_analysis",
            self.audio_path,
            "--hop-length", str(hop),
            "--bins-per-octave", str(bpo),
            "--cqt-fmin", str(fmin),
            "--cqt-fmax", str(fmax),
            "--outdir", self.analysis_dir,
        ]
        print(f"Broadening CQT: fmin={fmin:.1f} Hz (full range) ...")
        _sp.check_call(cmd,
                       cwd=os.path.dirname(os.path.abspath(__file__)))
        print("CQT broadening complete.")

    def _broaden_fb(self) -> None:
        """Re-run FB decomposition with the full frequency range, replacing
        the partial (Hybrid) filterbank with a complete one."""
        if not self.audio_path or not self.analysis_dir:
            return
        sr_file, raw = _load_audio(self.audio_path)
        def _to_f32(arr: np.ndarray) -> np.ndarray:
            if np.issubdtype(arr.dtype, np.integer):
                return arr.astype(np.float32) / max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)
            return arr.astype(np.float32)
        if raw.ndim == 2:
            mono = _to_f32(raw.mean(axis=1))
            left_ch = _to_f32(raw[:, 0])
            right_ch = _to_f32(raw[:, 1])
            is_stereo = True
        else:
            mono = _to_f32(raw)
            is_stereo = False
        ftype_idx = self.file_panel.fb_filter_type_idx if self.file_panel else 0
        ftype = _FILTER_TYPES[ftype_idx]
        bpo = self.file_panel.fb_bpo if self.file_panel else 12
        fmin = self.file_panel.fb_fmin if self.file_panel else 16.35
        fmax = self.file_panel.fb_fmax if self.file_panel else 19912.13
        print(f"Broadening FB: {fmin:.1f}-{fmax:.1f} Hz (full range) ...")
        bands = FilterBankDecomposition.bands_from_log_spacing(
            bpo, fmin, fmax, sr_file)  # no ceiling
        fb = FilterBankDecomposition(sr_file, ftype)
        fb.compute_and_save(mono, bands, self.analysis_dir, sr_file,
                            wav_path=self.audio_path)
        # Always produce L+R envelopes (mono input → L=R)
        fb_dir = os.path.join(self.analysis_dir, "filterbank")
        if is_stereo:
            ch_pairs = [("left", left_ch), ("right", right_ch)]
        else:
            ch_pairs = [("left", mono)]
        for ch_label, ch_sig in ch_pairs:
            fb_ch = FilterBankDecomposition(sr_file, ftype)
            fb_ch.compute(ch_sig, bands)
            mags_ch, phases_ch, hops_ch = fb_ch.compute_envelopes()
            data_ch: dict[str, np.ndarray] = {
                "n_bands": np.int64(len(mags_ch)),
                "hops": np.array(hops_ch, dtype=np.int64),
            }
            for ii, (m, p) in enumerate(zip(mags_ch, phases_ch)):
                data_ch[f"mag_{ii}"] = m
                data_ch[f"phase_{ii}"] = p
            np.savez_compressed(
                os.path.join(fb_dir,
                             f"fb_envelopes_{ch_label}.npz"),
                **data_ch)
        if not is_stereo:
            import shutil
            shutil.copy2(
                os.path.join(fb_dir, "fb_envelopes_left.npz"),
                os.path.join(fb_dir, "fb_envelopes_right.npz"))
        print("FB broadening complete.")

    def _start_broadening(self, synth_mode: str) -> None:
        """Kick off broadening in a background thread.  On completion the
        main loop will reload and auto-submit the requested synth job."""
        if self._broadening:
            return
        self._broadening = True
        self._broaden_pending_synth = synth_mode

        need_cqt = synth_mode == "cqt"
        need_fb = synth_mode == "fb_audio"

        def _worker() -> None:
            try:
                if need_cqt:
                    self._broaden_cqt()
                elif need_fb:
                    self._broaden_fb()
            except Exception as exc:
                print(f"Broadening error: {exc}")
                self._broaden_pending_synth = None
            finally:
                self._broadening = False

        self._broadening_thread = threading.Thread(
            target=_worker, daemon=True)
        self._broadening_thread.start()

    def _submit_synth_job(self) -> None:
        if not self.synth or not self._loaded:
            return
        # Auto-broaden Hybrid partial CQT before iCQT synthesis
        if self._is_cqt_partial():
            if not self._broadening:
                self._start_broadening("cqt")
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100
        label = (f"CQT {self.view_x0:.0f}-{self.view_x1:.0f} / "
                 f"{self.view_y0:.0f}-{self.view_y1:.0f}")
        job = SynthJob(
            job_id=0, label=label, source_type="cqt",
            params={
                "synth": self.synth,
                "raw_arrays": self._raw_arrays,
                "field_resolver": self._resolve_field_array,
                "freqs": self._freqs,
                "times": self.times,
                "view_x0": self.view_x0, "view_x1": self.view_x1,
                "view_y0": self.view_y0, "view_y1": self.view_y1,
                "n_frames": self.n_frames, "n_bins": self.n_bins,
                "t_start": self.t_start, "t_dur": self.t_dur,
                "sr": sr_val, "is_stereo": self.is_stereo,
                "normalize": self.normalize_loudness,
                "phase_mode": self.phase_mode,
                "routing": self.synth_panel.channel_routing if self.synth_panel else None,
                "field_sources": self.field_sources,
                "field_configs": self.field_configs,
                "use_adjusted": self.synth_panel.use_adjusted if self.synth_panel else False,
                "global_defaults": self.global_defaults,
            })
        if self.synth_panel:
            self.synth_panel.submit(job)

    def _submit_fb_audio_job(self) -> None:
        """Submit a filterbank-from-audio synthesis job for the current viewport."""
        if not self.synth or not self._loaded or not self.audio_path:
            return
        # Auto-broaden Hybrid partial FB before FB synthesis
        if self._is_fb_partial():
            if not self._broadening:
                self._start_broadening("fb_audio")
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100

        # Map viewport bin range to Hz
        y0 = max(0, int(math.floor(self.view_y0)))
        y1 = min(self.n_bins, int(math.ceil(self.view_y1)))
        if y1 <= y0:
            return
        f_lo = float(self._freqs[y0])
        f_hi = float(self._freqs[min(y1, len(self._freqs)) - 1])

        # Map viewport frame range to time
        x0 = max(0, int(math.floor(self.view_x0)))
        x1 = min(self.n_frames, int(math.ceil(self.view_x1)))
        t0 = self.t_start + x0 / self.n_frames * self.t_dur
        t1 = self.t_start + x1 / self.n_frames * self.t_dur

        label = (f"FB {f_lo:.0f}-{f_hi:.0f}Hz "
                 f"{t0:.2f}-{t1:.2f}s")
        job = SynthJob(
            job_id=0, label=label, source_type="fb_audio",
            params={
                "synth": self.synth,
                "audio_path": self.audio_path,
                "sr": sr_val,
                "t0": t0, "t1": t1,
                "f_lo": f_lo, "f_hi": f_hi,
                "n_bands": self.fb_n_bands,
                "is_stereo": self.is_stereo,
                "normalize": self.normalize_loudness,
            })
        if self.synth_panel:
            self.synth_panel.submit(job)

    def _submit_hybrid_synth_job(self) -> None:
        """Submit a hybrid synthesis job: FB below crossover + iCQT above."""
        if not self.synth or not self._loaded or not self.audio_path:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100

        cross_hz = hybrid_crossover_freq(
            self.synth.bins_per_octave, self.synth.hop_length, sr_val)

        y0 = max(0, int(math.floor(self.view_y0)))
        y1 = min(self.n_bins, int(math.ceil(self.view_y1)))
        if y1 <= y0:
            return
        f_lo = float(self._freqs[y0])

        label = (f"Hybrid x={cross_hz:.0f}Hz "
                 f"{self.view_x0:.0f}-{self.view_x1:.0f}")
        job = SynthJob(
            job_id=0, label=label, source_type="hybrid",
            params={
                "synth": self.synth,
                "raw_arrays": self._raw_arrays,
                "field_resolver": self._resolve_field_array,
                "freqs": self._freqs,
                "times": self.times,
                "view_x0": self.view_x0, "view_x1": self.view_x1,
                "view_y0": self.view_y0, "view_y1": self.view_y1,
                "n_frames": self.n_frames, "n_bins": self.n_bins,
                "t_start": self.t_start, "t_dur": self.t_dur,
                "sr": sr_val, "is_stereo": self.is_stereo,
                "normalize": self.normalize_loudness,
                "phase_mode": self.phase_mode,
                "audio_path": self.audio_path,
                "cross_hz": cross_hz,
                "f_lo": f_lo,
                "n_bands": self.fb_n_bands,
                "routing": self.synth_panel.channel_routing if self.synth_panel else None,
                "field_sources": self.field_sources,
                "field_configs": self.field_configs,
                "use_adjusted": self.synth_panel.use_adjusted if self.synth_panel else False,
                "global_defaults": self.global_defaults,
            })
        if self.synth_panel:
            self.synth_panel.submit(job)

    def _submit_fb_tf_job(self) -> None:
        """Submit an FB-from-TF synthesis job using routed viewport data."""
        if not self.synth or not self._loaded:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100
        label = (f"FB-TF {self.view_x0:.0f}-{self.view_x1:.0f} / "
                 f"{self.view_y0:.0f}-{self.view_y1:.0f}")
        sp = self.synth_panel
        job = SynthJob(
            job_id=0, label=label, source_type="fb_tf",
            params={
                "synth": self.synth,
                "raw_arrays": self._raw_arrays,
                "field_resolver": self._resolve_field_array,
                "freqs": self._freqs,
                "times": self.times,
                "view_x0": self.view_x0, "view_x1": self.view_x1,
                "view_y0": self.view_y0, "view_y1": self.view_y1,
                "n_frames": self.n_frames, "n_bins": self.n_bins,
                "t_start": self.t_start, "t_dur": self.t_dur,
                "sr": sr_val, "is_stereo": self.is_stereo,
                "normalize": self.normalize_loudness,
                "phase_mode": self.phase_mode,
                "routing": sp.channel_routing if sp else None,
                "field_sources": self.field_sources,
                "field_configs": self.field_configs,
                "use_adjusted": sp.use_adjusted if sp else False,
                "global_defaults": self.global_defaults,
            })
        if sp:
            sp.submit(job)

    def _submit_wavelet_synth_job(self) -> None:
        """Submit a wavelet reconstruction synthesis job (DWT or CWT)."""
        if not self._wv_coeffs and self._wv_W is None:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100

        # Map viewport frame range to time
        x0 = max(0, int(math.floor(self.view_x0)))
        x1 = min(self.n_frames, int(math.ceil(self.view_x1)))
        t0 = self.t_start + x0 / self.n_frames * self.t_dur
        t1 = self.t_start + x1 / self.n_frames * self.t_dur

        wv_type = self._wv_meta.get("type", "dwt")
        label = f"{'CWT' if wv_type == 'cwt' else 'DWT'} {t0:.2f}-{t1:.2f}s"
        job = SynthJob(
            job_id=0, label=label, source_type="wavelet",
            params={
                "wv_coeffs": self._wv_coeffs,
                "wv_W": self._wv_W,
                "wv_freqs": self._wv_freqs,
                "wv_meta": self._wv_meta,
                "sr": sr_val,
                "t0": t0, "t1": t1,
                "normalize": self.normalize_loudness,
            })
        if self.synth_panel:
            self.synth_panel.submit(job)

    def _submit_hybrid_cwt_synth_job(self) -> None:
        """Submit a Hybrid CWT synthesis job: CWT low + iCQT high."""
        if not self.synth or not self._loaded:
            return
        if self._wv_W is None:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100
        sp = self.synth_panel

        cross_hz = hybrid_crossover_freq(
            self.file_panel.bins_per_octave if self.file_panel else 1200,
            self.file_panel.hop_length if self.file_panel else 512,
            sr_val)

        x0 = max(0, int(math.floor(self.view_x0)))
        x1 = min(self.n_frames, int(math.ceil(self.view_x1)))
        y0 = max(0, int(math.floor(self.view_y0)))
        y1 = min(self.n_bins, int(math.ceil(self.view_y1)))
        t0 = self.t_start + x0 / self.n_frames * self.t_dur
        t1 = self.t_start + x1 / self.n_frames * self.t_dur

        label = f"HybridCWT {t0:.2f}-{t1:.2f}s  xf={cross_hz:.1f}Hz"
        job = SynthJob(
            job_id=0, label=label, source_type="hybrid_cwt",
            params={
                "raw_arrays": self._raw_arrays,
                "freqs": self._freqs,
                "times": self._times,
                "view_x0": x0, "view_x1": x1,
                "view_y0": y0, "view_y1": y1,
                "n_frames": self.n_frames, "n_bins": self.n_bins,
                "t_start": self.t_start, "t_dur": self.t_dur,
                "sr": sr_val,
                "cross_hz": cross_hz,
                "is_stereo": self._is_stereo,
                "normalize": self.normalize_loudness,
                "phase_mode": self.phase_mode,
                "routing": sp.channel_routing if sp else None,
                "field_sources": self.field_sources,
                "field_configs": self.field_configs,
                "use_adjusted": sp.use_adjusted if sp else False,
                "global_defaults": self.global_defaults,
                "wv_W": self._wv_W,
                "wv_freqs": self._wv_freqs,
                "wv_meta": self._wv_meta,
            })
        if sp:
            sp.submit(job)

    def _submit_transplant_job(self) -> None:
        """Submit a TF transplant synthesis job.

        Source region = current viewport (time × frequency).
        Destination frequency range = transplant_dst_f0..dst_f1 from panel.
        """
        if not self.synth or not self._loaded or not self.audio_path:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100

        # Source: viewport bin range → Hz
        y0 = max(0, int(math.floor(self.view_y0)))
        y1 = min(self.n_bins, int(math.ceil(self.view_y1)))
        if y1 <= y0:
            return
        src_f0 = float(self._freqs[y0])
        src_f1 = float(self._freqs[min(y1, len(self._freqs)) - 1])

        # Source: viewport frame range → time (seconds)
        x0 = max(0, int(math.floor(self.view_x0)))
        x1 = min(self.n_frames, int(math.ceil(self.view_x1)))
        src_t0 = self.t_start + x0 / self.n_frames * self.t_dur
        src_t1 = self.t_start + x1 / self.n_frames * self.t_dur

        # Destination frequency range from panel
        sp = self.synth_panel
        dst_f0 = sp.transplant_dst_f0 if sp else 200.0
        dst_f1 = sp.transplant_dst_f1 if sp else 800.0

        # Signal duration
        duration = self.player.duration if self.player else self.t_dur

        label = (f"Transplant {src_f0:.0f}-{src_f1:.0f}→"
                 f"{dst_f0:.0f}-{dst_f1:.0f}Hz")
        job = SynthJob(
            job_id=0, label=label, source_type="transplant",
            params={
                "sr": sr_val,
                "freqs": self._freqs,
                "duration": duration,
                "src_t0": src_t0, "src_t1": src_t1,
                "src_f0": src_f0, "src_f1": src_f1,
                "dst_f0": dst_f0, "dst_f1": dst_f1,
                "audio_path": self.audio_path,
                "normalize": self.normalize_loudness,
            })
        if sp:
            sp.submit(job)

    def _execute_region_transplant(self) -> None:
        """Submit a transplant job from the region-tool selection."""
        rd = self._region_data
        if rd is None or not self.audio_path:
            return
        sr_val = int(self._npz["sr"]) if self._npz is not None and "sr" in self._npz else 44100
        nf = len(self._freqs)
        if nf == 0:
            return

        y0_bin = max(0, int(math.floor(rd[1])))
        y1_bin = min(nf - 1, int(math.ceil(rd[3])))
        src_f0 = float(self._freqs[min(y0_bin, nf - 1)])
        src_f1 = float(self._freqs[min(y1_bin, nf - 1)])

        src_t0 = self._frame_to_time(rd[0])
        src_t1 = self._frame_to_time(rd[2])

        # Apply octave shift: multiply frequencies by 2^oct
        oct = self._region_octave_shift
        dst_f0 = src_f0 * (2.0 ** oct)
        dst_f1 = src_f1 * (2.0 ** oct)

        # Apply time shift (shift source window in time)
        dt = self._region_time_shift
        src_t0 += dt
        src_t1 += dt

        duration = self.player.duration if self.player else self.t_dur

        label = (f"Region {src_f0:.0f}-{src_f1:.0f}Hz "
                 f"oct{oct:+.0f} dt{dt:+.2f}s")
        job = SynthJob(
            job_id=0, label=label, source_type="transplant",
            params={
                "sr": sr_val,
                "freqs": self._freqs,
                "duration": duration,
                "src_t0": src_t0, "src_t1": src_t1,
                "src_f0": src_f0, "src_f1": src_f1,
                "dst_f0": dst_f0, "dst_f1": dst_f1,
                "audio_path": self.audio_path,
                "normalize": self.normalize_loudness,
            })
        sp = self.synth_panel
        if sp:
            sp.submit(job)

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
            if self._region_data is not None:
                self._region_data = None
                self._region_panel_rect = None
                self._region_hitboxes = {}
                return True
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
            self.gui.data_list.scroll_to_selected()
        elif key == K_f:
            self.show_freq_axis = not self.show_freq_axis
        elif key == K_a:
            self.show_amp_axis = not self.show_amp_axis
        elif key == K_t:
            self.show_time_axis = not self.show_time_axis
        elif key == K_d:
            _modes = ["tf", "tsc", "ts", "overlay"]
            idx = _modes.index(self.display_mode) if self.display_mode in _modes else 0
            self._set_display_mode(_modes[(idx + 1) % len(_modes)])
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
            if self._handle_region_panel_click(event.pos):
                return
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
                if self._tool_mode == "region":
                    self._finalize_region(self._rect_start, self._rect_end)
                else:
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
        if self.display_mode == "overlay":
            # Use CQT bin count as canonical Y extent
            return (0.0, float(self.n_bins) if self.n_bins > 0 else 1.0)
        if self.display_mode == "tsc":
            n = self._wv_n_levels if self._wv_n_levels > 0 else 1
            return (0.0, float(n))
        # TF mode — depends on source
        if self.tf_source == "fb":
            n = len(self._fb_bands) if self._fb_bands else 16
            return (0.0, float(n))
        if self.tf_source == "hybrid":
            n_fb = len(self._fb_bands) if self._fb_bands else 0
            skip = self._hybrid_cqt_skip_bins()
            n_cqt_above = max(0, self.n_bins - skip)
            return (0.0, float(n_fb + n_cqt_above))
        # cqt: CQT bin indices as Y axis
        return (0.0, float(self.n_bins))

    def _screen_to_data(self, sx: int, sy: int) -> tuple[float, float]:
        rx, ry, rw, rh = self._spec_rect()
        fx = (sx - rx) / rw
        fy = 1.0 - (sy - ry) / rh
        dx = self.view_x0 + fx * (self.view_x1 - self.view_x0)
        dy = self.view_y0 + fy * (self.view_y1 - self.view_y0)
        return dx, dy

    def _data_to_screen(self, dx: float, dy: float) -> tuple[int, int]:
        rx, ry, rw, rh = self._spec_rect()
        fx = (dx - self.view_x0) / max(self.view_x1 - self.view_x0, 1e-9)
        fy = (dy - self.view_y0) / max(self.view_y1 - self.view_y0, 1e-9)
        sx = rx + fx * rw
        sy = ry + (1.0 - fy) * rh
        return int(sx), int(sy)

    def _time_to_frame(self, t: float) -> float:
        if self.t_dur <= 0:
            return 0.0
        return (t - self.t_start) / self.t_dur * self.n_frames

    def _frame_to_time(self, f: float) -> float:
        if self.n_frames <= 0:
            return self.t_start
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
        x_max = float(self._wv_n_frames) if (
            self.display_mode == "tsc" and self._wv_n_frames > 0
        ) else float(self.n_frames)
        if self.view_x0 < 0:
            self.view_x0 = 0
            self.view_x1 = w
        if self.view_x1 > x_max:
            self.view_x1 = x_max
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
        if self.display_mode == "ts":
            min_y = 0.01
        elif self.display_mode == "tf" and self.tf_source == "fb":
            min_y = 1.0
        else:
            min_y = 4.0
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
        if self.display_mode == "ts":
            min_y = 0.01
        elif self.display_mode == "tf" and self.tf_source == "fb":
            min_y = 1.0
        else:
            min_y = 4.0
        if x1 - x0 < 4 or y1 - y0 < min_y:
            return
        if self.zoom_lock != "freq":
            self.view_x0, self.view_x1 = x0, x1
        if self.zoom_lock != "time":
            self.view_y0, self.view_y1 = y0, y1
        self._clamp_view()

    def _finalize_region(self, p0: tuple[int, int], p1: tuple[int, int]) -> None:
        d0 = self._screen_to_data(*p0)
        d1 = self._screen_to_data(*p1)
        x0, x1 = sorted([d0[0], d1[0]])
        y0, y1 = sorted([d0[1], d1[1]])
        if x1 - x0 < 1 or y1 - y0 < 0.5:
            return
        self._region_data = (x0, y0, x1, y1)
        self._region_octave_shift = 0.0
        self._region_time_shift = 0.0

    def _handle_region_panel_click(self, pos: tuple[int, int]) -> bool:
        """Check if *pos* hits a region-panel control.  Returns True if consumed."""
        pr = self._region_panel_rect
        if pr is None or self._region_data is None:
            return False
        if not pr.collidepoint(pos):
            return False
        lx = pos[0] - pr.x
        ly = pos[1] - pr.y
        for name, r in self._region_hitboxes.items():
            if r.collidepoint(lx, ly):
                if name == "oct_minus":
                    self._region_octave_shift = max(-4.0, self._region_octave_shift - 1.0)
                elif name == "oct_plus":
                    self._region_octave_shift = min(4.0, self._region_octave_shift + 1.0)
                elif name == "dt_minus":
                    self._region_time_shift -= 0.1
                elif name == "dt_plus":
                    self._region_time_shift += 0.1
                elif name == "execute":
                    self._execute_region_transplant()
                elif name == "clear":
                    self._region_data = None
                    self._region_panel_rect = None
                    self._region_hitboxes = {}
                return True
        return True  # inside panel but not on a button — consume anyway

    def _build_region_panel(self) -> tuple[pygame.Surface, dict[str, pygame.Rect]] | None:
        """Build the floating control panel for the active region."""
        rd = self._region_data
        if rd is None:
            return None
        font = pygame.font.SysFont("consolas", 13)
        W = 290
        ROW_H = 22
        PAD = 6
        BTN_W = 22
        rows = 5  # oct, dt, info1, info2, execute+clear
        H = rows * (ROW_H + 2) + PAD * 2
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill((25, 25, 35, 210))
        pygame.draw.rect(surf, (140, 110, 50), surf.get_rect(), 1)

        hitboxes: dict[str, pygame.Rect] = {}
        y = PAD

        # --- Octave shift ---
        lbl = font.render("Oct:", True, (200, 200, 200))
        surf.blit(lbl, (PAD, y + 3))
        lbl_w = 42

        minus_r = pygame.Rect(lbl_w, y + 2, BTN_W, ROW_H - 4)
        pygame.draw.rect(surf, (50, 50, 65), minus_r)
        pygame.draw.rect(surf, (80, 80, 100), minus_r, 1)
        mt = font.render("\u2212", True, (200, 200, 200))
        surf.blit(mt, (minus_r.x + (BTN_W - mt.get_width()) // 2, y + 2))
        hitboxes["oct_minus"] = minus_r

        plus_r = pygame.Rect(lbl_w + BTN_W + 4, y + 2, BTN_W, ROW_H - 4)
        pygame.draw.rect(surf, (50, 50, 65), plus_r)
        pygame.draw.rect(surf, (80, 80, 100), plus_r, 1)
        pt = font.render("+", True, (200, 200, 200))
        surf.blit(pt, (plus_r.x + (BTN_W - pt.get_width()) // 2, y + 2))
        hitboxes["oct_plus"] = plus_r

        val_txt = font.render(f"{self._region_octave_shift:+.0f}", True, (255, 200, 100))
        surf.blit(val_txt, (plus_r.right + 6, y + 3))
        y += ROW_H + 2

        # --- Time shift ---
        lbl = font.render("\u0394t:", True, (200, 200, 200))
        surf.blit(lbl, (PAD, y + 3))

        minus_r = pygame.Rect(lbl_w, y + 2, BTN_W, ROW_H - 4)
        pygame.draw.rect(surf, (50, 50, 65), minus_r)
        pygame.draw.rect(surf, (80, 80, 100), minus_r, 1)
        mt = font.render("\u2212", True, (200, 200, 200))
        surf.blit(mt, (minus_r.x + (BTN_W - mt.get_width()) // 2, y + 2))
        hitboxes["dt_minus"] = minus_r

        plus_r = pygame.Rect(lbl_w + BTN_W + 4, y + 2, BTN_W, ROW_H - 4)
        pygame.draw.rect(surf, (50, 50, 65), plus_r)
        pygame.draw.rect(surf, (80, 80, 100), plus_r, 1)
        pt = font.render("+", True, (200, 200, 200))
        surf.blit(pt, (plus_r.x + (BTN_W - pt.get_width()) // 2, y + 2))
        hitboxes["dt_plus"] = plus_r

        val_txt = font.render(f"{self._region_time_shift:+.2f}s", True, (255, 200, 100))
        surf.blit(val_txt, (plus_r.right + 6, y + 3))
        y += ROW_H + 2

        # --- Info lines ---
        y0_bin = max(0, int(rd[1]))
        y1_bin = int(rd[3])
        nf = len(self._freqs)
        if nf > 0:
            f_lo = float(self._freqs[min(y0_bin, nf - 1)])
            f_hi = float(self._freqs[min(y1_bin, nf - 1)])
        else:
            f_lo, f_hi = 0.0, 0.0
        fc = (f_lo + f_hi) / 2
        t0 = self._frame_to_time(rd[0])
        t1 = self._frame_to_time(rd[2])
        dur = t1 - t0

        it = font.render(f"{f_lo:.0f}\u2013{f_hi:.0f} Hz  Fc {fc:.0f} Hz",
                         True, (180, 180, 180))
        surf.blit(it, (PAD, y + 3))
        y += ROW_H + 2

        it2 = font.render(f"{t0:.2f}\u2013{t1:.2f}s  ({dur:.2f}s)",
                          True, (180, 180, 180))
        surf.blit(it2, (PAD, y + 3))
        y += ROW_H + 2

        # --- Execute / Clear buttons ---
        exec_w = (W - 3 * PAD) // 2
        exec_r = pygame.Rect(PAD, y + 1, exec_w, ROW_H - 2)
        pygame.draw.rect(surf, (50, 70, 50), exec_r)
        pygame.draw.rect(surf, (100, 140, 100), exec_r, 1)
        et = font.render("\u25b6 Execute", True, (200, 255, 200))
        surf.blit(et, (exec_r.x + (exec_r.width - et.get_width()) // 2, y + 2))
        hitboxes["execute"] = exec_r

        clr_r = pygame.Rect(PAD * 2 + exec_w, y + 1, exec_w, ROW_H - 2)
        pygame.draw.rect(surf, (70, 45, 45), clr_r)
        pygame.draw.rect(surf, (140, 90, 90), clr_r, 1)
        ct = font.render("\u2716 Clear", True, (255, 200, 200))
        surf.blit(ct, (clr_r.x + (clr_r.width - ct.get_width()) // 2, y + 2))
        hitboxes["clear"] = clr_r

        return surf, hitboxes

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

        # Pass 1: field gather (data textures → 4 channels + alpha via MRT)
        p1f = gl_shaders.compileShader(PASS1_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._pass1_program = gl_shaders.compileProgram(
            gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER), p1f)

        # Pass 2: HSL hue blend only (4 channels → raw unbounded RGB)
        p2f = gl_shaders.compileShader(PASS2_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._pass2_program = gl_shaders.compileProgram(
            gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER), p2f)

        # Reduce seed: copy pass-2 into min/max MRT pair
        rsf = gl_shaders.compileShader(REDUCE_SEED_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._reduce_seed_program = gl_shaders.compileProgram(
            gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER), rsf)

        # Reduce step: iterative 2×2 min/max downsample
        rrf = gl_shaders.compileShader(REDUCE_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._reduce_program = gl_shaders.compileProgram(
            gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER), rrf)

        # Pass 3: HDR normalize + out_gamma/scale + final clamp
        p3f = gl_shaders.compileShader(PASS3_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._pass3_program = gl_shaders.compileProgram(
            gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER), p3f)

        # Keep legacy name for _set_field_uniforms compatibility
        self._program = self._pass1_program

        # Overlay passthrough (only used for GUI overlays now)
        ov = gl_shaders.compileShader(OVERLAY_VERT_SRC, GL_VERTEX_SHADER)
        of = gl_shaders.compileShader(OVERLAY_FRAG_SRC, GL_FRAGMENT_SHADER)
        self._overlay_program = gl_shaders.compileProgram(ov, of)

        # FBOs for multi-pass rendering (created on demand)
        self._fbo1 = 0          # pass-1 MRT FBO
        self._fbo1_tex_ch = 0   # pass-1 color attachment 0: 4 data channels
        self._fbo1_tex_a = 0    # pass-1 color attachment 1: alpha
        self._fbo2 = 0          # pass-2 FBO
        self._fbo2_tex = 0      # pass-2 color attachment: blended RGB
        self._reduce_fbos: list[int] = []
        self._reduce_min_texs: list[int] = []
        self._reduce_max_texs: list[int] = []
        self._fbo_w = 0
        self._fbo_h = 0
        # Auto-detected input range from GPU reduction
        self._auto_hdr_min = np.zeros(3, dtype=np.float32)
        self._auto_hdr_max = np.ones(3, dtype=np.float32)

        # Invalidate textures (GL context may have been recreated)
        self._front_tex_ids = {}
        self._back_tex_ids = {}
        self._back_buffers = {}
        self._fb_shader_tex_ids = {}
        self._fb_dirty = True
        self._scan_key = None
        self._scan_complete = True

    def _ensure_fbos(self, w: int, h: int) -> None:
        """Create or resize all FBOs for the multi-pass pipeline."""
        if w == self._fbo_w and h == self._fbo_h and self._fbo1:
            return

        # --- Clean up old resources ---
        for old_fbo in [self._fbo1, self._fbo2] + self._reduce_fbos:
            if old_fbo:
                glDeleteFramebuffers(1, [old_fbo])
        for old_tex in ([self._fbo1_tex_ch, self._fbo1_tex_a, self._fbo2_tex]
                        + self._reduce_min_texs + self._reduce_max_texs):
            if old_tex:
                glDeleteTextures([old_tex])

        def _make_rgba32f_tex(tw: int, th: int) -> int:
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, tw, th, 0,
                         GL_RGBA, GL_FLOAT, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            return tex

        # --- FBO 1: pass-1 MRT (2 color attachments: channels + alpha) ---
        self._fbo1_tex_ch = _make_rgba32f_tex(w, h)
        self._fbo1_tex_a = _make_rgba32f_tex(w, h)
        self._fbo1 = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo1)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self._fbo1_tex_ch, 0)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                               GL_TEXTURE_2D, self._fbo1_tex_a, 0)
        glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # --- FBO 2: pass-2 single output (blended RGB + alpha) ---
        self._fbo2_tex = _make_rgba32f_tex(w, h)
        self._fbo2 = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo2)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self._fbo2_tex, 0)
        glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])
        assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE

        # --- Reduction chain: iterative 2× downsample until 1×1 ---
        self._reduce_fbos = []
        self._reduce_min_texs = []
        self._reduce_max_texs = []
        rw, rh = w, h
        while rw > 1 or rh > 1:
            rw = max(1, (rw + 1) // 2)
            rh = max(1, (rh + 1) // 2)
            min_tex = _make_rgba32f_tex(rw, rh)
            max_tex = _make_rgba32f_tex(rw, rh)
            fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, fbo)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_2D, min_tex, 0)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1,
                                   GL_TEXTURE_2D, max_tex, 0)
            glDrawBuffers(2, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])
            assert glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
            self._reduce_fbos.append(fbo)
            self._reduce_min_texs.append(min_tex)
            self._reduce_max_texs.append(max_tex)

        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        self._fbo_w = w
        self._fbo_h = h

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
        """Bind data textures and set pass-1 uniforms (field gather only)."""
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

        mmm = 0
        if self.synth_panel:
            mmm = self.synth_panel.channel_routing.mono_mix_mode
        glUniform1i(loc("uMonoMixMode"), mmm)

    # ---- Unified 3-pass spectral rendering --------------------------------

    def _render_spectral_passes(self, tex_ids: dict[str, int],
                                sx: int, sy: int, sw: int, sh: int,
                                u0: float, v0: float,
                                u1: float, v1: float) -> None:
        """Run the full GPU pipeline: gather → blend → reduce → display.

        All modes (CQT TF, filterbank TF, wavelet TSC, overlay) call
        this with their respective texture IDs and UV ranges.  The
        shader sequence is identical every time.
        """
        if sw <= 0 or sh <= 0:
            return
        self._ensure_fbos(sw, sh)

        ww, wh = self.win_w, self.win_h
        gd = self.global_defaults

        # NDC coordinates for the viewport quad
        nx0 = 2.0 * sx / ww - 1.0
        ny0 = 1.0 - 2.0 * (sy + sh) / wh
        nx1 = 2.0 * (sx + sw) / ww - 1.0
        ny1 = 1.0 - 2.0 * sy / wh

        def _fullscreen_quad():
            """Draw a [-1,1] fullscreen quad with UV [0,1]."""
            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 0.0); glVertex2f( 1.0, -1.0)
            glTexCoord2f(1.0, 1.0); glVertex2f( 1.0,  1.0)
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0,  1.0)
            glEnd()

        def _data_quad():
            """Draw a quad with the data UV coordinates."""
            glBegin(GL_QUADS)
            glTexCoord2f(u0, v0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(u1, v0); glVertex2f( 1.0, -1.0)
            glTexCoord2f(u1, v1); glVertex2f( 1.0,  1.0)
            glTexCoord2f(u0, v1); glVertex2f(-1.0,  1.0)
            glEnd()

        # ---- Pass 1: field gather → FBO1 (MRT: channels + alpha) --------
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo1)
        glViewport(0, 0, sw, sh)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)
        glDisable(GL_BLEND)

        glUseProgram(self._pass1_program)
        self._set_field_uniforms(tex_ids)
        _data_quad()

        # ---- Pass 2: HSL hue blend → FBO2 (raw unbounded RGB) -----------
        glBindFramebuffer(GL_FRAMEBUFFER, self._fbo2)
        glViewport(0, 0, sw, sh)
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(self._pass2_program)
        p2loc = lambda n: glGetUniformLocation(self._pass2_program, n)

        # Bind pass-1 MRT outputs as textures
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._fbo1_tex_ch)
        glUniform1i(p2loc("uPass1Channels"), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, self._fbo1_tex_a)
        glUniform1i(p2loc("uPass1Alpha"), 1)

        h = gd.channel_hues
        s = gd.channel_sats
        glUniform4f(p2loc("uChannelHue"), h[0], h[1], h[2], h[3])
        glUniform4f(p2loc("uChannelSat"), s[0], s[1], s[2], s[3])
        glUniform1i(p2loc("uBlendMode"), gd.blend_mode)
        _fullscreen_quad()

        # ---- GPU Reduction: auto-detect min/max of pass-2 output ---------
        self._gpu_reduce_minmax(sw, sh)

        # ---- Pass 3: HDR normalize + out_gamma/scale + clamp → screen ----
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, ww, wh)
        glEnable(GL_BLEND)

        glUseProgram(self._pass3_program)
        p3loc = lambda n: glGetUniformLocation(self._pass3_program, n)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._fbo2_tex)
        glUniform1i(p3loc("uPass2Tex"), 0)

        glUniform1i(p3loc("uHdrNormalizer"), gd.hdr_normalizer)
        # Input range is auto-detected from the data
        glUniform2f(p3loc("uHdrInputRange"),
                    float(self._auto_hdr_min.min()),
                    float(self._auto_hdr_max.max()))
        glUniform2f(p3loc("uHdrOutputRange"),
                    gd.hdr_output_lo, gd.hdr_output_hi)
        glUniform4f(p3loc("uOutGamma"),
                    gd.out_gamma, gd.out_gamma, gd.out_gamma, gd.out_gamma)
        glUniform4f(p3loc("uOutScale"),
                    gd.out_scale, gd.out_scale, gd.out_scale, gd.out_scale)

        # Draw final quad at the actual viewport position
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(nx0, ny0)
        glTexCoord2f(1.0, 0.0); glVertex2f(nx1, ny0)
        glTexCoord2f(1.0, 1.0); glVertex2f(nx1, ny1)
        glTexCoord2f(0.0, 1.0); glVertex2f(nx0, ny1)
        glEnd()

        glUseProgram(0)
        glActiveTexture(GL_TEXTURE0)

    def _gpu_reduce_minmax(self, w: int, h: int) -> None:
        """Run the GPU min/max reduction chain on the pass-2 FBO output.

        After this call, ``_auto_hdr_min`` and ``_auto_hdr_max`` hold
        the per-component (RGB) min and max of the blended image.
        """
        if not self._reduce_fbos:
            return

        # Seed pass: copy pass-2 texture into level-0 min/max MRT
        glBindFramebuffer(GL_FRAMEBUFFER, self._reduce_fbos[0])
        rw = max(1, (w + 1) // 2)
        rh = max(1, (h + 1) // 2)
        glViewport(0, 0, rw, rh)

        glUseProgram(self._reduce_seed_program)
        loc_s = lambda n: glGetUniformLocation(self._reduce_seed_program, n)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self._fbo2_tex)
        glUniform1i(loc_s("uSrcTex"), 0)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f( 1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f( 1.0,  1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0,  1.0)
        glEnd()

        # Iterative reduction steps
        src_w, src_h = rw, rh
        for i in range(1, len(self._reduce_fbos)):
            dst_w = max(1, (src_w + 1) // 2)
            dst_h = max(1, (src_h + 1) // 2)

            glBindFramebuffer(GL_FRAMEBUFFER, self._reduce_fbos[i])
            glViewport(0, 0, dst_w, dst_h)

            glUseProgram(self._reduce_program)
            loc_r = lambda n: glGetUniformLocation(self._reduce_program, n)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self._reduce_min_texs[i - 1])
            glUniform1i(loc_r("uMinTex"), 0)
            glActiveTexture(GL_TEXTURE1)
            glBindTexture(GL_TEXTURE_2D, self._reduce_max_texs[i - 1])
            glUniform1i(loc_r("uMaxTex"), 1)
            glUniform2f(loc_r("uTexelSize"),
                        1.0 / src_w, 1.0 / src_h)

            glBegin(GL_QUADS)
            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
            glTexCoord2f(1.0, 0.0); glVertex2f( 1.0, -1.0)
            glTexCoord2f(1.0, 1.0); glVertex2f( 1.0,  1.0)
            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0,  1.0)
            glEnd()

            src_w, src_h = dst_w, dst_h

        # Read back the final 1×1 min/max texels
        last = len(self._reduce_fbos) - 1
        glBindFramebuffer(GL_FRAMEBUFFER, self._reduce_fbos[last])
        # Read min (attachment 0)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        min_px = glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT)
        # Read max (attachment 1)
        glReadBuffer(GL_COLOR_ATTACHMENT1)
        max_px = glReadPixels(0, 0, 1, 1, GL_RGBA, GL_FLOAT)

        self._auto_hdr_min = np.frombuffer(min_px, dtype=np.float32)[:3].copy()
        self._auto_hdr_max = np.frombuffer(max_px, dtype=np.float32)[:3].copy()

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

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

        # --- Persistent region tool overlay ---
        if self._region_data is not None:
            rd = self._region_data
            s_lo = self._data_to_screen(rd[0], rd[1])
            s_hi = self._data_to_screen(rd[2], rd[3])
            sx0 = min(s_lo[0], s_hi[0])
            sx1 = max(s_lo[0], s_hi[0])
            sy0 = min(s_lo[1], s_hi[1])
            sy1 = max(s_lo[1], s_hi[1])

            rx0 = 2.0 * sx0 / self.win_w - 1.0
            rx1 = 2.0 * sx1 / self.win_w - 1.0
            ry0 = 1.0 - 2.0 * sy1 / self.win_h
            ry1 = 1.0 - 2.0 * sy0 / self.win_h

            glColor4f(1.0, 0.6, 0.2, 0.12)
            glBegin(GL_QUADS)
            glVertex2f(rx0, ry0); glVertex2f(rx1, ry0)
            glVertex2f(rx1, ry1); glVertex2f(rx0, ry1)
            glEnd()
            glLineWidth(2.0)
            glColor4f(1.0, 0.7, 0.3, 0.9)
            glBegin(GL_LINE_LOOP)
            glVertex2f(rx0, ry0); glVertex2f(rx1, ry0)
            glVertex2f(rx1, ry1); glVertex2f(rx0, ry1)
            glEnd()

            # Floating control panel anchored below the region
            panel_result = self._build_region_panel()
            if panel_result is not None:
                psurf, hitboxes = panel_result
                pw, ph = psurf.get_size()
                # Position: centred horizontally under the region, clamped to window
                pcx = (sx0 + sx1) // 2 - pw // 2
                pcy = sy1 + 4
                if pcy + ph > self.win_h:
                    pcy = sy0 - ph - 4  # above if no room below
                pcx = max(0, min(pcx, self.win_w - pw))
                pcy = max(0, min(pcy, self.win_h - ph))
                self._region_panel_rect = pygame.Rect(pcx, pcy, pw, ph)
                self._region_hitboxes = hitboxes

                tex = self._upload_pygame_surface(psurf)
                px0 = 2.0 * pcx / self.win_w - 1.0
                px1 = 2.0 * (pcx + pw) / self.win_w - 1.0
                py1 = 1.0 - 2.0 * pcy / self.win_h
                py0 = 1.0 - 2.0 * (pcy + ph) / self.win_h

                glEnable(GL_TEXTURE_2D)
                glBindTexture(GL_TEXTURE_2D, tex)
                glColor4f(1.0, 1.0, 1.0, 1.0)
                glBegin(GL_QUADS)
                glTexCoord2f(0, 0); glVertex2f(px0, py0)
                glTexCoord2f(1, 0); glVertex2f(px1, py0)
                glTexCoord2f(1, 1); glVertex2f(px1, py1)
                glTexCoord2f(0, 1); glVertex2f(px0, py1)
                glEnd()
                glDisable(GL_TEXTURE_2D)

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

                # Clip to available panel height with scroll offset
                panel_surf = panel._apply_panel_scroll(panel_surf)

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
        """Draw the time-frequency spectrogram via the unified 3-pass pipeline."""
        if not (self._pass1_program and (self._front_tex_ids or self._back_tex_ids)):
            return

        u0, u1 = 0.0, 1.0
        v0, v1 = 0.0, 1.0

        if self._front_tex_ids:
            self._render_spectral_passes(
                self._front_tex_ids, sx, sy, sw, sh, u0, v0, u1, v1)

        if self._back_tex_ids and not self._scan_complete:
            scan_frac = self._scan_col / max(1, self._scan_total)
            scan_px = int(scan_frac * sw)
            if scan_px > 0:
                glEnable(GL_SCISSOR_TEST)
                glScissor(sx, self.win_h - sy - sh,
                          max(1, scan_px), sh)
                self._render_spectral_passes(
                    self._back_tex_ids, sx, sy, sw, sh, u0, v0, u1, v1)
                glDisable(GL_SCISSOR_TEST)

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

    def _render_ts_band(self, sx: int, sy: int, sw: int, sh: int,
                        band_idx: int) -> None:
        """Draw a filterbank subband waveform in the given rect."""
        fb = self._ts_fb
        if fb is None or band_idx >= len(fb.subbands) or sw <= 0 or sh <= 0:
            return

        mono = fb.subbands[band_idx]
        sr = fb.sr
        n_samples = len(mono)
        if n_samples == 0:
            return

        t0 = self._frame_to_time(self.view_x0)
        t1 = self._frame_to_time(self.view_x1)
        s0 = max(0, int(t0 * sr))
        s1 = min(n_samples, int(t1 * sr))
        if s1 <= s0:
            return

        chunk = mono[s0:s1]

        ww, wh = float(self.win_w), float(self.win_h)
        amp_lo = self.view_y0
        amp_hi = self.view_y1
        amp_range = amp_hi - amp_lo
        if amp_range <= 0:
            return

        # Downsample: min/max envelope per pixel column
        n = len(chunk)
        cols = min(sw, n)
        if cols <= 0:
            return
        indices = np.linspace(0, n, cols + 1, dtype=np.int64)
        env_min = np.empty(cols, dtype=np.float32)
        env_max = np.empty(cols, dtype=np.float32)
        for c in range(cols):
            seg = chunk[indices[c]:max(indices[c] + 1, indices[c + 1])]
            env_min[c] = seg.min()
            env_max[c] = seg.max()

        def amp_to_screen_y(a: np.ndarray) -> np.ndarray:
            frac = (a - amp_lo) / amp_range
            return sy + sh * (1.0 - frac)

        x_screen = sx + np.arange(cols) * (sw / cols)
        x_ndc = 2.0 * x_screen / ww - 1.0
        y_min_ndc = 1.0 - 2.0 * amp_to_screen_y(env_min) / wh
        y_max_ndc = 1.0 - 2.0 * amp_to_screen_y(env_max) / wh

        # Per-band colour: cycle through distinct hues
        _BAND_COLORS = [
            (0.25, 0.65, 0.90), (0.90, 0.45, 0.25), (0.25, 0.85, 0.45),
            (0.85, 0.25, 0.65), (0.55, 0.45, 0.90), (0.90, 0.80, 0.25),
            (0.25, 0.90, 0.85), (0.90, 0.25, 0.25), (0.50, 0.80, 0.30),
        ]
        r, g, b = _BAND_COLORS[band_idx % len(_BAND_COLORS)]

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColor4f(r, g, b, 0.85)
        glBegin(GL_QUAD_STRIP)
        for i in range(cols):
            glVertex2f(float(x_ndc[i]), float(y_min_ndc[i]))
            glVertex2f(float(x_ndc[i]), float(y_max_ndc[i]))
        glEnd()

        # Zero line
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

    # ---- Wavelet time-scale rendering --------------------------------------

    def _load_wavelet_data(self) -> bool:
        """Load wavelet coefficients from the analysis folder.

        Handles both DWT (approx + detail_N keys) and CWT (W_real/W_imag
        + freqs keys) formats, detected via ``meta["type"]`` or npz keys.

        Returns True if data was successfully loaded.
        """
        if not self.analysis_dir:
            return False
        wv_dir = os.path.join(self.analysis_dir, "wavelet")
        npz_path = os.path.join(wv_dir, "wavelet_data.npz")
        meta_path = os.path.join(wv_dir, "wavelet_meta.json")
        if not os.path.isfile(npz_path):
            return False
        import json as _json
        meta: dict = {}
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = _json.load(f)
        npz = np.load(npz_path, allow_pickle=False)

        wv_type = meta.get("type", "dwt")  # default for legacy data
        if wv_type == "cwt" or "W_real" in npz:
            # ── CWT format ──
            W_real = npz["W_real"]
            W_imag = npz["W_imag"]
            self._wv_W = W_real + 1j * W_imag
            self._wv_freqs = npz["freqs"]
            self._wv_coeffs = []  # clear DWT data
            self._wv_meta = meta
            self._wv_n_levels = self._wv_W.shape[0]  # n_scales
            self._wv_n_frames = self._wv_W.shape[1]   # n_frames
            self._wv_dirty = True
            print(f"Wavelet CWT: loaded {self._wv_n_levels} scales, "
                  f"{self._wv_n_frames} frames from {wv_dir}")
        else:
            # ── DWT format ──
            n_levels = int(npz["n_levels"])
            coeffs: list[np.ndarray] = [npz["approx"]]
            for li in range(1, n_levels + 1):
                coeffs.append(npz[f"detail_{li}"])
            self._wv_coeffs = coeffs
            self._wv_W = None
            self._wv_freqs = None
            self._wv_meta = meta
            self._wv_n_levels = n_levels + 1  # approx + details
            self._wv_n_frames = max(len(c) for c in coeffs) if coeffs else 0
            self._wv_dirty = True
            print(f"Wavelet DWT: loaded {n_levels} detail levels + approx "
                  f"from {wv_dir}")
        return True

    def _compute_wavelet_scalogram(self) -> None:
        """Build a 2-D (n_levels, tex_w) scalogram texture from wavelet
        coefficients and upload as shader-compatible RGBA textures.

        Handles both DWT (list of coefficient arrays resampled to common
        width) and CWT (2-D complex scalogram already on a regular grid).
        """
        is_cwt = self._wv_W is not None
        if not is_cwt and not self._wv_coeffs:
            self._wv_dirty = False
            return

        if is_cwt:
            # ── CWT: scalogram is already (n_scales, n_frames) complex ──
            W = self._wv_W
            n_rows, n_cols = W.shape
            max_w = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
            tex_w = min(n_cols, max_w)
            self._wv_n_frames = tex_w
            native_dt = np.float32

            mag_grid = np.abs(W).astype(native_dt)
            phase_grid = np.angle(W).astype(native_dt)
            # Downsample columns if needed
            if tex_w < n_cols:
                dst_t = np.linspace(0.0, 1.0, tex_w)
                src_t = np.linspace(0.0, 1.0, n_cols)
                mg2 = np.zeros((n_rows, tex_w), dtype=native_dt)
                pg2 = np.zeros((n_rows, tex_w), dtype=native_dt)
                for i in range(n_rows):
                    mg2[i] = np.interp(dst_t, src_t, mag_grid[i])
                    pg2[i] = np.interp(dst_t, src_t, phase_grid[i])
                mag_grid, phase_grid = mg2, pg2
        else:
            # ── DWT: resampled ragged coefficient arrays ──
            coeffs = self._wv_coeffs
            n_rows = len(coeffs)
            max_len = max(len(c) for c in coeffs)
            max_w = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
            tex_w = min(max_len, max_w)
            self._wv_n_frames = tex_w
            native_dt = coeffs[0].dtype if len(coeffs) > 0 else np.float32

            mag_grid = np.zeros((n_rows, tex_w), dtype=native_dt)
            dst_t = np.linspace(0.0, 1.0, tex_w)
            for i, c in enumerate(coeffs):
                n_src = len(c)
                if n_src == 0:
                    continue
                absv = np.abs(c)
                if n_src == tex_w:
                    mag_grid[i] = absv
                else:
                    src_t = np.linspace(0.0, 1.0, n_src)
                    mag_grid[i] = np.interp(dst_t, src_t, absv)

            phase_grid = np.zeros((n_rows, tex_w), dtype=native_dt)
            for i, c in enumerate(coeffs):
                n_src = len(c)
                if n_src == 0:
                    continue
                sign_arr = np.where(c >= 0, 0.0, np.pi)
                if n_src == tex_w:
                    phase_grid[i] = sign_arr
                else:
                    src_t = np.linspace(0.0, 1.0, n_src)
                    phase_grid[i] = np.interp(dst_t, src_t, sign_arr)

        # Normalise magnitude to [0, 1]
        gmax = float(mag_grid.max()) if mag_grid.size else 1.0
        if gmax < 1e-12:
            gmax = 1.0
        mag_norm = np.clip(mag_grid / gmax, 0.0, 1.0)

        # Normalise phase to [0, 1]
        phase_norm = np.clip(
            (phase_grid + np.pi) / (2.0 * np.pi), 0.0, 1.0
        )

        # Clean up old wavelet textures
        for old_id in self._wv_shader_tex_ids.values():
            glDeleteTextures([old_id])
        self._wv_shader_tex_ids = {}

        zeros = np.zeros((n_rows, tex_w), dtype=np.float32)
        alpha = np.ones((n_rows, tex_w), dtype=np.float32)

        def _pack_and_upload(r, g, b, a) -> int:
            rgba = np.stack([r, g, b, a], axis=-1).astype(np.float32)
            rgba = np.ascontiguousarray(rgba)
            return self._upload_texture_float(rgba)

        self._wv_shader_tex_ids["wavelet_coeffs"] = _pack_and_upload(
            mag_norm, phase_norm, zeros, alpha)

        self._raw_arrays["wvlt_approx"] = mag_grid
        self._raw_arrays["wvlt_detail"] = phase_grid
        wvlt_mag_max = float(gmax) if gmax > 1e-12 else 1.0
        self._field_ranges["wvlt_approx"] = (0.0, wvlt_mag_max)
        self._field_ranges["wvlt_detail"] = (0.0, float(np.pi))

        self._wv_dirty = False
        wv_type = "CWT" if is_cwt else "DWT"
        print(f"Wavelet scalogram ({wv_type}): {n_rows} rows, tex {n_rows}×{tex_w}")

    def _render_tsc(self, sx: int, sy: int, sw: int, sh: int) -> None:
        """Render the wavelet time-scale scalogram through the unified
        3-pass pipeline."""
        if sw <= 0 or sh <= 0:
            return
        if self._wv_dirty:
            self._compute_wavelet_scalogram()
        if not self._wv_shader_tex_ids or not self._pass1_program:
            return

        n_levels = self._wv_n_levels if self._wv_n_levels > 0 else 1
        cqt_total = float(self.n_frames) if self.n_frames > 0 else 1.0

        u0 = self.view_x0 / cqt_total
        u1 = self.view_x1 / cqt_total
        v0 = self.view_y0 / n_levels
        v1 = self.view_y1 / n_levels

        needed: set[str] = set()
        for i, fs in enumerate(self.field_sources):
            if self.field_configs[i].target >= 0:
                needed.add(fs.tex_name)
        saved_tex_units = dict(self._tex_units)
        self._tex_units = {}
        unit = 0
        for tex_name in sorted(needed):
            if tex_name not in self._wv_shader_tex_ids:
                continue
            if unit >= 8:
                break
            self._tex_units[tex_name] = unit
            unit += 1

        self._render_spectral_passes(
            self._wv_shader_tex_ids, sx, sy, sw, sh, u0, v0, u1, v1)

        self._tex_units = saved_tex_units

    # ---- Overlay composite rendering ----------------------------------------

    def _render_overlay(self, sx: int, sy: int, sw: int, sh: int) -> None:
        """Render the overlay composite via the unified 3-pass GPU pipeline.

        All enabled layers are resampled to a common (CQT-sized) grid on
        the CPU, packed into RGBA float textures matching the field
        system's tex_name keys, and then rendered through the same
        pass-1/pass-2/pass-3 shader pipeline as every other mode.
        """
        if sw <= 0 or sh <= 0:
            return
        if not self._pass1_program:
            return

        # Canonical grid dims — use CQT shape as reference
        grid_h = self.n_bins if self.n_bins > 0 else 256
        grid_w = self.n_frames if self.n_frames > 0 else 256

        # Destination coordinate axes — physical units so every source
        # maps onto the same frequency / time grid accurately.
        if len(self.times) == grid_w:
            dst_x = np.asarray(self.times, dtype=np.float64)
        else:
            dst_x = np.linspace(self.t_start, self.t_end, grid_w,
                                dtype=np.float64)
        if len(self._freqs) == grid_h:
            dst_y = np.asarray(self._freqs, dtype=np.float64)
        else:
            dst_y = np.arange(grid_h, dtype=np.float64)

        # Detect when settings changed since last bake
        gd = self.global_defaults
        gd_snap = (tuple((fc.target, fc.norm_mode, fc.gamma, fc.scale,
                          fc.use_global) for fc in self.field_configs),)
        if gd_snap != self._overlay_gd_snap:
            self._overlay_dirty = True

        if self._overlay_dirty or not getattr(self, "_overlay_shader_tex_ids", None):
            # Ensure backing data is loaded for all source categories
            if self._fb_dirty:
                self._compute_fb_envelopes()
            if self._wv_dirty:
                if not self._wv_coeffs:
                    self._load_wavelet_data()
                self._compute_wavelet_scalogram()

            # Pre-compute per-category frequency & time axes
            _fb_freq_y: np.ndarray | None = None
            if hasattr(self, "_fb_bands") and self._fb_bands:
                _fb_freq_y = np.array(
                    [b.centre for b in self._fb_bands], dtype=np.float64)

            _wv_freq_y: np.ndarray | None = None
            if self._wv_meta and self._wv_coeffs:
                _wv_sr = float(self._wv_meta.get("sr", 48000))
                _wv_L = len(self._wv_coeffs) - 1
                _wvf = np.zeros(len(self._wv_coeffs), dtype=np.float64)
                _wvf[0] = _wv_sr / 2.0 ** (_wv_L + 1.5)
                for _k in range(1, _wv_L + 1):
                    _j = _wv_L - _k + 1
                    _wvf[_k] = _wv_sr / 2.0 ** (_j + 0.5)
                _wv_freq_y = _wvf

            # Build per-tex_name RGBA buffers on the common grid
            tex_bufs: dict[str, np.ndarray] = {}

            for fs, fc in zip(self.field_sources, self.field_configs):
                if fc.target < 0 or fc.target > 3:
                    continue
                arr = self._resolve_field_array(fs.name)
                if arr is None or arr.size == 0:
                    continue
                if arr.ndim != 2:
                    continue
                src_h, src_w = arr.shape
                cat = _FIELD_CATEGORY_MAP.get(fs.name, fs.category)

                # frequency axis (Y)
                if cat == "fb" and _fb_freq_y is not None \
                        and len(_fb_freq_y) == src_h:
                    src_y = _fb_freq_y
                elif cat == "wvlt" and _wv_freq_y is not None \
                        and len(_wv_freq_y) == src_h:
                    src_y = _wv_freq_y
                elif cat == "cqt" and len(self._freqs) == src_h:
                    src_y = np.asarray(self._freqs, dtype=np.float64)
                else:
                    src_y = np.arange(src_h, dtype=np.float64)

                # time axis (X)
                if cat == "cqt" and len(self.times) == src_w:
                    src_x = np.asarray(self.times, dtype=np.float64)
                else:
                    src_x = np.linspace(self.t_start, self.t_end, src_w,
                                        dtype=np.float64)

                resampled = _overlay_resample_to_grid(
                    arr, src_x, src_y, dst_x, dst_y, transpose=False)
                resampled = np.abs(resampled).astype(np.float32)

                # Rank-order norm must be done CPU-side
                fnorm = fc.norm_mode if not fc.use_global else gd.norm_mode
                if fnorm == 2:
                    flat = resampled.ravel()
                    n = flat.size
                    if n > 0:
                        order = flat.argsort().argsort()
                        resampled = (order.astype(np.float32) / max(n - 1, 1)).reshape(resampled.shape)
                    else:
                        resampled = np.zeros_like(resampled, dtype=np.float32)

                # Pack into the correct RGBA channel of the texture
                tname = fs.tex_name
                ch = fs.tex_channel
                if tname not in tex_bufs:
                    tex_bufs[tname] = np.zeros(
                        (grid_h, grid_w, 4), dtype=np.float32)
                tex_bufs[tname][:, :, ch] = resampled

            # Delete old overlay textures
            for old_id in getattr(self, "_overlay_shader_tex_ids", {}).values():
                if old_id:
                    glDeleteTextures([old_id])

            # Upload packed textures
            self._overlay_shader_tex_ids: dict[str, int] = {}
            for tname, buf in tex_bufs.items():
                self._overlay_shader_tex_ids[tname] = \
                    self._upload_texture_float(np.ascontiguousarray(buf))

            self._overlay_dirty = False
            self._overlay_gd_snap = gd_snap

        if not self._overlay_shader_tex_ids:
            return

        # Build overlay tex_units
        needed: set[str] = set()
        for i, fs in enumerate(self.field_sources):
            if self.field_configs[i].target >= 0:
                needed.add(fs.tex_name)
        saved_tex_units = dict(self._tex_units)
        self._tex_units = {}
        unit = 0
        for tex_name in sorted(needed):
            if tex_name not in self._overlay_shader_tex_ids:
                continue
            if unit >= 8:
                break
            self._tex_units[tex_name] = unit
            unit += 1

        # UV from viewport
        u0 = self.view_x0 / max(float(grid_w), 1.0)
        u1 = self.view_x1 / max(float(grid_w), 1.0)
        v0 = self.view_y0 / max(float(grid_h), 1.0)
        v1 = self.view_y1 / max(float(grid_h), 1.0)

        self._render_spectral_passes(
            self._overlay_shader_tex_ids, sx, sy, sw, sh, u0, v0, u1, v1)

        self._tex_units = saved_tex_units

    def _overlay_source_magnitude(self, layer: OverlayLayer) -> np.ndarray | None:
        """Return a 2-D magnitude array for the given overlay layer source.

        Uses the same pre-computed grids stored by the TF / TSC pipelines
        in ``_raw_arrays`` so that channel selection (left / right) works
        and no redundant recomputation is done.

        Shape is (rows, cols) in each source's native orientation:
        - CQT: (n_bins, n_frames) — ``|real + j*imag|``
        - FB: (n_bands, tex_w) — ``fb_mag_left`` / ``fb_mag_right``
        - Wavelet: (n_levels, tex_w) — ``wvlt_approx``  (mono only)
        """
        src = layer.source
        side = "left" if layer.channel == 0 else "right"

        if src == "cqt":
            r = self._raw_arrays.get(f"real_{side}")
            im = self._raw_arrays.get(f"imag_{side}")
            if r is None:
                return None
            mag = np.sqrt(r.astype(np.float64) ** 2 +
                          (im.astype(np.float64) ** 2 if im is not None
                           else 0.0))
            return mag.astype(r.dtype)

        if src == "fb":
            # Use the pre-resampled stereo grids stored by
            # _compute_fb_envelopes (fb_mag_left / fb_mag_right).
            grid = self._raw_arrays.get(f"fb_mag_{side}")
            if grid is not None:
                return grid
            # Fallback: try the other side or mono data
            grid = self._raw_arrays.get("fb_mag_left")
            return grid

        if src == "wavelet":
            # Wavelet data is mono — stored as wvlt_approx by
            # _compute_wavelet_scalogram.
            grid = self._raw_arrays.get("wvlt_approx")
            return grid

        return None

    # ---- Filterbank-as-TF rendering ----------------------------------------

    def _compute_fb_envelopes(self) -> None:
        """Load pre-baked per-band envelopes from the analysis folder.
        Falls back to a full Hilbert recompute only for legacy folders
        that lack the *fb_envelopes.npz* sidecar.
        """
        if not self.analysis_dir:
            print("Filterbank TF: no analysis folder loaded")
            self._fb_dirty = False
            return

        # --- Load band metadata -------------------------------------------
        fb_dir = os.path.join(self.analysis_dir, "filterbank")
        meta_path = os.path.join(fb_dir, "filterbank_meta.json")
        if os.path.isfile(meta_path):
            import json as _json
            with open(meta_path) as _f:
                _meta = _json.load(_f)
            self._fb_bands = [
                BandDef(fmin=bm["fmin"], fmax=bm["fmax"],
                        label=bm.get("label", ""))
                for bm in _meta["bands"]
            ]

        # --- Try cached envelopes first (instant) -------------------------
        cached = FilterBankDecomposition.load_envelopes(self.analysis_dir)
        if cached is not None:
            mags, phases, hops = cached
            self._fb_mags = mags
            self._fb_phases = phases
            self._fb_hops = hops
            self._fb_dirty = False
            desc = ", ".join(f"{len(m)}@h{h}"
                             for m, h in zip(mags, hops))
            print(f"Filterbank TF: loaded cached envelopes ({desc})")
        else:
            # Envelopes are baked during analysis — if missing, the
            # analysis folder predates envelope support.  Recompute from
            # the WAVs on disk (with progress) and cache for next time.
            print("Filterbank TF: no cached envelopes, loading WAVs "
                  "to recompute (one-time migration)...")
            fb = FilterBankDecomposition.load(self.analysis_dir)
            if fb is None or not fb.subbands:
                print("Filterbank TF: no filterbank data in "
                      f"{self.analysis_dir} (run Filter Bank analysis first)")
                self._fb_dirty = False
                return

            self._fb_bands = list(fb.bands)
            print("Filterbank TF: computing Hilbert envelopes ...")
            mags, phases, hops = fb.compute_envelopes()
            fb.save_envelopes(self.analysis_dir, mags, phases, hops)
            desc = ", ".join(f"{len(m)}@h{h}"
                             for m, h in zip(mags, hops))
            print(f"Filterbank TF: computed and cached ({desc})")

            self._fb_mags = mags
            self._fb_phases = phases
            self._fb_hops = hops
            self._fb_dirty = False

        # --- Resample ragged per-band arrays to a common texture grid -----
        # The texture is (n_bands, tex_w) where tex_w is the finest
        # band's frame count, clamped to the GPU's max texture width.
        n_bands = len(mags)
        if n_bands == 0:
            return
        max_frames = max(len(m) for m in mags)
        max_w = int(glGetIntegerv(GL_MAX_TEXTURE_SIZE))
        tex_w = min(max_frames, max_w)

        # Build regular (n_bands, tex_w) grids by resampling each
        # band from its native frame count onto the common grid.
        # Use nearest-neighbour (sample-and-hold) so that sparse low-freq
        # bands show stepped blocks rather than washed-out linear ramps.
        mag_grid = np.zeros((n_bands, tex_w), dtype=np.float32)
        phase_grid = np.zeros((n_bands, tex_w), dtype=np.float32)
        dst_idx = np.arange(tex_w)

        def _resample_envelopes(m_list, p_list):
            """Resample ragged per-band envelopes to (n_bands, tex_w)."""
            dt = m_list[0].dtype if len(m_list) > 0 else np.float32
            mg = np.zeros((n_bands, tex_w), dtype=dt)
            pg = np.zeros((n_bands, tex_w), dtype=dt)
            for i in range(n_bands):
                n_src = len(m_list[i])
                if n_src == 0:
                    continue
                if n_src == tex_w:
                    mg[i] = m_list[i]
                    pg[i] = p_list[i]
                else:
                    src_sel = np.clip(
                        (dst_idx * n_src + tex_w // 2) // tex_w,
                        0, n_src - 1)
                    mg[i] = m_list[i][src_sel]
                    pg[i] = p_list[i][src_sel]
            return mg, pg

        mag_grid, phase_grid = _resample_envelopes(mags, phases)

        # --- Always load L+R envelopes (mono input stored L=R) ----------
        mag_grid_l = mag_grid       # fallback: mono for both
        phase_grid_l = phase_grid
        mag_grid_r = mag_grid
        phase_grid_r = phase_grid

        for ch_label in ("left", "right"):
            ch_path = os.path.join(fb_dir, f"fb_envelopes_{ch_label}.npz")
            if os.path.isfile(ch_path):
                d = np.load(ch_path)
                if "n_bands" in d:
                    n_ch = int(d["n_bands"])
                    if n_ch == n_bands:
                        m_ch = [d[f"mag_{i}"] for i in range(n_ch)]
                        p_ch = [d[f"phase_{i}"] for i in range(n_ch)]
                        mg_ch, pg_ch = _resample_envelopes(m_ch, p_ch)
                        if ch_label == "left":
                            mag_grid_l = mg_ch
                            phase_grid_l = pg_ch
                        else:
                            mag_grid_r = mg_ch
                            phase_grid_r = pg_ch

        # --- Bandwidth energy regularization (optional) --------------------
        # In a log-spaced filterbank the higher bands are exponentially
        # wider so they capture proportionally more broadband energy.
        # Divide each band row by sqrt(bandwidth_hz) to convert raw
        # amplitude to spectral density before the global-max scaling.
        _do_q_norm = (self.file_panel is not None and self.file_panel.fb_q_norm)
        if _do_q_norm and hasattr(self, "_fb_bands") and self._fb_bands:
            fb_sr = int(_meta.get("sr", 44100)) if os.path.isfile(meta_path) else 44100
            _nyq = fb_sr / 2.0
            bw_divisors = np.ones(n_bands, dtype=np.float32)
            for i, band in enumerate(self._fb_bands):
                bw = min(band.fmax, _nyq) - max(band.fmin, 0.0)
                if bw > 0:
                    bw_divisors[i] = np.sqrt(bw)
            # Apply as column vector — divides every sample in each row
            bw_col = bw_divisors[:, np.newaxis]
            mag_grid_l = mag_grid_l / bw_col
            mag_grid_r = mag_grid_r / bw_col
        self._fb_q_norm_last = _do_q_norm

        # --- Upload shader-compatible RGBA float32 textures ---------------
        # Shared max across L/R so they scale identically.
        # Preserve native dtype through normalization; float32 conversion
        # happens only at the GPU upload boundary (_pack_and_upload).
        mag_max_lr = max(float(mag_grid_l.max()),
                         float(mag_grid_r.max()), 1e-12)
        mag_norm_l = np.clip(mag_grid_l / mag_max_lr, 0.0, 1.0)
        mag_norm_r = np.clip(mag_grid_r / mag_max_lr, 0.0, 1.0)
        phase_norm_l = np.clip(
            (phase_grid_l + np.pi) / (2.0 * np.pi), 0.0, 1.0)
        phase_norm_r = np.clip(
            (phase_grid_r + np.pi) / (2.0 * np.pi), 0.0, 1.0)

        # Clean up old FB textures
        for old_id in self._fb_shader_tex_ids.values():
            glDeleteTextures([old_id])
        self._fb_shader_tex_ids = {}

        def _pack_and_upload(r, g, b, a) -> int:
            """Stack channels and upload. Row 0 = band 0 = V=0 = bottom."""
            rgba = np.stack([r, g, b, a], axis=-1).astype(np.float32)
            rgba = np.ascontiguousarray(rgba)
            return self._upload_texture_float(rgba)

        alpha = np.ones((n_bands, tex_w), dtype=mag_grid_l.dtype)
        zeros = np.zeros((n_bands, tex_w), dtype=mag_grid_l.dtype)

        # Only L/R polar textures needed — mono is derived in the
        # shader by mapping both L and R to the same display colour.
        self._fb_shader_tex_ids["fb_left_polar"] = _pack_and_upload(
            mag_norm_l, phase_norm_l, zeros, alpha)
        self._fb_shader_tex_ids["fb_right_polar"] = _pack_and_upload(
            mag_norm_r, phase_norm_r, zeros, alpha)

        # Store pre-normalization FB grids so the routing / synthesis
        # pipeline can access them by canonical field name.
        self._raw_arrays["fb_mag_left"] = mag_grid_l
        self._raw_arrays["fb_phase_left"] = phase_grid_l
        self._raw_arrays["fb_mag_right"] = mag_grid_r
        self._raw_arrays["fb_phase_right"] = phase_grid_r

        print(f"Filterbank TF: {n_bands} bands, "
              f"tex {n_bands}\u00d7{tex_w}")

    def _render_fb_tf(self, sx: int, sy: int, sw: int, sh: int,
                      kind: str = "mag") -> None:
        """Render filterbank-as-TF heatmap through the unified 3-pass
        pipeline."""
        if sw <= 0 or sh <= 0:
            return
        # Detect Q-norm toggle and force recompute
        _want_q = (self.file_panel is not None and self.file_panel.fb_q_norm)
        if _want_q != getattr(self, "_fb_q_norm_last", False):
            self._fb_dirty = True
        if self._fb_dirty:
            self._compute_fb_envelopes()

        if not self._fb_shader_tex_ids or not self._pass1_program:
            return

        n_bands = len(self._fb_bands) if self._fb_bands else 1

        cqt_total = float(self.n_frames) if self.n_frames > 0 else 1.0
        u0 = self.view_x0 / cqt_total
        u1 = self.view_x1 / cqt_total
        v0 = self.view_y0 / n_bands
        v1 = self.view_y1 / n_bands

        # Build tex_units for the FB textures that are actually needed
        needed: set[str] = set()
        for i, fs in enumerate(self.field_sources):
            if self.field_configs[i].target >= 0:
                needed.add(fs.tex_name)
        saved_tex_units = dict(self._tex_units)
        self._tex_units = {}
        unit = 0
        for tex_name in sorted(needed):
            if tex_name not in self._fb_shader_tex_ids:
                continue
            if unit >= 8:
                break
            self._tex_units[tex_name] = unit
            unit += 1

        self._render_spectral_passes(
            self._fb_shader_tex_ids, sx, sy, sw, sh, u0, v0, u1, v1)

        self._tex_units = saved_tex_units

    def _hybrid_cqt_skip_bins(self) -> int:
        """Return how many low-frequency CQT bins to skip in hybrid mode.

        In hybrid mode, filterbank envelopes cover the low frequencies
        and CQT covers the highs.  CQT bins whose centre frequency
        falls below the highest FB band's upper edge are redundant and
        should be truncated so the two sources meet without overlap.
        """
        if not self._fb_bands or len(self._freqs) == 0:
            return 0
        # Upper edge of the topmost FB band is the crossover
        cross_hz = max(b.fmax for b in self._fb_bands)
        skip = int(np.searchsorted(self._freqs, cross_hz))
        return min(skip, self.n_bins)

    def _hybrid_cross_bin(self) -> float:
        """Return the unified-Y coordinate at the hybrid crossover.

        In the hybrid coordinate system FB bands occupy [0..n_fb) and
        CQT bins occupy [n_fb..n_fb+n_cqt_above).  The crossover sits
        at n_fb.
        """
        n_fb = len(self._fb_bands) if self._fb_bands else 0
        return float(n_fb)

    def _render_hybrid_tf(self, sx: int, sy: int,
                          sw: int, sh: int) -> None:
        """Composite FB (low freqs) and CQT (high freqs) into one view.

        The unified Y axis places FB bands at [0..n_fb) and CQT bins
        (above crossover only) at [n_fb..n_fb+n_cqt_above).  Below the
        crossover the filterbank Hilbert-envelope texture is drawn;
        above it the progressive-scan CQT texture is drawn.  CQT bins
        below the crossover frequency are truncated so the two sources
        meet without overlap.
        """
        if sw <= 0 or sh <= 0:
            return

        cross_bin = self._hybrid_cross_bin()

        # If crossover is outside the viewport, fall through to one source
        if cross_bin <= self.view_y0:
            self._render_tf(sx, sy, sw, sh)
            return
        if cross_bin >= self.view_y1:
            # Entire viewport is below crossover → render FB
            self._render_hybrid_fb_portion(sx, sy, sw, sh,
                                           self.view_y0, self.view_y1)
            return

        # --- Split the viewport -------------------------------------------
        view_h = self.view_y1 - self.view_y0
        frac_above = (self.view_y1 - cross_bin) / view_h
        split_px = int(frac_above * sh)     # pixels for CQT (top)
        bot_px = sh - split_px              # pixels for FB (bottom)

        ww, wh = self.win_w, self.win_h

        # --- Top half: CQT -----------------------------------------------
        if split_px > 0 and self._pass1_program and (
                self._front_tex_ids or self._back_tex_ids):
            if self._front_tex_ids:
                self._render_spectral_passes(
                    self._front_tex_ids, sx, sy, sw, split_px,
                    0.0, 0.0, 1.0, 1.0)

            if self._back_tex_ids and not self._scan_complete:
                scan_frac = self._scan_col / max(1, self._scan_total)
                scan_px = int(scan_frac * sw)
                if scan_px > 0:
                    glEnable(GL_SCISSOR_TEST)
                    glScissor(sx, wh - sy - split_px,
                              max(1, scan_px), split_px)
                    self._render_spectral_passes(
                        self._back_tex_ids, sx, sy, sw, split_px,
                        0.0, 0.0, 1.0, 1.0)
                    glDisable(GL_SCISSOR_TEST)

        # --- Bottom half: FB envelopes ------------------------------------
        if bot_px > 0:
            self._render_hybrid_fb_portion(
                sx, sy + split_px, sw, bot_px,
                self.view_y0, cross_bin)

    def _render_hybrid_fb_portion(
        self, sx: int, sy: int, sw: int, sh: int,
        bin_lo: float, bin_hi: float,
    ) -> None:
        """Render FB envelope data into a screen rect via unified pipeline.

        In the unified hybrid coordinate system ``bin_lo`` / ``bin_hi``
        are already expressed in FB-band-index units (band *i* occupies
        [i, i+1)), so no frequency conversion is needed.
        """
        if sw <= 0 or sh <= 0:
            return
        _want_q = (self.file_panel is not None and self.file_panel.fb_q_norm)
        if _want_q != getattr(self, "_fb_q_norm_last", False):
            self._fb_dirty = True
        if self._fb_dirty:
            self._compute_fb_envelopes()
        if not self._fb_shader_tex_ids or not self._pass1_program:
            return

        n_bands = len(self._fb_bands) if self._fb_bands else 1

        # Direct band-index → UV
        v0 = max(0.0, bin_lo) / n_bands
        v1 = min(float(n_bands), bin_hi) / n_bands

        # X UV: map CQT-frame viewport to time fraction
        cqt_total = float(self.n_frames) if self.n_frames > 0 else 1.0
        u0 = self.view_x0 / cqt_total
        u1 = self.view_x1 / cqt_total

        # Bind FB textures
        needed: set[str] = set()
        for i, fs in enumerate(self.field_sources):
            if self.field_configs[i].target >= 0:
                needed.add(fs.tex_name)
        saved_tex_units = dict(self._tex_units)
        self._tex_units = {}
        unit = 0
        for tex_name in sorted(needed):
            if tex_name not in self._fb_shader_tex_ids:
                continue
            if unit >= 8:
                break
            self._tex_units[tex_name] = unit
            unit += 1

        self._render_spectral_passes(
            self._fb_shader_tex_ids, sx, sy, sw, sh, u0, v0, u1, v1)

        self._tex_units = saved_tex_units

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

        style = _GRID_STYLES[self.grid_style_idx]
        grid_color, tick_color = _GRID_STYLE_COLORS[style]

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

        # --- Shared horizontal grid + tick drawing -------------------------
        def draw_h_grid(screen_ys: list[float]) -> None:
            if grid_color is None:
                return
            glBegin(GL_LINES)
            for scr_y in screen_ys:
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

        def draw_h_labels(labels: list[tuple[float, str]]) -> None:
            glEnable(GL_TEXTURE_2D)
            glUseProgram(0)
            for scr_y, text in labels:
                atlas.draw_string(text, float(sx) - 8, scr_y, ww, wh,
                                  anchor_x=1.0, anchor_y=0.5)
                atlas.draw_string(text, float(sx + sw) + 8, scr_y, ww, wh,
                                  anchor_x=0.0, anchor_y=0.5)
            glDisable(GL_TEXTURE_2D)

        # --- Shared vertical grid + tick drawing ---------------------------
        def draw_v_grid(screen_xs: list[float]) -> None:
            if grid_color is None:
                return
            glBegin(GL_LINES)
            for scr_x in screen_xs:
                nx = px_to_ndc_x(scr_x)
                ny_top = px_to_ndc_y(float(sy))
                ny_bot = px_to_ndc_y(float(sy + sh))
                glColor4f(*grid_color)
                glVertex2f(nx, ny_bot)
                glVertex2f(nx, ny_top)
                glColor4f(*tick_color)
                glVertex2f(nx, px_to_ndc_y(float(sy)))
                glVertex2f(nx, px_to_ndc_y(float(sy) - 4))
                glVertex2f(nx, px_to_ndc_y(float(sy + sh)))
                glVertex2f(nx, px_to_ndc_y(float(sy + sh) + 4))
            glEnd()

        def draw_v_labels(labels: list[tuple[float, str]]) -> None:
            glEnable(GL_TEXTURE_2D)
            glUseProgram(0)
            for scr_x, text in labels:
                atlas.draw_string(text, scr_x, float(sy) - 4, ww, wh,
                                  anchor_x=0.5, anchor_y=1.0)
                atlas.draw_string(text, scr_x, float(sy + sh) + 4, ww, wh,
                                  anchor_x=0.5, anchor_y=0.0)
            glDisable(GL_TEXTURE_2D)

        # ---- Frequency axis (horizontal grid lines, labels left + right) --
        if self.show_freq_axis and self.display_mode == "tf" and self.tf_source in ("cqt", "hybrid"):
            # In hybrid mode CQT bin i lives at unified Y = i + cross_bin
            cqt_y_off = self._hybrid_cross_bin() if self.tf_source == "hybrid" else 0.0
            # Decide which semitone rows are visible and not too dense
            visible_idxs: list[int] = []
            for i in np.flatnonzero(self._semitone_mask):
                if self.view_y0 <= i + cqt_y_off <= self.view_y1:
                    visible_idxs.append(int(i))

            # Thin out when too dense: skip labels closer than 14 px
            min_gap = 14
            shown_idxs: list[int] = []
            last_screen_y = -999.0
            for i in visible_idxs:
                scr_y = bin_to_sy(float(i) + cqt_y_off)
                if abs(scr_y - last_screen_y) >= min_gap:
                    shown_idxs.append(i)
                    last_screen_y = scr_y

            draw_h_grid([bin_to_sy(float(i) + cqt_y_off) for i in shown_idxs])

            labels: list[tuple[float, str]] = []
            for i in shown_idxs:
                name = self._note_names[i] if i < len(self._note_names) else ""
                if not name:
                    continue
                freq = float(self._freqs[i])
                labels.append((bin_to_sy(float(i) + cqt_y_off),
                               f"{name}  {freq:.1f} Hz"))
            draw_h_labels(labels)

            # ---- Hybrid crossover indicator line ----
            if self.tf_source == "hybrid":
                cross_bin = self._hybrid_cross_bin()
                if self.view_y0 < cross_bin < self.view_y1:
                    scr_y_cross = bin_to_sy(cross_bin)
                    ny_cross = px_to_ndc_y(scr_y_cross)
                    nx_l = px_to_ndc_x(float(sx))
                    nx_r = px_to_ndc_x(float(sx + sw))
                    glLineWidth(2.0)
                    glBegin(GL_LINES)
                    glColor4f(1.0, 0.6, 0.0, 0.7)
                    glVertex2f(nx_l, ny_cross)
                    glVertex2f(nx_r, ny_cross)
                    glEnd()
                    glLineWidth(1.0)
                    # Label
                    glEnable(GL_TEXTURE_2D)
                    glUseProgram(0)
                    synth = self.synth
                    if synth:
                        fc = hybrid_crossover_freq(
                            synth.bins_per_octave,
                            synth.hop_length, synth.sr)
                    else:
                        fc = 0.0
                    atlas.draw_string(
                        f"\u2500 hybrid {fc:.0f} Hz \u2500",
                        float(sx + sw // 2), scr_y_cross - 2,
                        ww, wh, anchor_x=0.5, anchor_y=1.0)
                    glDisable(GL_TEXTURE_2D)

        # ---- Filterbank-TF frequency axis (band boundary labels) ----------
        if self.show_freq_axis and self.display_mode == "tf" and self.tf_source in ("fb", "hybrid") and self._fb_bands:
            n_bands = len(self._fb_bands)
            min_gap = 14
            shown_bands: list[tuple[int, float, str]] = []
            last_screen_y = -999.0
            for i, band in enumerate(self._fb_bands):
                band_centre = float(i) + 0.5
                if not (self.view_y0 <= band_centre <= self.view_y1):
                    continue
                scr_y = bin_to_sy(band_centre)
                if abs(scr_y - last_screen_y) >= min_gap:
                    sr_for_label = self.synth.sr if self.synth else 44100
                    if _FB_LABEL_MODES[self.fb_label_mode_idx] == "Mid+BW":
                        label_text = band.mid_label(sr_for_label)
                    else:
                        label_text = band.label or band.auto_label(sr_for_label)
                    shown_bands.append((i, scr_y, label_text))
                    last_screen_y = scr_y

            boundary_ys: list[float] = []
            for i in range(n_bands + 1):
                boundary = float(i)
                if self.view_y0 <= boundary <= self.view_y1:
                    boundary_ys.append(bin_to_sy(boundary))
            draw_h_grid(boundary_ys)
            draw_h_labels([(scr_y, text) for _, scr_y, text in shown_bands])

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

                amp_ticks: list[float] = []
                a = a_first
                while a <= amp_hi + amp_step * 0.01:
                    amp_ticks.append(a)
                    a += amp_step

                draw_h_grid([amp_to_screen(a) for a in amp_ticks])

                amp_labels: list[tuple[float, str]] = []
                for a in amp_ticks:
                    if abs(a) < 1e-9:
                        label = "0"
                    else:
                        db = 20.0 * math.log10(max(abs(a), 1e-12))
                        label = f"{a:+.2f}  {db:.0f} dB"
                    amp_labels.append((amp_to_screen(a), label))
                draw_h_labels(amp_labels)

        # ---- Scale axis (horizontal grid, labels left + right) — TSC ------
        if self.show_freq_axis and self.display_mode == "tsc" and self._wv_n_levels > 0:
            n_levels = self._wv_n_levels
            min_gap = 14
            shown_levels: list[tuple[int, float, str]] = []
            last_screen_y = -999.0
            for i in range(n_levels):
                level_centre = float(i) + 0.5
                if not (self.view_y0 <= level_centre <= self.view_y1):
                    continue
                scr_y = bin_to_sy(level_centre)
                if abs(scr_y - last_screen_y) >= min_gap:
                    if i == 0:
                        lbl = "Approx"
                    else:
                        lbl = f"D{i}"
                    shown_levels.append((i, scr_y, lbl))
                    last_screen_y = scr_y

            level_boundary_ys: list[float] = []
            for i in range(n_levels + 1):
                boundary = float(i)
                if self.view_y0 <= boundary <= self.view_y1:
                    level_boundary_ys.append(bin_to_sy(boundary))
            draw_h_grid(level_boundary_ys)
            draw_h_labels([(scr_y, lbl) for _, scr_y, lbl in shown_levels])

        # ---- Time axis (vertical grid lines, labels top + bottom) ---------
        _eff_n_frames = (self._wv_n_frames
                         if self.display_mode == "tsc" and self._wv_n_frames > 0
                         else self.n_frames)
        if self.show_time_axis and _eff_n_frames > 0 and self.t_dur > 0:
            t0_view = self.t_start + (self.view_x0 / _eff_n_frames) * self.t_dur
            t1_view = self.t_start + (self.view_x1 / _eff_n_frames) * self.t_dur
            dur_view = t1_view - t0_view
            if dur_view > 0:
                tick_step = _nice_step(dur_view, sw)
                t_first = math.ceil(t0_view / tick_step) * tick_step

                if tick_step >= 1.0:
                    fmt = "{:.0f}"
                elif tick_step >= 0.1:
                    fmt = "{:.1f}"
                elif tick_step >= 0.01:
                    fmt = "{:.2f}"
                else:
                    fmt = "{:.3f}"

                ticks: list[tuple[float, str]] = []
                t = t_first
                while t <= t1_view + tick_step * 0.01:
                    frame = (t - self.t_start) / self.t_dur * _eff_n_frames
                    ticks.append((frame, fmt.format(t)))
                    t += tick_step

                draw_v_grid([frame_to_sx(frame) for frame, _ in ticks])
                draw_v_labels([(frame_to_sx(frame), label)
                               for frame, label in ticks])

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
    # Check filterbank metadata for stored wav_path
    fb_meta_path = os.path.join(analysis_dir, "filterbank",
                                "filterbank_meta.json")
    if os.path.isfile(fb_meta_path):
        import json as _json
        with open(fb_meta_path, "r") as _f:
            _fbm = _json.load(_f)
        wp = _fbm.get("wav_path", "")
        if wp and os.path.isfile(wp):
            return wp
    # Check wavelet metadata for stored wav_path
    wv_meta_path = os.path.join(analysis_dir, "wavelet",
                                "wavelet_meta.json")
    if os.path.isfile(wv_meta_path):
        import json as _json
        with open(wv_meta_path, "r") as _f:
            _wvm = _json.load(_f)
        wp = _wvm.get("wav_path", "")
        if wp and os.path.isfile(wp):
            return wp
    # Try to find any top-level audio file in the analysis dir
    audio_hits: list[str] = []
    if os.path.isdir(analysis_dir):
        for _fn in os.listdir(analysis_dir):
            if os.path.splitext(_fn)[1].lower() in _AUDIO_EXTS:
                audio_hits.append(os.path.join(analysis_dir, _fn))
    audio_hits.sort()
    if audio_hits:
        return audio_hits[0]
    # Infer from folder name: <name>_analysis -> <name>.wav in parent
    base = os.path.basename(analysis_dir)
    if base.endswith("_analysis"):
        stem = base[:-len("_analysis")]
        parent = os.path.dirname(analysis_dir)
        for ext in (".wav", ".flac", ".mp3", ".ogg"):
            candidate = os.path.join(parent, stem + ext)
            if os.path.isfile(candidate):
                return candidate
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
