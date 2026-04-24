#!/usr/bin/env python3
"""
Spectrogram Viewer — OpenGL/Pygame data-texture navigator with per-field control.

Loads 16-bit RGBA data textures produced by bass_plot.py.  Each data field
(e.g., real_left, mag_right, onset) is presented as a tab with its own
normalization, gamma, and scale controls.  Multiple fields can be mapped to
the same display channel (R, G, B, A); contributions are averaged and a
global output gamma / scale is applied after aggregation.

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
    Escape          Quit

Dependencies
------------
pip install numpy pygame PyOpenGL Pillow scipy
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, KEYDOWN, KEYUP, MOUSEBUTTONDOWN, MOUSEBUTTONUP,
    MOUSEMOTION, MOUSEWHEEL, OPENGL, QUIT, RESIZABLE, VIDEORESIZE,
    K_SPACE, K_ESCAPE, K_LEFT, K_RIGHT, K_HOME, K_END, K_TAB,
    K_r, K_f, K_t,
    K_LSHIFT, K_RSHIFT,
)
from OpenGL.GL import *
from OpenGL.GL import shaders as gl_shaders
from OpenGL.GLU import gluOrtho2D
from PIL import Image
from scipy.io import wavfile


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

AXIS_FRAG_SRC = """
#version 330 compatibility
in vec2 vUV;
out vec4 fragColor;
uniform sampler2D uTex;
void main() {
    fragColor = texture(uTex, vUV);
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

NORM_MODES = ["linear", "dB"]
TARGET_LABELS = ["none", "R", "G", "B", "A"]
TARGET_VALUES = [-1, 0, 1, 2, 3]


@dataclass
class FieldSource:
    name: str
    display_name: str
    tex_name: str
    tex_channel: int  # 0=R 1=G 2=B 3=A


@dataclass
class FieldConfig:
    target: int = -1       # display channel: 0=R 1=G 2=B 3=A, -1=disabled
    norm_mode: int = 0     # 0=linear 1=dB
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


def resolve_field_sources(meta: dict) -> list[FieldSource]:
    """Build deduplicated list of data fields from texture metadata."""
    seen: set[str] = set()
    sources: list[FieldSource] = []
    for tex_name, tex_info in meta.get("textures", {}).items():
        for ch_idx, ch_key in enumerate("RGBA"):
            ch_data = tex_info.get(ch_key)
            if ch_data is None or not isinstance(ch_data, dict):
                continue
            field_name = ch_data.get("field", "")
            if not field_name or field_name in seen:
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
# GUI Panel — per-field tab interface
# ---------------------------------------------------------------------------

class FieldTabPanel:
    """Panel with a tab per data field + global defaults + output controls."""

    PANEL_W = 280
    ROW_H = 22
    PAD = 6

    def __init__(self, fields: list[FieldSource],
                 configs: list[FieldConfig],
                 global_defaults: GlobalDefaults) -> None:
        self.fields = fields
        self.configs = configs
        self.global_defaults = global_defaults
        self.selected_idx = 0
        self._active_dropdown: str | None = None
        self._dropdown_opts: list[str] = []
        self._dropdown_rect: pygame.Rect = pygame.Rect(0, 0, 0, 0)
        self._dropdown_item_rects: list[pygame.Rect] = []
        self._dragging: str | None = None
        self.visible = True
        self.font: pygame.font.Font | None = None
        self._item_map: dict[str, tuple[pygame.Rect, Any]] = {}
        self._field_row_rects: list[pygame.Rect] = []

    def _ensure_font(self) -> None:
        if self.font is None:
            pygame.font.init()
            self.font = pygame.font.SysFont("consolas", 13)

    @property
    def panel_rect(self) -> pygame.Rect:
        sw = pygame.display.get_surface().get_width()
        sh = pygame.display.get_surface().get_height()
        return pygame.Rect(sw - self.PANEL_W, 0, self.PANEL_W, sh)

    # ---- Rendering --------------------------------------------------------

    def render(self) -> pygame.Surface | None:
        if not self.visible:
            return None
        self._ensure_font()
        font = self.font
        w = self.PANEL_W
        self._item_map = {}
        self._field_row_rects = []

        rows: list[tuple[str, Any]] = []

        # -- Field list --
        rows.append(("label", "\u2500\u2500 Data Fields \u2500\u2500"))
        for i, fs in enumerate(self.fields):
            rows.append(("field_row", (i, fs, self.configs[i])))

        rows.append(("label", ""))

        # -- Selected field controls --
        sel = self.selected_idx
        sel_fs = self.fields[sel]
        sel_cfg = self.configs[sel]
        rows.append(("label", f"\u2500\u2500 {sel_fs.display_name} \u2500\u2500"))

        tgt_label = TARGET_LABELS[TARGET_VALUES.index(sel_cfg.target)]
        rows.append(("dropdown", ("f_target", tgt_label, TARGET_LABELS)))
        rows.append(("toggle", ("f_custom", "Custom", not sel_cfg.use_global)))

        gd = self.global_defaults
        eff_norm = NORM_MODES[sel_cfg.norm_mode if not sel_cfg.use_global
                              else gd.norm_mode]
        eff_gamma = sel_cfg.gamma if not sel_cfg.use_global else gd.gamma
        eff_scale = sel_cfg.scale if not sel_cfg.use_global else gd.scale

        rows.append(("dropdown", ("f_norm", eff_norm, NORM_MODES)))
        rows.append(("slider", ("f_gamma", eff_gamma, 0.1, 5.0, "{:.2f}")))
        rows.append(("slider", ("f_scale", eff_scale, 0.01, 10.0, "{:.2f}")))

        rows.append(("label", ""))

        # -- Global defaults --
        rows.append(("label", "\u2500\u2500 Global Defaults \u2500\u2500"))
        rows.append(("dropdown", ("g_norm",
                                   NORM_MODES[gd.norm_mode], NORM_MODES)))
        rows.append(("slider", ("g_gamma", gd.gamma, 0.1, 5.0, "{:.2f}")))
        rows.append(("slider", ("g_scale", gd.scale, 0.01, 10.0, "{:.2f}")))

        rows.append(("label", ""))

        # -- Output post-aggregation --
        rows.append(("label", "\u2500\u2500 Output \u2500\u2500"))
        rows.append(("slider", ("o_gamma", gd.out_gamma, 0.1, 5.0, "{:.2f}")))
        rows.append(("slider", ("o_scale", gd.out_scale, 0.01, 10.0, "{:.2f}")))

        # -- Layout --
        total_h = self.PAD
        for rtype, _ in rows:
            total_h += self.ROW_H + 2
        total_h += self.PAD

        surf = pygame.Surface((w, total_h), pygame.SRCALPHA)
        surf.fill((30, 30, 30, 210))

        y = self.PAD
        field_uses_global = sel_cfg.use_global

        for rtype, rdata in rows:
            if rtype == "label":
                txt = font.render(rdata, True, (180, 180, 180))
                surf.blit(txt, (self.PAD, y + 2))

            elif rtype == "field_row":
                idx, fs, cfg = rdata
                rect = pygame.Rect(self.PAD, y, w - 2 * self.PAD, self.ROW_H)
                if idx == self.selected_idx:
                    pygame.draw.rect(surf, (50, 50, 70), rect)
                pygame.draw.rect(surf, (60, 60, 80), rect, 1)
                txt = font.render(fs.display_name, True, (200, 200, 200))
                surf.blit(txt, (self.PAD + 4, y + 3))
                tgt_text = TARGET_LABELS[TARGET_VALUES.index(cfg.target)]
                tgt_colors = {
                    "R": (255, 80, 80), "G": (80, 255, 80),
                    "B": (80, 80, 255), "A": (200, 200, 200),
                    "none": (80, 80, 80),
                }
                color = tgt_colors.get(tgt_text, (80, 80, 80))
                arrow = font.render(f"\u2192 {tgt_text}", True, color)
                surf.blit(arrow, (w - 60, y + 3))
                self._field_row_rects.append(rect)

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
                track_x = self.PAD
                track_w = w - 2 * self.PAD - 50
                track_rect = pygame.Rect(track_x, y + 8, track_w, 6)
                tc = (35, 35, 40) if greyed else (50, 50, 60)
                pygame.draw.rect(surf, tc, track_rect)
                frac = (val - vmin) / max(vmax - vmin, 1e-9)
                thumb_x = track_x + int(frac * track_w)
                thc = (70, 80, 100) if greyed else (120, 140, 200)
                pygame.draw.rect(surf, thc,
                                 pygame.Rect(thumb_x - 5, y + 3, 10, 16))
                vc = (100, 100, 100) if greyed else (200, 200, 200)
                vtxt = font.render(fmt.format(val), True, vc)
                surf.blit(vtxt, (track_x + track_w + 6, y + 2))
                if not greyed:
                    self._item_map[key] = (
                        pygame.Rect(track_x, y, track_w, self.ROW_H),
                        (vmin, vmax))

            y += self.ROW_H + 2

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

            # Field rows
            for i, rect in enumerate(self._field_row_rects):
                if rect.collidepoint(lx, ly):
                    self.selected_idx = i
                    return True

            # Toggle
            if "f_custom" in self._item_map:
                rect, _ = self._item_map["f_custom"]
                if rect.collidepoint(lx, ly):
                    cfg = self.configs[self.selected_idx]
                    if cfg.use_global:
                        # Copy globals so there is no visual jump
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

            # Sliders
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

        if event.type == MOUSEMOTION and self._dragging:
            mx, my = event.pos
            lx = mx - pr.x
            key = self._dragging
            if key in self._item_map:
                rect, (vmin, vmax) = self._item_map[key]
                self._update_slider(key, lx, rect, vmin, vmax)
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
# Viewer
# ---------------------------------------------------------------------------

class SpectrogramViewer:
    """OpenGL data-texture viewer with per-field channel mapping."""

    def __init__(self, analysis_dir: str, audio_path: str) -> None:
        self.analysis_dir = analysis_dir
        self.audio_path = audio_path

        # Load metadata
        meta_path = os.path.join(analysis_dir, "texture_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)

        self.n_bins = self.meta["n_bins"]
        self.n_frames = self.meta["n_frames"]

        # Resolve data fields from textures
        self.field_sources = resolve_field_sources(self.meta)
        self.field_configs = [FieldConfig() for _ in self.field_sources]
        self.global_defaults = GlobalDefaults()

        # Sensible initial assignments
        for i, fs in enumerate(self.field_sources):
            if fs.name == "mag_left":
                self.field_configs[i].target = 0   # R
            elif fs.name == "mag_right":
                self.field_configs[i].target = 1   # G

        # Time range from npz
        npz_path = os.path.join(analysis_dir, "cqt_data.npz")
        d = np.load(npz_path, allow_pickle=True)
        self.times = d["times"]
        self.t_start = float(self.times[0])
        self.t_end = float(self.times[-1])
        self.t_dur = self.t_end - self.t_start
        d.close()

        # Display state
        self.win_w = 1600
        self.win_h = 900
        self.view_x0: float = 0.0
        self.view_y0: float = 0.0
        self.view_x1: float = float(self.n_frames)
        self.view_y1: float = float(self.n_bins)

        self.show_freq_axis = True
        self.show_time_axis = True
        self.axis_margin_left = 0
        self.axis_margin_right = 0
        self.axis_margin_top = 0
        self.axis_margin_bottom = 0

        self._rect_start: tuple[int, int] | None = None
        self._rect_end: tuple[int, int] | None = None
        self._is_panning = False
        self._pan_last: tuple[int, int] | None = None

        # GL handles
        self._program = 0
        self._axis_program = 0
        self._tex_units: dict[str, int] = {}
        self._tex_ids: dict[str, int] = {}
        self._freq_axis_tex = 0
        self._time_axis_tex = 0
        self._gui_tex = 0

        self.player: AudioPlayer | None = None
        self.gui: FieldTabPanel | None = None

    # ---- Main loop --------------------------------------------------------

    def run(self) -> None:
        pygame.init()
        sr_hint = 44100
        try:
            d = np.load(os.path.join(self.analysis_dir, "cqt_data.npz"),
                        allow_pickle=True)
            sr_hint = int(d["sr"])
            d.close()
        except Exception:
            pass
        pygame.mixer.pre_init(frequency=sr_hint, size=-16, channels=2,
                              buffer=2048)
        pygame.mixer.init()
        pygame.display.set_mode(
            (self.win_w, self.win_h), DOUBLEBUF | OPENGL | RESIZABLE,
        )
        pygame.display.set_caption("Spectrogram Viewer")
        pygame.font.init()

        self._init_gl()
        self.gui = FieldTabPanel(self.field_sources, self.field_configs,
                                 self.global_defaults)
        self._load_all_textures()
        self._load_axis_textures()
        self.player = AudioPlayer(self.audio_path)

        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                    continue

                if self.gui and self.gui.handle_event(event):
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
                    self._load_all_textures()
                    self._load_axis_textures()

            # Keyboard-held scrolling
            keys = pygame.key.get_pressed()
            speed = (self.view_x1 - self.view_x0) * 0.02
            if keys[K_LSHIFT] or keys[K_RSHIFT]:
                speed *= 4
            if keys[K_LEFT]:
                self._pan_view(-speed, 0)
            if keys[K_RIGHT]:
                self._pan_view(speed, 0)

            # Auto-scroll to keep cursor visible
            if self.player and self.player.playing:
                cur_frame = self._time_to_frame(self.player.get_position())
                vw = self.view_x1 - self.view_x0
                if cur_frame < self.view_x0 or cur_frame > self.view_x1:
                    self.view_x0 = cur_frame - vw * 0.25
                    self.view_x1 = self.view_x0 + vw
                    self._clamp_view()

            self._render()
            pygame.display.flip()
            clock.tick(60)

        if self.player:
            self.player.stop()
        pygame.quit()

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
            self.view_x0 = 0.0
            self.view_y0 = 0.0
            self.view_x1 = float(self.n_frames)
            self.view_y1 = float(self.n_bins)
        elif key == K_TAB and self.gui:
            self.gui.selected_idx = (
                (self.gui.selected_idx + 1) % len(self.field_sources))
        elif key == K_f:
            self.show_freq_axis = not self.show_freq_axis
        elif key == K_t:
            self.show_time_axis = not self.show_time_axis
        return True

    # ---- Mouse handling ---------------------------------------------------

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        if event.button == 1:
            self._is_panning = True
            self._pan_last = event.pos
        elif event.button == 2:
            self._seek_to_screen_pos(event.pos)
        elif event.button == 3:
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
        ml = self.axis_margin_left if self.show_freq_axis else 0
        mr = self.axis_margin_right if self.show_freq_axis else 0
        mt = self.axis_margin_top if self.show_time_axis else 0
        mb = self.axis_margin_bottom if self.show_time_axis else 0
        gui_w = FieldTabPanel.PANEL_W if (self.gui and self.gui.visible) else 0
        x = ml
        y = mt
        w = max(1, self.win_w - ml - mr - gui_w)
        h = max(1, self.win_h - mt - mb)
        return x, y, w, h

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
        if self.view_x0 < 0:
            self.view_x0 = 0
            self.view_x1 = w
        if self.view_x1 > self.n_frames:
            self.view_x1 = float(self.n_frames)
            self.view_x0 = self.view_x1 - w
        if self.view_y0 < 0:
            self.view_y0 = 0
            self.view_y1 = h
        if self.view_y1 > self.n_bins:
            self.view_y1 = float(self.n_bins)
            self.view_y0 = self.view_y1 - h
        self.view_x0 = max(0.0, self.view_x0)
        self.view_y0 = max(0.0, self.view_y0)

    def _zoom_at_screen(self, sx: int, sy: int, factor: float) -> None:
        dx, dy = self._screen_to_data(sx, sy)
        w = self.view_x1 - self.view_x0
        h = self.view_y1 - self.view_y0
        nw = max(4.0, w / factor)
        nh = max(4.0, h / factor)
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
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        self.view_x0, self.view_x1 = x0, x1
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

        axis_frag = gl_shaders.compileShader(AXIS_FRAG_SRC, GL_FRAGMENT_SHADER)
        vert2 = gl_shaders.compileShader(VERT_SRC, GL_VERTEX_SHADER)
        self._axis_program = gl_shaders.compileProgram(vert2, axis_frag)

        freq_ax = self.meta.get("axes", {}).get("freq", {})
        time_ax = self.meta.get("axes", {}).get("time", {})
        half_w = freq_ax.get("width", 400) // 2
        half_h = time_ax.get("height", 64) // 2
        self.axis_margin_left = half_w
        self.axis_margin_right = half_w
        self.axis_margin_top = half_h
        self.axis_margin_bottom = half_h

    # ---- Texture management -----------------------------------------------

    def _upload_texture_rgba16(self, img: Image.Image) -> int:
        img8 = img.convert("RGBA")
        w, h = img8.size
        raw = img8.tobytes("raw", "RGBA")
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        return tex_id

    def _upload_texture_rgba8(self, img: Image.Image) -> int:
        img8 = img.convert("RGBA")
        w, h = img8.size
        raw = img8.tobytes("raw", "RGBA")
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, raw)
        return tex_id

    def _load_all_textures(self) -> None:
        """Load every unique data texture and assign GL texture units."""
        for tex_id in self._tex_ids.values():
            glDeleteTextures([tex_id])
        self._tex_ids.clear()
        self._tex_units.clear()

        needed: set[str] = set()
        for fs in self.field_sources:
            needed.add(fs.tex_name)

        for i, tex_name in enumerate(sorted(needed)):
            if i >= 8:
                print(f"WARNING: max 8 texture units, skipping {tex_name}")
                break
            path = os.path.join(self.analysis_dir, f"{tex_name}.png")
            if not os.path.isfile(path):
                print(f"WARNING: {path} not found")
                continue
            img = Image.open(path)
            tex_id = self._upload_texture_rgba16(img)
            self._tex_units[tex_name] = i
            self._tex_ids[tex_name] = tex_id

    def _load_axis_textures(self) -> None:
        axes = self.meta.get("axes", {})
        freq_info = axes.get("freq", {})
        fpath = os.path.join(self.analysis_dir, freq_info.get("file", ""))
        if os.path.isfile(fpath):
            img = Image.open(fpath)
            if self._freq_axis_tex:
                glDeleteTextures([self._freq_axis_tex])
            self._freq_axis_tex = self._upload_texture_rgba8(img)
        time_info = axes.get("time", {})
        tpath = os.path.join(self.analysis_dir, time_info.get("file", ""))
        if os.path.isfile(tpath):
            img = Image.open(tpath)
            if self._time_axis_tex:
                glDeleteTextures([self._time_axis_tex])
            self._time_axis_tex = self._upload_texture_rgba8(img)

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

    def _set_field_uniforms(self) -> None:
        """Bind all data textures and set per-field + output uniforms."""
        loc = lambda n: glGetUniformLocation(self._program, n)

        for tex_name, unit in self._tex_units.items():
            glActiveTexture(GL_TEXTURE0 + unit)
            glBindTexture(GL_TEXTURE_2D, self._tex_ids[tex_name])
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

        # --- Data textures with multi-field shader ---
        if self._tex_ids and self._program:
            glUseProgram(self._program)
            self._set_field_uniforms()

            u0 = self.view_x0 / self.n_frames
            u1 = self.view_x1 / self.n_frames
            v0 = self.view_y0 / self.n_bins
            v1 = self.view_y1 / self.n_bins

            nx0 = 2.0 * sx / self.win_w - 1.0
            ny0 = 1.0 - 2.0 * (sy + sh) / self.win_h
            nx1 = 2.0 * (sx + sw) / self.win_w - 1.0
            ny1 = 1.0 - 2.0 * sy / self.win_h

            glBegin(GL_QUADS)
            glTexCoord2f(u0, v0); glVertex2f(nx0, ny0)
            glTexCoord2f(u1, v0); glVertex2f(nx1, ny0)
            glTexCoord2f(u1, v1); glVertex2f(nx1, ny1)
            glTexCoord2f(u0, v1); glVertex2f(nx0, ny1)
            glEnd()

            glUseProgram(0)

        # --- Axis textures (passthrough shader) ---
        if self._axis_program:
            glUseProgram(self._axis_program)
            aloc = lambda n: glGetUniformLocation(self._axis_program, n)

            if self.show_freq_axis and self._freq_axis_tex:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self._freq_axis_tex)
                glUniform1i(aloc("uTex"), 0)

                fv0 = self.view_y0 / self.n_bins
                fv1 = self.view_y1 / self.n_bins
                lx0 = -1.0
                lx1 = 2.0 * sx / self.win_w - 1.0
                ly0 = 1.0 - 2.0 * (sy + sh) / self.win_h
                ly1 = 1.0 - 2.0 * sy / self.win_h

                glBegin(GL_QUADS)
                glTexCoord2f(0.0, fv0); glVertex2f(lx0, ly0)
                glTexCoord2f(0.5, fv0); glVertex2f(lx1, ly0)
                glTexCoord2f(0.5, fv1); glVertex2f(lx1, ly1)
                glTexCoord2f(0.0, fv1); glVertex2f(lx0, ly1)
                glEnd()

                rx0 = 2.0 * (sx + sw) / self.win_w - 1.0
                rx1_ndc = min(1.0, 2.0 * (sx + sw + self.axis_margin_right)
                              / self.win_w - 1.0)

                glBegin(GL_QUADS)
                glTexCoord2f(0.5, fv0); glVertex2f(rx0, ly0)
                glTexCoord2f(1.0, fv0); glVertex2f(rx1_ndc, ly0)
                glTexCoord2f(1.0, fv1); glVertex2f(rx1_ndc, ly1)
                glTexCoord2f(0.5, fv1); glVertex2f(rx0, ly1)
                glEnd()

            if self.show_time_axis and self._time_axis_tex:
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, self._time_axis_tex)
                glUniform1i(aloc("uTex"), 0)

                tu0 = self.view_x0 / self.n_frames
                tu1 = self.view_x1 / self.n_frames
                tx0 = 2.0 * sx / self.win_w - 1.0
                tx1 = 2.0 * (sx + sw) / self.win_w - 1.0

                ty1_ndc = 1.0
                ty0_ndc = 1.0 - 2.0 * sy / self.win_h

                glBegin(GL_QUADS)
                glTexCoord2f(tu0, 0.5); glVertex2f(tx0, ty0_ndc)
                glTexCoord2f(tu1, 0.5); glVertex2f(tx1, ty0_ndc)
                glTexCoord2f(tu1, 1.0); glVertex2f(tx1, ty1_ndc)
                glTexCoord2f(tu0, 1.0); glVertex2f(tx0, ty1_ndc)
                glEnd()

                by1_ndc = 1.0 - 2.0 * (sy + sh) / self.win_h
                by0_ndc = -1.0

                glBegin(GL_QUADS)
                glTexCoord2f(tu0, 0.0); glVertex2f(tx0, by0_ndc)
                glTexCoord2f(tu1, 0.0); glVertex2f(tx1, by0_ndc)
                glTexCoord2f(tu1, 0.5); glVertex2f(tx1, by1_ndc)
                glTexCoord2f(tu0, 0.5); glVertex2f(tx0, by1_ndc)
                glEnd()

            glUseProgram(0)

        # --- Playback cursor ---
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

        # --- GUI panel ---
        if self.gui and self.gui.visible:
            panel_surf = self.gui.render()
            if panel_surf:
                dd_surf = self.gui.render_dropdown_overlay()
                if dd_surf and self.gui._active_dropdown:
                    dr = self.gui._dropdown_rect
                    panel_surf.blit(dd_surf, (dr.x, dr.y))

                tex = self._upload_pygame_surface(panel_surf)
                pw, ph = panel_surf.get_size()
                px0 = 2.0 * (self.win_w - pw) / self.win_w - 1.0
                px1 = 1.0
                py0 = 1.0 - 2.0 * ph / self.win_h
                py1 = 1.0

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

        # --- HUD ---
        self._render_hud()

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

        actives = []
        for i, fs in enumerate(self.field_sources):
            cfg = self.field_configs[i]
            if cfg.target >= 0:
                ch = "RGBA"[cfg.target]
                actives.append(f"{fs.display_name}\u2192{ch}")
        mapping = ", ".join(actives) if actives else "no fields"
        txt = f"{state} {pos:.1f}s / {dur:.1f}s  |  {mapping}"
        surf = font.render(txt, True, (220, 220, 200))

        w, h = surf.get_size()
        bg = pygame.Surface((w + 8, h + 4), pygame.SRCALPHA)
        bg.fill((20, 20, 20, 180))
        bg.blit(surf, (4, 2))

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
    parser.add_argument("analysis_dir",
                        help="Path to the analysis output directory "
                             "(must contain cqt_data.npz + texture PNGs)")
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
    audio = _resolve_audio(args.analysis_dir, args.audio)
    viewer = SpectrogramViewer(args.analysis_dir, audio)
    viewer.run()


if __name__ == "__main__":
    main()
