"""arpeggio_viewer.py

Real-time animated Lissajous / vectorscope viewer for the analytic companion
sidecar produced by demo_arpeggio_outputs.py.

By default looks for output in  arpeggio_demo_out/  relative to CWD.

Keyboard
────────
  L / V       toggle Lissajous ↔ Vectorscope mode
  G           cycle colour: frequency → amplitude → time → white
  Space       pause / resume
  ← / →       scrub ±5 s
  [ / ]       narrow / widen the trace tail
  - / =       zoom out / in
  A           toggle audio (stereo_quadrature.wav in companion dir)
  F           toggle fullscreen
  Q / Esc     quit

Usage
─────
  python arpeggio_viewer.py                          # default output dir
  python arpeggio_viewer.py --dir my_run/
  python arpeggio_viewer.py --tail 1.5 --vectorscope
  python arpeggio_viewer.py --no-audio --color time
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pygame
from pygame.locals import (
    DOUBLEBUF, KEYDOWN, OPENGL, QUIT, RESIZABLE, VIDEORESIZE,
    K_ESCAPE, K_SPACE, K_LEFT, K_RIGHT,
    K_a, K_f, K_g, K_l, K_v, K_q,
    K_LEFTBRACKET, K_RIGHTBRACKET, K_EQUALS, K_MINUS,
    K_KP_PLUS, K_KP_MINUS,
)
from OpenGL.GL import (
    GL_ARRAY_BUFFER, GL_BLEND, GL_COLOR_BUFFER_BIT, GL_DEPTH_TEST,
    GL_FLOAT, GL_LINE_SMOOTH, GL_LINE_SMOOTH_HINT, GL_LINE_STRIP,
    GL_NICEST, GL_ONE, GL_ONE_MINUS_SRC_ALPHA, GL_POINTS,
    GL_RGBA, GL_SRC_ALPHA, GL_STREAM_DRAW, GL_TEXTURE_2D,
    GL_TRIANGLE_STRIP, GL_UNSIGNED_BYTE,
    GL_CLAMP_TO_EDGE, GL_LINEAR,
    GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
    GL_UNPACK_ALIGNMENT,
    glBindBuffer, glBindTexture, glBindVertexArray,
    glBlendFunc, glBufferData,
    glClear, glClearColor,
    glDeleteBuffers, glDeleteTextures, glDeleteVertexArrays,
    glDisable, glDrawArrays, glEnable,
    glEnableVertexAttribArray, glGenBuffers, glGenTextures,
    glGenVertexArrays, glGetUniformLocation,
    glHint, glLineWidth, glPixelStorei, glPointSize,
    glTexImage2D, glTexParameteri,
    glUniform1f, glUniform1i,
    glUseProgram, glVertexAttribPointer, glViewport,
)
from OpenGL.GL import shaders as gl_shaders


# ── Constants ─────────────────────────────────────────────────────────────────

DEMO_OUT_DIR = "arpeggio_demo_out"
META_FILE    = "companion_meta.json"
VIZ_FILE     = "companion_viz.f32"
FULL_FILE    = "companion_full.f32"

COLOR_MODES = ["f_inst", "amplitude", "time", "white"]
VIEW_MODES  = ["lissajous", "vectorscope"]

WAV_CANDIDATES = [
    "stereo_quadrature.wav",
    "quadrature.wav",
    "mono.wav",
]


# ── GLSL ──────────────────────────────────────────────────────────────────────

_SCOPE_VERT = """
#version 330 core
layout(location = 0) in vec2 xy;     // channel 0 = L, channel 1 = R
layout(location = 1) in vec2 af;     // af.x = amplitude, af.y = f_inst (Hz)

uniform int   u_mode;        // 0 = lissajous, 1 = vectorscope
uniform int   u_color_mode;  // 0 = f_inst, 1 = amplitude, 2 = time, 3 = white
uniform int   u_tail_len;
uniform float u_f_min;
uniform float u_f_max;
uniform float u_scale;

out vec4 v_color;

vec3 hsv2rgb(float h, float s, float v_) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(vec3(h) + K.xyz) * 6.0 - K.www);
    return v_ * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), s);
}

void main() {
    vec2 pos;
    if (u_mode == 0) {
        // Lissajous: L on X, R on Y
        pos = xy;
    } else {
        // Vectorscope: mid/side 45-degree rotation
        pos = vec2((xy.x + xy.y) * 0.70710678,
                   (xy.x - xy.y) * 0.70710678);
    }
    gl_Position  = vec4(pos * u_scale, 0.0, 1.0);
    gl_PointSize = 5.0;

    // t=0 is oldest vertex, t=1 is newest
    float t     = float(gl_VertexID) / float(max(u_tail_len - 1, 1));
    float alpha = pow(t, 0.35);   // older samples fade to black

    vec3 rgb;
    if (u_color_mode == 0) {
        // Hue mapped to instantaneous frequency (blue=low, red=high)
        float frac = (u_f_max > u_f_min)
            ? clamp((af.y - u_f_min) / (u_f_max - u_f_min), 0.0, 1.0)
            : 0.5;
        rgb = hsv2rgb(0.67 - frac * 0.67, 1.0, 1.0);
    } else if (u_color_mode == 1) {
        // Amplitude → cyan brightness
        float a = clamp(af.x, 0.0, 1.0);
        rgb = hsv2rgb(0.55, 0.75, a);
    } else if (u_color_mode == 2) {
        // Time → rainbow hue
        rgb = hsv2rgb(t * 0.85, 1.0, 1.0);
    } else {
        rgb = vec3(1.0);
    }
    v_color = vec4(rgb, alpha);
}
"""

_SCOPE_FRAG = """
#version 330 core
in  vec4 v_color;
out vec4 frag_color;
void main() { frag_color = v_color; }
"""

# HUD: textured full-screen quad for on-screen text overlay
_HUD_VERT = """
#version 330 core
layout(location=0) in vec2 pos;
layout(location=1) in vec2 uv;
out vec2 v_uv;
void main() { gl_Position = vec4(pos, 0.0, 1.0); v_uv = uv; }
"""

_HUD_FRAG = """
#version 330 core
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 frag_color;
void main() { frag_color = texture(u_tex, v_uv); }
"""


# ── Loader ────────────────────────────────────────────────────────────────────

def load_companion(companion_dir: str):
    """Return (data_f32 [N,4], meta dict, viz_rate_hz)."""
    meta_path = os.path.join(companion_dir, META_FILE)
    viz_path  = os.path.join(companion_dir, VIZ_FILE)
    full_path = os.path.join(companion_dir, FULL_FILE)

    if not os.path.isfile(meta_path):
        sys.exit(
            f"ERROR: {META_FILE!r} not found in {companion_dir!r}.\n"
            f"       Run:  python demo_arpeggio_outputs.py"
            f"  (output goes to '{DEMO_OUT_DIR}/' by default)"
        )

    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    if os.path.isfile(viz_path):
        raw  = np.fromfile(viz_path, dtype=np.float32)
        rate = float(meta.get("viz_rate_hz", meta.get("sample_rate", 120)))
        print(f"Loaded viz sidecar:  {len(raw) // 4:,} samples @ {rate:.0f} Hz"
              f"  ({viz_path})")
    elif os.path.isfile(full_path):
        raw  = np.fromfile(full_path, dtype=np.float32)
        rate = float(meta.get("sample_rate", 48000))
        print(f"Loaded full sidecar: {len(raw) // 4:,} samples @ {rate:.0f} Hz"
              f"  (this is large; consider re-running with --viz-fps 120)")
    else:
        sys.exit(
            f"ERROR: no .f32 sidecar files found in {companion_dir!r}.\n"
            f"       Run:  python demo_arpeggio_outputs.py"
        )

    data = raw.reshape(-1, 4)   # columns: [L, R, amplitude, f_inst]
    return data, meta, rate


def find_wav(companion_dir: str) -> Optional[str]:
    for name in WAV_CANDIDATES:
        p = os.path.join(companion_dir, name)
        if os.path.isfile(p):
            return p
    return None


# ── Viewer ────────────────────────────────────────────────────────────────────

class LissajousViewer:
    def __init__(
        self,
        data:       np.ndarray,     # (N, 4) float32
        viz_rate:   float,          # samples per second
        meta:       dict,
        wav_path:   Optional[str] = None,
        tail_s:     float = 0.5,
        width:      int   = 900,
        height:     int   = 900,
        fps_cap:    float = 60.0,
        audio:      bool  = True,
    ) -> None:
        self._data     = data
        self._N        = len(data)
        self._rate     = viz_rate
        self._meta     = meta
        self._wav_path = wav_path
        self._tail_s   = tail_s
        self._width    = width
        self._height   = height
        self._fps_cap  = fps_cap

        self._view_mode  = 0     # 0=lissajous  1=vectorscope
        self._color_mode = 0     # 0=f_inst  1=amp  2=time  3=white
        self._scale      = 0.90
        self._paused     = False
        self._t          = 0.0
        self._t0         = time.perf_counter()
        self._duration   = self._N / self._rate
        self._audio      = audio and wav_path is not None

        # Frequency range for colour mapping (5th–95th percentile of nonzero)
        f_col = data[:, 3]
        valid = f_col[f_col > 0.0]
        if valid.size > 0:
            self._f_min = float(np.percentile(valid,  5))
            self._f_max = float(np.percentile(valid, 95))
        else:
            self._f_min, self._f_max = 20.0, 2000.0

    # ── GL initialisation ────────────────────────────────────────────────────

    def _init_gl(self) -> None:
        self._scope_prog = gl_shaders.compileProgram(
            gl_shaders.compileShader(_SCOPE_VERT, 0x8B31),  # GL_VERTEX_SHADER
            gl_shaders.compileShader(_SCOPE_FRAG, 0x8B30),  # GL_FRAGMENT_SHADER
        )
        self._hud_prog = gl_shaders.compileProgram(
            gl_shaders.compileShader(_HUD_VERT, 0x8B31),
            gl_shaders.compileShader(_HUD_FRAG, 0x8B30),
        )

        # --- Scope VAO / VBO ---
        self._scope_vao = glGenVertexArrays(1)
        glBindVertexArray(self._scope_vao)

        self._scope_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._scope_vbo)
        # Pre-allocate for the maximum possible tail (full buffer)
        glBufferData(GL_ARRAY_BUFFER, self._N * 16, None, GL_STREAM_DRAW)
        # attrib 0: L, R  (offset 0, stride 16)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 16, ctypes.c_void_p(0))
        # attrib 1: amplitude, f_inst  (offset 8, stride 16)
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        # --- HUD fullscreen quad VAO ---
        # Positions and UVs for a -1..+1 quad
        quad = np.array([
            -1.0, -1.0,  0.0, 0.0,
             1.0, -1.0,  1.0, 0.0,
            -1.0,  1.0,  0.0, 1.0,
             1.0,  1.0,  1.0, 1.0,
        ], dtype=np.float32)
        self._hud_vao = glGenVertexArrays(1)
        glBindVertexArray(self._hud_vao)
        self._hud_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._hud_vbo)
        glBufferData(GL_ARRAY_BUFFER, quad.nbytes, quad, GL_STREAM_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 2, GL_FLOAT, False, 16, ctypes.c_void_p(0))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 2, GL_FLOAT, False, 16, ctypes.c_void_p(8))
        glBindVertexArray(0)

        # --- HUD texture ---
        self._hud_tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D, 0)

        # --- GL state ---
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
        glLineWidth(1.5)
        glPointSize(4.0)
        glDisable(GL_DEPTH_TEST)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw_scope(self, window: np.ndarray) -> None:
        """Upload and render the current tail window."""
        n = len(window)
        if n < 2:
            return

        glBindBuffer(GL_ARRAY_BUFFER, self._scope_vbo)
        glBufferData(GL_ARRAY_BUFFER, window.nbytes, window, GL_STREAM_DRAW)

        glUseProgram(self._scope_prog)
        glUniform1i(glGetUniformLocation(self._scope_prog, "u_mode"),       self._view_mode)
        glUniform1i(glGetUniformLocation(self._scope_prog, "u_color_mode"), self._color_mode)
        glUniform1i(glGetUniformLocation(self._scope_prog, "u_tail_len"),   n)
        glUniform1f(glGetUniformLocation(self._scope_prog, "u_f_min"),      self._f_min)
        glUniform1f(glGetUniformLocation(self._scope_prog, "u_f_max"),      self._f_max)
        glUniform1f(glGetUniformLocation(self._scope_prog, "u_scale"),      self._scale)

        glBindVertexArray(self._scope_vao)

        # Additive blending gives glowing trace
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)
        glDrawArrays(GL_LINE_STRIP, 0, n)

        # Bright dot at the current tip
        glDrawArrays(GL_POINTS, n - 1, 1)

        # Restore normal blending for HUD
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glBindVertexArray(0)
        glUseProgram(0)

    def _update_hud_texture(self, i_cur: int, window_len: int) -> None:
        """Render HUD text to a pygame surface, upload as GL texture."""
        W, H = self._width, self._height
        surf = pygame.Surface((W, H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        mode_str  = VIEW_MODES[self._view_mode].upper()
        color_str = COLOR_MODES[self._color_mode]
        status    = "PAUSED" if self._paused else "PLAY  "
        i_safe    = min(i_cur, self._N - 1)
        f_cur     = float(self._data[i_safe, 3])
        a_cur     = float(self._data[i_safe, 2])

        lines = [
            f"{status}  {mode_str}  col={color_str}",
            f"t={self._t:6.2f}s / {self._duration:.2f}s   f={f_cur:.1f} Hz   A={a_cur:.3f}",
            f"tail={self._tail_s:.2f}s ({window_len} pts)   zoom={self._scale:.2f}",
            "L/V mode · G colour · Space pause · ←→ scrub · [] tail · -/= zoom · A audio · Q quit",
        ]

        font = pygame.font.SysFont("monospace", 13)
        y = 6
        for line in lines:
            shadow  = font.render(line, True, (0, 0, 0))
            text    = font.render(line, True, (180, 220, 180))
            surf.blit(shadow, (9, y + 1))
            surf.blit(text,   (8, y))
            y += 18

        # Upload to GL texture (flip vertically: pygame Y-down, GL Y-up)
        data_str = pygame.image.tostring(surf, "RGBA", True)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(
            GL_TEXTURE_2D, 0, GL_RGBA,
            W, H, 0, GL_RGBA, GL_UNSIGNED_BYTE, data_str,
        )
        glBindTexture(GL_TEXTURE_2D, 0)

    def _draw_hud(self) -> None:
        """Draw the HUD texture as a transparent fullscreen quad."""
        glUseProgram(self._hud_prog)
        glBindTexture(GL_TEXTURE_2D, self._hud_tex)
        glUniform1i(glGetUniformLocation(self._hud_prog, "u_tex"), 0)
        glBindVertexArray(self._hud_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glUseProgram(0)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> None:
        pygame.init()
        pygame.font.init()

        pygame.display.set_mode(
            (self._width, self._height),
            DOUBLEBUF | OPENGL | RESIZABLE,
        )
        pygame.display.set_caption("Lissajous Viewer — arpeggio analytic companion")

        self._init_gl()
        self._t0 = time.perf_counter()

        if self._audio:
            self._start_audio(offset=0.0)

        clock = pygame.time.Clock()

        while True:
            # ── advance playback time ─────────────────────────────────────
            if not self._paused:
                self._t = time.perf_counter() - self._t0
                if self._t >= self._duration:
                    self._t = 0.0
                    self._t0 = time.perf_counter()
                    if self._audio:
                        self._start_audio(offset=0.0)

            # ── events ───────────────────────────────────────────────────
            for ev in pygame.event.get():
                if ev.type == QUIT:
                    self._cleanup()
                    return
                elif ev.type == VIDEORESIZE:
                    self._width, self._height = ev.w, ev.h
                    glViewport(0, 0, ev.w, ev.h)
                elif ev.type == KEYDOWN:
                    self._handle_key(ev.key)

            # ── compute tail window ───────────────────────────────────────
            i_cur   = int(self._t * self._rate)
            tail_n  = max(2, int(self._tail_s * self._rate))
            i_start = max(0, i_cur - tail_n)
            window  = self._data[i_start : i_cur + 1]   # contiguous slice

            # ── draw ─────────────────────────────────────────────────────
            glClear(GL_COLOR_BUFFER_BIT)
            glViewport(0, 0, self._width, self._height)

            self._draw_scope(window)

            self._update_hud_texture(i_cur, len(window))
            self._draw_hud()

            pygame.display.flip()
            clock.tick(self._fps_cap)

    # ── Input handling ───────────────────────────────────────────────────────

    def _handle_key(self, key: int) -> None:
        if key in (K_ESCAPE, K_q):
            self._cleanup()
            sys.exit(0)
        elif key == K_SPACE:
            self._toggle_pause()
        elif key in (K_l, K_v):
            self._view_mode = 1 - self._view_mode
        elif key == K_g:
            self._color_mode = (self._color_mode + 1) % len(COLOR_MODES)
        elif key == K_LEFT:
            self._seek(self._t - 5.0)
        elif key == K_RIGHT:
            self._seek(self._t + 5.0)
        elif key == K_LEFTBRACKET:
            self._tail_s = max(0.02, self._tail_s * 0.8)
        elif key == K_RIGHTBRACKET:
            self._tail_s = min(60.0, self._tail_s * 1.25)
        elif key in (K_MINUS, K_KP_MINUS):
            self._scale = max(0.1, self._scale * 0.9)
        elif key in (K_EQUALS, K_KP_PLUS):
            self._scale = min(4.0, self._scale * 1.1)
        elif key == K_a:
            self._toggle_audio()
        elif key == K_f:
            self._toggle_fullscreen()

    # ── Playback helpers ─────────────────────────────────────────────────────

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        if self._audio:
            if self._paused:
                pygame.mixer.music.pause()
            else:
                pygame.mixer.music.unpause()
                self._t0 = time.perf_counter() - self._t
        else:
            if not self._paused:
                self._t0 = time.perf_counter() - self._t

    def _seek(self, new_t: float) -> None:
        new_t = max(0.0, min(self._duration, new_t))
        self._t  = new_t
        self._t0 = time.perf_counter() - new_t
        if self._audio:
            self._start_audio(offset=new_t)

    def _start_audio(self, offset: float) -> None:
        try:
            pygame.mixer.init(frequency=48000, size=-16, channels=2, buffer=2048)
            pygame.mixer.music.load(self._wav_path)
            pygame.mixer.music.play(loops=0, start=offset)
            self._t0 = time.perf_counter() - offset
        except Exception as exc:
            print(f"Audio start failed: {exc}")
            self._audio = False

    def _toggle_audio(self) -> None:
        if not self._wav_path:
            print("No WAV file found in companion directory.")
            return
        if self._audio:
            pygame.mixer.music.stop()
            self._audio = False
        else:
            self._audio = True
            self._start_audio(offset=self._t)

    def _toggle_fullscreen(self) -> None:
        flags = pygame.display.get_surface().get_flags()
        if flags & pygame.FULLSCREEN:
            pygame.display.set_mode(
                (self._width, self._height),
                DOUBLEBUF | OPENGL | RESIZABLE,
            )
        else:
            pygame.display.set_mode(
                (0, 0),
                DOUBLEBUF | OPENGL | pygame.FULLSCREEN,
            )

    def _cleanup(self) -> None:
        try:
            glDeleteBuffers(1, [self._scope_vbo])
            glDeleteBuffers(1, [self._hud_vbo])
            glDeleteVertexArrays(1, [self._scope_vao])
            glDeleteVertexArrays(1, [self._hud_vao])
            glDeleteTextures([self._hud_tex])
        except Exception:
            pass
        if self._audio:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass
        pygame.quit()


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Lissajous / vectorscope viewer for the arpeggio analytic companion sidecar.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--dir", default=DEMO_OUT_DIR,
        help="Directory containing companion_viz.f32 / companion_meta.json",
    )
    ap.add_argument("--tail",        type=float, default=0.5,
                    help="Initial trace tail length in seconds")
    ap.add_argument("--fps",         type=float, default=60.0,
                    help="Display frame rate cap")
    ap.add_argument("--width",       type=int,   default=900)
    ap.add_argument("--height",      type=int,   default=900)
    ap.add_argument("--no-audio",    dest="audio", action="store_false", default=True,
                    help="Disable audio playback")
    ap.add_argument("--vectorscope", action="store_true", default=False,
                    help="Start in vectorscope mode instead of lissajous")
    ap.add_argument("--color",       choices=COLOR_MODES, default="f_inst",
                    help="Initial colour mode")
    args = ap.parse_args()

    data, meta, viz_rate = load_companion(args.dir)
    wav_path = find_wav(args.dir) if args.audio else None
    if args.audio and wav_path is None:
        print(f"Note: no WAV found in {args.dir!r} — audio disabled.")

    viewer = LissajousViewer(
        data      = data,
        viz_rate  = viz_rate,
        meta      = meta,
        wav_path  = wav_path,
        tail_s    = args.tail,
        width     = args.width,
        height    = args.height,
        fps_cap   = args.fps,
        audio     = args.audio,
    )
    viewer._view_mode  = 1 if args.vectorscope else 0
    viewer._color_mode = COLOR_MODES.index(args.color)
    viewer.run()


if __name__ == "__main__":
    main()
