"""compose_apng — assemble test_basic_shader_frames/*.png into an animated PNG.

Uses Pillow's APNG writer.  Run after `python test_basic_shader.py` has
populated ``test_basic_shader_frames/``.
"""
from __future__ import annotations
import os, sys, glob

from PIL import Image

FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_basic_shader_frames")
OUT_PATH   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "test_basic_shader_animation.png")
TARGET_FPS = 30


def main() -> int:
    paths = sorted(glob.glob(os.path.join(FRAMES_DIR, "frame_*.png")))
    if not paths:
        print(f"[err] no frames found in {FRAMES_DIR}", file=sys.stderr)
        return 1
    print(f"[apng] reading {len(paths)} frames from {FRAMES_DIR}")
    frames = [Image.open(p).convert("RGBA") for p in paths]
    duration_ms = int(round(1000.0 / TARGET_FPS))
    head, *tail = frames
    head.save(
        OUT_PATH,
        format="PNG",
        save_all=True,
        append_images=tail,
        duration=duration_ms,
        loop=0,
        disposal=2,        # restore-to-bg between frames
        default_image=False,
    )
    sz = os.path.getsize(OUT_PATH)
    print(f"[apng] wrote {OUT_PATH}  ({sz/1024:.1f} KiB, {len(frames)} frames @ {TARGET_FPS} fps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
