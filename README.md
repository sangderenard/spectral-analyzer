# Spectral Analyzer

GPU-accelerated Constant-Q Transform analyzer with 1-cent frequency resolution, stereo PIL spectrogram rendering, and interactive OpenGL viewer.

## Pipeline

```
bass_analysis.py  →  cqt_data.npz  →  bass_plot.py   →  spectrogram PNGs
                                    →  bass_viewer.py  →  interactive playback
```

1. **`bass_analysis.py`** — Computes stereo CQT on GPU, onset enhancement, pad influence map, bass metrics. Saves compressed `.npz` with power, phase, real, imaginary per channel plus all metadata.

2. **`bass_plot.py`** — Reads the `.npz` and renders spectrogram images (whole/bass/treble views + summary plot). Can re-render with different gamma/DPI without re-computing the CQT.

3. **`bass_viewer.py`** — Loads spectrogram PNGs as OpenGL textures and plays the source audio in sync with a cursor overlay. Pan, zoom, seek, and switch views interactively.

## Quick Start

```bash
pip install -e ".[all]"

# Analyze
python bass_analysis.py song.wav --composite "r:0:L:r"

# Plot
python bass_plot.py song_analysis/

# View
python bass_viewer.py song_analysis/ song.wav
```

## Launchers (Windows)

```
launchers\launch_reflect.bat song.wav      # full file, reflect padding
launchers\launch_compare.bat song.wav      # default A/B composite
```

## Dependencies

- **Required:** numpy, scipy, librosa, torch (CUDA), Pillow, matplotlib
- **MP3 support:** miniaudio
- **Viewer:** pygame, PyOpenGL
