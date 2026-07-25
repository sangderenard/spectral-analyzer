# Pluck / Spectral Analyzer

Pluck is a graph-driven audio, instrument, acoustic, optical, camera, and real-time rendering research workspace. The repository began as a GPU-accelerated Constant-Q spectral analyzer; that analyzer remains available, but it is now one subsystem of a much larger simulation environment.

## System map

- Graph audio and control: analytic voices, routing, state-machine plugins, materialization, SCC-based solving, and timing.
- Instrument simulation: playable guitar, strings, bodies, pickups, microphones, rooms, acoustic FDTD, and coupled evolution.
- Optical transport: rasterization, ray tracing, BDPT, coherent/complex transport, apertures, lenses, sensors, films, and light-field assets.
- Camera and rendering: camera models, exposure and calibration workflows, OpenGL viewers, stations, and image/export pipelines.
- Spectral analysis: stereo CQT analysis, plotting, and synchronized playback.
- Nodus integration: graph-tool contracts, optical KPN boundaries, and shared-runtime experiments.

[`ARCHITECTURE.md`](ARCHITECTURE.md) is a focused graph-solver reference, not a complete atlas. For broader navigation, see [`VITRUVIAN_REPOSITORY_ATLAS.md`](VITRUVIAN_REPOSITORY_ATLAS.md) and [`RAY_TRACER_INVENTORY.md`](RAY_TRACER_INVENTORY.md).

## Common entry points

The repository contains many research demos rather than one canonical executable. Choose an entry point for the subsystem you are working on and read its adjacent design or handoff document.

The original analyzer remains available through `bass_analysis.py`, `bass_plot.py`, and `bass_viewer.py`. For current guitar/renderer work, `demo_note_playback.py`, the camera-station scripts, and focused acceptance documents are better entry points than the legacy analyzer pipeline.

## Dependencies

Dependencies vary by subsystem. Python numerical/audio work commonly uses NumPy, SciPy, librosa, PyTorch, Pillow, and matplotlib; interactive views commonly use pygame and PyOpenGL. Native optical and GPU paths have additional requirements. Consult the selected subsystem documentation before installing or rebuilding dependencies.

## Nodus relationship

Pluck remains independently runnable. Nodus supplies graph/runtime and tool-integration surfaces; it does not own Pluck's simulation implementation.

- [`../NODUS_PLUCK_HANDOFF.md`](../NODUS_PLUCK_HANDOFF.md)
- [`NODUS_OPTICAL_KPN_INTEGRATION.md`](NODUS_OPTICAL_KPN_INTEGRATION.md)
- [`NODUS_TOOL_BRIEF.md`](NODUS_TOOL_BRIEF.md)
