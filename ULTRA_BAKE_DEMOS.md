# Ultra-baked optical showcase images

`wave_transform_visual_demo.py` can bake high-investment image plates from the
same physical-aperture material and production T4 propagation kernels used by
the live OpenGL qualification view. It does not introduce a second solver or
an ideal aperture mask.

List the installed recipes and their exact numerical investment:

```powershell
python wave_transform_visual_demo.py --list-ultra-bakes
```

## Showcase commands

Perceptually divided 32-band visible spectrum through the moving material iris:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-spectrum-fixed --output-dir exposures/ultra_bakes
```

Resampled 32-lane continuous-frequency cohorts through the same iris:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-spectrum-continuous --output-dir exposures/ultra_bakes
```

Canonical linear, circular, radial, azimuthal, partial, and unpolarized source
states at one useful aperture position:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-polarization --output-dir exposures/ultra_bakes
```

Piston-free coherent s/p phase at the blade material and after propagation,
including the reverse-direction polarization result:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-coherent-phase --output-dir exposures/ultra_bakes
```

Run the complete collection:

```powershell
python wave_transform_visual_demo.py --ultra-bake all --output-dir exposures/ultra_bakes
```

The defaults deliberately use `quality=bake`: a 64x64 published field is
surrounded by a 512x512 hidden absorbing solve domain, with 16 propagation
substeps. Spectrum recipes use 32 lanes; the polarization and coherent-phase
recipes use 16.

These are final-quality jobs, not preview commands. On the current development
workstation, the validation bake of one default fixed-spectrum frame completed
in approximately 164 seconds. The CLI prints `start` before every solve and
`done` with elapsed time afterward.

## Rehearsal and override controls

Use this smaller rehearsal before committing to the full collection:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-spectrum-fixed --output-dir exposures/ultra_rehearsal --ultra-size 32 --ultra-frames 3 --ultra-scale 3 --ultra-quality high --ultra-lanes 8
```

Available overrides are:

- `--ultra-size`: visible power-of-two field width;
- `--ultra-frames`: number of aperture/source states;
- `--ultra-scale`: PNG presentation scale;
- `--ultra-quality`: `balanced`, `high`, or `bake`;
- `--ultra-lanes`: exact width `1`, `3`, `4`, `8`, `16`, or `32`.

Lane width and spectral semantics remain separate. A continuous 32-lane bake
contains 32 stratified continuous-frequency samples; it does not silently
become the fixed 32-band discretization.

## Products

Each recipe directory contains:

- `hero.png`: the middle showcase state;
- `frame_NNN.png`: a 3x2 scientific image plate for every state;
- `manifest.json`: wavelengths, solve dimensions, aperture radius, boundary
  retention, border fraction, and source state for every frame.

The collection root also receives `ultra_bakes.json`.

The spectral RGB panel is a display projection using the renderer's canonical
wavelength-to-RGB weights. The Stokes, phase, power, and boundary values remain
the scientific products; RGB is not represented as a replacement for the
underlying spectral lanes.
