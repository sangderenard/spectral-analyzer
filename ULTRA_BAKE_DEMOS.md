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

Resampled 32-lane continuous-frequency cohorts with an azimuthal source through
the same iris:

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
- `--ultra-pattern`: finite-material `iris`, `circular`, `hole-array`,
  `slot-array`, or `grating`.
- `--ultra-output-size`: publish each composite as an exact square RGBA
  texture while preserving the scientific plate's aspect ratio.
- `--ultra-panel`: publish one chosen scientific panel as a raw square instead
  of making the six-panel plate. It accepts indices `0..5` or names including
  `physical`, `spectral`, `power`, `polarization`, `stokes-q`, and `stokes-v`.

For example, bake the same continuous-spectrum recipe through a physical
transmission grating:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-spectrum-continuous --ultra-pattern grating --output-dir exposures/ultra_bakes_grating
```

One 2048×2048 composite texture:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-spectrum-continuous --ultra-frames 1 --ultra-output-size 2048 --output-dir exposures/ultra_bakes_2048
```

One genuinely computed 2048×2048 spectral field—not an enlarged contact
sheet:

```powershell
python wave_transform_visual_demo.py --ultra-bake iris-spectrum-continuous --ultra-frames 1 --ultra-size 2048 --ultra-scale 1 --ultra-quality balanced --ultra-lanes 1 --ultra-panel spectral --output-dir exposures/ultra_field_2048
```

At `balanced`, the 2048 published field uses a 4096×4096 absorbing solve.
Selecting `bake` would use a 16384×16384 solve and should only be attempted
after spectral/state streaming is implemented or on a substantially larger
machine.

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
