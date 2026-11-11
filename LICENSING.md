# Licensing

## Scope

The MIT grant in [LICENSE](LICENSE) applies to the first-party code and
documentation for which sangderenard holds licensing rights. Existing
copyright and permission notices must be retained. Third-party software,
submodules, dependencies, assets and independently authored material retain
their own terms. This grant does not claim ownership of those works or
supply missing permission for unverified material.

The owner selected this first-party MIT policy on 2026-11-11 to express the
projects' continuity with SpeakToMe. Repository history and author names were
checked as provenance evidence; they are not proof of ownership of every
imported file. This document is a scoped inventory, not a complete dependency
audit or a statement that every possible combined distribution is compatible.

## fftfree snapshot

`third_party/fftfree/VENDORED_FROM.md` records the source checkout and commit
`642dcf442664e3f9e3ae3e4165180b6302c457fe`. Its source history attributes commits
to Albert and sangderenard. The owner-authored parts of this vendored snapshot
are included in this repository's MIT grant. The snapshot's provenance
document now records this decision instead of deferring license selection.

This declaration does not change the separate fftfree repository or license
Eigen or any other third-party dependency. Existing third-party notices take
precedence for their respective files.

## Dependencies and assets

`pyproject.toml` declares NumPy, SciPy, librosa, Torch, Pillow, matplotlib,
soundfile, PyYAML and usd-core, with optional miniaudio, pygame, PyOpenGL and
mido. Native build files reference pybind11 and OpenMP. These retain their own
terms. Fonts, models, audio, images, other imported assets and transitive
dependencies require their own provenance checks before redistribution.
No claim of a complete release inventory is made here.
