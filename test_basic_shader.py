#!/usr/bin/env python3
"""
test_basic_shader.py
--------------------
Standalone stress-test for the C BaseRasterizer + material/emission pipeline.

Scene
-----
  * Central showcase:  Fibonacci-distributed sphere (triangulated via convex
                       hull of unit-sphere Fibonacci spiral points) at
                       (0, 0, -3.2), radius 0.65, satin gold + lacquer enamel.
  * Orbiter spheres:   Four small icosphere-style triangulated spheres
                       arranged around the central sphere:
                         - chrome mirror   (warm side)
                         - ruby glass      (lower-warm)
                         - emerald gem     (HDR emissive, cool side)
                         - sapphire gem    (HDR emissive, lower-cool)
                         - obsidian polish (above central, dark Fresnel test)
  * Background:        Hyperbolic saddle plane z = (x² − y²) / k, sampled on
                       a square grid in (x, y), centred behind the spheres
                       at z ≈ -7.0 and looking down at it from the camera's
                       perspective.  The infinite extents of the saddle are
                       deliberately cropped to a finite square so only the
                       interesting central region appears in frame.

Camera
------
  Origin at (0,0,0) looking toward -Z.
  Identity view matrix → geometry is authored directly in view-space.
  45° vertical FOV, 4:3 aspect, near=0.1, far=100.

Materials
---------
  All six materials are loaded from configs/materials/test_*.yaml and
  registered into MaterialDatabase via the rich PBR dict path (which also
  honours the optional `enamel:` block).  The YAML files carry both the
  legacy refl/diff/abso fields (so the BVH/raytracer pipeline can still
  consume them) and the PBR fields (albedo_rgb, roughness, metallic,
  emission_rgb, ambient, spec_strength, shininess, inner_color, enamel).

Logging
-------
  Every frame: lit-pixel count, alpha coverage, per-channel mean/max.
  Final frame: full 5×5 pixel grid sample + optional PNG save.

No OpenGL, no pygame, no physics – pure C rasterizer path.
"""

import sys
import os
import time
import argparse
import logging
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
_cli = argparse.ArgumentParser(add_help=True,
    description="C BaseRasterizer scene stress-test.")
_cli.add_argument("--spheres", "-n", type=int, default=5,
    help="number of orbiter spheres (>=1).  Emitters are placed first; "
         "non-emitters fill the remainder; if N exceeds the named pool, "
         "additional spheres are synthesised by cycling the emitter set.")
_cli.add_argument("--max-lights", type=int, default=8,
    help="cap on cluster-lights emitted per frame from the group cache "
         "(clamped to [1, 32]).  Default 8.")
_CLI_ARGS, _ = _cli.parse_known_args()
N_ORBITERS_REQUESTED = max(1, int(_CLI_ARGS.spheres))
MAX_LIGHTS_REQUESTED = max(1, min(32, int(_CLI_ARGS.max_lights)))

# Ensure stdout/stderr can carry the box-drawing characters used in section
# headers below — cp1252 (the default Windows console encoding) cannot.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("test_basic_shader")

# ─────────────────────────────────────────────────────────────────────────────
# 1.  C extension
# ─────────────────────────────────────────────────────────────────────────────
log.info("─── Importing _spectral_kernels ───────────────────────────────")
try:
    import _spectral_kernels as _sk
    log.info("  OK  BaseRasterizer present: %s", hasattr(_sk, "BaseRasterizer"))
except ImportError as exc:
    log.error("FATAL: _spectral_kernels not available — rebuild the C extension")
    log.error("  %s", exc)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Material DB
# ─────────────────────────────────────────────────────────────────────────────
log.info("─── Importing material_db ─────────────────────────────────────")
try:
    from material_db import MaterialDatabase, EmissionProfileDatabase
    log.info("  OK")
except ImportError as exc:
    log.error("FATAL: material_db not importable: %s", exc)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Emission profile database snapshot
# ─────────────────────────────────────────────────────────────────────────────
ep_db = EmissionProfileDatabase.instance()

# ── Load the central spectral library ────────────────────────────────────────
# Every named ColorProfile / EmissionProfile / RemissionProfile referenced by
# any material YAML is defined in configs/profiles/spectral_profiles.yaml.
# The loader registers each entry into the global EmissionProfileDatabase and
# bakes the spectral→sRGB triple before any material is registered.  No
# inline profile registration happens in this test.
import spectral_library
spectral_library.load_default_library()

log.info("─── EmissionProfileDatabase ───────────────────────────────────")
log.info("  %d profiles registered", len(ep_db))
_ep_rgb_tensor = ep_db.build_rgb_tensor()
for _i, _name in enumerate(ep_db._order):
    _rgb = _ep_rgb_tensor[_i] if _i < len(_ep_rgb_tensor) else "?"
    log.info("  [%d] %-32s  baked_rgb = %s", _i, _name, _rgb)

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Register test materials (loaded from configs/materials/test_*.yaml)
# ─────────────────────────────────────────────────────────────────────────────
try:
    import yaml as _yaml
except ImportError as exc:
    log.error("FATAL: PyYAML not available — install with `pip install pyyaml`")
    log.error("  %s", exc)
    sys.exit(1)

_CONFIGS_MATERIALS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "configs", "materials")

# Names map to configs/materials/<name>.yaml; values become the dict registered
# into MaterialDatabase.  Order is fixed so per-material indices are stable.
_SCENE_MATERIAL_NAMES = [
    "fibonacci_satin_gold",   # central showcase sphere
    "chrome_mirror",          # orbiter
    "ruby_glass",             # orbiter — has remit_profile (frame-delayed re-emission demo)
    "emerald_emissive",       # orbiter (HDR emitter)
    "sapphire_emissive",      # orbiter (HDR emitter)
    "obsidian_polish",        # orbiter
    "hyperbolic_slate",       # background saddle
]

def _load_material_yaml_dict(name: str) -> dict:
    """Load configs/materials/<name>.yaml as a plain dict.  Strips top-level
    metadata keys (`name`, `description`, `notes`, `gameplay`) that the
    renderer does not consume."""
    path = os.path.join(_CONFIGS_MATERIALS, f"{name}.yaml")
    with open(path, "r", encoding="utf-8") as fh:
        d = _yaml.safe_load(fh) or {}
    for k in ("name", "description", "notes", "gameplay"):
        d.pop(k, None)
    return d

def _yaml_to_mat16(d: dict) -> np.ndarray:
    """Compose a 16-float authored material vector from a YAML dict.

    Resolves `emit_profile_name` and `color_profile_name` against the
    EmissionProfileDatabase (which holds ALL profile types — emission,
    reflective color, etc.) so build_tensors() can bake the spectral
    triple into pbr at registration time.
    """
    refl  = float(d.get("reflectivity", 0.5))
    diff  = float(d.get("diffusion",    0.0))
    abso  = float(d.get("absorption",   0.0))
    # Authored albedo_rgb is no longer permitted; fall through to the
    # spectral bake.  A neutral grey placeholder is written so the slot is
    # well-defined for any inspector that reads the row before the bake.
    alb   = d.get("albedo_rgb", [0.5, 0.5, 0.5])
    ior   = float(d.get("ior",     1.5))
    opac  = float(d.get("opacity", 1.0))

    _ep_name = d.get("emit_profile_name")
    if _ep_name is not None and _ep_name in ep_db:
        emit_idx = ep_db.index_of(_ep_name)
    else:
        emit_idx = -1

    remit_idx = -1
    _rp_name = d.get("remit_profile_name")
    if _rp_name is not None and _rp_name in ep_db:
        remit_idx = ep_db.index_of(_rp_name)
    elif "remit_profile_idx" in d:
        remit_idx = int(d.get("remit_profile_idx", -1))

    _cp_name = d.get("color_profile_name")
    if _cp_name is not None and _cp_name in ep_db:
        color_idx = ep_db.index_of(_cp_name)
    else:
        color_idx = -1

    react = float(d.get("reactive_shift_hz", 0.0))

    flags = np.uint32(0)
    if emit_idx >= 0:
        flags |= np.uint32(1)   # MAT_FLAG_EMISSIVE
    if react != 0.0:
        flags |= np.uint32(2)   # MAT_FLAG_REACTIVE

    arr = np.zeros(16, np.float32)
    arr[:11] = [refl, diff, abso,  refl, diff, abso,
                float(alb[0]), float(alb[1]), float(alb[2]), ior, opac]
    arr[11]  = np.frombuffer(np.array([flags], np.uint32).tobytes(), np.float32)[0]
    arr[12]  = float(emit_idx)
    arr[13]  = float(remit_idx)
    arr[14]  = float(color_idx)
    arr[15]  = react
    return arr

db = MaterialDatabase.instance()
mat_idx: dict[str, int] = {}
for _n in _SCENE_MATERIAL_NAMES:
    _yaml_dict = _load_material_yaml_dict(_n)
    _mat16 = _yaml_to_mat16(_yaml_dict)
    mat_idx[_n] = db.register_from_mat16(_n, _mat16)
    # Re-register the dict on top so the rich-PBR fields (enamel, phong
    # overrides, etc.) that the mat16 path discards are still available
    # to _fill_enamel and _pbr_to_phong_record at bake time.  The mat16
    # ingest above already populated the profile-idx slots that
    # build_tensors() needs to overwrite albedo + emission spectrally.
    _yaml_dict["emit_profile_idx"]  = float(_mat16[12])
    _yaml_dict["remit_profile_idx"] = float(_mat16[13])
    _yaml_dict["color_profile_idx"] = float(_mat16[14])
    db.register(_n, _yaml_dict)

log.info("─── Registered materials ──────────────────────────────────────")
for _n, _i in mat_idx.items():
    log.info("  %-32s  index = %d", _n, _i)

# Aliases for downstream readability
GOLD_NAME    = "fibonacci_satin_gold"
CHROME_NAME  = "chrome_mirror"
RUBY_NAME    = "ruby_glass"
EMERALD_NAME = "emerald_emissive"
SAPPHIRE_NAME= "sapphire_emissive"
OBSIDIAN_NAME= "obsidian_polish"
SLATE_NAME   = "hyperbolic_slate"

gold_idx     = mat_idx[GOLD_NAME]
chrome_idx   = mat_idx[CHROME_NAME]
ruby_idx     = mat_idx[RUBY_NAME]
emerald_idx  = mat_idx[EMERALD_NAME]
sapphire_idx = mat_idx[SAPPHIRE_NAME]
obsidian_idx = mat_idx[OBSIDIAN_NAME]
slate_idx    = mat_idx[SLATE_NAME]

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Build + inspect tensors
# ─────────────────────────────────────────────────────────────────────────────
tensors  = db.build_tensors()
pbr_t    = tensors["pbr"]            # (N, 16) float32 — authoritative
phon_t   = tensors["phong_compat"]   # (N,  8) float32 — downstream Phong sink
enam_t   = tensors["enamel"]         # (N,  8) float32
index_map = dict(tensors.get("index", {}))

log.info("─── Tensor shapes ─────────────────────────────────────────────")
log.info("  pbr   : %s   phong : %s   enamel : %s",
         pbr_t.shape, phon_t.shape, enam_t.shape)
log.info("  index : %s", index_map)

for mat_name in _SCENE_MATERIAL_NAMES:
    idx = index_map.get(mat_name, -1)
    if idx < 0 or idx >= len(pbr_t):
        log.warning("  [!] %s missing from index_map (idx=%d)", mat_name, idx)
        continue
    row = pbr_t[idx]
    pr  = phon_t[idx]
    en  = enam_t[idx]
    log.info("  PBR [%d]  %s", idx, mat_name)
    log.info("    albedo     = [%.4f, %.4f, %.4f]  roughness=%.3f  metallic=%.3f",
             row[0], row[1], row[2], row[3], row[4])
    log.info("    ior=%.3f  opacity=%.3f  mat_flags=%.0f",
             row[6], row[7], row[14] if len(row) > 14 else 0.0)
    log.info("    emission   = [%.4f, %.4f, %.4f]",
             row[8], row[9], row[10])
    log.info("    phong: ambient=%.3f  spec_str=%.3f  shininess=%.1f  grain=%.4f",
             pr[0], pr[1], pr[2], pr[3])
    log.info("    enamel: thickness_nm=%.1f  ior=(%.3f,%.3f)  rough=%.3f  tint=[%.2f,%.2f,%.2f]",
             en[0], en[1], en[2], en[3], en[4], en[5], en[6])

# ─────────────────────────────────────────────────────────────────────────────
# 6.  BaseRasterizer
# ─────────────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 320, 240
log.info("─── BaseRasterizer(%d × %d) ────────────────────────────────────",
         WIDTH, HEIGHT)
rdr = _sk.BaseRasterizer(WIDTH, HEIGHT, 16)
log.info("  Created OK")
rdr.set_max_lights(MAX_LIGHTS_REQUESTED)
log.info("  max_lights = %d", MAX_LIGHTS_REQUESTED)

rdr.set_pbr_chunk(np.ascontiguousarray(pbr_t,   dtype=np.float32))
rdr.set_phong_chunk(np.ascontiguousarray(phon_t, dtype=np.float32))
rdr.set_enamel_chunk(np.ascontiguousarray(enam_t, dtype=np.float32))
log.info("  Chunks uploaded  (N=%d materials)", len(pbr_t))

# ─────────────────────────────────────────────────────────────────────────────
# 7.  Geometry helpers — Fibonacci sphere, hyperbolic saddle, small spheres
# ─────────────────────────────────────────────────────────────────────────────
from scipy.spatial import ConvexHull  # Delaunay-on-sphere via 3D convex hull


def _fibonacci_sphere_points(n: int) -> np.ndarray:
    """Generate n points distributed on the unit sphere using the Fibonacci
    spiral.  Returns float32 (n, 3) on the unit sphere."""
    n = max(int(n), 4)
    i = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)                 # polar angle
    golden = np.pi * (1.0 + 5.0 ** 0.5)                # golden-angle increment
    theta = golden * i                                 # azimuth
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def _sphere_triangulation(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (points (n,3) float32 on the unit sphere, triangles (Nt,3) int32
    indexing into points).  Triangles are oriented CCW when viewed from
    outside the sphere — the rasterizer's back-face cull treats positive
    screen-space signed area as back, so we orient outward-facing triangles
    accordingly when building the (Nt, 3, 3) vertex array."""
    pts = _fibonacci_sphere_points(n)
    hull = ConvexHull(pts.astype(np.float64))
    tris = hull.simplices.astype(np.int32)
    # Re-orient each triangle so its face normal points outward (i.e. the
    # cross product of two edges aligns with the outward radial direction).
    centroids = pts[tris].mean(axis=1)
    e1 = pts[tris[:, 1]] - pts[tris[:, 0]]
    e2 = pts[tris[:, 2]] - pts[tris[:, 0]]
    fn = np.cross(e1, e2)
    flip = np.einsum("ij,ij->i", fn, centroids) < 0.0
    tris[flip] = tris[flip][:, [0, 2, 1]]
    return pts, tris


def _sphere_triangles(center: tuple[float, float, float], radius: float,
                      n_points: int, mat_id: int):
    """Build (Nt, 3, 3) world-space triangles for a triangulated sphere.

    Returns (tris float32, mat_ids int32 (Nt,))."""
    pts, tri_idx = _sphere_triangulation(n_points)
    cx, cy, cz = center
    world_pts = pts * radius + np.array([cx, cy, cz], dtype=np.float32)
    tris = world_pts[tri_idx].astype(np.float32)         # (Nt, 3, 3)
    mids = np.full(len(tris), int(mat_id), dtype=np.int32)
    return tris, mids


def _saddle_triangles(center: tuple[float, float, float],
                      half_extent: float,
                      curvature_k: float,
                      grid_n: int,
                      mat_id: int):
    """Hyperbolic-paraboloid (saddle) patch:  z = (x² − y²) / k.

    Sampled on a (grid_n × grid_n) regular grid in (x, y) over the square
    [-half_extent, +half_extent] then translated to `center`.  Each grid cell
    yields two triangles, oriented so face normals point upward (toward +y in
    view-space when rotated; the test geometry leaves the saddle in its
    natural orientation and lets the camera look down at it from above).

    Returns (tris float32 (Nt, 3, 3), mat_ids int32 (Nt,))."""
    n = max(int(grid_n), 2)
    cx, cy, cz = center
    lin = np.linspace(-half_extent, +half_extent, n, dtype=np.float32)
    xs, ys = np.meshgrid(lin, lin, indexing="xy")
    zs = (xs * xs - ys * ys) / float(curvature_k)
    # World-space grid:  put the saddle "below" the spheres by treating the
    # sampled (x, y) as (x_world, z_world_offset) and the saddle height zs as
    # y_world (so the camera looking down -Z sees a saddle below the orbs).
    grid_x = xs + cx
    grid_y = -zs + cy            # negate so positive-curvature lobes sit lower
    grid_z = ys + cz             # ys becomes depth; centre at cz
    grid = np.stack([grid_x, grid_y, grid_z], axis=-1).astype(np.float32)

    # Two triangles per cell, indexed (i, j), (i+1, j), (i, j+1), (i+1, j+1).
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            v00 = grid[i,     j]
            v10 = grid[i + 1, j]
            v01 = grid[i,     j + 1]
            v11 = grid[i + 1, j + 1]
            # Orient CW in screen-space-after-Y-flip → the rasterizer culls
            # positive signed area, so we want the visible (top) face as the
            # negative-area side.  Build both triangles facing "up" toward
            # the camera looking down.
            tris.append([v00, v01, v10])
            tris.append([v10, v01, v11])
    tri_arr = np.asarray(tris, dtype=np.float32)
    mids = np.full(len(tri_arr), int(mat_id), dtype=np.int32)
    return tri_arr, mids


def _build_verts_view(tris: np.ndarray, smooth_normals_for_sphere: bool = False
                      ) -> np.ndarray:
    """Convert (Nt, 3, 3) world tris to (Nt*3, 6) view-space verts+normals.

    Camera at origin, -Z forward → geometry is already in view-space here.
    Default normal = flat face normal (cross(e1, e2) normalised, broadcast to
    all 3 vertices per tri).  For sphere meshes pass smooth_normals_for_sphere
    = True to use the radial (per-vertex) normal instead — this is what gives
    the orbs a smoothly shaded, faceted-free appearance."""
    pts  = tris.reshape(-1, 3)                             # (Nt*3, 3)
    e1   = tris[:, 1, :] - tris[:, 0, :]
    e2   = tris[:, 2, :] - tris[:, 0, :]
    fn_raw = np.cross(e1, e2).astype(np.float32)
    fn_len = np.linalg.norm(fn_raw, axis=1, keepdims=True)
    fn     = fn_raw / np.where(fn_len > 1e-8, fn_len, 1.0)
    nrm    = np.repeat(fn, 3, axis=0)                     # flat normals (Nt*3, 3)
    if smooth_normals_for_sphere:
        # Radial normals: for a sphere centred at the centroid of all verts,
        # the outward normal at vertex v is (v - centre) / |v - centre|.
        centre = pts.mean(axis=0, keepdims=True)
        rad = pts - centre
        rad_len = np.linalg.norm(rad, axis=1, keepdims=True)
        nrm = (rad / np.where(rad_len > 1e-8, rad_len, 1.0)).astype(np.float32)
    verts  = np.concatenate([pts, nrm], axis=1)           # (Nt*3, 6)
    return np.ascontiguousarray(verts, dtype=np.float32)


# ── Scene assembly ──────────────────────────────────────────────────────────
# Central showcase: Fibonacci-distributed sphere
central_tris, central_mids = _sphere_triangles(
    center=(0.0, 0.0, -3.2), radius=0.65, n_points=512, mat_id=gold_idx)

# Orbiter spheres around the central sphere — emitters first, then the
# non-emitting decorative spheres.  If --spheres N exceeds the length of
# this named pool we synthesise extra orbiters by cycling the emitter set
# (so adding spheres always *adds light*, never just dead chrome).
ORBIT_R = 1.35

# (centre, radius, mat_id, label) — emitters listed first.
_EMITTER_POOL = [
    ((+ORBIT_R,  0.10, -3.2), 0.30, ruby_idx,     "ruby"),       # remissive
    ((-0.85,    -0.95, -3.4), 0.28, emerald_idx,  "emerald"),    # HDR emit
    ((+0.85,    -0.95, -3.4), 0.28, sapphire_idx, "sapphire"),   # HDR emit
]
_NON_EMITTER_POOL = [
    ((-ORBIT_R,  0.10, -3.2), 0.30, chrome_idx,   "chrome"),
    (( 0.00,    +1.05, -3.2), 0.26, obsidian_idx, "obsidian"),
]
_NAMED_POOL = _EMITTER_POOL + _NON_EMITTER_POOL

orbiters: list = []
for k in range(N_ORBITERS_REQUESTED):
    if k < len(_NAMED_POOL):
        orbiters.append(_NAMED_POOL[k])
    else:
        # Synthesise extra emitter spheres beyond the named pool by cycling
        # through the emitter pool.  Centre/radius are placeholders; the
        # actual per-frame centre comes from the orbit basis below.
        src = _EMITTER_POOL[(k - len(_NAMED_POOL)) % len(_EMITTER_POOL)]
        _ctr, _rad, _mid, _lbl = src
        orbiters.append((_ctr, _rad, _mid, f"{_lbl}#{k}"))

orbiter_tris_list = []
orbiter_mids_list = []
orbiter_vert_offsets = []   # (start_vertex, end_vertex) per orbiter
_running_vert = central_tris.shape[0] * 3
for (ctr, rad, mid, _label) in orbiters:
    _t, _m = _sphere_triangles(center=ctr, radius=rad, n_points=128, mat_id=mid)
    orbiter_tris_list.append(_t)
    orbiter_mids_list.append(_m)

# Hyperbolic-saddle background plane.  The saddle's natural infinite extents
# are bounded by `half_extent`; everything outside that square is simply not
# part of the mesh — it is "cropped out of the geometry" as requested.
saddle_tris, saddle_mids = _saddle_triangles(
    center=(0.0, -0.55, -7.0),
    half_extent=4.0,
    curvature_k=4.5,
    grid_n=24,
    mat_id=slate_idx)

# ── Animation: phased circular orbits passing across the centre ─────────────
# Each orbiter travels a circular orbit of radius ORBIT_R centred on the
# gold sphere's position.  Orbit planes are tilted differently per orbiter so
# their paths visually cross the central showcase from independent angles.
# Phases are spaced 2π·k/N around the cycle so the moments of closest-approach
# to the centre never coincide — minimising visual overlap during the loop.
ORBIT_CENTER = np.array([0.0, 0.0, -3.2], dtype=np.float32)
ORBIT_PERIOD_FRAMES = 120
N_ORBITERS = len(orbiters)

# Per-orbiter (u, v) basis vectors defining the orbit plane.  Position at
# phase θ is  ORBIT_CENTER + ORBIT_R * (cos θ · u + sin θ · v).
def _orbit_basis(plane_normal: tuple[float, float, float],
                 spin_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """Build an (u, v) orthonormal pair spanning the orbit plane whose
    normal is `plane_normal`.  `spin_deg` rotates the (u, v) frame inside
    that plane (around `plane_normal`) by the given angle — purely a
    visual phase choice, NOT a separate axis tilt.

    Position at phase θ is  ORBIT_CENTER + ORBIT_R * (cosθ·u + sinθ·v),
    which lies in the plane perpendicular to `plane_normal` for any spin.
    """
    n = np.asarray(plane_normal, dtype=np.float32)
    n /= max(float(np.linalg.norm(n)), 1e-8)
    # u: any vector perpendicular to n
    helper = np.array([0.0, 1.0, 0.0], dtype=np.float32) \
        if abs(n[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(n, helper); u /= max(float(np.linalg.norm(u)), 1e-8)
    v = np.cross(n, u);      v /= max(float(np.linalg.norm(v)), 1e-8)
    # Rodrigues rotation of (u, v) about n by `spin_deg` keeps the plane
    # invariant — we're only re-phasing the basis inside its own plane.
    a = float(np.radians(spin_deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    u_rot = (c * u + s * v).astype(np.float32)
    v_rot = (-s * u + c * v).astype(np.float32)
    return u_rot, v_rot

# N distinct orbit planes — generated via a Fibonacci sphere of plane
# normals so the set spreads visually no matter how many spheres the user
# requests.  The first 5 entries are the original hand-tuned normals so
# the default-N=5 behaviour is preserved bit-for-bit.
_HAND_TUNED_PLANES = [
    _orbit_basis((0.0, 0.0, 1.0),   0.0),    # chrome / 1st emitter
    _orbit_basis((0.0, 1.0, 0.0),  20.0),    # ruby
    _orbit_basis((1.0, 0.0, 0.0), -15.0),    # emerald
    _orbit_basis((1.0, 1.0, 0.0),  10.0),    # sapphire
    _orbit_basis((1.0,-1.0, 0.5), -25.0),    # obsidian
]

def _fib_orbit_planes(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate N orbit planes whose normals are Fibonacci-distributed on
    the unit sphere, with a small per-plane Y-tilt for visual variety."""
    planes = []
    phi = float(np.pi * (3.0 - np.sqrt(5.0)))   # golden angle
    for k in range(n):
        # Spherical Fibonacci point — evenly spaced on the sphere.
        z = 1.0 - 2.0 * (k + 0.5) / float(n)
        r = float(np.sqrt(max(0.0, 1.0 - z * z)))
        a = phi * k
        nx, ny, nz = r * float(np.cos(a)), r * float(np.sin(a)), z
        # Vary the Y-tilt around the cycle so consecutive planes don't
        # look like rigid rotations of each other.
        tilt = float(15.0 * np.sin(a * 0.5))
        planes.append(_orbit_basis((nx, ny, nz), tilt))
    return planes

if N_ORBITERS_REQUESTED <= len(_HAND_TUNED_PLANES):
    _orbit_planes = _HAND_TUNED_PLANES[:N_ORBITERS_REQUESTED]
else:
    # Keep the hand-tuned 5 in their original slots; append Fibonacci-spaced
    # extras for the remaining orbiters.
    _extra = _fib_orbit_planes(N_ORBITERS_REQUESTED - len(_HAND_TUNED_PLANES))
    _orbit_planes = _HAND_TUNED_PLANES + _extra

# Pre-build each orbiter's *local* (centred-at-origin) sphere triangulation,
# then translate per-frame.  This avoids re-triangulating every frame.
orbiter_local_tris   = []   # list of (Nt_i, 3, 3) float32, centred at origin
orbiter_local_mids   = []   # list of (Nt_i,)  int32
orbiter_local_radii  = []   # list of float
for (ctr, rad, mid, _label) in orbiters:
    _t, _m = _sphere_triangles(center=(0.0, 0.0, 0.0),
                               radius=rad, n_points=128, mat_id=mid)
    orbiter_local_tris.append(_t)
    orbiter_local_mids.append(_m)
    orbiter_local_radii.append(rad)

def _orbiter_centers_for_phase(theta_base: float) -> list[np.ndarray]:
    """Compute world-space centres for all orbiters at base phase angle θ.
    Each orbiter k is offset by 2π·k/N so closest-approaches to ORBIT_CENTER
    (which happen as the path sweeps past) are temporally interleaved."""
    out = []
    for k, (u, v) in enumerate(_orbit_planes):
        theta = theta_base + (2.0 * np.pi * k / N_ORBITERS)
        c = ORBIT_CENTER + ORBIT_R * (np.cos(theta) * u + np.sin(theta) * v)
        out.append(c.astype(np.float32))
    return out

def _assemble_scene_for_phase(theta_base: float
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translate each orbiter to its current orbit position and concatenate
    with the static saddle + central sphere.  Returns
    (all_tris (Nt,3,3) float32, all_mids (Nt,) int32, verts_view (Nt*3,6) float32)."""
    centres = _orbiter_centers_for_phase(theta_base)
    moved_tris = []
    moved_mids = []
    moved_verts = []
    for local_t, local_m, c in zip(orbiter_local_tris, orbiter_local_mids, centres):
        wt = local_t + c                                  # broadcast (Nt,3,3) + (3,)
        moved_tris.append(wt)
        moved_mids.append(local_m)
        moved_verts.append(_build_verts_view(wt, smooth_normals_for_sphere=True))
    a_tris = np.concatenate([saddle_tris, central_tris] + moved_tris, axis=0)
    a_mids = np.concatenate([saddle_mids, central_mids] + moved_mids, axis=0)
    vv = np.ascontiguousarray(
        np.concatenate([saddle_verts, central_verts] + moved_verts, axis=0),
        dtype=np.float32)
    return a_tris, np.ascontiguousarray(a_mids, dtype=np.int32), vv

# Build static-mesh verts once.
saddle_verts  = _build_verts_view(saddle_tris,  smooth_normals_for_sphere=False)
central_verts = _build_verts_view(central_tris, smooth_normals_for_sphere=True)

# Initial scene at θ=0 — used for the lighting derivation below and as the
# first frame's geometry.
all_tris, mat_ids, verts_view = _assemble_scene_for_phase(0.0)
Nt = int(all_tris.shape[0])
all_mids = mat_ids

log.info("─── Geometry ──────────────────────────────────────────────────")
log.info("  saddle plane    : %5d tris  (mat_id=%d, %s)",
         len(saddle_tris), slate_idx, SLATE_NAME)
log.info("  central sphere  : %5d tris  (mat_id=%d, %s, fib_n=512, r=0.65)",
         len(central_tris), gold_idx, GOLD_NAME)
for (ctr, rad, mid, label), _t in zip(orbiters, orbiter_tris_list):
    log.info("  orbiter [%-9s]: %5d tris  (mat_id=%d, r=%.2f, c=%s)",
             label, len(_t), mid, rad, ctr)
log.info("  total           : %5d tris", Nt)
log.info("  verts_view  : shape=%s  dtype=%s", verts_view.shape, verts_view.dtype)
log.info("  mat_ids     : shape=%s  dtype=%s  unique=%s",
         mat_ids.shape, mat_ids.dtype, sorted(np.unique(mat_ids).tolist()))

# ─────────────────────────────────────────────────────────────────────────────
# 8.  Projection matrix
# ─────────────────────────────────────────────────────────────────────────────
def _perspective_proj(fov_y_deg, aspect, near, far):
    f = 1.0 / np.tan(np.radians(fov_y_deg) * 0.5)
    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] = (2.0 * far * near) / (near - far)
    P[3, 2] = -1.0
    return P

P_mat = _perspective_proj(fov_y_deg=45.0, aspect=WIDTH / HEIGHT,
                          near=0.1, far=100.0)
# C rasterizer expects column-major flattened (= P transposed, then reshape)
proj = np.ascontiguousarray(P_mat.T.reshape(-1), dtype=np.float32)

log.info("─── Projection matrix ─────────────────────────────────────────")
for _ri in range(4):
    log.info("  row %d: %s", _ri, P_mat[_ri])

# ─────────────────────────────────────────────────────────────────────────────
# 9.  Object-group descriptors for the rasterizer's emissive-light cache
#
#     The host scene graph already partitions every triangle into objects.
#     We hand that partition to the C kernel verbatim via `rdr.set_groups()`.
#     The kernel caches one cluster-light per emissive object across frames,
#     refreshing only what the per-frame `dirty` bitmask says changed.
#
#     Group ordering MUST match the triangle ordering in
#     `_assemble_scene_for_phase` (saddle, central sphere, then orbiters).
# ─────────────────────────────────────────────────────────────────────────────
_GRP_SADDLE  = 0
_GRP_CENTRAL = 1
_GRP_ORBITER_BASE = 2          # orbiter k ⇒ group_id = _GRP_ORBITER_BASE + k

_saddle_n  = int(saddle_tris.shape[0])
_central_n = int(central_tris.shape[0])
_orbiter_n = [int(t.shape[0]) for t in orbiter_local_tris]

# Triangle offsets in the assembled buffer.
_off_saddle  = 0
_off_central = _off_saddle + _saddle_n
_off_orbiters = []
_acc = _off_central + _central_n
for _n in _orbiter_n:
    _off_orbiters.append(_acc)
    _acc += _n

# Identity mv for static groups; orbiters get translation-only mv each frame.
_IDEN_MV = np.eye(4, dtype=np.float32).reshape(-1)   # column-major == row-major for I

def _mv_translation(centre: np.ndarray) -> np.ndarray:
    """Column-major flattened 4×4 translation matrix (no rotation/scale).
    The cached centroid is in the orbiter's local frame at origin; multiplying
    by this mv reproduces the world-space centre.  View transform is identity
    in this test (camera at origin looking down -Z), so view ≡ world."""
    M = np.eye(4, dtype=np.float32)
    M[:3, 3] = centre.astype(np.float32)
    return np.ascontiguousarray(M.T.reshape(-1), dtype=np.float32)

def _push_groups(theta_base: float, *, first: bool) -> None:
    """Build and upload the per-frame group descriptor arrays.

    On the first call we mark every group BR_DIRTY_GEOM so the kernel
    seeds its cache.  Thereafter the static saddle + central sphere are
    clean (dirty=0) and the 5 orbiters carry BR_DIRTY_MV only — the
    kernel just transports their cached centroid by the new mv, no
    triangle work."""
    BR_DIRTY_GEOM = 1
    BR_DIRTY_MV   = 2

    centres = _orbiter_centers_for_phase(theta_base)

    n_groups = 2 + len(orbiters)
    gids   = np.empty((n_groups,), dtype=np.int32)
    mids_a = np.empty((n_groups,), dtype=np.int32)
    offs   = np.empty((n_groups,), dtype=np.int32)
    cnts   = np.empty((n_groups,), dtype=np.int32)
    mvs    = np.empty((n_groups, 16), dtype=np.float32)
    drts   = np.empty((n_groups,), dtype=np.int32)

    # Saddle (static).
    gids[0]   = _GRP_SADDLE
    mids_a[0] = slate_idx
    offs[0]   = _off_saddle
    cnts[0]   = _saddle_n
    mvs[0]    = _IDEN_MV
    drts[0]   = BR_DIRTY_GEOM if first else 0

    # Central sphere (static).
    gids[1]   = _GRP_CENTRAL
    mids_a[1] = gold_idx
    offs[1]   = _off_central
    cnts[1]   = _central_n
    mvs[1]    = _IDEN_MV
    drts[1]   = BR_DIRTY_GEOM if first else 0

    # Orbiters (mv changes per frame; geometry only on the first frame).
    for k, ((_ctr0, _rad, _mid, _label), _ofs, _cnt) in enumerate(
            zip(orbiters, _off_orbiters, _orbiter_n)):
        gids[2 + k]   = _GRP_ORBITER_BASE + k
        mids_a[2 + k] = _mid
        offs[2 + k]   = _ofs
        cnts[2 + k]   = _cnt
        mvs[2 + k]    = _mv_translation(centres[k])
        drts[2 + k]   = BR_DIRTY_GEOM if first else BR_DIRTY_MV

    rdr.set_groups(gids, mids_a, offs, cnts, mvs, drts)

# ─────────────────────────────────────────────────────────────────────────────
# 10.  Animated render loop — 120 frames, one full orbit per loop
# ─────────────────────────────────────────────────────────────────────────────
N_FRAMES = ORBIT_PERIOD_FRAMES                # 120
TARGET_FPS = 30
DURATION_S = N_FRAMES / TARGET_FPS

_FRAMES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "test_basic_shader_frames")
os.makedirs(_FRAMES_DIR, exist_ok=True)

# Pre-build a tiny PNG writer (re-used for every frame).
import struct as _struct, zlib as _zlib

def _write_png_rgba(path: str, arr_rgba: np.ndarray) -> None:
    h, w = arr_rgba.shape[:2]
    raw_rows = bytearray()
    for row_i in range(h):
        raw_rows += b"\x00"
        raw_rows += arr_rgba[row_i, :, :3].tobytes()
    compressed = _zlib.compress(bytes(raw_rows), 6)
    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = _zlib.crc32(tag + data) & 0xFFFFFFFF
        return _struct.pack(">I", len(data)) + tag + data + _struct.pack(">I", crc)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_chunk(b"IHDR", _struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        fh.write(_chunk(b"IDAT", compressed))
        fh.write(_chunk(b"IEND", b""))

frame_count = 0
rgba = None

# ── Frame-delayed remission engine ───────────────────────────────────────────
# Materials whose YAML carries a `remit_profile_name` referring to a
# RemissionProfile with a `response_envelope` are picked up here.  In the
# default scene that's `ruby_glass` (→ `ruby_afterglow`, 12-frame exponential
# tail).  Per frame we synthesise a scalar stimulus per remissive material
# (here, a simple proxy: the orbital position drives a stimulus pulse twice
# per orbit when the ruby crosses the high-irradiance "front" hemisphere),
# advance the engine, and overwrite next frame's emission slot before
# uploading the updated PBR chunk.
from material_db import RemissionFeedbackEngine
remit_engine = RemissionFeedbackEngine(db, ep_db)
remit_engine.configure()
log.info("─── RemissionFeedbackEngine ───────────────────────────────────")
log.info("  remissive material count = %d", len(remit_engine.remissive_mat_ids()))
for _mid in remit_engine.remissive_mat_ids():
    _mname = next((n for n, i in mat_idx.items() if i == _mid), f"<mat{_mid}>")
    log.info("  driving %-32s  ir_len=%d", _mname, len(remit_engine._ir[_mid]))

log.info("─── Animated render  (%d frames, %.1fs @ %d fps) ──────────────",
         N_FRAMES, DURATION_S, TARGET_FPS)
log.info("  frames out                     = %s", _FRAMES_DIR)

t0 = time.perf_counter()

for _f in range(N_FRAMES):
    # Phase advances one full revolution over the loop.
    theta_base = 2.0 * np.pi * (_f / float(N_FRAMES))

    # ── Drive the remission engine ───────────────────────────────────────────
    # Stimulus model: each remissive orbiter receives a pulse proportional to
    # how close it is to the +Z hemisphere (proxy for "facing the bright side
    # of the saddle").  This is a stand-in for a real per-material irradiance
    # integral; the FIR convolution + frame-delay are the same regardless.
    _stim: dict[int, float] = {}
    for _mid in remit_engine.remissive_mat_ids():
        # Crude stimulus: 0.5*(1 + cos(2θ)) gives two pulses per orbit, peak 1.
        _stim[_mid] = float(0.5 * (1.0 + np.cos(2.0 * theta_base)))
    remit_engine.push_stimulus(_stim)
    remit_engine.apply_to_pbr(pbr_t)
    rdr.set_pbr_chunk(np.ascontiguousarray(pbr_t, dtype=np.float32))

    a_tris, m_ids, vv = _assemble_scene_for_phase(theta_base)
    # Tell the kernel which objects exist and what changed since last call.
    # (Materials emit light; the kernel derives one cluster-light per emissive
    # group from this declaration + cached state, never from per-frame work.)
    _push_groups(theta_base, first=(_f == 0))
    rdr.clear(0.0, 0.0, 0.0, 0.0)
    try:
        rdr.render(vv, m_ids, proj)
    except Exception as exc:
        log.error("  frame %d: render() raised: %s", _f, exc)
        break

    rgba = rdr.readback_u8()
    if rgba is None:
        log.warning("  frame %4d: readback_u8() returned None", _f)
        frame_count += 1
        continue

    rgba_np = np.asarray(rgba, dtype=np.uint8)
    if rgba_np.ndim == 1:
        rgba_np = rgba_np.reshape(HEIGHT, WIDTH, 4)

    # Save every frame.
    _path = os.path.join(_FRAMES_DIR, f"frame_{_f:04d}.png")
    try:
        _write_png_rgba(_path, rgba_np)
    except Exception as _png_exc:
        log.warning("  frame %4d: PNG save failed: %s", _f, _png_exc)

    a_ch     = rgba_np[:, :, 3]
    lit_mask = a_ch > 0
    lit_count = int(np.count_nonzero(lit_mask))
    alpha_pct = 100.0 * lit_count / (WIDTH * HEIGHT)
    if lit_count > 0:
        r_lit = rgba_np[:, :, 0][lit_mask].astype(np.float32)
        g_lit = rgba_np[:, :, 1][lit_mask].astype(np.float32)
        b_lit = rgba_np[:, :, 2][lit_mask].astype(np.float32)
        mean_r, mean_g, mean_b = float(np.mean(r_lit)), float(np.mean(g_lit)), float(np.mean(b_lit))
        max_r,  max_g,  max_b  = float(np.max(r_lit)),  float(np.max(g_lit)),  float(np.max(b_lit))
        rgb_sum_max = float(np.max(r_lit + g_lit + b_lit))
    else:
        mean_r = mean_g = mean_b = 0.0
        max_r  = max_g  = max_b  = 0.0
        rgb_sum_max = 0.0

    # Trim per-frame log volume — emit every 10th frame plus the first/last.
    if (_f % 10 == 0) or (_f == N_FRAMES - 1):
        log.info(
            "  frame %4d  θ=%5.2frad  lit=%5d (%5.1f%%)  "
            "mean=[%5.1f,%5.1f,%5.1f]  max=[%3.0f,%3.0f,%3.0f]  rgb_sum_max=%5.0f",
            _f, theta_base, lit_count, alpha_pct,
            mean_r, mean_g, mean_b, max_r, max_g, max_b, rgb_sum_max,
        )

    frame_count += 1

# ─────────────────────────────────────────────────────────────────────────────
# 11.  Final frame deep-inspection
# ─────────────────────────────────────────────────────────────────────────────
total_elapsed = time.perf_counter() - t0
log.info("─── Render complete  frames=%d  elapsed=%.2fs ──────────────────",
         frame_count, total_elapsed)

if rgba is not None:
    rgba_np = np.asarray(rgba, dtype=np.uint8)
    if rgba_np.ndim == 1:
        rgba_np = rgba_np.reshape(HEIGHT, WIDTH, 4)

    lit_mask  = rgba_np[:, :, 3] > 0
    lit_count = int(np.count_nonzero(lit_mask))

    log.info("─── Final frame pixel statistics ──────────────────────────────")
    log.info("  Dimensions   : %d × %d  (%d px total)", WIDTH, HEIGHT, WIDTH * HEIGHT)
    log.info("  Lit pixels   : %d  (%.1f%%)", lit_count, 100.0 * lit_count / (WIDTH * HEIGHT))

    if lit_count > 0:
        for ch_idx, ch_name in enumerate("RGBA"):
            ch = rgba_np[:, :, ch_idx]
            lit_vals = ch[lit_mask].astype(np.float32)
            log.info(
                "  %-4s : min=%3.0f  max=%3.0f  mean=%5.1f  median=%5.1f",
                ch_name,
                float(np.min(lit_vals)),
                float(np.max(lit_vals)),
                float(np.mean(lit_vals)),
                float(np.median(lit_vals)),
            )

        # Heuristic: the central showcase sphere occupies the image centre.
        # Compare the warm content (R+G) at the centre against the cool slate
        # background at the periphery — the gold sphere should be markedly
        # warmer than the saddle backdrop.
        cx, cy = WIDTH // 2, HEIGHT // 2
        r_inner = 40
        ys, xs  = np.mgrid[0:HEIGHT, 0:WIDTH]
        centre_mask = ((ys - cy) ** 2 + (xs - cx) ** 2) < r_inner ** 2
        outer_mask  = ~centre_mask & lit_mask

        if np.any(centre_mask & lit_mask) and np.any(outer_mask):
            ic_pixels = rgba_np[centre_mask & lit_mask].astype(np.float32)
            op_pixels = rgba_np[outer_mask].astype(np.float32)
            inner_warm = float(np.mean(ic_pixels[:, 0] + ic_pixels[:, 1]))
            outer_warm = float(np.mean(op_pixels[:, 0] + op_pixels[:, 1]))
            log.info("  Warm (R+G) — centre (gold orb): %.1f  periphery (saddle): %.1f",
                     inner_warm, outer_warm)
            if inner_warm > outer_warm:
                log.info("  [PASS] Warm-tinted central sphere reads brighter than backdrop")
            else:
                log.warning("  [WARN] Centre not noticeably warmer than backdrop")
        else:
            log.info("  (centre/periphery split unavailable — coverage too low)")
    else:
        log.warning("  [FAIL] Zero lit pixels in final frame — shader produced no output")

    # 5 × 5 grid sample
    log.info("─── 5×5 grid sample (R, G, B, A) ─────────────────────────────")
    ys_grid = np.linspace(0, HEIGHT - 1, 5, dtype=int)
    xs_grid = np.linspace(0, WIDTH  - 1, 5, dtype=int)
    for _y in ys_grid:
        row_str = "  y=%3d: " % _y
        for _x in xs_grid:
            px = rgba_np[_y, _x]
            row_str += "(%3d,%3d,%3d,%3d) " % (px[0], px[1], px[2], px[3])
        log.info(row_str)

    # Save PNG (minimal pure-Python writer — no Pillow dependency)
    _out_dir = os.path.dirname(os.path.abspath(__file__))
    _out_path = os.path.join(_out_dir, "test_basic_shader_out.png")
    try:
        import struct, zlib

        def _write_png(path, arr_rgba):
            h, w = arr_rgba.shape[:2]
            # Build raw scan-lines: filter byte 0x00 (None) + RGB bytes per row
            raw_rows = bytearray()
            for row_i in range(h):
                raw_rows += b"\x00"
                raw_rows += arr_rgba[row_i, :, :3].tobytes()
            compressed = zlib.compress(bytes(raw_rows), 9)

            def _chunk(tag: bytes, data: bytes) -> bytes:
                crc = zlib.crc32(tag + data) & 0xFFFFFFFF
                return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

            with open(path, "wb") as fh:
                fh.write(b"\x89PNG\r\n\x1a\n")
                fh.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
                fh.write(_chunk(b"IDAT", compressed))
                fh.write(_chunk(b"IEND", b""))

        _write_png(_out_path, rgba_np)
        log.info("─── Saved final frame PNG ─────────────────────────────────────")
        log.info("  %s", _out_path)
    except Exception as _png_exc:
        log.warning("  Could not save PNG: %s", _png_exc)

else:
    log.warning("[FAIL] No frames produced (rgba is None at end of loop)")

# ─────────────────────────────────────────────────────────────────────────────
# 12.  Compose animated PNG from per-frame dumps
# ─────────────────────────────────────────────────────────────────────────────
try:
    import glob as _glob
    from PIL import Image as _PILImage

    _apng_paths = sorted(_glob.glob(os.path.join(_FRAMES_DIR, "frame_*.png")))
    if not _apng_paths:
        log.warning("[apng] no frames in %s — skipping", _FRAMES_DIR)
    else:
        _apng_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "test_basic_shader_animation.png")
        _APNG_FPS = 30
        _frames = [_PILImage.open(p).convert("RGBA") for p in _apng_paths]
        _duration_ms = int(round(1000.0 / _APNG_FPS))
        _head, *_tail = _frames
        _head.save(
            _apng_out,
            format="PNG",
            save_all=True,
            append_images=_tail,
            duration=_duration_ms,
            loop=0,
            disposal=2,
            default_image=False,
        )
        _sz = os.path.getsize(_apng_out)
        log.info("─── Composed APNG ─────────────────────────────────────────────")
        log.info("  %s", _apng_out)
        log.info("  %d frames @ %d fps  (%.1f KiB)",
                 len(_frames), _APNG_FPS, _sz / 1024.0)
except Exception as _apng_exc:
    log.warning("[apng] compose failed: %s", _apng_exc)

log.info("─── Done ──────────────────────────────────────────────────────")
