"""camera_software/camera_body.py
----------------------------------
Physical camera body enclosure — the light-tight box that holds the optical
system together.

Component hierarchy (front → back along the optical axis):

    ┌─ LENS MOUNT FACE ──────────────────────────────────────────────────┐
    │  Annular front plate: outer_r → mount_r_inner (clear aperture)     │
    │  The lens barrel back face rests flush here.                        │
    └────────────────────────────────────────────────────────────────────┘
    │  OUTER SHELL — cylindrical or box walls connecting front to back    │
    ├─ CHAMBER SEGMENTS ─────────────────────────────────────────────────┤
    │  Each chamber is a named volume defined by (x_start, x_end,        │
    │  r_inner).  Typical sequence:                                       │
    │    "mirror_box"   — reserved space for reflex mirror / prism        │
    │    "shutter"      — focal-plane or leaf shutter light gate          │
    │    "sensor_space" — volume between shutter and exposure plane       │
    │  No mesh is generated inside a chamber; geometry surrounds it.     │
    ├─ INNER LIGHT GUIDE ────────────────────────────────────────────────┤
    │  A tapered cone (or cylinder) from the mount clear aperture to the  │
    │  sensor radius.  Matte black.  Prevents stray light reaching the    │
    │  sensor from outside the designed light cone.                       │
    ├─ SENSOR HOUSING ───────────────────────────────────────────────────┤
    │  Annular ring around the sensor active area at sensor_x.           │
    └─ CAMERA BACK FACE ─────────────────────────────────────────────────┘
       Disc or ring at back_x.  If secondary_back_depth > 0, a second
       light-tight extension box is generated here for accessories
       (rear projection, digital back swaps, film holders, etc.).

Design philosophy
-----------------
*  All geometry is expressed in the simulation's global coordinate system:
   the optical axis runs along the X-axis; the sensor is at higher X values
   (object/scene is at lower X).
*  This module is independent of thick_lens_focus_lab.py.  It produces
   standalone numpy arrays that any scene builder can merge.
*  Material assignment is the caller's responsibility — pass mat_idx for
   the desired interior-black material.
*  ChamberSpec defines reserved internal space.  Geometry only wraps around
   chambers; nothing is built inside them so they remain extensible.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

__all__ = [
    "ChamberSpec",
    "CameraBodySpec",
    "build_camera_body_mesh",
    "camera_body_from_assembly",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ChamberSpec:
    """One named segment of the internal camera volume.

    Chambers are ordered front-to-back (ascending x) and together span the
    full interior from the mount face to the sensor.  The light guide cone
    is built from the chamber inner radii — if omitted the guide is straight
    (mount_r_inner at both ends), which is conservative.

    Parameters
    ----------
    name        : human-readable label ("mirror_box", "shutter", …)
    x_start     : front X of this chamber (metres)
    x_end       : back  X of this chamber (metres)
    r_inner     : clear inner radius through this segment (metres)
                  Determines the light guide at this junction.
    note        : free-text description (design rationale, constraints)
    """
    name: str
    x_start: float
    x_end: float
    r_inner: float
    note: str = ""


@dataclass
class CameraBodySpec:
    """Complete specification for a camera body enclosure.

    Coordinate system: optical axis = +X, sensor at higher X than lens.

    Parameters
    ----------
    lens_mount_x     : X position of the front face / lens mount flange.
                       The lens barrel's back face sits flush here.
    sensor_x         : X position of the exposure plane (sensor / film).
    back_x           : X position of the camera back face.  Must be >= sensor_x.
    outer_r          : Outer body radius (cylinder) or largest cross-section.
    mount_r_inner    : Clear aperture radius at the lens mount (the hole).
    mount_r_flange   : Outer radius of the mount ring.  Typically equals or
                       slightly exceeds the lens barrel outer radius.
    sensor_r         : Active sensor area radius.
    sensor_housing_r : Annular ring outer radius around the sensor.  Defaults
                       to outer_r when 0.
    chambers         : Ordered list of internal chamber segments.  If empty
                       the interior is treated as one open volume.
    body_shape       : "cylinder" or "box".  Only "cylinder" is fully
                       implemented; "box" is reserved for later.
    n_theta          : Number of azimuthal segments for all cylindrical surfaces.
    secondary_back_depth : If > 0, an extension box (accessory bay) is added
                       behind back_x with this depth.
    secondary_back_r : Outer radius of the extension.  0 → same as outer_r.
    """
    lens_mount_x: float
    sensor_x: float
    back_x: float
    outer_r: float
    mount_r_inner: float
    mount_r_flange: float
    sensor_r: float
    sensor_housing_r: float = 0.0
    chambers: List[ChamberSpec] = field(default_factory=list)
    body_shape: str = "cylinder"
    n_theta: int = 96
    secondary_back_depth: float = 0.0
    secondary_back_r: float = 0.0

    def effective_sensor_housing_r(self) -> float:
        return self.sensor_housing_r if self.sensor_housing_r > 0.0 else self.outer_r

    def effective_secondary_r(self) -> float:
        return self.secondary_back_r if self.secondary_back_r > 0.0 else self.outer_r


# ---------------------------------------------------------------------------
# Low-level mesh primitives
# ---------------------------------------------------------------------------

def _annular_ring(x: float,
                  r_inner: float,
                  r_outer: float,
                  n_theta: int,
                  normal_sign: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Flat annular disc at position x, facing +X (normal_sign=+1) or -X (-1).

    Returns
    -------
    verts : (V, 6) float32  — x,y,z, nx,ny,nz
    tris  : (T, 3) int32
    """
    verts, tris = [], []
    n = int(n_theta)
    # Two rings of vertices: inner radius and outer radius.
    for ring, r in enumerate((r_inner, r_outer)):
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            y = r * math.cos(theta)
            z = r * math.sin(theta)
            verts.append([x, y, z, normal_sign, 0.0, 0.0])
    # Connect inner ring (0..n-1) to outer ring (n..2n-1) with quads.
    for i in range(n):
        i0 = i
        i1 = (i + 1) % n
        o0 = n + i
        o1 = n + (i + 1) % n
        # Two triangles per quad, winding depends on normal direction.
        if normal_sign > 0:
            tris += [[i0, o0, i1], [i1, o0, o1]]
        else:
            tris += [[i0, i1, o0], [i1, o1, o0]]
    return (np.array(verts,  dtype=np.float32),
            np.array(tris,   dtype=np.int32))


def _cylinder_wall(x0: float, r0: float,
                   x1: float, r1: float,
                   n_theta: int,
                   inward: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Tapered cylindrical wall (frustum) from (x0, r0) to (x1, r1).

    Normals point outward radially (inward=False) or inward (inward=True).

    Returns
    -------
    verts : (V, 6) float32  — x,y,z, nx,ny,nz
    tris  : (T, 3) int32
    """
    n = int(n_theta)
    verts, tris = [], []
    sign = -1.0 if inward else 1.0

    for ring_idx, (x, r) in enumerate(((x0, r0), (x1, r1))):
        for i in range(n):
            theta = 2.0 * math.pi * i / n
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            y = r * cos_t
            z = r * sin_t
            nx = 0.0
            ny = sign * cos_t
            nz = sign * sin_t
            verts.append([x, y, z, nx, ny, nz])

    # Quads between ring 0 (front) and ring 1 (back).
    for i in range(n):
        a = i
        b = (i + 1) % n
        c = n + i
        d = n + (i + 1) % n
        if not inward:
            tris += [[a, c, b], [b, c, d]]
        else:
            tris += [[a, b, c], [b, d, c]]

    return (np.array(verts, dtype=np.float32),
            np.array(tris,  dtype=np.int32))


def _disc(x: float,
          r: float,
          n_theta: int,
          normal_sign: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Solid flat disc at position x.

    Returns
    -------
    verts : (V, 6) float32
    tris  : (T, 3) int32
    """
    n = int(n_theta)
    verts = [[x, 0.0, 0.0, normal_sign, 0.0, 0.0]]  # centre
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        verts.append([x, r * math.cos(theta), r * math.sin(theta),
                      normal_sign, 0.0, 0.0])
    tris = []
    for i in range(n):
        a = 0
        b = 1 + i
        c = 1 + (i + 1) % n
        if normal_sign > 0:
            tris.append([a, b, c])
        else:
            tris.append([a, c, b])
    return (np.array(verts, dtype=np.float32),
            np.array(tris,  dtype=np.int32))


def _merge(parts: list) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate (verts, tris) pairs into a single indexed mesh."""
    all_verts = []
    all_tris  = []
    offset = 0
    for v, t in parts:
        all_verts.append(v)
        all_tris.append(t + offset)
        offset += len(v)
    if not all_verts:
        return (np.zeros((0, 6), dtype=np.float32),
                np.zeros((0, 3), dtype=np.int32))
    return (np.concatenate(all_verts, axis=0),
            np.concatenate(all_tris,  axis=0))


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_camera_body_mesh(spec: CameraBodySpec) -> dict:
    """Build all camera body meshes from a CameraBodySpec.

    Returns a dict mapping component name → (verts (V,6) float32, tris (T,3) int32).
    Each entry is a standalone indexed mesh in world space (optical-axis = X).

    Components returned
    -------------------
    "mount_face"       — front annular ring (mount hole to body outer wall)
    "outer_shell"      — cylindrical outer wall from mount to back
    "inner_light_guide"— tapered inner cone from mount aperture to sensor
    "sensor_housing"   — annular ring around the sensor plane
    "back_face"        — rear disc / ring
    "secondary_back"   — extension bay walls + face (if spec.secondary_back_depth > 0)

    All faces use outward-facing normals (suitable for ray intersection from
    outside the body).  The inner_light_guide has inward-facing normals
    (the camera interior sees it from inside).
    """
    n = spec.n_theta
    meshes: dict = {}

    # ── 1.  Front mount face — annular ring sealing lens barrel to body ──
    meshes["mount_face"] = _merge([
        _annular_ring(spec.lens_mount_x,
                      spec.mount_r_inner,
                      spec.outer_r,
                      n,
                      normal_sign=-1.0)  # normal points toward scene (-X)
    ])

    # ── 2.  Outer shell — tapered or straight wall, mount to back face ───
    #   May taper if outer_r != effective_secondary_r, but typically straight.
    meshes["outer_shell"] = _merge([
        _cylinder_wall(spec.lens_mount_x, spec.outer_r,
                       spec.back_x,       spec.outer_r,
                       n,
                       inward=False)
    ])

    # ── 3.  Inner light guide — tapered cone, mount hole → sensor area ───
    #   This is the "utilitarian black plastic guide" that channels the light
    #   cone.  If chambers are defined, we honour each segment's r_inner so
    #   the guide follows the prescribed clear aperture through mirror/shutter
    #   spaces.  With no chambers it collapses to a single frustum.
    guide_parts = []
    segments: List[Tuple[float, float, float, float]] = []  # (x0, r0, x1, r1)

    if spec.chambers:
        sorted_ch = sorted(spec.chambers, key=lambda c: c.x_start)
        # From mount to first chamber
        prev_x = spec.lens_mount_x
        prev_r = spec.mount_r_inner
        for ch in sorted_ch:
            segments.append((prev_x, prev_r, ch.x_start, ch.r_inner))
            segments.append((ch.x_start, ch.r_inner, ch.x_end, ch.r_inner))
            prev_x = ch.x_end
            prev_r = ch.r_inner
        # From last chamber to sensor
        segments.append((prev_x, prev_r, spec.sensor_x, spec.sensor_r))
    else:
        segments = [(spec.lens_mount_x, spec.mount_r_inner,
                     spec.sensor_x,     spec.sensor_r)]

    for x0, r0, x1, r1 in segments:
        if abs(x1 - x0) > 1e-9 and (r0 > 1e-9 or r1 > 1e-9):
            guide_parts.append(
                _cylinder_wall(x0, max(r0, 1e-9),
                               x1, max(r1, 1e-9),
                               n, inward=True)  # faces inward (interior surface)
            )
    meshes["inner_light_guide"] = _merge(guide_parts)

    # ── 4.  Sensor housing — annular ring around the exposure plane ───────
    hs_r = spec.effective_sensor_housing_r()
    meshes["sensor_housing"] = _merge([
        _annular_ring(spec.sensor_x,
                      spec.sensor_r,
                      hs_r,
                      n,
                      normal_sign=1.0)  # normal toward sensor (+X, toward scene)
    ])

    # ── 5.  Back face — closes the camera body ────────────────────────────
    if spec.sensor_x < spec.back_x - 1e-9:
        # Outer shell extends past sensor: need an annular ring at back face
        # from sensor housing outward.
        back_parts = [
            _annular_ring(spec.back_x,
                          hs_r,
                          spec.outer_r,
                          n,
                          normal_sign=1.0)  # normal pointing away from scene (+X)
        ]
    else:
        # Sensor is flush with back — full solid back disc.
        back_parts = [
            _annular_ring(spec.back_x,
                          spec.sensor_r,
                          spec.outer_r,
                          n,
                          normal_sign=1.0)
        ]
    meshes["back_face"] = _merge(back_parts)

    # ── 6.  Secondary back / accessory bay (optional) ────────────────────
    if spec.secondary_back_depth > 1e-9:
        sec_r = spec.effective_secondary_r()
        sec_back_x = spec.back_x + spec.secondary_back_depth
        sec_parts = [
            # Side walls of the extension bay
            _cylinder_wall(spec.back_x, sec_r,
                           sec_back_x,  sec_r,
                           n, inward=False),
            # Rear cap of the extension — a full disc (or annulus for pass-through)
            _annular_ring(sec_back_x,
                          0.0,      # closed cap for now
                          sec_r,
                          n,
                          normal_sign=1.0),
        ]
        # Transition annulus on the body back face, from outer_r to sec_r
        if sec_r > spec.outer_r + 1e-6:
            sec_parts.insert(0,
                _annular_ring(spec.back_x,
                              spec.outer_r,
                              sec_r,
                              n,
                              normal_sign=1.0))
        meshes["secondary_back"] = _merge(sec_parts)
    else:
        meshes["secondary_back"] = (_zeros_mesh())

    return meshes


def _zeros_mesh() -> Tuple[np.ndarray, np.ndarray]:
    return (np.zeros((0, 6), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32))


# ---------------------------------------------------------------------------
# Convenience factory: derive spec from LensAssembly + scene parameters
# ---------------------------------------------------------------------------

def camera_body_from_assembly(
    lens_mount_x: float,
    sensor_x: float,
    mount_r_inner: float,
    mount_r_flange: float,
    outer_r: float,
    sensor_r: float,
    has_mirror_box: bool = False,
    mirror_box_depth: float = 0.045,
    has_focal_plane_shutter: bool = True,
    shutter_thickness: float = 0.004,
    back_clearance: float = 0.005,
    n_theta: int = 96,
) -> CameraBodySpec:
    """Construct a CameraBodySpec from the assembly's key dimensions.

    Parameters
    ----------
    lens_mount_x          : X of the lens mount flange (= last lens group back + gap)
    sensor_x              : X of the exposure plane
    mount_r_inner         : Clear aperture at the mount face (lens barrel inner bore)
    mount_r_flange        : Outer radius of the mount ring (typically = outer_r or slightly less)
    outer_r               : Camera body outer radius
    sensor_r              : Sensor active area radius
    has_mirror_box        : Reserve space for a reflex mirror / prism
    mirror_box_depth      : Depth (in X) of the mirror box if present
    has_focal_plane_shutter: Include a shutter chamber immediately before the sensor
    shutter_thickness     : Shutter mechanism thickness in X
    back_clearance        : Extra space behind sensor before the camera back face
    n_theta               : Azimuthal tessellation count
    """
    x = lens_mount_x
    chambers: List[ChamberSpec] = []

    # Mirror box — immediately behind mount face.
    if has_mirror_box:
        mb_depth = max(0.005, float(mirror_box_depth))
        chambers.append(ChamberSpec(
            name="mirror_box",
            x_start=x,
            x_end=x + mb_depth,
            r_inner=mount_r_inner,
            note="Reserved for reflex mirror, beam splitter, or prism.",
        ))
        x += mb_depth

    # Main interior volume — from end of mirror box (or mount) to shutter.
    sensor_side_of_interior = sensor_x - (shutter_thickness if has_focal_plane_shutter else 0.0)
    if sensor_side_of_interior > x + 1e-9:
        chambers.append(ChamberSpec(
            name="interior",
            x_start=x,
            x_end=sensor_side_of_interior,
            r_inner=max(sensor_r, mount_r_inner * 0.5),
            note="Main interior volume — light path from mount to shutter / sensor.",
        ))
        x = sensor_side_of_interior

    # Focal-plane shutter — immediately before sensor.
    if has_focal_plane_shutter:
        sh = max(0.001, float(shutter_thickness))
        chambers.append(ChamberSpec(
            name="shutter",
            x_start=x,
            x_end=min(x + sh, sensor_x),
            r_inner=sensor_r,
            note="Focal-plane shutter mechanism — light gate just before sensor.",
        ))

    back_x = sensor_x + max(0.0, float(back_clearance))

    return CameraBodySpec(
        lens_mount_x=lens_mount_x,
        sensor_x=sensor_x,
        back_x=back_x,
        outer_r=outer_r,
        mount_r_inner=mount_r_inner,
        mount_r_flange=mount_r_flange,
        sensor_r=sensor_r,
        sensor_housing_r=outer_r,
        chambers=chambers,
        n_theta=n_theta,
    )
