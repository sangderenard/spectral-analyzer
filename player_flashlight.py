"""
player_flashlight.py — Physical flashlight: faceted parabolic cup + tungsten bulb.

Geometry coordinate system (local space)
-----------------------------------------
  +Z = forward  (open face of the cup, toward the scene)
  Origin at the apex of the parabolic reflector (back of cup, z = 0).

Cup optics
----------
Inner reflector surface is a paraboloid of revolution:

    r² = 4 · focal_length · z

where  focal_length = cup_radius² / (4 · cup_depth).

Each cell of the paraboloid mesh is explicitly flattened onto the tangent
plane at its centre point, producing a segmented "square-faceted" reflector.
In a ray-tracer each flat facet acts as a tiny plane mirror aimed slightly
differently from its neighbours, creating a characteristic multi-spot or
ring-of-spots beam.  In the rasteriser the facets produce hard-edged
specular lobes.

Focal-point semantics
---------------------
  bulb_z == focal_length   →  parallel output beam
  bulb_z  < focal_length   →  converging beam (then diverges past focus)
  bulb_z  > focal_length   →  diverging (wider) beam

Glass front element
-------------------
A borosilicate glass disk at z = cup_depth covers the open face.  It
diffuses the output beam slightly and acts as a lens seat for the camera-
station upgrade (PlacedLens, not included here).

Materials (register in configs/materials/)
------------------------------------------
  tungsten_filament     — high-emission warm-white bulb sphere
  flashlight_reflector  — polished-aluminium inner cup surface
  flashlight_shell      — matte-black outer shell, back cap, rim
  borosilicate_glass    — front glass disk (already exists)

Key bindings (wired in PlayerController)
-----------------------------------------
  G          — toggle flashlight on / off
  [          — move bulb backward (narrow / sharpen beam)
  ]          — move bulb forward  (widen beam)
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _proj_onto_plane(P: np.ndarray, Pc: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Project point P onto the plane (Pc, N)."""
    return (P - np.dot(P - Pc, N) * N).astype(np.float32)


def _translate_tris(tris: np.ndarray, dx: float, dy: float, dz: float) -> np.ndarray:
    """Translate a triangle soup (N,3,3) by a local offset."""
    if tris is None or len(tris) == 0:
        return np.zeros((0, 3, 3), np.float32)
    out = np.asarray(tris, dtype=np.float32).copy()
    out[:, :, 0] += float(dx)
    out[:, :, 1] += float(dy)
    out[:, :, 2] += float(dz)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Primitive generators  (local flashlight space)
# ─────────────────────────────────────────────────────────────────────────────

def _faceted_parabolic_bowl(cup_radius: float, cup_depth: float,
                            n_rings: int = 12, n_sides: int = 20) -> np.ndarray:
    """
    Square-faceted inner paraboloid of revolution, inward normals.

    Each ring × side cell is projected onto the tangent plane at its centre
    so it forms a single flat rectangular mirror face.  In the ray-tracer
    each facet redirects light with its own constant normal, producing the
    characteristic multi-spot/ring pattern of segmented reflectors.

    Returns (N, 3, 3) float32 with winding chosen for inward normals
    (normal pointing toward the optical axis).
    """
    R, D = float(cup_radius), float(cup_depth)
    f    = R * R / (4.0 * D)   # paraboloid focal length
    tris = []

    for ri in range(n_rings):
        z0 = D * ri       / n_rings
        z1 = D * (ri + 1) / n_rings
        zc = (z0 + z1) * 0.5

        r0 = R * math.sqrt(z0 / D) if z0 > 1e-9 else 0.0
        r1 = R * math.sqrt(z1 / D)
        rc = R * math.sqrt(zc / D)

        for si in range(n_sides):
            a0 = 2.0 * math.pi * si       / n_sides
            a1 = 2.0 * math.pi * (si + 1) / n_sides
            ac = (a0 + a1) * 0.5

            # Tangent-plane centre and inward normal for this cell.
            # Paraboloid gradient (outward): (rc·cos(ac), rc·sin(ac), -2f)
            # Inward (toward axis): negate, then normalise.
            Pc = np.array([rc*math.cos(ac), rc*math.sin(ac), zc], np.float64)
            N_out = np.array([rc*math.cos(ac), rc*math.sin(ac), -2.0*f], np.float64)
            N_len = float(np.linalg.norm(N_out))
            if N_len < 1e-9:
                N = np.array([0.0, 0.0, 1.0])
            else:
                N = -N_out / N_len  # inward

            # Raw corners A(z0,a0), B(z0,a1), C(z1,a1), D(z1,a0) on paraboloid
            def corner(z, r, a):
                rr = R * math.sqrt(z / D) if z > 1e-9 else 0.0
                return np.array([rr*math.cos(a), rr*math.sin(a), z], np.float64)

            A = corner(z0, r0, a0)
            B = corner(z0, r0, a1)
            C = corner(z1, r1, a1)
            Dv = corner(z1, r1, a0)

            if ri == 0:
                # First ring: apex at origin, project and fan to first ring.
                apex = _proj_onto_plane(np.zeros(3, np.float64), Pc, N)
                C_p  = _proj_onto_plane(C,  Pc, N)
                D_p  = _proj_onto_plane(Dv, Pc, N)
                # Winding: inward normal (cross(D_p-apex, C_p-apex) ≈ inward)
                tris.append([apex, D_p, C_p])
            else:
                # Project all 4 corners onto the cell tangent plane
                A_p  = _proj_onto_plane(A,  Pc, N)
                B_p  = _proj_onto_plane(B,  Pc, N)
                C_p  = _proj_onto_plane(C,  Pc, N)
                D_p  = _proj_onto_plane(Dv, Pc, N)
                # Two triangles forming a flat facet quad
                # Winding [A,D,B] and [D,C,B] gives inward normals (verified)
                tris.append([A_p, D_p, B_p])
                tris.append([D_p, C_p, B_p])

    return np.array(tris, dtype=np.float32)


def _cylinder_lateral(radius: float, z_back: float, z_front: float,
                      n_sides: int = 24, outward: bool = True) -> np.ndarray:
    """(N, 3, 3) lateral surface of a right circular cylinder."""
    tris = []
    for i in range(n_sides):
        a0 = 2.0 * math.pi * i       / n_sides
        a1 = 2.0 * math.pi * (i + 1) / n_sides
        ca0, sa0 = math.cos(a0), math.sin(a0)
        ca1, sa1 = math.cos(a1), math.sin(a1)

        p00 = np.array([radius*ca0, radius*sa0, z_back],  np.float32)
        p10 = np.array([radius*ca1, radius*sa1, z_back],  np.float32)
        p01 = np.array([radius*ca0, radius*sa0, z_front], np.float32)
        p11 = np.array([radius*ca1, radius*sa1, z_front], np.float32)

        if outward:
            tris.append([p00, p10, p01])
            tris.append([p10, p11, p01])
        else:
            tris.append([p10, p00, p01])
            tris.append([p11, p10, p01])
    return np.array(tris, dtype=np.float32)


def _disk(radius: float, z: float, n_sides: int = 24,
          normal_fwd: bool = True) -> np.ndarray:
    """(N, 3, 3) solid disk at height z.  normal_fwd → normal +Z."""
    center = np.array([0.0, 0.0, z], np.float32)
    tris = []
    for i in range(n_sides):
        a0 = 2.0 * math.pi * i       / n_sides
        a1 = 2.0 * math.pi * (i + 1) / n_sides
        p0 = np.array([radius*math.cos(a0), radius*math.sin(a0), z], np.float32)
        p1 = np.array([radius*math.cos(a1), radius*math.sin(a1), z], np.float32)
        if normal_fwd:
            tris.append([center, p0, p1])
        else:
            tris.append([center, p1, p0])
    return np.array(tris, dtype=np.float32)


def _annulus(r_inner: float, r_outer: float, z: float,
             n_sides: int = 24, normal_fwd: bool = True) -> np.ndarray:
    """(N, 3, 3) annular disk (washer) at height z."""
    tris = []
    for i in range(n_sides):
        a0 = 2.0 * math.pi * i       / n_sides
        a1 = 2.0 * math.pi * (i + 1) / n_sides
        ca0, sa0 = math.cos(a0), math.sin(a0)
        ca1, sa1 = math.cos(a1), math.sin(a1)

        pi0 = np.array([r_inner*ca0, r_inner*sa0, z], np.float32)
        pi1 = np.array([r_inner*ca1, r_inner*sa1, z], np.float32)
        po0 = np.array([r_outer*ca0, r_outer*sa0, z], np.float32)
        po1 = np.array([r_outer*ca1, r_outer*sa1, z], np.float32)

        if normal_fwd:
            tris.append([pi0, po0, pi1])
            tris.append([po0, po1, pi1])
        else:
            tris.append([po0, pi0, pi1])
            tris.append([po1, po0, pi1])
    return np.array(tris, dtype=np.float32)


def _sphere(radius: float, cx: float, cy: float, cz: float,
            n_lat: int = 8, n_lon: int = 16) -> np.ndarray:
    """(N, 3, 3) UV sphere centred at (cx, cy, cz) with outward normals."""
    tris = []
    for li in range(n_lat):
        phi0 = math.pi * li       / n_lat
        phi1 = math.pi * (li + 1) / n_lat
        sp0, cp0 = math.sin(phi0), math.cos(phi0)
        sp1, cp1 = math.sin(phi1), math.cos(phi1)
        for gi in range(n_lon):
            th0 = 2.0 * math.pi * gi       / n_lon
            th1 = 2.0 * math.pi * (gi + 1) / n_lon
            ct0, st0 = math.cos(th0), math.sin(th0)
            ct1, st1 = math.cos(th1), math.sin(th1)

            p00 = np.array([cx+radius*sp0*ct0, cy+radius*sp0*st0, cz+radius*cp0], np.float32)
            p10 = np.array([cx+radius*sp0*ct1, cy+radius*sp0*st1, cz+radius*cp0], np.float32)
            p01 = np.array([cx+radius*sp1*ct0, cy+radius*sp1*st0, cz+radius*cp1], np.float32)
            p11 = np.array([cx+radius*sp1*ct1, cy+radius*sp1*st1, cz+radius*cp1], np.float32)

            if li > 0:
                tris.append([p00, p10, p01])
            if li < n_lat - 1:
                tris.append([p10, p11, p01])

    return np.array(tris, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PlayerFlashlight
# ─────────────────────────────────────────────────────────────────────────────

class PlayerFlashlight:
    """
    Physical flashlight: square-faceted parabolic reflector cup, tungsten bulb,
    and a borosilicate glass front element.

    Parameters
    ----------
    cup_radius : float
        Radius of the cup's open face (metres).  Default 40 mm.
    cup_depth : float
        Axial depth from apex (z=0) to open face (z=cup_depth).  Default 50 mm.
    shell_thickness : float
        Outer wall thickness (metres).  Default 3 mm.
    bulb_z : float | None
        Bulb centre axial position from the apex.  None → placed at focal_length
        (tight parallel beam).
    bulb_radius : float
        Radius of the emissive tungsten sphere (metres).  Default 5 mm.
    n_sides : int
        Polygon count around the circumference.
    n_rings : int
        Number of facet rings along the paraboloid.
    """

    # Material constants — must be registered in configs/materials/
    MAT_BULB      = "tungsten_filament"
    MAT_REFLECTOR = "flashlight_reflector"
    MAT_SHELL     = "flashlight_shell"
    MAT_GLASS     = "borosilicate_glass"   # front element


    def __init__(
        self,
        cup_radius:      float = 0.052,
        cup_depth:       float = 0.110,
        shell_thickness: float = 0.003,
        bulb_z:          Optional[float] = None,
        bulb_radius:     float = 0.0025,
        n_sides:         int   = 24,
        n_rings:         int   = 12,
    ):
        self.cup_radius      = float(cup_radius)
        self.cup_depth       = float(cup_depth)
        self.shell_thickness = float(shell_thickness)
        self.bulb_radius     = float(bulb_radius)
        self.n_sides         = int(n_sides)
        self.n_rings         = int(n_rings)

        # r² = 4·f·z  →  f = R²/(4·D)
        self._focal_length = self.cup_radius ** 2 / (4.0 * self.cup_depth)

        # Rear aperture + bulb enclosure constraint.
        self._rear_aperture_radius = max(0.0015, self.bulb_radius * 0.75)
        self._bulb_open_clearance = self.bulb_radius * 2.0
        _z_min = self._rear_aperture_radius + self.bulb_radius + self._bulb_open_clearance
        _z_max = self.cup_depth - (self.bulb_radius + self._bulb_open_clearance)
        _default_z = max(_z_min, min(self._focal_length, _z_max))
        self.bulb_z = float(np.clip((_default_z if bulb_z is None else float(bulb_z)), _z_min, _z_max))
        self._enabled = False

        self._build_static()
        self._build_bulb()

    # ── Geometry ───────────────────────────────────────────────────────────────

    def _build_static(self) -> None:
        R  = self.cup_radius
        D  = self.cup_depth
        t  = self.shell_thickness
        Ro = R + t
        ns = self.n_sides
        nr = self.n_rings

        # Inner reflector: square-faceted paraboloid, inward normals
        self._inner = _faceted_parabolic_bowl(R, D, n_rings=nr, n_sides=ns)

        # Outer cylindrical shell (outward normals)
        self._outer = _cylinder_lateral(Ro, 0.0, D, n_sides=ns, outward=True)

        # Back cap with centered aperture ring (normal -Z).
        self._back = _annulus(self._rear_aperture_radius, Ro, 0.0,
                      n_sides=ns, normal_fwd=False)

        # Front annular rim — washer bridging inner to outer radius (normal +Z)
        self._rim   = _annulus(R, Ro, D, n_sides=ns, normal_fwd=True)

        # Front glass disk — covers the open cup face (both sides visible)
        self._glass = _disk(R, D, n_sides=ns, normal_fwd=True)

        # Handle: coaxial with flashlight axis (centered on bulb/reflector axis).
        # Local +Z remains the flashlight aim axis.
        handle_r = max(0.008, R * 0.24)
        handle_z0 = -max(0.11, D * 1.5)
        handle_z1 = 0.0
        h_outer = _cylinder_lateral(handle_r, handle_z0, handle_z1, n_sides=ns, outward=True)
        h_back  = _disk(handle_r, handle_z0, n_sides=ns, normal_fwd=False)
        h_front = _disk(handle_r, handle_z1, n_sides=ns, normal_fwd=True)
        self._handle = np.concatenate([
            _translate_tris(h_outer, 0.0, 0.0, 0.0),
            _translate_tris(h_back,  0.0, 0.0, 0.0),
            _translate_tris(h_front, 0.0, 0.0, 0.0),
        ], axis=0).astype(np.float32)

    def _build_bulb(self) -> None:
        self._bulb = _sphere(
            self.bulb_radius,
            0.0, 0.0, float(self.bulb_z),
            n_lat=8, n_lon=16,
        )

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def toggle(self) -> bool:
        self._enabled = not self._enabled
        return self._enabled

    def enable(self)  -> None: self._enabled = True
    def disable(self) -> None: self._enabled = False

    @property
    def focal_length(self) -> float:
        """Focal length of the paraboloid (metres).  Bulb here → parallel beam."""
        return self._focal_length

    def set_bulb_z(self, z: float) -> None:
        """Move bulb along axis while keeping diameter clearance from open areas."""
        z_min = self._rear_aperture_radius + self.bulb_radius + self._bulb_open_clearance
        z_max = self.cup_depth - (self.bulb_radius + self._bulb_open_clearance)
        self.bulb_z = float(np.clip(z, z_min, z_max))
        self._build_bulb()

    @property
    def bulb_local_pos(self) -> np.ndarray:
        """Bulb centre in flashlight-local space (+Z = forward)."""
        return np.array([0.0, 0.0, self.bulb_z], dtype=np.float32)

    # ── World-space geometry ───────────────────────────────────────────────────

    @staticmethod
    def _xform(tris: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Apply a 4×4 column-major world transform to (N, 3, 3) float32 triangles."""
        n = len(tris)
        if n == 0:
            return tris
        pts = tris.reshape(-1, 3)
        hom = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)
        out = (M.astype(np.float32) @ hom.T).T[:, :3]
        return out.reshape(n, 3, 3).astype(np.float32)

    def world_tris(self, transform: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply a world transform and return per-component triangle soups.

        Parameters
        ----------
        transform : (4, 4) float32
            Column-major world transform.  Local +Z maps to the aim direction.

        Returns
        -------
        dict with keys (suggested materials in parentheses):
          "inner_reflector" → MAT_REFLECTOR  (polished aluminium)
          "outer_shell"     → MAT_SHELL      (matte black)
                    "handle"          → MAT_SHELL      (matte black)
          "back_cap"        → MAT_SHELL
          "front_rim"       → MAT_SHELL
          "front_glass"     → MAT_GLASS      (borosilicate glass)
          "bulb"            → MAT_BULB       (tungsten filament)
        """
        M = np.asarray(transform, dtype=np.float32)
        return {
            "inner_reflector": self._xform(self._inner, M),
            "outer_shell":     self._xform(self._outer, M),
            "handle":          self._xform(self._handle, M),
            "back_cap":        self._xform(self._back,  M),
            "front_rim":       self._xform(self._rim,   M),
            "front_glass":     self._xform(self._glass, M),
            "bulb":            self._xform(self._bulb,  M),
        }
