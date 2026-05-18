"""camera_designer/bake_worker.py
==================================
64-bit parametric ray tracer that bakes a CameraPreset into a LensManifold
noodle LUT.

Architecture
------------
All ray tracing uses the ParametricSurface CPU intersect() path — float64,
no meshes, exact conic/polynomial geometry.  Snell's law is applied at each
surface interface using the GlassSpec.n_at(wavelength) Sellmeier formula.

The aperture plane (aperture_stop) is the manifold coordinate origin.  Rays
that pass through the aperture are parameterised by their normalised aperture
hit position (u,v) ∈ [-1,1]² and their scene-side incident direction
(in_dx, in_dy, in_dz).  The LUT output is the sensor-side exit direction
(out_dx, out_dy, out_dz) and the optical path length (OPL).

Noodle schema matches camera_software/lens_manifold.py exactly:
  col 0,1  : u, v         aperture normalised hit position
  col 2,3  : fu, fv       scene field-angle factors (tan of angular deviation)
  col 4,5,6: in_dir       unit incident ray direction (scene → aperture)
  col 7,8,9: out_dir      unit exit ray direction (aperture → sensor)
  col 10   : opl          optical path length (metres)

Usage
-----
    from camera_designer import CameraPreset, BakeWorker

    preset = CameraPreset.load("my_lens.camera.json")
    worker = BakeWorker(preset, n_rays=65536, n_wavelengths=3)
    manifold = worker.bake()          # returns LensManifold
    manifold.save("my_lens.npz")
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .camera_preset import CameraPreset
from .parametric_surfaces import ParametricSurface, ApertureStop

__all__ = ["BakeWorker", "trace_ray", "trace_ray_backward"]

_INF = np.inf


# ─────────────────────────────────────────────────────────────────────────────
# Snell's law in vector form (64-bit)
# ─────────────────────────────────────────────────────────────────────────────

def _snell(rd: np.ndarray, n: np.ndarray, n1: float, n2: float) -> Optional[np.ndarray]:
    """Refract *rd* (unit, pointing into surface) through normal *n* (unit, facing ray).

    Returns the refracted direction or None on total internal reflection.
    ``n`` must face the incoming ray (dot(rd, n) < 0).
    """
    n = np.asarray(n, np.float64)
    rd = np.asarray(rd, np.float64)
    cos_i = -np.dot(rd, n)
    ratio = n1 / n2
    sin2_t = ratio**2 * (1.0 - cos_i**2)
    if sin2_t > 1.0:
        return None  # TIR
    cos_t = math.sqrt(max(0., 1.0 - sin2_t))
    return ratio * rd + (ratio * cos_i - cos_t) * n


# ─────────────────────────────────────────────────────────────────────────────
# Single-ray tracer
# ─────────────────────────────────────────────────────────────────────────────

def trace_ray(
    preset: CameraPreset,
    ro: np.ndarray,
    rd: np.ndarray,
    wavelength_um: float = 0.587,
) -> Optional[dict]:
    """Trace one ray through the full lens group.

    Parameters
    ----------
    preset        : fully specified CameraPreset
    ro            : ray origin in camera space (float64, metres)
    rd            : unit ray direction (float64, scene → lens)
    wavelength_um : wavelength in micrometres for dispersion

    Returns
    -------
    dict with keys:
      'aperture_uv'   : (2,) normalised aperture hit  [-1,1]
      'aperture_hit'  : (3,) world-space aperture point
      'in_dir'        : (3,) incident direction at aperture (= rd)
      'out_dir'       : (3,) exit direction at sensor
      'opl'           : float  optical path length metres
      'sensor_hit'    : (3,) sensor hit point
    or None if the ray misses, hits the mount, or undergoes TIR.
    """
    ro = np.asarray(ro, np.float64)
    rd = np.asarray(rd, np.float64)
    rd = rd / max(np.linalg.norm(rd), 1e-30)

    opl         = 0.0
    n_current   = 1.0       # index of medium the ray is currently in
    ap_hit      = None      # aperture plane hit point (set on first ap hit)
    in_dir_ap   = None

    # Check aperture stop first — it gates the manifold key
    ap_surf = preset.aperture_stop

    # ── Propagate through lens elements (front-to-back) ────────────────────
    elements = sorted(preset.lens_group.elements,
                      key=lambda e: e.z_vertex, reverse=True)  # front = largest z

    for el in elements:
        # Transform ray into surface-local frame (vertex at z=0 local)
        ro_local = ro.copy()
        ro_local[2] -= el.z_vertex
        t, hit_local, normal = el.surface.intersect(ro_local, rd)
        if not math.isfinite(t):
            return None  # missed this element — ray lost
        hit_world = hit_local.copy()
        hit_world[2] += el.z_vertex

        # Accumulate OPL: n * distance
        opl += n_current * t

        # Refract at the surface
        n_out = el.glass_out.n_at(wavelength_um)
        rd_new = _snell(rd, normal, n_current, n_out)
        if rd_new is None:
            return None  # TIR
        rd       = rd_new / max(np.linalg.norm(rd_new), 1e-30)
        n_current = n_out
        ro        = hit_world

        # Check if we just passed through the aperture plane
        if ap_hit is None:
            z_ap = ap_surf.z_pos
            if (ro[2] - z_ap) * (hit_world[2] - z_ap) <= 0:
                # Interpolate aperture crossing
                dz = rd[2]
                if abs(dz) > 1e-9:
                    t_ap = (z_ap - ro[2]) / dz
                    pt_ap = ro + t_ap * rd
                    r_ap  = math.sqrt(pt_ap[0]**2 + pt_ap[1]**2)
                    if (ap_surf.r_inner <= r_ap <= ap_surf.r_outer):
                        ap_hit   = pt_ap
                        in_dir_ap = rd.copy()

    # ── Hit the sensor ────────────────────────────────────────────────────
    ro_local = ro.copy()
    ro_local[2] -= preset.sensor.z_pos
    t_s, hit_s_local, _ = preset.sensor.intersect(ro_local, rd)
    if not math.isfinite(t_s):
        return None  # missed sensor
    hit_sensor = hit_s_local.copy()
    hit_sensor[2] += preset.sensor.z_pos
    opl += n_current * t_s

    # ── Aperture UV ────────────────────────────────────────────────────────
    if ap_hit is None:
        # Fallback: intersect the aperture plane directly with incoming ray
        if abs(rd[2]) > 1e-9:
            t_ap = (ap_surf.z_pos - ro[2]) / rd[2]
            ap_hit = ro + t_ap * rd
            in_dir_ap = rd.copy()
        else:
            return None

    r_max  = ap_surf.r_outer
    u = ap_hit[0] / max(r_max, 1e-12)
    v = ap_hit[1] / max(r_max, 1e-12)
    if abs(u) > 1.0 or abs(v) > 1.0:
        return None  # outside aperture

    # fu, fv — field-angle factors (tangent of angle in scene cone)
    safe_z = rd[2] if abs(rd[2]) > 1e-9 else 1e-9
    fu = rd[0] / safe_z
    fv = rd[1] / safe_z

    return {
        "aperture_uv":  np.array([u, v],         np.float64),
        "field_angle":  np.array([fu, fv],        np.float64),
        "aperture_hit": ap_hit,
        "in_dir":       in_dir_ap / max(np.linalg.norm(in_dir_ap), 1e-30),
        "out_dir":      rd / max(np.linalg.norm(rd), 1e-30),
        "opl":          opl,
        "sensor_hit":   hit_sensor,
    }


def trace_ray_backward(
    preset: CameraPreset,
    ro: np.ndarray,
    rd: np.ndarray,
    wavelength_um: float = 0.587,
) -> Optional[dict]:
    """Trace a ray from sensor side backward through the lens to scene.

    Applies Snell's law with swapped n1/n2 through elements in reverse order
    (sensor-side first), giving the time-reversed optical path.

    Parameters
    ----------
    ro : (3,) array — ray origin near the sensor plane (camera local space)
    rd : (3,) unit direction pointing toward the lens / scene (+z nominally)

    Returns
    -------
    dict with same keys as trace_ray, or None on miss or TIR.
      'aperture_hit' : (3,) world-space aperture crossing point
      'in_dir'       : (3,) ray direction AT the aperture crossing
      'out_dir'      : (3,) final scene-side direction after exiting the lens
      'aperture_uv'  : (2,) normalised aperture UV [-1, 1]
      'field_angle'  : (2,) (fu, fv) tangent-field factors
      'opl'          : float  optical path length (metres)
    """
    ro = np.asarray(ro, np.float64)
    rd = np.asarray(rd, np.float64)
    rd = rd / max(np.linalg.norm(rd), 1e-30)

    opl      = 0.0
    ap_surf  = preset.aperture_stop
    z_ap     = float(ap_surf.z_pos)
    ap_hit   = None
    ap_dir   = None

    # Build forward n-sequence then reverse it for backward traversal.
    elements_fwd = sorted(preset.lens_group.elements,
                          key=lambda e: e.z_vertex, reverse=True)
    n_seq = [1.0] + [el.glass_out.n_at(wavelength_um) for el in elements_fwd]
    N = len(elements_fwd)
    elements_bwd = list(reversed(elements_fwd))

    # Start in sensor-side air (n_seq[N] should be 1.0).
    n_current = n_seq[N]
    prev_ro   = ro.copy()

    for bwd_k, el in enumerate(elements_bwd):
        fwd_k  = N - 1 - bwd_k
        n_exit = n_seq[fwd_k]

        ro_local        = ro.copy()
        ro_local[2]    -= el.z_vertex
        t, hit_local, normal = el.surface.intersect(ro_local, rd)
        if not math.isfinite(t):
            return None

        hit_world        = hit_local.copy()
        hit_world[2]    += el.z_vertex

        # Detect aperture crossing on this ray segment (prev_ro → hit_world).
        if ap_hit is None:
            dz_seg = hit_world[2] - prev_ro[2]
            if abs(dz_seg) > 1e-12:
                s = (z_ap - prev_ro[2]) / dz_seg
                if 0.0 < s < 1.0:
                    ap_pt = prev_ro + s * (hit_world - prev_ro)
                    r_ap  = math.sqrt(ap_pt[0] ** 2 + ap_pt[1] ** 2)
                    if ap_surf.r_inner <= r_ap <= ap_surf.r_outer:
                        ap_hit = ap_pt
                        ap_dir = rd.copy()

        opl += n_current * t

        # Snell's law — ensure normal faces the incoming ray.
        if np.dot(rd, normal) > 0:
            normal = -normal
        rd_new = _snell(rd, normal, n_current, n_exit)
        if rd_new is None:
            return None
        rd        = rd_new / max(np.linalg.norm(rd_new), 1e-30)
        n_current = n_exit
        prev_ro   = hit_world
        ro        = hit_world

    # Fallback: extrapolate scene-side ray back to aperture plane.
    if ap_hit is None:
        if abs(rd[2]) > 1e-9:
            t_ap  = (z_ap - ro[2]) / rd[2]   # negative (aperture is behind scene pos)
            ap_pt = ro + t_ap * rd
            r_ap  = math.sqrt(ap_pt[0] ** 2 + ap_pt[1] ** 2)
            if r_ap <= ap_surf.r_outer:
                ap_hit = ap_pt
                ap_dir = rd.copy()
        if ap_hit is None:
            return None

    r_max = ap_surf.r_outer
    u = ap_hit[0] / max(r_max, 1e-12)
    v = ap_hit[1] / max(r_max, 1e-12)
    if abs(u) > 1.0 or abs(v) > 1.0:
        return None

    safe_z = rd[2] if abs(rd[2]) > 1e-9 else 1e-9
    return {
        "aperture_uv":  np.array([u, v],       np.float64),
        "field_angle":  np.array([rd[0] / safe_z, rd[1] / safe_z], np.float64),
        "aperture_hit": ap_hit,
        "in_dir":       ap_dir / max(np.linalg.norm(ap_dir), 1e-30),
        "out_dir":      rd    / max(np.linalg.norm(rd),      1e-30),
        "opl":          opl,
        "sensor_hit":   prev_ro,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BakeWorker
# ─────────────────────────────────────────────────────────────────────────────

class BakeWorker:
    """Bake a CameraPreset into a LensManifold noodle LUT.

    Parameters
    ----------
    preset        : CameraPreset defining the full optical system
    n_rays        : base number of rays to trace (before adaptive refinement)
    n_wavelengths : number of wavelength samples (uses preset.wavelengths list)
    n_refine      : adaptive refinement passes
    threshold     : out-direction variance threshold to trigger refinement
    seed          : RNG seed for reproducibility
    verbose       : print progress to stdout
    """

    def __init__(
        self,
        preset:        CameraPreset,
        n_rays:        int   = 65_536,
        n_wavelengths: int   = 3,
        n_refine:      int   = 2,
        threshold:     float = 1e-4,
        seed:          int   = 42,
        verbose:       bool  = True,
    ) -> None:
        self.preset        = preset
        self.n_rays        = n_rays
        self.n_wavelengths = n_wavelengths
        self.n_refine      = n_refine
        self.threshold     = threshold
        self.rng           = np.random.default_rng(seed)
        self.verbose       = verbose

    # ── Internal helpers ───────────────────────────────────────────────────

    def _sample_rays(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample n (ro, rd) pairs: origins on the aperture disk, directions
        spanning the scene-side FOV cone.

        Returns (ro_arr, rd_arr) each (n, 3) float64.
        """
        ap   = self.preset.aperture_stop
        fov  = math.radians(self.preset.fov_deg * 0.5)
        ftan = math.tan(fov)

        # Aperture disk: uniform Halton-like sampling
        r2  = self.rng.uniform(0., ap.r_outer**2, n)
        ang = self.rng.uniform(0., 2*math.pi, n)
        r   = np.sqrt(r2)
        ax  = r * np.cos(ang)
        ay  = r * np.sin(ang)
        z_front = max(el.z_vertex
                      for el in self.preset.lens_group.elements) + 0.001
        az  = np.full(n, z_front)  # start 1 mm in front of frontmost element

        ro_arr = np.stack([ax, ay, az], axis=1)  # (n, 3)

        # Scene directions: uniform over cone
        fu = self.rng.uniform(-ftan, ftan, n)
        fv = self.rng.uniform(-ftan, ftan, n)
        rd_raw = np.stack([fu, fv, -np.ones(n)], axis=1)  # pointing -Z (toward sensor)
        norms  = np.linalg.norm(rd_raw, axis=1, keepdims=True)
        rd_arr = rd_raw / np.maximum(norms, 1e-30)

        return ro_arr, rd_arr

    def _trace_batch(
        self,
        ro_arr: np.ndarray,
        rd_arr: np.ndarray,
    ) -> np.ndarray:
        """Trace a batch of rays; return noodle rows (M, 11) float64.

        Rows for missed/TIR rays are silently dropped.
        """
        wls = self.preset.wavelengths[:self.n_wavelengths]
        rows = []
        for i in range(len(ro_arr)):
            ro = ro_arr[i]
            rd = rd_arr[i]
            # Average over wavelengths for chromatic averaging
            results = []
            for wl in wls:
                r = trace_ray(self.preset, ro, rd, wl)
                if r is not None:
                    results.append(r)
            if not results:
                continue
            # Mean over wavelengths
            u    = float(np.mean([r["aperture_uv"][0]  for r in results]))
            v    = float(np.mean([r["aperture_uv"][1]  for r in results]))
            fu   = float(np.mean([r["field_angle"][0]  for r in results]))
            fv   = float(np.mean([r["field_angle"][1]  for r in results]))
            in_d = np.mean([r["in_dir"]  for r in results], axis=0)
            out_d= np.mean([r["out_dir"] for r in results], axis=0)
            opl  = float(np.mean([r["opl"] for r in results]))
            in_d  /= max(np.linalg.norm(in_d), 1e-30)
            out_d /= max(np.linalg.norm(out_d), 1e-30)
            rows.append([u, v, fu, fv,
                         in_d[0], in_d[1], in_d[2],
                         out_d[0], out_d[1], out_d[2],
                         opl])
        if not rows:
            return np.zeros((0, 11), np.float64)
        return np.array(rows, np.float64)

    # ── Public API ─────────────────────────────────────────────────────────

    def bake(self):
        """Trace rays and build a LensManifold.

        Returns
        -------
        LensManifold
        """
        try:
            from camera_software.lens_manifold import LensManifold
        except ImportError:
            raise ImportError(
                "camera_software.lens_manifold is required for baking. "
                "Ensure camera_software/ is on the Python path."
            )

        if self.verbose:
            print(f"[BakeWorker] baking {self.n_rays} rays for "
                  f"'{self.preset.name}'...", flush=True)

        ro_arr, rd_arr = self._sample_rays(self.n_rays)
        data = self._trace_batch(ro_arr, rd_arr)

        if self.verbose:
            print(f"[BakeWorker]   {len(data)} noodles traced", flush=True)

        # Adaptive refinement: split (u,v) regions with high out-dir variance
        for pass_i in range(self.n_refine):
            if len(data) == 0:
                break
            # Partition by quadrant in aperture UV
            quads = [
                (data[:,0] >= 0) & (data[:,1] >= 0),
                (data[:,0] <  0) & (data[:,1] >= 0),
                (data[:,0] >= 0) & (data[:,1] <  0),
                (data[:,0] <  0) & (data[:,1] <  0),
            ]
            extras = []
            for mask in quads:
                chunk = data[mask]
                if len(chunk) < 4:
                    continue
                var = float(np.mean(np.var(chunk[:, 7:10], axis=0)))
                if var > self.threshold:
                    n_extra = max(64, len(chunk) // 2)
                    ro2, rd2 = self._sample_rays(n_extra)
                    extra = self._trace_batch(ro2, rd2)
                    if len(extra):
                        extras.append(extra)
            if extras:
                data = np.concatenate([data, *extras], axis=0)
                if self.verbose:
                    print(f"[BakeWorker]   pass {pass_i+1}: "
                          f"{len(data)} noodles after refinement", flush=True)

        if len(data) == 0:
            raise RuntimeError(
                "BakeWorker: no rays traced successfully. "
                "Check the CameraPreset optical geometry."
            )

        # Build the manifold from the noodle array using from_pairs()
        manifold = LensManifold.from_pairs(
            uv       = data[:, :2],
            fufv     = data[:, 2:4],
            in_dirs  = data[:, 4:7],
            out_dirs = data[:, 7:10],
            opls     = data[:, 10],
            meta     = {
                "preset_name": self.preset.name,
                "focal_mm":    self.preset.focal_mm,
                "f_number":    self.preset.f_number,
                "fov_deg":     self.preset.fov_deg,
                "n_noodles":   len(data),
            },
        )

        if self.verbose:
            print(f"[BakeWorker] done. {len(data)} noodles in manifold.", flush=True)

        return manifold

    def bake_glsl_source(self) -> str:
        """Emit a GLSL compute shader that traces one ray per invocation.

        This is the GPU bake path for when the number of rays is very large
        (> 1M).  Returns a complete GLSL 460 compute shader source string.
        The shader reads ray (ro, rd) from binding 0 SSBO and writes noodle
        rows to binding 1 SSBO.

        Surface intercept functions are inlined for every element in the
        lens group, indexed by element order.
        """
        lines = [
            "#version 460 core",
            "layout(local_size_x = 64) in;",
            "",
            "// Input rays (ro.xyz + rd.xyz per ray, float64 emulated as two float32 pairs)",
            "layout(std430, binding=0) readonly buffer RayBuf { float rays[]; };",
            "// Output noodle rows: 11 float64 per noodle → 22 float32",
            "layout(std430, binding=1) writeonly buffer NoodleBuf { float noodles[]; };",
            "layout(std430, binding=2) writeonly buffer HitCountBuf { uint hit_count; };",
            "",
        ]

        # Inline all surface intercept functions
        for idx, el in enumerate(self.preset.lens_group.elements):
            fn_name = f"surf_{idx}"
            lines.append(f"// Element {idx}: {el.label}")
            lines.append(el.surface.glsl_intercept_fn(fn_name))
            lines.append("")

        # Inline aperture stop
        lines.append("// Aperture stop")
        lines.append(self.preset.aperture_stop.glsl_intercept_fn("surf_aperture"))
        lines.append("")

        # Inline sensor surface
        lines.append("// Sensor surface")
        lines.append(self.preset.sensor.glsl_intercept_fn("surf_sensor"))
        lines.append("")

        # Snell's law helper
        lines += [
            "vec3 snell(vec3 rd, vec3 n, float n1, float n2) {",
            "    float cos_i = -dot(rd, n);",
            "    float ratio = n1 / n2;",
            "    float sin2t = ratio*ratio*(1.0 - cos_i*cos_i);",
            "    if (sin2t > 1.0) return vec3(0.0); // TIR sentinel",
            "    float cos_t = sqrt(max(0.0, 1.0 - sin2t));",
            "    return normalize(ratio*rd + (ratio*cos_i - cos_t)*n);",
            "}",
            "",
        ]

        # Main trace function
        n_els = len(self.preset.lens_group.elements)
        wl_list = ", ".join(f"{w}" for w in self.preset.wavelengths[:self.n_wavelengths])

        lines += [
            "void main() {",
            "    uint gid = gl_GlobalInvocationID.x;",
            "    vec3 ro = vec3(rays[gid*6+0], rays[gid*6+1], rays[gid*6+2]);",
            "    vec3 rd = normalize(vec3(rays[gid*6+3], rays[gid*6+4], rays[gid*6+5]));",
            "",
            f"    float[{self.n_wavelengths}] wls = float[]({wl_list});",
            "    vec3 out_dir_sum = vec3(0.0);",
            "    float opl_sum = 0.0;",
            "    int   hit_count_wl = 0;",
            "    vec3  ap_hit = vec3(0.0);",
            "    vec3  in_dir = rd;",
            "",
            f"    for (int wi = 0; wi < {self.n_wavelengths}; wi++) {{",
            "        vec3  r  = ro; vec3 d = rd;",
            "        float n1 = 1.0; float opl = 0.0;",
            "        float t; vec3 normal;",
            "        bool ok = true;",
        ]

        # Unroll element loop
        for idx, el in enumerate(self.preset.lens_group.elements):
            n_out = el.glass_out.n_d  # simplified: no Sellmeier in GLSL path
            lines += [
                f"        if (ok) {{",
                f"            vec3 r_local = r; r_local.z -= {el.z_vertex};",
                f"            if (!surf_{idx}(r_local, d, t, normal)) {{ ok = false; }}",
                f"            else {{",
                f"                opl += n1 * t;",
                f"                vec3 nd = snell(d, normal, n1, {n_out});",
                f"                if (length(nd) < 0.5) {{ ok = false; }}",
                f"                else {{ d = normalize(nd); n1 = {n_out}; r = r_local + t*d; r.z += {el.z_vertex}; }}",
                f"            }}",
                f"        }}",
            ]

        lines += [
            "        if (ok) {",
            f"            vec3 r_s = r; r_s.z -= {self.preset.sensor.z_pos};",
            "            float t_s; vec3 ns;",
            "            if (surf_sensor(r_s, d, t_s, ns)) {",
            "                out_dir_sum += d; opl_sum += opl + n1*t_s;",
            "                hit_count_wl++;",
            "            }",
            "        }",
            "    }",
            "",
            "    if (hit_count_wl == 0) return;",
            "",
            "    // Aperture UV",
            f"    float r_max = {self.preset.aperture_stop.r_outer};",
            f"    float z_ap  = {self.preset.aperture_stop.z_pos};",
            "    float t_ap = (z_ap - ro.z) / max(abs(rd.z), 1e-9);",
            "    vec3  ap   = ro + t_ap * rd;",
            "    float u    = ap.x / max(r_max, 1e-9);",
            "    float v    = ap.y / max(r_max, 1e-9);",
            "    if (abs(u) > 1.0 || abs(v) > 1.0) return;",
            "",
            "    vec3  out_dir = normalize(out_dir_sum / float(hit_count_wl));",
            "    float opl_avg = opl_sum / float(hit_count_wl);",
            "    float fu  = rd.x / max(abs(rd.z), 1e-9);",
            "    float fv  = rd.y / max(abs(rd.z), 1e-9);",
            "",
            "    uint slot = atomicAdd(hit_count, 1u);",
            "    uint base = slot * 11u;",
            "    noodles[base+0]  = u;",
            "    noodles[base+1]  = v;",
            "    noodles[base+2]  = fu;",
            "    noodles[base+3]  = fv;",
            "    noodles[base+4]  = in_dir.x;",
            "    noodles[base+5]  = in_dir.y;",
            "    noodles[base+6]  = in_dir.z;",
            "    noodles[base+7]  = out_dir.x;",
            "    noodles[base+8]  = out_dir.y;",
            "    noodles[base+9]  = out_dir.z;",
            "    noodles[base+10] = opl_avg;",
            "}",
        ]

        return "\n".join(lines)
