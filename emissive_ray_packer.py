"""Distribute pre-baked ray sources randomly across emissive triangle areas.

Unified scene API
-----------------
The packer consumes the SAME inputs that ``BaseGLRenderer`` and
``BaseRasterizer`` already read every frame:

    * ``verts8``         — (3*Ntri, 8) float32 vertex stream (pos.xyz, n.xyz, uv.xy)
    * ``groups``         — the 6-tuple (gids, mids, offsets, counts, mvs, dirty)
                           returned by the harness's ``scene_for_phase`` and
                           passed verbatim to ``derive_emissive_area_lights``.
    * ``material_db``    — ``material_db.MaterialDatabase`` instance (its
                           ``build_tensors()['pbr']`` row [8:11] is the
                           authoritative emission RGB; ``emit_profile_idx`` in
                           the dict/Material refines spectral sampling).

There is **no separate "ray-tracer scene"** struct. Whatever the rasterizer
sees, the packer sees — full data, accessed through the same tensor build
the shaders already use.

Output
------
A ``(N_rays, 12) float32`` array matching
``camera_designer.ray_order.RayOrder.bake_rays`` row layout exactly:

    [0:3]  pos_xyz       — uniform barycentric point on emissive triangle
    [3]    amp_weight    — per-source amplitude weight (1.0)
    [4:7]  dir_xyz       — cosine-weighted hemisphere about triangle normal
    [7]    0.0           — reserved (dir_kind.w)
    [8]    freq_hz       — c / λ; λ from EmissionProfile if registered, else
                           luminance-weighted RGB channel center
    [9]    phase_rad     — uniform [0, 2π)
    [10]   energy        — per-ray energy share so Σ energy per triangle
                           equals (area × emission_luminance × power_scale)
    [11]   0.0           — reserved (packet.w)

Energy normalization
--------------------
``Σ row.energy`` over all rays equals the sum over emissive triangles of
``area_i × luma(emission_rgb_i) × power_scale``.  For materials with a
registered ``EmissionProfile`` we additionally multiply by
``profile.total_power_W * profile.emissivity`` so blackbody/parametric
emitters carry their natural-unit power into the ray budget.
"""

from __future__ import annotations

import math
from typing import Any, Sequence, Tuple

import numpy as np

# Channel centers (nm) — matches camera_designer's RGB-to-wavelength convention.
_R_NM = 620.0
_G_NM = 540.0
_B_NM = 460.0
_C_UM_S = 2.998e14   # speed of light in µm / s — same constant ray_order uses
_TWO_PI = 2.0 * math.pi
# Rec. 709 luma weights — matches the calibration script's pearl luma metric.
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _emission_profile_for(material_db, mat_id: int):
    """Return (EmissionProfile or None) for a material via the same DB lookup
    the shaders use — never a hand-rolled simpler struct."""
    try:
        from material_db import EmissionProfileDatabase
    except Exception:
        return None
    name = None
    try:
        name = material_db._order[mat_id]   # type: ignore[attr-defined]
    except Exception:
        return None
    mat = material_db._materials.get(name)  # type: ignore[attr-defined]
    if mat is None:
        return None
    raw = None
    if isinstance(mat, dict):
        raw = mat.get("emit_profile_idx", mat.get("emit_profile_name", None))
    else:
        raw = getattr(mat, "emit_profile_idx",
                      getattr(mat, "emit_profile_name", None))
    ep_db = EmissionProfileDatabase.instance()
    try:
        if isinstance(raw, str):
            idx = ep_db.index_of(raw)
        elif raw is None:
            return None
        else:
            idx = int(raw)
        if idx < 0:
            return None
        return ep_db.get(idx)
    except Exception:
        return None


def _sample_wavelength_nm(profile, rng: np.random.Generator,
                          rgb_pmf: np.ndarray) -> float:
    """Sample one wavelength (nm).  Profile (peak, fwhm) wins when present;
    otherwise pick R/G/B channel center weighted by the material's emission RGB."""
    if profile is not None:
        peak = float(getattr(profile, "peak_wavelength_nm", 550.0))
        fwhm = float(getattr(profile, "fwhm_nm", 30.0))
        sigma = max(1.0e-3, fwhm / 2.3548200450309493)   # 2*sqrt(2 ln 2)
        wl = peak + float(rng.standard_normal()) * sigma
        return max(80.0, min(3000.0, wl))
    # 3-channel discrete fallback.
    k = int(rng.choice(3, p=rgb_pmf))
    return (_R_NM, _G_NM, _B_NM)[k]


def _build_emissive_tri_table(verts8: np.ndarray,
                              groups: Sequence,
                              pbr: np.ndarray,
                              material_db,
                              min_emitter_group_id: int) -> Tuple[np.ndarray, ...]:
    """Walk groups exactly like ``BaseGLRenderer.derive_emissive_area_lights``,
    but keep PER-TRIANGLE rows instead of collapsing to one centroid per group.

    Returns
    -------
    pts0, edge1, edge2, normal_unit  : (M, 3) float32
    area, weight, power_scale_W      : (M,) float32     (weight = area*luma*power_scale)
    mat_id                           : (M,) int32       (per-triangle material id)
    """
    gids, mids, offs, cnts = (np.asarray(groups[i]).reshape(-1)
                              for i in range(4))
    pts0_l, e1_l, e2_l, n_l = [], [], [], []
    area_l, w_l, pwr_l, mid_l = [], [], [], []

    n_groups = min(len(gids), len(mids), len(offs), len(cnts))
    for i in range(n_groups):
        gid = int(gids[i])
        if gid < min_emitter_group_id:
            continue
        mat_id = int(mids[i])
        if mat_id < 0 or mat_id >= pbr.shape[0]:
            continue
        emis = np.asarray(pbr[mat_id, 8:11], np.float32)
        luma = float(_LUMA @ emis)
        if luma <= 1.0e-8:
            continue

        # Optional spectral profile gives natural-unit power.
        prof = _emission_profile_for(material_db, mat_id)
        if prof is not None:
            power_scale = max(0.0, float(getattr(prof, "total_power_W", 1.0)) *
                                   float(getattr(prof, "emissivity", 1.0)))
            if power_scale <= 0.0:
                power_scale = 1.0
        else:
            power_scale = 1.0

        s = int(offs[i]) * 3
        e = s + int(cnts[i]) * 3
        if s < 0 or e > verts8.shape[0]:
            continue
        tri = verts8[s:e, 0:3].reshape(-1, 3, 3)
        if tri.shape[0] == 0:
            continue
        e1 = (tri[:, 1] - tri[:, 0]).astype(np.float32)
        e2 = (tri[:, 2] - tri[:, 0]).astype(np.float32)
        cr = np.cross(e1, e2)
        nrm = np.linalg.norm(cr, axis=1)
        keep = nrm > 1.0e-12
        if not np.any(keep):
            continue
        e1, e2, cr, nrm = e1[keep], e2[keep], cr[keep], nrm[keep]
        n_unit = (cr / nrm[:, None]).astype(np.float32)
        area = (0.5 * nrm).astype(np.float32)

        pts0_l.append(tri[keep, 0].astype(np.float32))
        e1_l.append(e1)
        e2_l.append(e2)
        n_l.append(n_unit)
        area_l.append(area)
        w_l.append(area * luma * power_scale)
        pwr_l.append(np.full(area.shape[0], power_scale, np.float32))
        mid_l.append(np.full(area.shape[0], mat_id, np.int32))

    if not area_l:
        z3 = np.zeros((0, 3), np.float32)
        z1 = np.zeros((0,), np.float32)
        z1i = np.zeros((0,), np.int32)
        return z3, z3, z3, z3, z1, z1, z1, z1i

    return (
        np.concatenate(pts0_l, axis=0),
        np.concatenate(e1_l,   axis=0),
        np.concatenate(e2_l,   axis=0),
        np.concatenate(n_l,    axis=0),
        np.concatenate(area_l, axis=0),
        np.concatenate(w_l,    axis=0),
        np.concatenate(pwr_l,  axis=0),
        np.concatenate(mid_l,  axis=0),
    )


def pack_emissive_area_rays(
    verts8: np.ndarray,
    groups: Sequence,
    material_db: Any,
    *,
    n_rays: int,
    min_emitter_group_id: int = 10,
    seed: int = 0,
    total_energy_J: float | None = None,
) -> np.ndarray:
    """Distribute ``n_rays`` random ray sources across emissive triangle areas.

    Returns ``(N, 12) float32`` in the ``RayOrder.bake_rays`` schema.  The
    output dtype follows the input ``verts8`` dtype precision class — both
    float32 and float64 inputs are preserved.

    Parameters
    ----------
    verts8 : (3*Ntri, 8) float32 / float64
        Same vertex stream uploaded to the GL VAO and the C rasterizer.
    groups : 6-tuple
        (gids, mids, offsets, counts, mvs, dirty) as produced by the
        scene builder and passed to ``derive_emissive_area_lights``.
    material_db : MaterialDatabase
        Same instance used to bake PBR/spectral tensors for the shaders.
    n_rays : int
        Total number of rays to emit (>= 1).
    min_emitter_group_id : int
        Group ids strictly below this value are receivers, not emitters.
    seed : int
        RNG seed for reproducibility.
    total_energy_J : float | None
        When supplied, rescale ``rows[:,10]`` (per-ray energy) so that
        ``Σ energy == total_energy_J`` — the camera exposure budget from
        ``camera_exposure_budget.plan_ray_budget``.  Per-tri area·luma
        weighting is preserved; only the global scale changes.
    """
    if n_rays <= 0:
        return np.zeros((0, 12), np.float32)

    in_dtype = np.asarray(verts8).dtype
    out_dtype = np.float32 if in_dtype == np.float32 else np.float64

    pbr = np.ascontiguousarray(
        material_db.build_tensors().get("pbr",
                                        np.zeros((0, 16), np.float32)),
        np.float32,
    )
    pts0, e1, e2, n_unit, area, weight, pwr_scale, _mat_id = (
        _build_emissive_tri_table(np.asarray(verts8), groups, pbr,
                                  material_db, int(min_emitter_group_id))
    )
    M = area.shape[0]
    if M == 0:
        return np.zeros((0, 12), out_dtype)

    rng = np.random.default_rng(int(seed))

    # ── Multinomial: rays per triangle ∝ area × luma × power_scale ────────
    w_sum = float(weight.sum())
    if w_sum <= 0.0:
        return np.zeros((0, 12), out_dtype)
    pmf = weight / w_sum
    counts = rng.multinomial(int(n_rays), pmf).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros((0, 12), out_dtype)

    # Per-triangle energy share = total weight bucket / count_for_tri,
    # so Σ energies on triangle i  ==  area_i * luma_i * power_scale_i.
    safe_counts = np.where(counts > 0, counts, 1)
    energy_per_ray_per_tri = (weight / safe_counts.astype(np.float32))

    # ── Expand per-triangle data to per-ray rows ──────────────────────────
    tri_idx = np.repeat(np.arange(M, dtype=np.int64), counts)
    n_tot = tri_idx.shape[0]

    # Uniform barycentric (u,v) on triangle: fold the unit square.
    u = rng.random(n_tot, dtype=np.float64)
    v = rng.random(n_tot, dtype=np.float64)
    fold = (u + v) > 1.0
    u[fold] = 1.0 - u[fold]
    v[fold] = 1.0 - v[fold]
    u32 = u.astype(np.float32)[:, None]
    v32 = v.astype(np.float32)[:, None]
    pos = pts0[tri_idx] + e1[tri_idx] * u32 + e2[tri_idx] * v32

    # Cosine-weighted hemisphere about the triangle normal — Lambertian area
    # source. tx, ty form an orthonormal basis around n.
    n = n_unit[tri_idx]
    ref = np.where(np.abs(n[:, 2:3]) > 0.9,
                   np.array([[1.0, 0.0, 0.0]], np.float32),
                   np.array([[0.0, 0.0, 1.0]], np.float32))
    tx = np.cross(ref, n).astype(np.float32)
    tx /= (np.linalg.norm(tx, axis=1, keepdims=True) + 1.0e-30)
    ty = np.cross(n, tx).astype(np.float32)

    r1 = rng.random(n_tot, dtype=np.float64)
    r2 = rng.random(n_tot, dtype=np.float64)
    cos_t = np.sqrt(r1).astype(np.float32)
    sin_t = np.sqrt(np.maximum(0.0, 1.0 - r1)).astype(np.float32)
    phi = (_TWO_PI * r2).astype(np.float32)
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    dirs = (tx * (sin_t * cphi)[:, None]
            + ty * (sin_t * sphi)[:, None]
            + n  * cos_t[:, None])
    dnrm = np.linalg.norm(dirs, axis=1, keepdims=True)
    dirs = dirs / np.where(dnrm > 1.0e-12, dnrm, 1.0)

    # ── Spectral sampling per ray (uses material's own EmissionProfile) ──
    # Build per-tri RGB pmf once, then expand by tri_idx.
    emis_per_tri = pbr[_mat_id, 8:11].astype(np.float32)
    emis_per_tri = np.maximum(emis_per_tri, 0.0)
    rgb_lum = emis_per_tri * _LUMA[None, :]
    rgb_sum = rgb_lum.sum(axis=1, keepdims=True)
    rgb_pmf_tri = np.where(rgb_sum > 0,
                           rgb_lum / np.where(rgb_sum > 0, rgb_sum, 1.0),
                           np.full_like(rgb_lum, 1.0 / 3.0))

    # Cache one EmissionProfile lookup per unique material id.
    unique_mats = np.unique(_mat_id)
    prof_by_mat = {int(m): _emission_profile_for(material_db, int(m))
                   for m in unique_mats}

    freq_hz = np.empty(n_tot, np.float32)
    # Vectorize per material to keep the loop cheap.
    for mid in unique_mats:
        mid_int = int(mid)
        sel_tri = (_mat_id == mid)
        if not np.any(sel_tri):
            continue
        sel_ray = sel_tri[tri_idx]
        if not np.any(sel_ray):
            continue
        prof = prof_by_mat[mid_int]
        if prof is not None:
            peak = float(getattr(prof, "peak_wavelength_nm", 550.0))
            fwhm = float(getattr(prof, "fwhm_nm", 30.0))
            sigma = max(1.0e-3, fwhm / 2.3548200450309493)
            n_sel = int(sel_ray.sum())
            wl_nm = peak + rng.standard_normal(n_sel).astype(np.float32) * sigma
            wl_nm = np.clip(wl_nm, 80.0, 3000.0)
        else:
            # 3-channel discrete sample.  Use the per-triangle pmf for the
            # triangles inside this material (they all share the same row, but
            # this stays correct if a future material edits per-tri RGB).
            ray_tri = tri_idx[sel_ray]
            pmf_ray = rgb_pmf_tri[ray_tri]
            cdf = np.cumsum(pmf_ray, axis=1)
            u_pick = rng.random(pmf_ray.shape[0], dtype=np.float64).astype(np.float32)
            ch = np.argmax(cdf >= u_pick[:, None], axis=1)
            wl_nm = np.where(ch == 0, _R_NM,
                    np.where(ch == 1, _G_NM, _B_NM)).astype(np.float32)
        wl_um = wl_nm * 1.0e-3
        freq_hz[sel_ray] = (_C_UM_S / np.maximum(wl_um, 1.0e-6)).astype(np.float32)

    phase = (rng.random(n_tot, dtype=np.float64) * _TWO_PI).astype(np.float32)
    energy = energy_per_ray_per_tri[tri_idx]
    amp_weight = np.ones(n_tot, np.float32)

    rows = np.empty((n_tot, 12), out_dtype)
    rows[:, 0:3]  = pos
    rows[:, 3]    = amp_weight
    rows[:, 4:7]  = dirs
    rows[:, 7]    = 0.0
    rows[:, 8]    = freq_hz
    rows[:, 9]    = phase
    rows[:, 10]   = energy
    rows[:, 11]   = 0.0

    if total_energy_J is not None and float(total_energy_J) > 0.0:
        cur_total = float(rows[:, 10].sum())
        if cur_total > 0.0:
            rows[:, 10] *= np.asarray(
                float(total_energy_J) / cur_total, out_dtype
            )
    return rows


def summarize_packed_rays(rays: np.ndarray,
                          groups: Sequence | None = None) -> dict:
    """Return calibration stats for a baked ray batch.

    Useful in test harnesses to assert energy conservation, hemisphere
    coverage, and spectral distribution sanity.
    """
    if rays.shape[0] == 0:
        return {"n_rays": 0}
    dirs = rays[:, 4:7]
    energy = rays[:, 10]
    freq = rays[:, 8]
    wl_nm = (_C_UM_S / np.maximum(freq, 1.0)) * 1.0e3
    return {
        "n_rays":            int(rays.shape[0]),
        "energy_total":      float(energy.sum()),
        "energy_mean":       float(energy.mean()),
        "energy_min":        float(energy.min()),
        "energy_max":        float(energy.max()),
        "dir_norm_mean":     float(np.linalg.norm(dirs, axis=1).mean()),
        "dir_z_mean":        float(dirs[:, 2].mean()),
        "wavelength_mean_nm": float(wl_nm.mean()),
        "wavelength_p05_nm": float(np.percentile(wl_nm, 5.0)),
        "wavelength_p95_nm": float(np.percentile(wl_nm, 95.0)),
    }
