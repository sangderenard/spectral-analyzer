"""camera_software/lens_manifold.py
-----------------------------------
Noodle-manifold lens model.

A ``LensManifold`` bakes an arbitrary number of (aperture-position, scene-ray)
→ output-ray vector triples through a compiled ``LensTransform``, organises
them in a KD-tree indexed by aperture (u, v) and scene (fu, fv) coordinates,
and exposes them for nearly-instant broadcasted tensor indexing::

    dirs = manifold.data[idxs, _C_OUT]   # (M, k, 3) — one numpy line

Baking is *adaptive*: an internal quadtree measures per-cell output-direction
variance and oversamples high-aberration regions, keeping total disk footprint
modest while preserving fidelity where the optics are complex.

Terminology
-----------
noodle   — one baked ray record: a single float64 row in ``_data``
manifold — the full collection of noodles, spatially indexed

Noodle schema (float64, 11 columns)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 col  0, 1   : u, v          — normalised aperture position in [-1, 1]
 col  2, 3   : fu, fv        — scene field-angle factors (pixel × fov_tan)
 col  4, 5, 6: in_dx/dy/dz  — unit input ray direction  (scene → lens)
 col  7, 8, 9: out_dx/dy/dz — unit output ray direction (lens → sensor)
 col 10      : opl           — optical path length (metres) from aperture
                               to the focal point / retina, accumulated as
                               Σ n_i · d_i across all refracting segments.
                               For the thin-lens analytic bake this equals
                               |aperture_pos → focus_pt| (n_air = 1).
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from .base import LensTransform

__all__ = ["LensManifold", "SphereCoords", "CylinderCoords", "PolarChain",
           "LensAlignmentBuffer"]

# ---------------------------------------------------------------------------
# Column layout
# ---------------------------------------------------------------------------
_C_U   = 0
_C_V   = 1
_C_FU  = 2
_C_FV  = 3
_C_IN  = slice(4, 7)
_C_OUT = slice(7, 10)
_C_OPL = 10       # optical path length (metres)
_NCOLS = 11


# ---------------------------------------------------------------------------
# Internal adaptive quadtree over (u, v) aperture space
# ---------------------------------------------------------------------------

class _QuadNode:
    """One node of the 2-D aperture quadtree.

    Leaf nodes own ``indices`` (row indices into the manifold ``_data`` array).
    Branch nodes own two ``children``; they hold no indices of their own.
    """
    __slots__ = ("u0", "u1", "v0", "v1", "children", "indices")

    def __init__(self, u0: float, u1: float, v0: float, v1: float) -> None:
        self.u0 = u0;  self.u1 = u1
        self.v0 = v0;  self.v1 = v1
        self.children: Optional[list[_QuadNode]] = None
        self.indices:  Optional[np.ndarray]      = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None

    def _contains(self, u: float, v: float) -> bool:
        return self.u0 <= u <= self.u1 and self.v0 <= v <= self.v1

    def route(self, u: float, v: float) -> "_QuadNode":
        """Descend to the leaf responsible for aperture point (u, v)."""
        node = self
        while not node.is_leaf:
            node = (node.children[0]
                    if node.children[0]._contains(u, v)
                    else node.children[1])
        return node

    def split(self, data: np.ndarray, threshold: float,
              max_depth: int, depth: int = 0) -> None:
        """Recursively split leaves whose out-direction variance exceeds
        *threshold* (mean variance across the xyz components)."""
        if self.indices is None or len(self.indices) < 4:
            return
        local = data[self.indices, _C_OUT]          # (N, 3)
        var = float(np.mean(np.var(local, axis=0)))
        if var <= threshold or depth >= max_depth:
            return
        # Split along the longer axis
        if (self.u1 - self.u0) >= (self.v1 - self.v0):
            mid  = (self.u0 + self.u1) * 0.5
            cA   = _QuadNode(self.u0, mid,     self.v0, self.v1)
            cB   = _QuadNode(mid,     self.u1, self.v0, self.v1)
            mask = data[self.indices, _C_U] <= mid
        else:
            mid  = (self.v0 + self.v1) * 0.5
            cA   = _QuadNode(self.u0, self.u1, self.v0, mid)
            cB   = _QuadNode(self.u0, self.u1, mid, self.v1)
            mask = data[self.indices, _C_V] <= mid
        cA.indices  = self.indices[ mask]
        cB.indices  = self.indices[~mask]
        self.children = [cA, cB]
        self.indices  = None           # branch nodes carry no direct data
        cA.split(data, threshold, max_depth, depth + 1)
        cB.split(data, threshold, max_depth, depth + 1)


def _collect_leaves(node: _QuadNode, out: list) -> None:
    if node.is_leaf:
        out.append(node)
    else:
        for child in node.children:
            _collect_leaves(child, out)


# ---------------------------------------------------------------------------
# Main manifold class
# ---------------------------------------------------------------------------

class LensManifold:
    """Baked noodle-manifold: aperture + scene-ray → output-ray lookup.

    Workflow::

        from camera_software import LensTransform, LensManifold

        lt = LensTransform()
        lt.lens_tilt[0]  = 0.08   # Scheimpflug nod (radians)
        lt.gimbal[0]     = 0.05   # pan (radians)
        lt.extension_mm  = 12.0

        mf = LensManifold.bake_from_transform(
            lt,
            focal_mm        = 85.0,
            focus_m         = 3.0,
            sensor_h_mm     = 24.0,
            aperture_radius = 0.015,   # metres
            fov_tan         = 0.28,    # half-width of sensor in focal-length units
            n_base          = 65536,
            n_refine        = 4,
            threshold       = 1e-5,
        )

        mf.save("lens_85mm.npz")

    Fast query (broadcasted tensor indexing)::

        uv    = rng.uniform(-1, 1, (N, 2))
        dirs, w = mf.query(uv, k=16)   # (N, 16, 3)  float64
        blended  = mf.interpolate(uv)  # (N, 3)       float64  IDW-blended
    """

    def __init__(self) -> None:
        self._data:  Optional[np.ndarray] = None   # (N, 10) float64
        self._tree                        = None   # scipy.spatial.KDTree over (u,v,fu,fv)
        self._root:  Optional[_QuadNode]  = None   # aperture quadtree
        self.meta:   dict                 = {}

    # ------------------------------------------------------------------ bake

    @classmethod
    def bake_from_transform(
        cls,
        lens_transform: "LensTransform",
        *,
        focal_mm:        float = 50.0,
        focus_m:         float = 2.0,
        sensor_h_mm:     float = 24.0,
        aperture_radius: float = 0.015,
        fov_tan:         float = 0.24,
        n_base:          int   = 16_384,
        n_refine:        int   = 4,
        threshold:       float = 1e-5,
        seed:            int   = 0,
    ) -> "LensManifold":
        """Analytically bake *n_base* noodles through *lens_transform*, then
        adaptively refine high-variance aperture regions for up to *n_refine*
        passes.

        All arithmetic is float64; no downcast is ever applied.

        Parameters
        ----------
        fov_tan         : tan(half_fov) — width of the scene sampling cone.
                          Matches the ``uFovTan`` value sent to the GPU shader.
        n_base          : initial noodle count.
        n_refine        : adaptive refinement passes.
        threshold       : per-cell mean-variance threshold for quadtree splits.
        """
        rng = np.random.default_rng(seed)
        mf  = cls()
        mf.meta = {
            "focal_mm":        focal_mm,
            "focus_m":         focus_m,
            "sensor_h_mm":     sensor_h_mm,
            "aperture_radius": aperture_radius,
            "fov_tan":         fov_tan,
            "lens_tilt":       list(np.asarray(lens_transform.lens_tilt,   np.float64)),
            "extension_mm":    float(lens_transform.extension_mm),
            "gimbal":          list(np.asarray(lens_transform.gimbal,      np.float64)),
            "shift":           list(np.asarray(lens_transform.shift,       np.float64)),
        }

        # Compile the transform (canonical sensor-space basis)
        R0 = np.array([1., 0., 0.], np.float64)
        U0 = np.array([0., 1., 0.], np.float64)
        F0 = np.array([0., 0., 1.], np.float64)
        payload = lens_transform.compile(focal_mm, focus_m, sensor_h_mm,
                                         R0, U0, F0)
        R   = np.asarray(payload["right"],          np.float64)
        U   = np.asarray(payload["up"],             np.float64)
        F   = np.asarray(payload["fwd"],            np.float64)
        ts  = np.asarray(payload["tilt_shift"],     np.float64)
        lt  = np.asarray(payload["lens_tilt"],      np.float64)  # (nod, pan)
        eff_focal_m = float(payload["effective_focal_mm"]) * 1e-3

        # Pack bake context into a dict to keep helper signatures clean
        ctx = dict(R=R, U=U, F=F, ts=ts, lt=lt,
                   eff_focal_m=eff_focal_m, focus_m=focus_m,
                   fov_tan=fov_tan, ap_r=aperture_radius)

        # ---- base pass: uniform aperture disk × scene cone sampling --------
        data = _bake_region(-1., 1., -1., 1., n_base, ctx, rng)

        # ---- build aperture quadtree ----------------------------------------
        root         = _QuadNode(-1., 1., -1., 1.)
        root.indices = np.arange(len(data))

        # ---- adaptive refinement passes ------------------------------------
        for _ in range(n_refine):
            leaves: list[_QuadNode] = []
            _collect_leaves(root, leaves)
            extras: list[np.ndarray] = []
            for leaf in leaves:
                if leaf.indices is None or len(leaf.indices) < 4:
                    continue
                var = float(np.mean(np.var(data[leaf.indices, _C_OUT], axis=0)))
                if var > threshold:
                    n_extra = max(32, len(leaf.indices))
                    chunk   = _bake_region(leaf.u0, leaf.u1,
                                           leaf.v0, leaf.v1,
                                           n_extra, ctx, rng)
                    if chunk is not None:
                        extras.append(chunk)
            if extras:
                data         = np.concatenate([data, *extras], axis=0)
                root.indices = np.arange(len(data))
                root.children = None          # rebuild tree next pass
            root.split(data, threshold, max_depth=14)

        mf._data = data
        mf._root = root
        mf._tree = _build_tree(data)
        return mf

    # ------------------------------------------------------------------ ingest external pairs

    @classmethod
    def from_pairs(
        cls,
        uv:       np.ndarray,
        fufv:     np.ndarray,
        in_dirs:  np.ndarray,
        out_dirs: np.ndarray,
        meta:     Optional[dict] = None,
        opls:     Optional[np.ndarray] = None,
    ) -> "LensManifold":
        """Build a manifold from externally supplied vector pairs.

        Use this to ingest pairs generated by the GPU ray tracer (run the
        sensor compute pass with ``uCapture=1`` and read back the output).
        Use ``opls`` to supply optical path lengths computed by :class:`PolarChain`
        via its :attr:`chain_opl` property after tracing.

        Parameters
        ----------
        uv       : (N, 2) float64 — normalised aperture positions
        fufv     : (N, 2) float64 — scene field-angle factors
        in_dirs  : (N, 3) float64 — input ray unit directions
        out_dirs : (N, 3) float64 — output ray unit directions
        opls     : (N,)   float64 — optical path lengths (metres); zeros if None
        """
        uv       = np.asarray(uv,       np.float64)
        fufv     = np.asarray(fufv,     np.float64)
        in_dirs  = np.asarray(in_dirs,  np.float64)
        out_dirs = np.asarray(out_dirs, np.float64)
        N = len(uv)
        data = np.empty((N, _NCOLS), np.float64)
        data[:, _C_U]   = uv[:, 0]
        data[:, _C_V]   = uv[:, 1]
        data[:, _C_FU]  = fufv[:, 0]
        data[:, _C_FV]  = fufv[:, 1]
        data[:, _C_IN]  = in_dirs
        data[:, _C_OUT] = out_dirs
        data[:, _C_OPL] = (np.asarray(opls, np.float64).ravel()
                           if opls is not None
                           else np.zeros(N, np.float64))
        mf        = cls()
        mf._data  = data
        mf.meta   = meta or {}
        mf._root  = _QuadNode(-1., 1., -1., 1.)
        mf._root.indices = np.arange(N)
        mf._tree  = _build_tree(data)
        return mf

    # ------------------------------------------------------------------ query

    def query(self, uv: np.ndarray, k: int = 8,
              in_dir: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
        """Return the *k* nearest noodles' output directions for each query.

        Lookup is keyed on (u, v, in_dx, in_dy, in_dz) when *in_dir* is
        supplied — aperture position AND approach angle.  Falls back to
        (u, v)-only when *in_dir* is None (legacy path).

        Parameters
        ----------
        uv     : (M, 2) float64 — normalised aperture positions in [-1, 1]
        k      : number of nearest neighbours
        in_dir : (M, 3) float64 — unit approach directions (scene → aperture)

        Returns
        -------
        dirs    : (M, k, 3) float64 — output ray directions (unit vectors)
        weights : (M, k)    float64 — IDW weights summing to 1 per row
        """
        assert self._data is not None, "Manifold is empty — bake first."
        uv = np.asarray(uv, np.float64)
        if uv.ndim == 1:
            uv = uv[np.newaxis, :]
        M = len(uv)
        k = min(k, len(self._data))

        if in_dir is not None:
            ind = np.asarray(in_dir, np.float64)
            if ind.ndim == 1:
                ind = ind[np.newaxis, :]
            query_pts = np.column_stack([uv[:, :2], ind[:, :3]])  # (M, 5)
        else:
            query_pts = uv[:, :2]  # (M, 2) legacy

        if self._tree is not None:
            dists, idxs = self._tree.query(query_pts, k=k, workers=-1)
        else:
            # Pure NumPy fallback — O(M · N)
            noodle_key = np.column_stack([self._data[:, :2], self._data[:, 4:7]])
            diff  = noodle_key[np.newaxis, :, :query_pts.shape[1]] - query_pts[:, np.newaxis, :]
            sq_d  = np.einsum("mni,mni->mn", diff, diff)
            idxs  = np.argpartition(sq_d, kth=k - 1, axis=1)[:, :k]
            dists = np.sqrt(sq_d[np.arange(M)[:, None], idxs])

        # (M, k, 3) via broadcasted advanced indexing — one line
        dirs    = self._data[idxs, _C_OUT]

        # Inverse-distance weights
        eps     = 1e-12
        inv_d   = 1.0 / (dists + eps)
        weights = inv_d / inv_d.sum(axis=1, keepdims=True)

        return dirs, weights

    def query_full(self, uv: np.ndarray, k: int = 8,
                   in_dir: np.ndarray | None = None) -> np.ndarray:
        """Single tensor-index fetch: returns raw ``(M, k, NCOLS)`` float64.

        One fancy-index operation on the flat data array gives every column
        for every neighbour at once — no second index pass needed::

            batch = mf.query_full(uv, k=16, in_dir=approach_dirs)  # (M, k, 11)
            dirs  = batch[:, :, _C_OUT]   # (M, k, 3)  view
            opls  = batch[:, :, _C_OPL]   # (M, k)     view

        Parameters
        ----------
        uv     : (M, 2) float64 — normalised aperture positions in [-1, 1]
        k      : number of nearest neighbours
        in_dir : (M, 3) float64 — unit approach directions; when supplied the
                 tree is queried on the 5D key (u, v, in_dx, in_dy, in_dz).

        Returns
        -------
        batch : (M, k, NCOLS) float64 — raw noodle rows; no copy unless
                the underlying array is non-contiguous.
        """
        assert self._data is not None, "Manifold is empty — bake first."
        uv = np.asarray(uv, np.float64)
        if uv.ndim == 1:
            uv = uv[np.newaxis, :]
        k = min(k, len(self._data))

        if in_dir is not None:
            ind = np.asarray(in_dir, np.float64)
            if ind.ndim == 1:
                ind = ind[np.newaxis, :]
            query_pts = np.column_stack([uv[:, :2], ind[:, :3]])  # (M, 5)
        else:
            query_pts = uv[:, :2]  # (M, 2) legacy

        if self._tree is not None:
            _, idxs = self._tree.query(query_pts, k=k, workers=-1)
        else:
            noodle_key = np.column_stack([self._data[:, :2], self._data[:, 4:7]])
            diff = noodle_key[np.newaxis, :, :query_pts.shape[1]] - query_pts[:, np.newaxis, :]
            sq_d = np.einsum("mni,mni->mn", diff, diff)
            idxs = np.argpartition(sq_d, kth=k - 1, axis=1)[:, :k]
        # Single advanced index — the whole record batch, no per-column copies
        return self._data[idxs]          # (M, k, NCOLS)

    def interpolate(self, uv: np.ndarray, k: int = 8,
                    in_dir: np.ndarray | None = None) -> np.ndarray:
        """IDW-blended single output direction per query point.

        Returns (M, 3) float64 normalised unit vectors.
        """
        dirs, w = self.query(uv, k=k, in_dir=in_dir)        # (M, k, 3), (M, k)
        blended = np.einsum("mk,mki->mi", w, dirs)          # (M, 3)
        norms   = np.linalg.norm(blended, axis=1, keepdims=True)
        return blended / np.maximum(norms, 1e-15)

    def interpolate_with_opl(self, uv: np.ndarray,
                              k: int = 8,
                              in_dir: np.ndarray | None = None) -> tuple:
        """IDW-blended output direction *and* optical path length per query.

        Uses :meth:`query_full` so both direction and OPL are fetched in a
        single tensor-index operation, with no extra lookup pass.

        Returns
        -------
        dirs : (M, 3) float64 \u2014 IDW-blended unit output directions
        opl  : (M,)   float64 \u2014 IDW-blended optical path length (metres)
        """
        batch = self.query_full(uv, k=k, in_dir=in_dir)     # (M, k, NCOLS)
        raw_dirs = batch[:, :, _C_OUT]                      # (M, k, 3)
        raw_opl  = batch[:, :, _C_OPL]                      # (M, k)

        # Compute IDW weights from the 5D key distances
        query_pts = np.asarray(uv, np.float64)
        if query_pts.ndim == 1:
            query_pts = query_pts[np.newaxis, :]
        uv_k  = batch[:, :, :2]                              # (M, k, 2)
        uv_diff = uv_k - query_pts[:, np.newaxis, :]         # (M, k, 2)
        if in_dir is not None:
            ind = np.asarray(in_dir, np.float64)
            if ind.ndim == 1:
                ind = ind[np.newaxis, :]
            in_k   = batch[:, :, 4:7]                        # (M, k, 3)
            in_diff = in_k - ind[:, np.newaxis, :]           # (M, k, 3)
            diff    = np.concatenate([uv_diff, in_diff], axis=-1)  # (M, k, 5)
        else:
            diff = uv_diff                                   # (M, k, 2)
        dists = np.sqrt(np.einsum("mki,mki->mk", diff, diff))  # (M, k)
        inv_d = 1.0 / (dists + 1e-12)
        w     = inv_d / inv_d.sum(axis=1, keepdims=True)    # (M, k)

        blended = np.einsum("mk,mki->mi", w, raw_dirs)      # (M, 3)
        norms   = np.linalg.norm(blended, axis=1, keepdims=True)
        dirs    = blended / np.maximum(norms, 1e-15)
        opl     = np.einsum("mk,mk->m", w, raw_opl)         # (M,)
        return dirs, opl

    @staticmethod
    def transform_batch(dirs: np.ndarray,
                        opls: np.ndarray,
                        R:    np.ndarray,
                        wavelength: float = 550e-9) -> tuple:
        """Batch-translate direction vectors and compute phase in one pass.

        Rotates all *dirs* from sensor-local space to world space via *R*,
        and converts *opls* to wave phase angles for *wavelength*.

        This is the canonical "tensor translate in phase and space" operation.
        Everything is a single broadcast / matmul — no per-ray Python loop.

        Parameters
        ----------
        dirs       : (M, 3) or (M, k, 3) float64 \u2014 sensor-local unit directions
        opls       : (M,)   or (M, k)     float64 \u2014 optical path lengths (metres)
        R          : (3, 3)               float64 \u2014 sensor\u2192world rotation matrix
        wavelength : float (metres) \u2014 for phase computation (default 550 nm)

        Returns
        -------
        world_dirs : same shape as *dirs* -- rotated to world frame
        phase      : same leading shape as *opls* -- wave phase in radians
                     phase = 2*pi * OPL / wavelength
        """
        R = np.asarray(R, np.float64)
        d = np.asarray(dirs, np.float64)
        o = np.asarray(opls, np.float64)
        # matmul broadcast: (..., 3) @ (3, 3).T -> (..., 3)
        world_dirs = d @ R.T
        # Renormalise to correct any accumulated float error
        norms = np.linalg.norm(world_dirs, axis=-1, keepdims=True)
        world_dirs = world_dirs / np.maximum(norms, 1e-15)
        phase = (2.0 * math.pi / max(wavelength, 1e-30)) * o
        return world_dirs, phase

    # ------------------------------------------------------------------ I/O

    def save(self, path: str) -> None:
        """Persist manifold to a .npz file, preserving float64 throughout."""
        import json
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez_compressed(
            path,
            data = self._data,
            meta = np.frombuffer(json.dumps(self.meta).encode(), dtype=np.uint8),
        )

    @classmethod
    def load(cls, path: str) -> "LensManifold":
        """Load a manifold saved with :meth:`save`.

        Handles files saved with the legacy 10-column schema by zero-filling
        the OPL column so older bakes remain usable without rebaking.
        """
        import json
        mf  = cls()
        npz = np.load(path, allow_pickle=False)
        raw = npz["data"].astype(np.float64, copy=False)
        if raw.shape[1] < _NCOLS:
            # Legacy 10-col file: append zero OPL column
            pad = np.zeros((len(raw), _NCOLS - raw.shape[1]), np.float64)
            raw = np.concatenate([raw, pad], axis=1)
        mf._data = raw
        try:
            mf.meta = json.loads(bytes(npz["meta"]).decode())
        except Exception:
            mf.meta = {}
        mf._root = _QuadNode(-1., 1., -1., 1.)
        mf._root.indices = np.arange(len(mf._data))
        mf._tree = _build_tree(mf._data)
        return mf

    # ------------------------------------------------------------------ info

    @property
    def n_noodles(self) -> int:
        return 0 if self._data is None else len(self._data)

    def __repr__(self) -> str:
        return (
            f"LensManifold(n={self.n_noodles:,}, "
            f"focal={self.meta.get('focal_mm', '?')}mm, "
            f"focus={self.meta.get('focus_m', '?')}m)"
        )


# ---------------------------------------------------------------------------
# Analytic bake helpers
# ---------------------------------------------------------------------------

def _build_tree(data: np.ndarray):
    """Build a KD-tree over (u, v, in_dx, in_dy, in_dz) — 5D key.

    Aperture position alone is insufficient: two rays arriving at the same
    aperture point from different scene angles produce different exit angles.
    Including in_dir cols 4-6 in the key ensures the lookup matches on both
    where the ray hits the aperture AND what direction it arrived from.
    """
    try:
        from scipy.spatial import KDTree
        pts = np.column_stack([data[:, :2], data[:, 4:7]])  # (N, 5)
        return KDTree(pts)
    except ImportError:
        return None


def _bake_region(
    u0: float, u1: float, v0: float, v1: float,
    n: int,
    ctx: dict,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """Rejection-sample *n* noodles inside the rectangular aperture sub-region
    [u0,u1] x [v0,v1] intersected with the unit-disk, for all scene directions
    in the FOV cone.

    Each noodle samples a random scene direction (fu, fv) independently, so
    the manifold captures the full per-aperture-point ray transform.
    """
    R, U, F   = ctx["R"], ctx["U"], ctx["F"]
    ts        = ctx["ts"]
    lt        = ctx["lt"]
    eff_focal_m = ctx["eff_focal_m"]
    focus_m   = ctx["focus_m"]
    fov_tan   = ctx["fov_tan"]
    ap_r      = ctx["ap_r"]

    # --- aperture disk sampling --------------------------------------------
    rows: list[np.ndarray] = []
    budget = n * 12
    while len(rows) < n and budget > 0:
        need = (n - len(rows)) * 4
        uu   = rng.uniform(u0, u1, need)
        vv   = rng.uniform(v0, v1, need)
        mask = uu * uu + vv * vv <= 1.0
        uu, vv = uu[mask], vv[mask]
        if len(uu):
            rows.append(np.stack([uu, vv], axis=1))
        budget -= need

    if not rows:
        return None

    uv_ap = np.concatenate(rows, axis=0)[:n]   # (N, 2)
    if len(uv_ap) == 0:
        return None
    N = len(uv_ap)

    u = uv_ap[:, 0]
    v = uv_ap[:, 1]

    # --- scene direction sampling (uniform over sensor rectangle) ----------
    fu = rng.uniform(-fov_tan, fov_tan, N)
    fv = rng.uniform(-fov_tan * 0.667, fov_tan * 0.667, N)  # 3:2 aspect

    # --- aperture world position (relative to eye) -------------------------
    ap_pos = (u[:, None] * ap_r * R[None, :]
              + v[:, None] * ap_r * U[None, :])              # (N, 3)

    # --- base scene ray (through principal point) --------------------------
    rd0 = F[None, :] + fu[:, None] * R[None, :] + fv[:, None] * U[None, :]
    rd0_n = np.linalg.norm(rd0, axis=1, keepdims=True)
    rd0 /= np.maximum(rd0_n, 1e-15)                         # (N, 3) — in_dir

    # --- Scheimpflug effective focus distance (mirrors GLSL exactly) -------
    lt_denom = 1.0 - fu * math.tan(lt[1]) - fv * math.tan(lt[0])
    fd_eff   = focus_m / np.maximum(0.005, lt_denom)
    fd_eff   = np.maximum(fd_eff, 0.01)

    # --- tilt_shift principal-point offset on focus plane ------------------
    focus_pt = rd0 * fd_eff[:, None]   # (N, 3) — focus world position

    # --- output ray: aperture point → focus point --------------------------
    out_dir  = focus_pt - ap_pos
    out_n    = np.linalg.norm(out_dir, axis=1, keepdims=True)
    out_dir /= np.maximum(out_n, 1e-15)

    # --- pack ---------------------------------------------------------------
    data = np.empty((N, _NCOLS), np.float64)
    data[:, _C_U]   = u
    data[:, _C_V]   = v
    data[:, _C_FU]  = fu
    data[:, _C_FV]  = fv
    data[:, _C_IN]  = rd0
    data[:, _C_OUT] = out_dir
    # OPL = geometric path length from aperture to focus point.
    # Thin-lens model: n_air = 1 everywhere, so OPL = |ap_pos → focus_pt|.
    # out_n holds that distance before normalisation was applied.
    data[:, _C_OPL] = out_n.ravel()
    return data


# ---------------------------------------------------------------------------
# Spherical coordinate utilities
# ---------------------------------------------------------------------------

class SphereCoords:
    """Spherical coordinate ↔ Cartesian unit-vector conversions (float64).

    Convention
    ----------
    Standard Z-up spherical:
        theta (θ) : polar angle from +Z axis,   range [0, π]
        phi   (φ) : azimuth from +X in XY plane, range [0, 2π)

    Local-frame spherical (used for surface-normal frames):
        theta : angle from surface normal N
        phi   : azimuth around N in the tangent plane

    All routines preserve the native dtype of inputs and never force a
    specific float width.
    """

    @staticmethod
    def to_vec(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """(θ, φ) → unit Cartesian vector(s) in the standard Z-up frame.

        Inputs may be scalars or (...) arrays; output has shape (..., 3).
        """
        theta = np.asarray(theta, np.float64)
        phi   = np.asarray(phi,   np.float64)
        s     = np.sin(theta)
        return np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)],
                        axis=-1)

    @staticmethod
    def from_vec(v: np.ndarray) -> tuple:
        """Unit vector(s) (..., 3) → (θ, φ) each shape (...) float64."""
        v  = np.asarray(v, np.float64)
        vn = v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-15)
        theta = np.arccos(np.clip(vn[..., 2], -1.0, 1.0))
        phi   = np.arctan2(vn[..., 1], vn[..., 0]) % (2.0 * math.pi)
        return theta, phi

    @staticmethod
    def _tangent_frame(normal: np.ndarray) -> tuple:
        """Build a right-handed (T, B, N) orthonormal frame around a unit normal."""
        normal = np.asarray(normal, np.float64)
        normal = normal / np.maximum(np.linalg.norm(normal), 1e-15)
        ref    = np.array([0.0, 0.0, 1.0], np.float64)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([1.0, 0.0, 0.0], np.float64)
        T = np.cross(normal, ref);  T /= np.maximum(np.linalg.norm(T), 1e-15)
        B = np.cross(normal, T);    B /= np.maximum(np.linalg.norm(B), 1e-15)
        return T, B, normal

    @staticmethod
    def local_to_world(theta: np.ndarray, phi: np.ndarray,
                       normal: np.ndarray) -> np.ndarray:
        """(θ from normal, φ around normal) → world unit vectors (N, 3) float64."""
        T, B, N = SphereCoords._tangent_frame(normal)
        theta = np.asarray(theta, np.float64)
        phi   = np.asarray(phi,   np.float64)
        s     = np.sin(theta)
        result = (s * np.cos(phi))[:, None] * T[None, :] \
               + (s * np.sin(phi))[:, None] * B[None, :] \
               + np.cos(theta)[:, None]     * N[None, :]
        return result

    @staticmethod
    def world_to_local(vecs: np.ndarray,
                       normal: np.ndarray) -> tuple:
        """World unit vectors (N, 3) → (θ, φ) in local frame of `normal`."""
        T, B, N = SphereCoords._tangent_frame(normal)
        vecs    = np.asarray(vecs, np.float64)
        cos_t   = np.clip(np.einsum('ni,i->n', vecs, N), -1.0, 1.0)
        theta   = np.arccos(cos_t)
        tx      = np.einsum('ni,i->n', vecs, T)
        ty      = np.einsum('ni,i->n', vecs, B)
        phi     = np.arctan2(ty, tx) % (2.0 * math.pi)
        return theta, phi

    @staticmethod
    def sphere_sample_uniform(N: int, rng: np.random.Generator) -> np.ndarray:
        """Uniform random unit vectors on the full sphere. Returns (N, 3) float64."""
        z    = rng.uniform(-1.0, 1.0, N)
        phi  = rng.uniform(0.0, 2.0 * math.pi, N)
        r    = np.sqrt(np.maximum(0.0, 1.0 - z**2))
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)

    @staticmethod
    def hemisphere_sample(N: int, normal: np.ndarray,
                          rng: np.random.Generator,
                          max_theta: float = math.pi / 2.0) -> np.ndarray:
        """Cosine-weighted hemisphere sample up to *max_theta* from *normal*.

        Returns (N, 3) float64 unit vectors biased toward the normal direction.
        """
        # Polar: theta from Beta distribution biased toward 0
        u1 = rng.uniform(0.0, 1.0, N)
        u2 = rng.uniform(0.0, 2.0 * math.pi, N)
        # Map u1 to theta in [0, max_theta] with cosine weighting
        cos_max = math.cos(max_theta)
        cos_t   = 1.0 - u1 * (1.0 - cos_max)
        theta   = np.arccos(np.clip(cos_t, -1.0, 1.0))
        return SphereCoords.local_to_world(theta, u2, normal)

    @staticmethod
    def foveal_sample(N: int, normal: np.ndarray,
                      rng: np.random.Generator,
                      foveal_sigma: float = 0.26,
                      peripheral_fraction: float = 0.35,
                      max_theta: float = math.pi / 2.0) -> np.ndarray:
        """Sample scene directions with realistic foveal density weighting.

        Concentrates samples near the optical axis (small theta) to match
        human cone-density falloff.  Two-zone model:
          - (1 - peripheral_fraction) of rays sampled within foveal Gaussian
            (sigma = foveal_sigma radians ≈ 15° by default)
          - peripheral_fraction sampled uniformly over [0, max_theta]

        Returns (N, 3) float64 unit vectors.
        """
        n_fov  = N - int(N * peripheral_fraction)
        n_peri = N - n_fov

        # Foveal zone: Rayleigh-distributed theta (Gaussian projected to sphere)
        u1_f   = rng.uniform(0.0, 1.0, n_fov)
        theta_f = foveal_sigma * np.sqrt(-2.0 * np.log(np.maximum(u1_f, 1e-12)))
        theta_f = np.minimum(theta_f, max_theta)
        phi_f   = rng.uniform(0.0, 2.0 * math.pi, n_fov)

        # Peripheral zone: uniform in solid angle up to max_theta
        u1_p    = rng.uniform(0.0, 1.0, n_peri)
        cos_max = math.cos(max_theta)
        cos_p   = 1.0 - u1_p * (1.0 - cos_max)
        theta_p = np.arccos(np.clip(cos_p, -1.0, 1.0))
        phi_p   = rng.uniform(0.0, 2.0 * math.pi, n_peri)

        theta = np.concatenate([theta_f, theta_p])
        phi   = np.concatenate([phi_f,   phi_p])
        return SphereCoords.local_to_world(theta, phi, normal)


# ---------------------------------------------------------------------------
# Cylindrical coordinate utilities
# ---------------------------------------------------------------------------

class CylinderCoords:
    """Cylindrical coordinate ↔ Cartesian conversions (float64).

    Convention: (r, φ, z) where r ≥ 0, φ ∈ [0, 2π), z along +Z axis.
    All routines preserve float64.
    """

    @staticmethod
    def to_cart(r: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
        """(r, φ, z) → Cartesian (..., 3) float64."""
        r, phi, z = (np.asarray(a, np.float64) for a in (r, phi, z))
        return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)

    @staticmethod
    def from_cart(xyz: np.ndarray) -> tuple:
        """Cartesian (..., 3) → (r, φ, z) each (...) float64."""
        xyz = np.asarray(xyz, np.float64)
        r   = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
        phi = np.arctan2(xyz[..., 1], xyz[..., 0]) % (2.0 * math.pi)
        return r, phi, xyz[..., 2]

    @staticmethod
    def disk_sample(N: int, r_max: float, rng: np.random.Generator,
                    r_min: float = 0.0) -> np.ndarray:
        """Uniform random points on a disk annulus [r_min, r_max].

        Returns (N, 2) float64 (x, y) with uniform area density.
        """
        u   = rng.uniform(r_min**2 / r_max**2, 1.0, N)
        phi = rng.uniform(0.0, 2.0 * math.pi, N)
        r   = r_max * np.sqrt(u)
        return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=1)

    @staticmethod
    def project_onto_axis(pts: np.ndarray, axis: np.ndarray,
                          origin: np.ndarray) -> tuple:
        """Decompose points into (r⊥, φ, z‖) relative to an oriented axis.

        pts    : (N, 3) float64
        axis   : (3,)   float64 — unit vector of cylinder axis
        origin : (3,)   float64 — point on the axis

        Returns (r, phi, z) each (N,) float64.
        """
        axis   = np.asarray(axis, np.float64)
        axis  /= np.maximum(np.linalg.norm(axis), 1e-15)
        origin = np.asarray(origin, np.float64)
        dp     = np.asarray(pts, np.float64) - origin[None, :]
        z      = np.einsum('ni,i->n', dp, axis)
        perp   = dp - z[:, None] * axis[None, :]
        r      = np.linalg.norm(perp, axis=1)
        T, B, _ = SphereCoords._tangent_frame(axis)
        phi    = np.arctan2(np.einsum('ni,i->n', perp, B),
                            np.einsum('ni,i->n', perp, T)) % (2.0 * math.pi)
        return r, phi, z


# ---------------------------------------------------------------------------
# Piecewise polar ray-chain tracer (Snell's law through N optical surfaces)
# ---------------------------------------------------------------------------

class PolarChain:
    """Batch piecewise polar ray-path tracer through a sequence of optical surfaces.

    Models a ray path as segments described in the local polar frame of each
    surface crossed.  Applies vectorised Snell's law at every interface so
    that millions of rays can be traced in a single NumPy pass.

    Noodle-compatible output
    ------------------------
    ``trace()`` returns ``(out_dirs, valid_mask)``.  The caller feeds
    ``out_dirs`` directly into the ``_C_OUT`` columns of a manifold.

    Chain table (available after ``trace()``)
    ------------------------------------------
    Shape ``(N_rays, N_surfaces, 4)`` float64:
        [..., 0]  theta      — polar angle from surface optical axis (radians)
        [..., 1]  phi        — azimuth around optical axis (radians)
        [..., 2]  path_len   — geometric path length of the segment
                               *entering* this surface (m)
        [..., 3]  n_in       — refractive index of the *incoming* medium
                               (the medium the segment travels through)

    Optical path length per ray is therefore::

        opl = np.sum(chain[:, :, 2] * chain[:, :, 3], axis=1)

    Surface geometry
    ----------------
    Each surface is a sphere of curvature defined by:
        origin : vertex position on the optical axis
        normal : optical axis direction at that surface (unit vector)
        radius : signed radius of curvature (positive → centre behind surface)

    The centre of curvature for surface i is:
        centre_i = origin_i + radius_i * normal_i
    """

    def __init__(self, n_media: list, normals: list,
                 origins: list, radii: list) -> None:
        """
        n_media : list of M+1 floats — refractive indices
                  [n_before_srf_0, n_between_0_and_1, ..., n_after_srf_M-1]
        normals : list of M (3,) arrays — optical axis unit vector per surface
        origins : list of M (3,) arrays — vertex positions (metres) per surface
        radii   : list of M floats      — signed radii of curvature (metres)
        """
        M = len(normals)
        assert len(n_media) == M + 1
        assert len(origins) == M
        assert len(radii)   == M
        self.n_media = [float(n) for n in n_media]
        self.normals = [
            np.asarray(v, np.float64) / max(float(np.linalg.norm(np.asarray(v))), 1e-15)
            for v in normals
        ]
        self.origins = [np.asarray(v, np.float64) for v in origins]
        self.radii   = [float(r) for r in radii]
        self._chain: Optional[np.ndarray] = None   # (N, M, 4) after trace()

    @property
    def n_surfaces(self) -> int:
        return len(self.normals)

    # ------------------------------------------------------------------ geometry helpers

    def _intersect_sphere(self, ro: np.ndarray, rd: np.ndarray,
                          centre: np.ndarray, radius: float) -> np.ndarray:
        """Batch ray–sphere intersection. Returns t (N,) float64, NaN on miss."""
        oc   = ro - centre[None, :]                          # (N, 3)
        b    = 2.0 * np.einsum('ni,ni->n', oc, rd)          # (N,)
        c    = np.einsum('ni,ni->n', oc, oc) - radius ** 2  # (N,)
        disc = b ** 2 - 4.0 * c                              # a=1 for unit rd
        t    = np.full(len(ro), np.nan, np.float64)
        hit  = disc >= 0.0
        sd   = np.where(hit, np.sqrt(np.maximum(disc, 0.0)), 0.0)
        t1   = (-b - sd) * 0.5
        t2   = (-b + sd) * 0.5
        eps  = 1e-9
        t    = np.where(hit & (t1 > eps), t1,
               np.where(hit & (t2 > eps), t2, np.nan))
        return t

    def _surface_normals_at(self, pts: np.ndarray, i: int) -> np.ndarray:
        """Outward unit normals at hit points on spherical surface i."""
        centre  = self.origins[i] + self.radii[i] * self.normals[i]
        normals = pts - centre[None, :]
        return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True),
                                    1e-15)

    # ------------------------------------------------------------------ Snell's law

    @staticmethod
    def snell_batch(n1: float, n2: float,
                    in_vecs: np.ndarray,
                    surf_normals: np.ndarray) -> tuple:
        """Vectorised Snell's law refraction.

        Uses the vector form:
            out = (n1/n2)*in + (n1/n2 * cosI - cosT) * N̂

        where N̂ points into the medium of incidence (against the ray).

        Returns
        -------
        out_vecs  : (N, 3) float64 — refracted unit directions (TIR rays zeroed)
        valid     : (N,)   bool    — False where total internal reflection occurs
        """
        # Ensure surface normal faces the incoming ray
        cos_i = -np.einsum('ni,ni->n', in_vecs, surf_normals)
        flip  = cos_i < 0.0
        n_hat = np.where(flip[:, None], -surf_normals, surf_normals)
        cos_i = np.where(flip, -cos_i, cos_i)
        cos_i = np.clip(cos_i, 0.0, 1.0)

        ratio  = n1 / n2
        sin2_t = ratio ** 2 * (1.0 - cos_i ** 2)
        tir    = sin2_t > 1.0
        cos_t  = np.sqrt(np.maximum(0.0, 1.0 - sin2_t))

        out_vecs  = ratio * in_vecs + (ratio * cos_i - cos_t)[:, None] * n_hat
        norms     = np.linalg.norm(out_vecs, axis=1, keepdims=True)
        out_vecs /= np.maximum(norms, 1e-15)
        out_vecs  = np.where(tir[:, None], np.zeros_like(out_vecs), out_vecs)
        return out_vecs, ~tir

    # ------------------------------------------------------------------ main trace

    def trace(self, ray_origins: np.ndarray,
              in_dirs: np.ndarray) -> tuple:
        """Trace N rays through every surface.

        Parameters
        ----------
        ray_origins : (N, 3) float64 — starting positions (e.g. on the iris plane)
        in_dirs     : (N, 3) float64 — unit direction vectors (scene → first surface)

        Returns
        -------
        out_dirs : (N, 3) float64 — final propagation directions after last surface
        valid    : (N,)   bool    — False if any surface was missed or TIR occurred
        """
        N      = len(ray_origins)
        M      = self.n_surfaces
        chain  = np.zeros((N, M, 4), np.float64)

        cur_o  = np.asarray(ray_origins, np.float64).copy()
        cur_d  = np.asarray(in_dirs,     np.float64).copy()
        cur_d /= np.maximum(np.linalg.norm(cur_d, axis=1, keepdims=True), 1e-15)
        valid  = np.ones(N, dtype=bool)

        for i in range(M):
            n1     = self.n_media[i]
            n2     = self.n_media[i + 1]
            origin = self.origins[i]
            normal = self.normals[i]
            radius = self.radii[i]

            centre = origin + radius * normal
            t      = self._intersect_sphere(cur_o, cur_d, centre, abs(radius))

            missed  = np.isnan(t) | (t <= 1e-9)
            valid  &= ~missed
            t       = np.where(missed, 0.0, t)

            hit_pts  = cur_o + t[:, None] * cur_d
            s_norms  = self._surface_normals_at(hit_pts, i)
            out_d, tir_ok = PolarChain.snell_batch(n1, n2, cur_d, s_norms)
            valid   &= tir_ok

            # Record chain table in optical-axis-local polar coords
            th, ph              = SphereCoords.world_to_local(cur_d, normal)
            chain[:, i, 0]      = th
            chain[:, i, 1]      = ph
            chain[:, i, 2]      = t
            chain[:, i, 3]      = n1   # incoming medium for this segment

            cur_o = hit_pts
            cur_d = out_d

        self._chain = chain
        return cur_d, valid

    @property
    def chain_table(self) -> Optional[np.ndarray]:
        """(N, M, 4) float64 from last trace(). Cols: theta, phi, path_len, n_in."""
        return self._chain

    @property
    def chain_opl(self) -> Optional[np.ndarray]:
        """(N,) float64 — total optical path length (Σ n_i · d_i) per ray.

        Computed from the chain table recorded by the last call to
        :meth:`trace`.  Returns ``None`` if no trace has been run yet.
        """
        if self._chain is None:
            return None
        # chain[:, i, 2] = segment geometric length d_i
        # chain[:, i, 3] = n_in for that segment
        return np.sum(self._chain[:, :, 2] * self._chain[:, :, 3], axis=1)


# ---------------------------------------------------------------------------
# LensAlignmentBuffer
# ---------------------------------------------------------------------------

class LensAlignmentBuffer:
    """Delay-line buffer that realigns lens-transform output by optical path length.

    Rays produced by a ``ManifoldBack`` at different aperture positions travel
    different geometric paths through the lens system, accumulating different
    OPLs and therefore arriving at the sensor at physically distinct times.
    When building a temporally consistent accumulation (e.g. real-time raycast
    integration or wavefront reconstruction), batches must be held until their
    OPL-implied delay has elapsed before being splatted.

    Usage::

        buf = LensAlignmentBuffer(opl_range=0.1e-3, n_slots=512)

        # Each frame — push outgoing rays with their per-ray OPL:
        dirs, opl = manifold.interpolate_with_opl(uv_ap)
        buf.push(dirs, values, weights, opl, opl_reference=opl.mean())

        # Retrieve rays whose delay has been satisfied at current reference OPL:
        for dirs_out, vals_out, w_out in buf.flush(current_opl_front):
            sensor_back.accumulate(dirs_out, vals_out, w_out)

    The buffer is OPL-ordered, not time-ordered — callers should advance the
    ``current_opl_front`` monotonically.  In a frame-based renderer pass the
    reference advances by ``c * frame_dt``.

    Parameters
    ----------
    opl_range : float
        Maximum OPL spread to accommodate (metres).  Rays beyond this range
        relative to the current front are held at the boundary slot until
        flushed.  A typical camera pupil spans < 0.1 mm in OPL, so the
        default of 1 mm covers even highly aberrated lenses.
    n_slots : int
        Number of quantisation slots across *opl_range*.  Finer slots give
        more precise temporal resolution at the cost of more internal queues.
        Must be a power of two for efficient modular indexing; if not, it is
        rounded up.
    c : float
        Speed of light (m/s) used only for the optional time-domain helper
        :meth:`flush_at_time`.  Does not affect OPL-domain operations.
    """

    def __init__(self, opl_range: float = 1e-3,
                 n_slots: int = 512,
                 c: float = 299_792_458.0) -> None:
        # Round n_slots up to the next power of two
        ns = 1
        while ns < n_slots:
            ns <<= 1
        self.n_slots   = ns
        self.opl_range = float(opl_range)
        self.c         = float(c)
        self._slot_w   = opl_range / ns          # OPL width per slot (metres)
        self._mask     = ns - 1                  # fast modular index
        # Each slot is a list of (dirs, values, weights, opl_arr) tuples
        self._slots: list = [[] for _ in range(ns)]
        # The current OPL front — slots with opl_slot < this are ready to flush
        self._front: float = 0.0

    # ------------------------------------------------------------------ push

    def push(self, dirs: np.ndarray,
             values: np.ndarray,
             weights: Optional[np.ndarray],
             opls: np.ndarray,
             opl_reference: Optional[float] = None) -> None:
        """Queue a batch of rays by their optical path length.

        Parameters
        ----------
        dirs          : (N, 3) float64 \u2014 sensor-local output directions
        values        : (N, C) or (N,) float64 \u2014 per-ray radiance / energy
        weights       : (N,)   float64 or None
        opls          : (N,)   float64 \u2014 per-ray OPL (metres)
        opl_reference : float or None \u2014 if given, shift all OPLs by
                        ``-opl_reference`` before slotting (normalises the
                        chief-ray OPL to slot 0, so only aberration delay is
                        stored).
        """
        opls = np.asarray(opls, np.float64).ravel()
        if opl_reference is not None:
            opls = opls - float(opl_reference)

        # Bin rays into slots by their relative OPL
        slot_f = np.clip(opls / self._slot_w, 0, self.n_slots - 1)
        slot_i = slot_f.astype(np.intp)          # (N,) integer slot indices

        dirs    = np.asarray(dirs,   np.float64)
        values  = np.asarray(values, np.float64)
        w_arr   = (np.asarray(weights, np.float64)
                   if weights is not None
                   else np.ones(len(dirs), np.float64))

        # Scatter rays into per-slot accumulators using unique-slot trick
        unique_slots, inv_idx = np.unique(slot_i, return_inverse=True)
        for k, s in enumerate(unique_slots):
            mask = (inv_idx == k)
            self._slots[int(s) & self._mask].append((
                dirs[mask],
                values[mask],
                w_arr[mask],
                opls[mask],
            ))

    # ------------------------------------------------------------------ flush

    def flush(self, opl_front: Optional[float] = None) -> list:
        """Yield and drain all slots whose OPL ≤ *opl_front*.

        Parameters
        ----------
        opl_front : float or None
            The current OPL reference (metres).  If ``None``, all slots are
            flushed regardless of delay.

        Returns
        -------
        list of (dirs, values, weights) tuples — one per non-empty slot,
        in ascending OPL order.  Safe to pass directly to
        ``CameraBack.accumulate()``.
        """
        if opl_front is None:
            limit = self.n_slots
        else:
            self._front = float(opl_front)
            limit = int(self._front / self._slot_w) + 1
            limit = min(limit, self.n_slots)

        out = []
        for s in range(limit):
            batch_list = self._slots[s & self._mask]
            if not batch_list:
                continue
            # Concatenate all pushes into this slot
            all_dirs = np.concatenate([b[0] for b in batch_list], axis=0)
            all_vals = np.concatenate([b[1] for b in batch_list], axis=0)
            all_w    = np.concatenate([b[2] for b in batch_list], axis=0)
            out.append((all_dirs, all_vals, all_w))
            self._slots[s & self._mask] = []
        return out

    def flush_at_time(self, t_now: float,
                      t_zero: float = 0.0) -> list:
        """Convenience wrapper: flush up to OPL front = c × (t_now - t_zero)."""
        opl_front = self.c * (float(t_now) - float(t_zero))
        return self.flush(opl_front)

    def clear(self) -> None:
        """Discard all pending rays."""
        self._slots = [[] for _ in range(self.n_slots)]
        self._front = 0.0

    @property
    def pending_count(self) -> int:
        """Total number of pending ray batches across all slots."""
        return sum(len(sl) for sl in self._slots)
