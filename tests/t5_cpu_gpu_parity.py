"""Small deterministic BDPT connection parity fixture.

This deliberately avoids the production thick-lens scene: exhaustive CPU T5
must remain small enough to finish in seconds.  Both variants trace the same
diffuse receiver, area emitter and sensor geometry; one dispatches T5 on the
GPU and the other forces the corrected C++ all-pairs implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _spectral_kernels as sk
from ray_tracer_bridge import per_tri_spectral_to_mat_buf


def quad_x(x: float, yc: float, zc: float, hy: float, hz: float, positive: bool):
    q = np.array([[x,yc-hy,zc-hz],[x,yc+hy,zc-hz],
                  [x,yc+hy,zc+hz],[x,yc-hy,zc+hz]], np.float64)
    if positive:
        return np.array([[q[0],q[1],q[2]],[q[0],q[2],q[3]]])
    return np.array([[q[0],q[2],q[1]],[q[0],q[3],q[2]]])


@dataclass
class Result:
    image: np.ndarray
    connections: int
    visible_connections: int
    overflow: dict


def run(force_cpu_t5: bool, use_glass: bool = False) -> Result:
    # receiver faces camera (-X), emitter faces receiver (+X), sensor faces out (-X)
    parts = [
        # The emitter is outside the camera cone, eliminating direct terminal
        # splats; image energy must therefore come from the T5 connection.
        quad_x(1.0, 0.0, 0.0, 1.00, 0.55, False),
        quad_x(0.55, 0.70, 0.0, 0.10, 0.10, True),
        quad_x(0.0, 0.0, 0.0, 0.20, 0.20, True),
    ]
    if use_glass:
        parts.append(quad_x(0.30, 0.0, 0.0, 1.10, 0.65, True))
    tris = np.concatenate(parts)
    normals = np.stack([np.cross(t[1]-t[0],t[2]-t[0]) for t in tris])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    nb = 4
    freq = (299_792_458.0 / np.linspace(680e-9, 440e-9, nb)).astype(np.float64)
    nt = len(tris)
    refl = np.zeros((nt,nb)); diff = np.zeros((nt,nb)); emission = np.zeros((nt,nb))
    transm = np.zeros((nt,nb)); ior = np.ones((nt,nb)); ior_im = np.zeros((nt,nb))
    refl[0:2] = 0.8; diff[0:2] = 1.0
    emission[2:4] = np.array([0.7, 1.0, 0.8, 0.5])
    if use_glass:
        refl[6:8] = 0.04
        transm[6:8] = 1.0
        ior[6:8] = np.array([1.50, 1.505, 1.515, 1.525])
    mat_idx, mat_buf, nm = per_tri_spectral_to_mat_buf(
        refl, np.zeros_like(refl), diff, freq, emission_bands=emission,
        transmittance_bands=transm, ior_real_bands=ior, ior_imag_bands=ior_im)
    tr = sk.RayTracer(nt, tris.reshape(nt,9), normals, mat_idx, mat_buf, nm,
                      freq, 299_792_458.0, np.zeros(nb))
    tr.clear_tri_groups()
    tr.set_tri_ior(2, 2, int(sk.MAT_FLAG_EMISSIVE))
    if use_glass:
        tr.set_tri_ior(6, 2, int(sk.MAT_FLAG_TRANSMISSIVE))
        tr.set_tri_boundary_media(6, 2, int(mat_idx[6]), -1)
    tr.register_tri_group(int(sk.TRI_GROUP_ROLE_EMISSIVE),
                          int(sk.TRI_GROUP_SAMPLE_AREA), np.array([2,3],np.int32))
    tr.register_tri_group(
        int(sk.TRI_GROUP_ROLE_SENSOR), int(sk.TRI_GROUP_SAMPLE_PIXEL_CONE),
        np.array([4,5],np.int32), sensor_camera={
            "pos":np.array([0.,0.,0.]), "fwd":np.array([1.,0.,0.]),
            "up":np.array([0.,0.,1.]), "sensor_w_m":0.4, "sensor_h_m":0.4,
            "focal_m":0.05, "aperture_radius_m":0.01,
            "n_px":8, "n_py":8, "n_aperture_samples":2,
            "pixel_stream_divisor":1, "pixel_stream_phase":0,
            "pixel_stream_phase_from_seed":0, "aperture_stop_group_id":-1,
        })
    tr.configure_sensor_image(0.0,0.2,0.2,8,0.002,1.0,0.4,0.0,0.0,0)
    tr.set_vcm(False,0.002,0.7)
    tr.set_force_cpu_t5(force_cpu_t5)
    if not force_cpu_t5:
        tr.set_t5_profile(True)
    nf = tr.submit_emissive_triangles(np.array([2,3],np.int32),512,1.0,1.0,2,0.0,2,17,
                                      True,False,"csrc/shaders",1.0,0.0,0.0,0.4)
    ns = tr.submit_sensor_sweep(max_bounces=2,min_amplitude=0.0,max_rays=0,
                           pix_offset=0,max_children=2,aperture_samples=2,
                           seed=17,exposure_weight=1.0)
    print({"force_cpu_t5":force_cpu_t5,"glass":use_glass,
           "submitted_flash":nf,"submitted_sensor":ns})
    deadline = time.monotonic() + 20.0
    while tr.in_flight_count() != 0 and time.monotonic() < deadline:
        time.sleep(0.002)
    if tr.in_flight_count() != 0:
        raise RuntimeError("transport did not become idle before parity snapshot")
    # Clear direct/terminal accumulation after tracing while retaining BDPT
    # records.  The post-join image is therefore an isolated T5 result.
    tr.configure_sensor_image(0.0,0.2,0.2,8,0.002,1.0,0.4,0.0,0.0,0)
    tr.signal_flash_dispatched()
    tr.signal_sensor_dispatched(); tr.join_t5()
    image = np.asarray(tr.get_sensor_image(),np.float64)
    conn_rows = np.asarray(tr.drain_bdpt_connections(1_000_000))
    conns = int(conn_rows.shape[0])
    visible = int(np.count_nonzero(
        conn_rows[:,60:64].copy().view(np.float32) > 0.0)) if conns else 0
    overflow = dict(tr.get_bdpt_overflow())
    tr.stop_pipeline()
    return Result(image,conns,visible,overflow)


def main() -> int:
    failed = False
    for glass in (False, True):
        gpu = run(False, glass)
        cpu = run(True, glass)
        g = gpu.image
        c = cpu.image
        scale = max(float(np.linalg.norm(g)), float(np.linalg.norm(c)), 1e-20)
        rel_l2 = float(np.linalg.norm(g-c)/scale)
        print({"glass":glass,"gpu_connections":gpu.connections,
               "cpu_connections":cpu.connections,
               "cpu_visible_connections":cpu.visible_connections,
               "gpu_sum":float(g.sum()),"cpu_sum":float(c.sum()),
               "gpu_channel_sums":gpu.image.sum(axis=(0,1)).tolist(),
               "cpu_channel_sums":cpu.image.sum(axis=(0,1)).tolist(),
               "relative_l2":rel_l2,"gpu_overflow":gpu.overflow,
               "cpu_overflow":cpu.overflow})
        failed |= (not np.isfinite(rel_l2) or not np.any(g>0) or
                   not np.any(c>0) or rel_l2 >= 0.15)
    return 3 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
