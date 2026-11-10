"""Opt-in, isolated native housing intersections plus still/live GPU receipts.

Run directly only after a fresh process check confirms an idle exposure slot.
Never contacts the production API, alters a user's files, or starts a server.
"""
from pathlib import Path
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import demo_pluck_gl as renderer
    from OpenGL import GL as gl
    from camera_designer.camera_frame import sensor_camera_basis, thin_lens_housing_frame
    from camera_designer.camera_preset import thin_lens_housing_preset
    from camera_designer.scene_builder import build_gpu_scene
    from camera_designer.marble_preview_worker import run

    output = ROOT / "build/dot-camera-geometry-20261109"
    output.mkdir(parents=True, exist_ok=True)
    renderer._ensure_worker_gl_context()
    probe = '''
    layout(std430,binding=13) readonly buffer ProbeIn {vec4 rays[];};
    layout(std430,binding=14) buffer ProbeOut {vec4 results[];};
    uniform int probeCount;
    void main(){int i=int(gl_GlobalInvocationID.x);if(i>=probeCount||gl_GlobalInvocationID.y!=0u)return;
    float distance;int hit=nearest_hit(rays[2*i].xyz,rays[2*i+1].xyz,distance);
    results[i]=vec4(float(hit),distance,0,0);}
    '''
    receipts = []
    poses = [([.1621047867, .0685409191, .4767308217], [0, 0, .18]),
             ([.9, 1.7, -.5], [.9, .1, -.5])]
    for eye, target in poses:
        preset = thin_lens_housing_preset({"aperture_radius_m": .002})
        rotation, translation = thin_lens_housing_frame(preset, eye, target)
        scene = build_gpu_scene(preset, [], camera_frame=(rotation, translation))
        geom, shade, mat, triangles, contexts, _, bounds, _ = scene
        nodes, ids = renderer._build_gpu_bvh(triangles)
        eye, right, up, fwd = sensor_camera_basis(eye, target)
        rays = []
        for theta in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            rays.extend(([*(eye + .002 * (right * np.cos(theta) + up * np.sin(theta))), 0], [*fwd, 0]))
        for origin, direction in (([.2, .011, .01], [-1, 0, 0]), ([.011, .013, -.2], [0, 0, 1])):
            rays.extend(([*(np.asarray(origin) @ rotation.T + translation), 0], [*(np.asarray(direction) @ rotation.T), 0]))
        rays = np.asarray(rays, np.float32)
        outputs = []
        for name, source in (("forward", renderer._GPU_RAY_FIELD_CS), ("sensor", renderer._GPU_SENSOR_CS)):
            program = renderer._prog((source.replace("void main() {", "void unused_production_main() {", 1) + probe, gl.GL_COMPUTE_SHADER))
            buffers = []
            try:
                assert gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
                data = ((0, geom), (1, nodes), (2, ids), (9, shade), (10, mat),
                        (12, np.zeros((1, 8), np.float32)), (13, rays),
                        (14, np.full((18, 4), -99, np.float32)))
                for binding, value in data:
                    buffer = int(gl.glGenBuffers(1)); buffers.append(buffer)
                    gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer)
                    gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, value.nbytes, value, gl.GL_STATIC_DRAW)
                    gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding, buffer)
                gl.glUseProgram(program)
                for uniform, value in (("uNodeCount", len(nodes)), ("uTriCount", len(geom)),
                                       ("uStudioSphereCount", 0), ("probeCount", 18)):
                    gl.glUniform1i(gl.glGetUniformLocation(program, uniform), value)
                # Sensor uses 16 X lanes, forward uses 128. Two groups cover
                # all 18 rays in either existing shader's real local layout.
                gl.glDispatchCompute(2, 1, 1)
                gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT); gl.glFinish()
                gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffers[-1])
                result = np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, 18 * 16), np.float32).reshape(18, 4).copy()
                assert np.all(result[:16, 0] == -1), result
                assert np.all(result[16:, 0] >= 0), result
                np.testing.assert_allclose(result[16:, 1], [.1305, .173], atol=3e-7)
                outputs.append(result)
                print(f"PASS {name}: 16 pupil misses, side/back reflected-direction hits, eye={eye.tolist()}", flush=True)
            finally:
                gl.glUseProgram(0); gl.glDeleteBuffers(len(buffers), buffers); gl.glDeleteProgram(program)
        np.testing.assert_array_equal(outputs[0], outputs[1])
        receipts.append({"eye_m": eye.tolist(), "target_m": list(target),
                         "bounds": [b.tolist() for b in bounds], "triangles": len(triangles),
                         "pupil_misses_both_tracers": 16, "side_back_hit_distance_m": outputs[0][16:, 1].tolist()})

    # Actual worker still path and finite live ring; tiny isolated inputs only.
    vertices = np.array([[[-.2, -.2, 0], [.2, -.2, 0], [0, .2, 0]]], np.float32)
    pbr = np.zeros((1, 16), np.float32); pbr[:, :3] = .8; pbr[:, 6] = 1.5; pbr[:, 7] = 1
    payload = output / "payload.npz"
    np.savez(payload, vertices=vertices, material_ids=np.zeros(1, np.int32), pbr=pbr)
    camera = {"eye_m": [0, 0, .07], "target_m": [0, 0, .18], "resolution": 128,
              "fov_degrees": 18, "aperture_radius_m": .002, "focus_distance_m": .11}
    config = {"camera": camera, "transport": "fixed:1", "scale_m": .025, "max_bounces": 1,
              "bounce_mode": "manual"}
    config_path = output / "still-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    still = run(payload, config_path, output / "still", seconds=30, max_epochs=1)
    assert still["state"] == "complete"
    live = dict(width=16, height=16, aspect=1, history_samples=2, checkpoint=False,
                lease_seconds=30, ray_batch_size=128, display={})
    config.update(transport="fixed:3", live=live)
    config_path = output / "live-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    live_dir = output / "live"; live_dir.mkdir(exist_ok=True)
    moved = {**camera, "eye_m": [.5, .4, -.4], "target_m": [0, 0, .18], "aperture_radius_m": .004}
    (live_dir / "live-control.json").write_text(json.dumps({"camera": moved, "live": live,
        "revision": 1, "expires_at": time.time() + 60}), encoding="utf-8")
    active = run(payload, config_path, live_dir, seconds=30, max_epochs=2)
    assert active["state"] == "canceled" and active["reason"] == "test epoch limit"
    assert active["live_revision"] == 1
    assert active["geometry_stats"]["camera_geometry"]["eye_m"] == moved["eye_m"]
    assert active["geometry_stats"]["camera_geometry"]["pupil_radius_m"] == .004
    assert active["total_samples"] > 0
    receipt = {"renderer": gl.glGetString(gl.GL_RENDERER).decode(), "intersections": receipts,
               "still_state": still["state"], "live_state": active["state"],
               "live_revision": active["live_revision"], "live_samples": active["total_samples"],
               "live_camera": active["geometry_stats"]["camera_geometry"]}
    (output / "gpu-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
