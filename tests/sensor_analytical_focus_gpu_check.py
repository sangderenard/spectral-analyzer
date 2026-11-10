"""Opt-in, bounded real-GL camera sampling check; never contacts a live service.

Run directly, not through pytest's GL stubs, after reserving an idle GPU slot.
Both complete composed sensor shader variants are compiled. The probe executes
their unchanged analytical camera block with controlled pixel inputs and reads
back focus points, pupil origins, and ray directions. It does not trace a scene
or claim to validate radiometry or a physical minimum focusing distance.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import demo_pluck_gl as renderer
    from OpenGL import GL as gl

    started = time.monotonic()
    output = ROOT / "build/dot-analytical-focus-20261109"
    output.mkdir(parents=True, exist_ok=True)
    renderer._ensure_worker_gl_context()
    samples = np.array([[x, y, 0, 0] for x in (0., .5, 1.) for y in (0., .5, 1.)], np.float32)
    receipts = []
    for name, source in (("triangle_sensor", renderer._GPU_SENSOR_CS),
                         ("region_sensor", renderer._GPU_SENSOR_REGION_CS)):
        block = source.split("// Analytical thin-lens path.", 1)[1].split("rd = normalize(focus_pt - ro);", 1)[0]
        block += "rd = normalize(focus_pt - ro);"
        probe = '''
        layout(std430,binding=13) readonly buffer FocusProbeIn {vec4 pixels[];};
        layout(std430,binding=14) buffer FocusProbeOut {vec4 result[];};
        void main() {
            int i=int(gl_GlobalInvocationID.x);
            if(i>=9 || gl_GlobalInvocationID.y!=0u)return;
            float pu=pixels[i].x,pv=pixels[i].y;
            uint rng=hash_u(uint(i)+uint(uSeed));
            vec3 ro=uCamEye,rd;
        ''' + block + '''
            result[3*i]=vec4(focus_pt,1);
            result[3*i+1]=vec4(ro,1);
            result[3*i+2]=vec4(rd,1);
        }
        '''
        program = renderer._prog((source.replace("void main() {", "void unused_production_main() {", 1) + probe,
                                  gl.GL_COMPUTE_SHADER))
        buffers = []
        cases = []
        try:
            assert gl.glGetProgramiv(program, gl.GL_LINK_STATUS)
            for binding, data in ((13, samples), (14, np.full((27, 4), np.nan, np.float32))):
                buffer = int(gl.glGenBuffers(1))
                buffers.append(buffer)
                gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffer)
                gl.glBufferData(gl.GL_SHADER_STORAGE_BUFFER, data.nbytes, data, gl.GL_STATIC_DRAW)
                gl.glBindBufferBase(gl.GL_SHADER_STORAGE_BUFFER, binding, buffer)
            gl.glUseProgram(program)
            def u1f(key, value):
                gl.glUniform1f(gl.glGetUniformLocation(program, key), value)
            def u1i(key, value):
                gl.glUniform1i(gl.glGetUniformLocation(program, key), value)
            def u2f(key, value):
                gl.glUniform2f(gl.glGetUniformLocation(program, key), *value)
            for key, vector in (("uCamEye", [0, 0, 0]), ("uCamRight", [1, 0, 0]),
                                ("uCamUp", [0, 1, 0]), ("uCamFwd", [0, 0, 1])):
                gl.glUniform3f(gl.glGetUniformLocation(program, key), *vector)
            tangent = np.float32(math.tan(math.radians(78.) / 2))
            aspect = np.float32(1.77)
            shift = np.array([.13, -.07], np.float32)
            u1f("uCamFovTan", tangent); u1f("uCamAspect", aspect)
            u2f("uTiltShift", shift); u1f("uApertureRot", .23); u1i("uSeed", 9187)
            pinhole = np.column_stack((((samples[:, 0] - .5) * 2 + shift[0]) * tangent * aspect,
                                      ((.5 - samples[:, 1]) * 2 + shift[1]) * tangent,
                                      np.ones(9))).astype(float)
            for focus in (.0005, .001, .005, .019, 10.):
                for tilt in ((0., 0.), (.11, -.09)):
                    tilt = np.asarray(tilt, np.float32)
                    normal = np.array([-math.tan(tilt[1]), -math.tan(tilt[0]), 1.])
                    intercept = max(.001, float(np.float32(focus)))
                    # Independent geometric ray/plane intersection, not a
                    # CPU implementation of the production GLSL assignments.
                    expected = pinhole * (intercept / (pinhole @ normal))[:, None]
                    u1f("uFocusDist", focus); u2f("uLensTilt", tilt)
                    for aperture, blades in ((0., 0), (.002, 0), (.002, 7)):
                        u1f("uApertureRadius", aperture); u1i("uNBlades", blades)
                        gl.glDispatchCompute(1, 1, 1)
                        gl.glMemoryBarrier(gl.GL_SHADER_STORAGE_BARRIER_BIT)
                        gl.glFinish()
                        gl.glBindBuffer(gl.GL_SHADER_STORAGE_BUFFER, buffers[-1])
                        result = np.frombuffer(gl.glGetBufferSubData(gl.GL_SHADER_STORAGE_BUFFER, 0, 27 * 16), np.float32).reshape(9, 3, 4).copy()
                        assert np.isfinite(result).all()
                        points, origins, directions = (result[:, i, :3].astype(float) for i in range(3))
                        np.testing.assert_allclose(points, expected, rtol=3e-6, atol=1e-9)
                        np.testing.assert_allclose(points @ normal, intercept, rtol=3e-6, atol=1e-9)
                        expected_direction = expected - origins
                        expected_direction /= np.linalg.norm(expected_direction, axis=1)[:, None]
                        np.testing.assert_allclose(directions, expected_direction, rtol=3e-6, atol=3e-7)
                        assert np.all(np.linalg.norm(origins, axis=1) <= aperture + 1e-9)
                        if aperture == 0:
                            assert np.all(origins == 0)
                        cases.append(dict(focus_m=focus, tilt_radians=tilt.tolist(), aperture_m=aperture,
                                          blades=blades, samples=9,
                                          max_plane_residual_m=float(np.max(abs(points @ normal - intercept)))))
            receipts.append(dict(shader=name, shader_sha256=hashlib.sha256(source.encode()).hexdigest(),
                                 compiled=True, cases=cases, samples=sum(case["samples"] for case in cases)))
            print(f"PASS {name}: {len(cases)} cases / {receipts[-1]['samples']} actual GLSL camera samples", flush=True)
        finally:
            gl.glUseProgram(0)
            if buffers:
                gl.glDeleteBuffers(len(buffers), buffers)
            gl.glDeleteProgram(program)
    receipt = dict(renderer=gl.glGetString(gl.GL_RENDERER).decode(), elapsed_seconds=time.monotonic() - started,
                   scope="composed shader compilation and analytical camera sampling only", shaders=receipts)
    (output / "gpu-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "shaders"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
