"""Analytic GPU extinction proof. Run only when the shared GPU is unoccupied."""
from pathlib import Path
import json
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def oracle(sigma, bounds, a, b):
    """Independent cell-plane event integration, not the GPU DDA algorithm."""
    a, b, bounds = (np.asarray(x, np.float64) for x in (a, b, bounds))
    delta = b - a
    length = np.linalg.norm(delta)
    if length == 0:
        return [1, 0]
    events = [0., 1.]
    for k, n in enumerate(sigma.shape[::-1]):
        if delta[k] != 0:
            events.extend(t for t in (np.linspace(bounds[0, k], bounds[1, k], n + 1) - a[k]) / delta[k] if 0 < t < 1)
    events = sorted(set(events))
    trans, integral = 1., 0.
    for left, right in zip(events, events[1:]):
        midpoint = a + delta * ((left + right) / 2)
        q = np.floor((midpoint - bounds[0]) / (bounds[1] - bounds[0]) * sigma.shape[::-1]).astype(int)
        value = sigma[tuple(q[::-1])] if np.all(q >= 0) and np.all(q < sigma.shape[::-1]) else 0.
        distance = (right - left) * length
        weight = -np.expm1(-value * distance) / value if value else distance
        integral += trans * weight
        trans *= np.exp(-value * distance)
    return [trans, integral]


def host_transport_check(r, GL):
    from camera_designer.studio_volume_extinction import ExtinctionGrid
    from camera_designer.film_response import build_film_stack, normalize_film_response
    triangles = np.array([[[-2, -2, 0], [2, -2, 0], [0, 2, 0]]], np.float32)
    geom = np.zeros((1, 16), np.float32)
    geom[:, :3] = triangles[:, 0]
    geom[:, 4:7] = triangles[:, 1] - triangles[:, 0]
    geom[:, 8:11] = triangles[:, 2] - triangles[:, 0]
    geom[:, 14] = 1
    geom[:, 15] = np.array([1], np.uint32).view(np.float32)  # real emissive surface
    shade = np.zeros((1, 16), np.float32)
    shade[:, :4] = [.2, 0, 1, 1.5]
    shade[:, 4:8] = [.2, 0, 1, 1]
    shade[:, 8:13] = 1
    pbr = np.zeros((1, 16), np.float32)
    pbr[:, :3] = 1; pbr[:, 3] = .5; pbr[:, 6] = 1.5; pbr[:, 7] = 1
    sources = np.zeros((3, 12), np.float32)
    sources[:, :3] = [0, 0, .4]; sources[:, 3] = 1; sources[:, 4:7] = [0, 0, -1]
    sources[:, 8] = 2.998e17 / np.array([650, 550, 450]); sources[:, 10] = 1
    film = build_film_stack(r.FILM_STACKS['em_rgb_fixed3'], normalize_film_response())
    nodes, ids = r._build_gpu_bvh(triangles)
    results = []
    for coefficient in (None, 0., 2.):
        grid = None if coefficient is None else ExtinctionGrid(np.full((3, 4, 5), coefficient), [[-.5]*3, [.5]*3])
        state = r._gpu_ray_field_prebuilt(geom, shade, np.zeros((32, 12), np.float32), nodes, ids,
            np.zeros((0, 8), np.float32), sources, ([-.5]*3, [.5]*3), dims=(8, 8, 8),
            dispatch_batch=128, max_bounces=1, initial_rays=0, film_stack=film,
            native_pbr=pbr, record_directional=True, volume_extinction=grid)
        assert state is not None
        accumulator = None
        try:
            r._gpu_pump_prebuilt(state, n_rays=128)
            GL.glFinish()
            scalar = r._gpu_readback_prebuilt(state['tex_bands'], state['dims'])
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, state['directional_record']['buffer'])
            size = int(GL.glGetBufferParameteriv(GL.GL_SHADER_STORAGE_BUFFER, GL.GL_BUFFER_SIZE))
            raw = np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, size), np.int32)
            assert raw[-1] == 0
            track = raw[:-1].reshape(-1, 6)[:, :3].sum(dtype=np.float64)
            accumulator = r.SensorAccumulator(state['ssbo'][0], state['ssbo'][1], state['ssbo'][2],
                state['ssbo'][9], state['ssbo'][10], 1, len(nodes), state['tex_bands'],
                state['bmin'], state['bmax'], state['dims'], [0, 0, .4], [0, 0, 0], 16, 16,
                rows_per_frame=16, max_bounces=1, aperture_radius=0, vol_alpha=0,
                n_sources=len(sources),
                ssbo_native_pbr=state['native_pbr_ssbo'], full_sensor_size=(16, 16),
                volume_extinction=state['volume_extinction'])
            assert accumulator._active
            accumulator.film = film
            accumulator._fov_tan = 0.  # identical known 0.4 m test rays
            accumulator.tick()
            GL.glFinish()
            GL.glBindTexture(GL.GL_TEXTURE_2D, accumulator._textures[0])
            pixels = np.frombuffer(GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_FLOAT), np.float32).reshape(16, 16, 4).copy()
            assert np.all(pixels[..., 0] > 0)
            # A non-emissive receiving face, zero forward field, and the real
            # source connection isolate camera T times direct-light-path T.
            GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, state['ssbo'][0])
            GL.glBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 15*4, 4, np.zeros(1, np.uint32))
            r._clear_prebuilt(state)
            accumulator.clear()
            accumulator.set_nee_spectral_sources(sources, state['ssbo'][5])
            accumulator.tick()
            GL.glFinish()
            GL.glBindTexture(GL.GL_TEXTURE_2D, accumulator._textures[0])
            nee_pixels = np.frombuffer(GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_FLOAT), np.float32).reshape(16, 16, 4).copy()
            assert np.all(nee_pixels[..., 0] > 0)
            results.append((scalar, track, pixels, nee_pixels))
        finally:
            if accumulator is not None:
                accumulator.destroy()
            buffer = state['volume_extinction']['buffer'] if state['volume_extinction'] else 0
            r._destroy_prebuilt(state)
            if buffer:
                assert not GL.glIsBuffer(buffer)
    np.testing.assert_array_equal(results[0][0], results[1][0])
    assert results[0][1] == results[1][1]
    np.testing.assert_array_equal(results[0][2], results[1][2])
    np.testing.assert_array_equal(results[0][3], results[1][3])
    assert np.all(results[2][0] <= results[0][0]) and np.any(results[2][0] < results[0][0])
    ratio = results[2][2][..., 0] / results[0][2][..., 0]
    np.testing.assert_allclose(ratio, np.exp(-2*.4), rtol=2e-5)
    track_ratio = results[2][1] / results[0][1]
    np.testing.assert_allclose(track_ratio, -np.expm1(-2*.4)/(2*.4), rtol=.001)
    nee_ratio = results[2][3][..., 0] / results[0][3][..., 0]
    np.testing.assert_allclose(nee_ratio, np.exp(-2*.8), rtol=3e-5)
    return dict(zero_grid_bit_identical=True, camera_transmittance=float(ratio.mean()),
                expected_camera_transmittance=float(np.exp(-.8)),
                camera_times_source_transmittance=float(nee_ratio.mean()),
                measured_track_length_ratio=float(track_ratio), resources_released=True)


def main():
    import demo_pluck_gl as r
    from OpenGL import GL
    from camera_designer.studio_volume_extinction import ExtinctionGrid, GLSL, allocate, bind, transport_shader
    from camera_designer.studio_directional_field import recording_shader
    r._ensure_worker_gl_context()
    for name in ("_GPU_RAY_FIELD_CS", "_GPU_RAY_FIELD_REGION_CS", "_GPU_SENSOR_CS", "_GPU_SENSOR_REGION_CS"):
        sources = [getattr(r, name)]
        if "RAY" in name:
            sources.append(recording_shader(sources[0]))
        for source in sources:
            program = r._prog((transport_shader(source, sensor="SENSOR" in name), GL.GL_COMPUTE_SHADER))
            try:
                assert GL.glGetProgramiv(program, GL.GL_LINK_STATUS), GL.glGetProgramInfoLog(program)
            finally:
                GL.glDeleteProgram(program)
    source = "#version 430 core\n" + GLSL + r'''
layout(local_size_x=1) in;
layout(std430,binding=0) readonly buffer Rays { vec4 ends[]; };
layout(std430,binding=1) writeonly buffer Results { vec2 results[]; };
void main(){uint i=gl_GlobalInvocationID.x;results[i]=studio_volume_path(ends[2*i].xyz,ends[2*i+1].xyz);}
'''
    program = r._prog((source, GL.GL_COMPUTE_SHADER))
    assert GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    rng = np.random.default_rng(9327)
    queries = rng.uniform(-1.5, 1.5, (128, 2, 3)).astype(np.float32)
    special = np.array([[[-2,0,0],[2,0,0]], [[2,0,0],[-2,0,0]],
                        [[-1,-1,-1],[1,1,1]], [[1,1,1],[-1,-1,-1]],
                        [[0,0,0],[0,0,0]], [[-2,1,0],[2,1,0]],
                        [[0,0,0],[1,0,0]], [[1,0,0],[0,0,0]]], np.float32)
    queries = np.concatenate([queries, special])
    packed = np.zeros((len(queries), 2, 4), np.float32)
    packed[:, :, :3] = queries
    buffers = [int(i) for i in GL.glGenBuffers(2)]
    try:
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffers[0])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL.GL_STATIC_DRAW)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 0, buffers[0])
        GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffers[1])
        GL.glBufferData(GL.GL_SHADER_STORAGE_BUFFER, len(queries)*8, None, GL.GL_DYNAMIC_READ)
        GL.glBindBufferBase(GL.GL_SHADER_STORAGE_BUFFER, 1, buffers[1])
        max_error = 0.
        for sigma in (np.zeros((2, 3, 4)), np.full((3, 4, 5), 2.),
                      rng.uniform(0, 4, (5, 3, 7)), np.full((64, 64, 64), 0.00001)):
            grid = ExtinctionGrid(sigma, [[-1]*3, [1]*3])
            record = allocate(grid)
            try:
                GL.glUseProgram(program)
                bind(record, program)
                GL.glDispatchCompute(len(queries), 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT | GL.GL_BUFFER_UPDATE_BARRIER_BIT)
                GL.glBindBuffer(GL.GL_SHADER_STORAGE_BUFFER, buffers[1])
                actual = np.frombuffer(GL.glGetBufferSubData(GL.GL_SHADER_STORAGE_BUFFER, 0, len(queries)*8), np.float32).reshape(-1, 2)
                expected = np.array([oracle(grid.sigma_t_m_inv, grid.bounds_m, *q) for q in queries])
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-6)
                max_error = max(max_error, float(np.abs(actual-expected).max()))
            finally:
                GL.glDeleteBuffers(1, [record["buffer"]])
    finally:
        GL.glDeleteBuffers(len(buffers), buffers)
        GL.glDeleteProgram(program)
    host = host_transport_check(r, GL)
    print(json.dumps(dict(result="PASS", renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                          variants=6, analytic_rays=len(queries)*4, maximum_absolute_error=max_error,
                          host_transport=host)))


if __name__ == "__main__":
    main()
