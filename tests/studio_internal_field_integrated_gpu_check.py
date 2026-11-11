"""Actual canonical forward transport proof; run only in a released GPU window.

This installs the observer helper in this process only. Production/source shader
assembly is not patched. Rays cross the canonical explicit glass boundaries.
"""
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import demo_pluck_gl as renderer
    from OpenGL import GL
    import camera_designer.studio_directional_field as directional
    from camera_designer.studio_directional_media import with_internal_medium_recording
    from camera_designer.film_response import build_film_stack, normalize_film_response
    from camera_designer.marble_color_profile import XYZ_TO_SRGB, _LINE_TO_XYZ

    renderer._ensure_worker_gl_context()
    # Two actual dielectric boundaries, followed by one absorbing stop.
    triangles = np.array([
        [[-.01, -.01, -.001], [-.01, .01, -.001], [.01, 0, -.001]],
        [[-.01, -.01, .001], [.01, -.01, .001], [0, .01, .001]],
        [[-.02, -.02, .004], [-.02, .02, .004], [.02, 0, .004]],
    ], np.float32)
    geom = np.zeros((3, 16), np.float32)
    geom[:, :3] = triangles[:, 0]
    geom[:, 4:7] = triangles[:, 1]-triangles[:, 0]
    geom[:, 8:11] = triangles[:, 2]-triangles[:, 0]
    geom[:, 12:15] = [[0, 0, -1], [0, 0, 1], [0, 0, -1]]
    geom[:2, 3] = 1  # positive side air
    geom[:2, 7] = 2  # negative side material0
    geom[0, 11] = 4097
    geom[2, 15] = np.array([4], np.uint32).view(np.float32)[0]
    shade = np.zeros((3, 16), np.float32)
    shade[:, :4] = [.2, 0, 1, 1]
    shade[:, 4:8] = [.2, 0, 1, 0]
    shade[2, 7] = 1
    shade[:, 8:11] = 1
    nodes, ids = renderer._build_gpu_bvh(triangles)
    sources = np.zeros((1, 12), np.float32)
    sources[0, :3] = [0, 0, -.004]
    sources[0, 3] = 1
    sources[0, 4:7] = [0, 0, 1]
    frequency = float(np.float32(299792458./650e-9))
    sources[0, 8] = frequency
    sources[0, 10] = 1
    materials = np.zeros((32, 12), np.float32)
    materials[0, 0] = frequency
    pbr = np.zeros((1, 16), np.float32)
    pbr[0, :3] = 1
    pbr[0, 5:8] = [1, 1, 0]
    film = build_film_stack(renderer.FILM_STACKS['em_rgb_fixed3'], normalize_film_response())
    bounds = ([-.008]*3, [.008]*3)
    inverse_color = np.linalg.inv(XYZ_TO_SRGB @ _LINE_TO_XYZ)
    original_recording = directional.recording_shader
    checks = []

    def observer(source, *args, **kwargs):
        return with_internal_medium_recording(original_recording(source, *args, **kwargs))

    def trace(index, sigma, enabled, angle=0.0):
        materials[0, 7] = index
        materials[0, 8] = sigma*299792458./(4*np.pi*frequency)
        sources[0, 4:7] = [np.sin(angle), 0, np.cos(angle)]
        pbr[0, 6] = index
        state = None
        try:
            state = renderer._gpu_ray_field_prebuilt(
                geom, shade, materials, nodes, ids, np.zeros((0, 8), np.float32), sources,
                bounds, dims=(8, 8, 8), dispatch_batch=128, max_bounces=8,
                initial_rays=0, film_stack=film, native_pbr=pbr, record_directional=enabled,light_field_internal_media=enabled,light_field_estimator=directional.ROOT_ESTIMATOR,light_field_grid_size=32,directional_quantization_divisor=16777216)
            renderer._clear_prebuilt(state)
            renderer._gpu_pump_prebuilt(state, n_rays=64)
            GL.glFinish()
            scalar = renderer._gpu_readback_prebuilt(state['tex_bands'], state['dims'])
            if not enabled:
                return scalar, None, None, None
            record = state['directional_record']
            atlas = directional.readback(record)
            energy = np.maximum(atlas[..., :3] @ inverse_color.T, 0).sum(axis=-1)
            # Cell centres in the known slab interior, not the exterior field.
            z = -.008+(np.arange(32)+.5)*.0005
            inside = abs(z) < .001
            internal = energy[inside]
            integral = float(internal.sum()*record['voxel_volume_m3'])
            moments = atlas[inside, ..., 3:]
            measured_direction = (moments*internal[..., None]).sum(axis=(0, 1, 2, 3))/internal.sum()
            dominant_direction = moments.reshape(-1, 3)[np.argmax(internal)]
            return scalar, integral, measured_direction, dominant_direction
        finally:
            directional.recording_shader = original_recording
            if state is not None:
                renderer._destroy_prebuilt(state)

    for transmission in (1., .1):
        sigma = -np.log(transmission)/.001
        ordinary, _, _, _ = trace(1., sigma, False)
        observed, integral, direction, _ = trace(1., sigma, True)
        np.testing.assert_array_equal(observed, ordinary)
        # Native boundary continuation starts 1 micrometre after entry.
        distance = .002-1e-6
        expected = (distance if sigma == 0 else -np.expm1(-sigma*distance)/sigma)
        # Independent per-cell integer rounding bound plus float32 intersections.
        error_bound = 64*6*.5/16777216 + 2e-6
        assert abs(integral-expected) <= error_bound, (transmission, integral, expected, error_bound)
        assert integral > 0
        np.testing.assert_allclose(direction, [0, 0, 1], rtol=0, atol=2e-5)
        checks.append(dict(transmission_per_mm=transmission, internal_integral=integral,
                           expected_integral=expected, scalar_bit_identical=True))
    _, integral, direction, dominant = trace(1.5, 0., True, np.pi/6)
    assert integral > 0
    # The direction is produced by the canonical Snell step, not a guessed bin.
    # An all-path average includes real internally reflected paths and is not
    # itself a Snell ray. Check the strongest observed bin's actual direction.
    np.testing.assert_allclose(dominant[:2], [1/3, 0], rtol=0, atol=.03)
    np.testing.assert_allclose(abs(dominant[2]), np.sqrt(8/9), rtol=0, atol=.03)
    checks.append(dict(refracted_bin_direction=dominant.tolist(), all_path_mean=direction.tolist(),
                       expected_forward=[1/3, 0, np.sqrt(8/9)]))
    print(json.dumps(dict(result="PASS", renderer=GL.glGetString(GL.GL_RENDERER).decode(), checks=checks)))


if __name__ == '__main__':
    main()
