"""Explicit bounded real-producer export test; run only with no active exposure."""
from pathlib import Path
import json
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import demo_pluck_gl as renderer
    from OpenGL import GL
    from camera_designer.film_response import build_film_stack, normalize_film_response
    from camera_designer.studio_forward_field import ForwardFieldPublisher, field_to_linear_srgb
    renderer._ensure_worker_gl_context()
    triangles = np.array([[[-2,-2,0],[2,-2,0],[0,2,0]]], np.float32)
    geom = np.zeros((1,16), np.float32)
    geom[:,:3] = triangles[:,0]
    geom[:,4:7] = triangles[:,1]-triangles[:,0]
    geom[:,8:11] = triangles[:,2]-triangles[:,0]
    geom[:,14] = 1
    shade = np.zeros((1,16), np.float32)
    shade[:,:4] = [.2,0,1,1.5]
    shade[:,4:8] = [.2,0,1,1]
    shade[:,8:11] = 1
    sources = np.zeros((3,12), np.float32)
    sources[:,:3] = [0,0,.4]
    sources[:,3] = 1
    sources[:,4:7] = [0,0,-1]
    sources[:,8] = 2.998e17/np.array([650,550,450])
    sources[:,10] = 1
    nodes, ids = renderer._build_gpu_bvh(triangles)
    film = build_film_stack(renderer.FILM_STACKS['em_rgb_fixed3'], normalize_film_response())
    state = renderer._gpu_ray_field_prebuilt(geom, shade, np.zeros((32,12),np.float32),
        nodes, ids, np.zeros((0,8),np.float32), sources,
        ([-.5,-.5,-.5],[.5,.5,.5]), dims=(8,8,8), dispatch_batch=128,
        max_bounces=1, initial_rays=0, film_stack=film)
    assert state is not None
    try:
        renderer._clear_prebuilt(state)
        renderer._gpu_pump_prebuilt(state, n_rays=128)
        GL.glFinish()
        with tempfile.TemporaryDirectory(prefix='studio-field-export-') as directory:
            publisher = ForwardFieldPublisher(directory, 'gpu-fixture')
            metadata = publisher.snapshot(renderer, state, now=0, sweep=1, ray_jobs=128)
            exported = np.fromfile(Path(directory)/metadata['name'], '<f4').reshape(8,8,8,3)
            before = renderer._gpu_readback_prebuilt(state['tex_bands'], state['dims'])
            assert np.count_nonzero(before) > 0
            np.testing.assert_array_equal(exported, field_to_linear_srgb(before, (8,8,8)))
            # Readback/export must neither clear nor mutate the producer.
            after = renderer._gpu_readback_prebuilt(state['tex_bands'], state['dims'])
            np.testing.assert_array_equal(before, after)
            renderer._clear_prebuilt(state)
            publisher.invalidate()
            assert publisher.snapshot(renderer,state,now=.5) is None
            zero = publisher.snapshot(renderer,state,now=1,sweep=2,ray_jobs=0)
            assert not np.fromfile(Path(directory)/zero['name'],'<f4').any()
            assert (Path(directory)/metadata['name']).read_bytes() == exported.tobytes()
            print(json.dumps(dict(renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                result='PASS actual forward producer, exact exported color/layout, immutable snapshot, fresh-field reset',
                source_rays=128, dimensions=[8,8,8], byte_length=metadata['byte_length'],
                nonzero_counts=int(np.count_nonzero(before)), min_rgb=float(exported.min()),
                max_rgb=float(exported.max()), metadata=metadata), indent=2))
    finally:
        renderer._destroy_prebuilt(state)


if __name__ == '__main__':
    main()
