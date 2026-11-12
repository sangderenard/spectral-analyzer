"""CPU-only source assembly control. No renderer import, GL or compiler calls."""
import ast
from pathlib import Path


def native_control_source(root):
    from camera_designer.spectral_sensor_basis import FIXED3_SPECTRAL_GLSL
    from camera_designer.studio_diffuse_surface import STUDIO_DIFFUSE_GLSL
    from camera_designer.studio_spectral_reflector import STUDIO_REFLECTOR_GLSL
    from camera_designer.studio_boundary_media import STUDIO_BOUNDARY_GLSL
    from camera_designer.studio_region_gpu import with_region_media,with_region_spheres
    from camera_designer.studio_sphere_regions import with_sphere_regions
    from camera_designer.studio_opaque_pbr import with_opaque_pbr
    from camera_designer.studio_directional_field import recording_shader
    from camera_designer.studio_directional_media import with_internal_medium_recording
    import mat_flags
    scope=locals().copy()
    for name in ('MAT_FLAG_SPECTRAL_REFLECTOR','MAT_FLAG_ROUGH_CONDUCTOR','MAT_FLAG_NATIVE_CONDUCTOR_NK','BINDING_STUDIO_NATIVE_PBR'):
        scope[name]=getattr(mat_flags,name)
    tree=ast.parse((Path(root)/'demo_pluck_gl.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('STUDIO_REFLECTOR_GLSL','_GPU_RAY_FIELD_CS','_GPU_RAY_FIELD_REGION_CS') for t in node.targets):
            exec(compile(ast.Module(body=[node],type_ignores=[]),'<native-source-assignment>','exec'),scope)
    # Same existing prebuilt assembly branches, information option absent.
    source=with_opaque_pbr(scope['_GPU_RAY_FIELD_REGION_CS'])
    source=recording_shader(source,stencil=12,grid_size=64,estimator='track-length-root-sweep-v1',independent_bounds=True,quantization_divisor=16777216)
    return with_internal_medium_recording(source)
