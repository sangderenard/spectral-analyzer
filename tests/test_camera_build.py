import numpy as np

from camera_designer.compound_optics import CompoundLens, ConicSurface
from camera_designer.lens_assembly import LensAssemblySpec
from camera_software.camera_build import RebuiltCameraArtifact


def _artifact():
    lens = CompoundLens()
    lens.add(ConicSurface(
        x_pos=0.1,
        R_curvature=0.05,
        n_before=1.0,
        n_after=1.52,
        aperture_r=0.02,
        n_before_spectral=[1.0, 1.0],
        n_after_spectral=[1.53, 1.51],
    ))
    assembly = LensAssemblySpec()
    assembly.set_optics(lens, mode=LensAssemblySpec.MODE_PARAMETRIC)
    return RebuiltCameraArtifact.create(
        scene_config=object(),
        lens_assembly=assembly,
        lens_surface_groups=[
            (np.array([2, 3]), np.array([4, 5]), 0.05, 0.05),
        ],
        scene_lenses=[],
        wavelengths_nm=[450.0, 650.0],
        sensor_center=[0.2, 0.0, 0.0],
    )


def test_rebuilt_camera_provenance_declares_exact_dispersive_transport():
    artifact = _artifact()

    assert artifact.provenance.intersection_model == "exact_parametric_conic"
    assert artifact.provenance.dispersion_model == "sampled_wavelength_sellmeier"
    assert artifact.provenance.wavelength_count == 2
    assert artifact.provenance.diffraction_model == "disabled"


def test_rebuilt_camera_remaps_lens_proxy_groups_without_losing_optics():
    artifact = _artifact()
    remap = np.array([0, -1, 1, 2, 3, 4], np.int64)

    remapped = artifact.remap_triangles(remap)

    assert remapped.lens_assembly is artifact.lens_assembly
    assert remapped.lens_surface_groups[0][0].tolist() == [1, 2]
    assert remapped.lens_surface_groups[0][1].tolist() == [3, 4]
