from __future__ import annotations

import math

import numpy as np

from camera_designer.compound_optics import RayBundle, TerminationReason
from camera_software.specialty_optics import (
    DramaticFisheyeSpec,
    PentaprismSpec,
    build_pentaprism_diagnostic,
)
from thick_lens_focus_lab import (
    SceneConfig,
    _build_lens_mesh,
    _compound_lens_from_stack,
    _lens_back_x,
    _lens_front_x,
    _lens_is_valid,
)


def _reflected(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    normal = normal / np.linalg.norm(normal)
    return direction - 2.0 * float(np.dot(direction, normal)) * normal


def test_pentaprism_is_closed_and_turns_chief_ray_ninety_degrees() -> None:
    assembly = PentaprismSpec((0.0, 0.0, 0.0)).build()
    assert assembly.triangles.shape == (16, 3, 3)

    # Every geometric edge of the closed triangular shell is shared twice.
    edge_counts: dict[tuple[tuple[float, ...], tuple[float, ...]], int] = {}
    for triangle in assembly.triangles:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((tuple(triangle[first]), tuple(triangle[second]))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    assert set(edge_counts.values()) == {2}

    p0, p1, p2, p3 = assembly.primary_path
    incoming = (p1 - p0) / np.linalg.norm(p1 - p0)
    middle = (p2 - p1) / np.linalg.norm(p2 - p1)
    outgoing = (p3 - p2) / np.linalg.norm(p3 - p2)
    for role, before, after in (
        ("silvered_reflector_1", incoming, middle),
        ("silvered_reflector_2", middle, outgoing),
    ):
        triangle = assembly.role_triangles(role)[0]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        reflected = _reflected(before, normal)
        assert np.allclose(reflected, after, atol=1.0e-12)
    assert abs(float(np.dot(incoming, outgoing))) < 1.0e-12


def test_diagnostic_assembly_has_real_splitter_prism_and_receiver_surfaces() -> None:
    assembly = build_pentaprism_diagnostic(
        pickoff_center_m=(1.0, 0.0, 0.0)
    )
    assert assembly.triangles.shape == (20, 3, 3)
    assert assembly.roles.count("diagnostic_beam_splitter") == 2
    assert assembly.roles.count("silvered_reflector_1") == 2
    assert assembly.roles.count("silvered_reflector_2") == 2
    assert assembly.roles.count("instrument_receiver") == 2
    indices = assembly.material_indices({
        "diagnostic_beam_splitter": 1,
        "entrance_glass": 2,
        "exit_glass": 2,
        "silvered_reflector_1": 3,
        "silvered_reflector_2": 3,
        "blackened_prism_face": 4,
        "blackened_prism_side": 4,
        "instrument_receiver": 5,
    })
    assert indices.shape == (20,)


def test_fisheye_mesh_and_exact_transport_share_every_conic() -> None:
    spec = DramaticFisheyeSpec()
    stack = spec.lens_stack()
    assert len(stack) == 7
    assert all(_lens_is_valid(lens) for lens in stack)
    assert any(abs(lens.conic_front) > 0.0 for lens in stack)
    assert any(abs(lens.conic_back) > 0.0 for lens in stack)

    triangles: list[np.ndarray] = []
    materials: list[int] = []
    front_ids: list[int] = []
    back_ids: list[int] = []
    _build_lens_mesh(
        stack[0], triangles, materials, 0, n_theta=12, n_radial=8,
        front_tri_ids=front_ids, back_tri_ids=back_ids,
    )
    mesh = np.stack(triangles)[front_ids + back_ids]
    radii = np.linalg.norm(mesh[..., 1:3], axis=-1)
    x = mesh[..., 0]
    expected_front = _lens_front_x(stack[0], radii)
    expected_back = _lens_back_x(stack[0], radii)
    on_a_surface = np.isclose(x, expected_front) | np.isclose(x, expected_back)
    assert np.all(on_a_surface)

    compound = _compound_lens_from_stack(stack)
    conics = [element.conic_k for element in compound.elements if hasattr(element, "conic_k")]
    expected = [value for lens in stack for value in (lens.conic_front, lens.conic_back)]
    assert np.allclose(conics, expected)


def test_fisheye_transmits_a_real_120_degree_field_without_image_warp() -> None:
    stack = DramaticFisheyeSpec().lens_stack()
    compound = _compound_lens_from_stack(stack)
    vertex_x = stack[0].x_front
    launch_x = vertex_x - 0.20
    half_field = math.radians(60.0)
    direction = np.array([math.cos(half_field), math.sin(half_field), 0.0])
    target_offsets = np.linspace(-0.12, 0.12, 481)
    origins_y = target_offsets - math.tan(half_field) * (vertex_x - launch_x)
    origins = np.column_stack([
        np.full(target_offsets.size, launch_x),
        origins_y,
        np.zeros(target_offsets.size),
    ])
    result = compound.evaluate_bundle(
        RayBundle(origins, np.tile(direction, (target_offsets.size, 1)))
    )
    assert np.count_nonzero(
        result.status == int(TerminationReason.PASSED.value)
    ) > 0

    scene = DramaticFisheyeSpec().apply_to_scene(SceneConfig())
    assert scene.optical_design is None
    assert scene.lens_stack == stack
    assert scene.image_plate.x > stack[-1].x_back
    assert scene.lens_hood_min_half_field_deg == 60.0
    assert scene.ring_light_enabled is False
