"""Integration guide: OpticalAssembly with exposure_render_demo.py

Phase B: Data Model Implementation

This file documents the integration points where OpticalAssembly will be used
in the exposure render pipeline to drive tier-specific behavior.
"""

# ── Phase B Integration Points ──────────────────────────────────────────────
#
# 1. ASSEMBLY CREATION (in ExposureSession)
#
#    When ExposureSession is initialized with a solved camera (from
#    camera_parametric_solver.py), construct an OpticalAssembly:
#
#        from optical_assembly import assembly_from_solved_camera_package
#        if solved is not None:
#            self._optical_assembly = assembly_from_solved_camera_package(
#                solved_package=solved,
#                name=f"camera_frame_{self._frame_index}"
#            )
#
#    Validation at construction time ensures tier consistency:
#        err = self._optical_assembly.validate()
#        if err: raise RuntimeError(err)
#
# 2. TIER-MODE VALIDATION (via camera_mode_validation.py)
#
#    At camera descriptor building (~line 4178), use assembly to verify
#    that the selected tier mode matches available elements:
#
#        TIER_ELEMENT_REQUIREMENTS = {
#            CAMERA_MODE_ORACLE_PINHOLE_REFERENCE: [],  # No optical elements
#            CAMERA_MODE_PHYSICAL_PINHOLE: [STOP],
#            CAMERA_MODE_APERTURE_CONE: [STOP],
#            CAMERA_MODE_THIN_LENS_GEOMETRIC: [STOP, REFRACTOR],
#            CAMERA_MODE_GEOMETRIC_ASSEMBLY: [STOP, REFRACTOR, MIRROR, FILTER],
#            CAMERA_MODE_WAVE_ASSEMBLY: [WAVE_REGION],
#            CAMERA_MODE_BAKED_TRANSFORM: [],  # Pre-computed LUT
#        }
#
#    Then validate:
#        assembly_roles = {e.role for e in assembly.active_elements()}
#        required_roles = TIER_ELEMENT_REQUIREMENTS[camera_mode]
#        if not required_roles.issubset(assembly_roles):
#            raise RuntimeError("Mode {} requires elements: {}".format(...))
#
# 3. OPTICAL HANDLER DISPATCH (in C++ ray tracer)
#
#    For each optical element in assembly, register with tracer:
#
#        for elem in assembly.active_elements():
#            if elem.role == OpticalElementRole.STOP:
#                tracer.register_optical_handler(
#                    handler_type="APERTURE_STOP",
#                    element_id=elem.id,
#                    geometry=elem.geometry,
#                    params={"diameter_m": elem.geometry.diameter_m}
#                )
#
#    This populates the C++ kernel's optical handler table, which
#    produces event counts in CameraEventTelemetry during integration.
#
# 4. TELEMETRY AGGREGATION (post-render)
#
#    After ray tracing completes, extract event counts from C++ backend
#    and correlate with assembly structure:
#
#        for elem_id in assembly_elem_ids:
#            telemetry = back.camera_event_telemetry
#            elem_hits = tracer.query_handler_event_count(elem_id)
#            print(f"  {elem_id}: {elem_hits} rays")
#
#    This validates that optical elements are actually participating,
#    not silently bypassed.
#
# 5. MODE-TO-ASSEMBLY MAPPING (for tier selection)
#
#    When rendering with different tiers, construct appropriate assembly:
#
#        if camera_mode == CAMERA_MODE_APERTURE_CONE:
#            assembly = assembly_from_simple_thin_lens(
#                name="aperture_cone_pinhole",
#                focal_length_m=0.0,  # Pinhole (infinite)
#                aperture_diameter_m=0.003,
#                sensor_distance_m=0.05,
#            )
#        elif camera_mode == CAMERA_MODE_THIN_LENS_GEOMETRIC:
#            assembly = assembly_from_solved_camera_package(solved)
#
# ── Phase C: Optical Handler Implementation ─────────────────────────────────
#
# After Phase B (assembly model) is validated, Phase C implements the C++
# optical handlers that read from OpticalAssembly and produce event counts.
#
# See csrc/include/optical_handlers.h and csrc/optical_handlers.cpp for the
# handler implementations (MIRROR, REFRACTOR, STOP, etc.).
#


# ── Task 5: Separate Oracle / Physical Photon Buffers ─────────────────────
#
# Task 5 (Phase A unfinished item) requires splitting the sensor photon
# accumulator into separate buffers for ORACLE_PINHOLE vs PHYSICAL_PINHOLE:
#
# In exposure_render_demo.py, method _make_sensor_integral(), circa line 3500:
#
#    def _make_sensor_integral(self, ...):
#        # Current: single photons_per_pixel array
#        photons_per_pixel = np.zeros((self.height, self.width), dtype=np.float32)
#
#        # After Task 5: separate oracle and physical buffers
#        if self.camera_mode == CAMERA_MODE_ORACLE_PINHOLE_REFERENCE:
#            photons_per_pixel = ...  # Oracle reference (perfect optics)
#        elif self.camera_mode == CAMERA_MODE_PHYSICAL_PINHOLE:
#            photons_per_pixel = ...  # Physical (with photon noise penalty)
#
# This deferred because it requires distinguishing ray paths at the C++ kernel
# level (oracle vs physical), which ties into optical handler roles.
#
